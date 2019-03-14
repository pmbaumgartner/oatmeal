import click

import torch

from models import (
    get_bert_binary_model,  # remove
    get_bert_multiclass_model,
    get_bert_multilabel_model,
    get_bert_opt,
    run_model_training,
    run_prediction_sigmoid,
    run_prediction_softmax,
)
from persistance import (
    create_training_parameters,
    load_classification_data,
    load_evaluation_data,
    load_model_classification,
    load_model_multilabel,
    load_multilabel_data,
    load_training_config,
    save_model,
)
from processing import (
    build_binary_predictions_df,
    build_multi_predictions_df,
    create_prediction_dataloader,
    create_training_dataloader,
)

TRAIN_EPOCHS = 3
PRED_BATCH_SIZE = 64


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if device == "cpu":
    TRAIN_BATCH_SIZE = 16
    MAX_SEQ_LEN = 64
else:
    TRAIN_BATCH_SIZE = 32
    MAX_SEQ_LEN = 128


print(TRAIN_BATCH_SIZE, MAX_SEQ_LEN)


@click.group()
def cli():
    pass


@cli.group("train")
@click.option(
    "-i",
    "--input-data",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    nargs=1,
)
@click.option(
    "-e",
    "--eval-data",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    nargs=1,
)
@click.option(
    "-o",
    "--export-path",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True),
    nargs=1,
)
@click.option("-n", "--model-name", type=str, nargs=1)
@click.option("-b", "--batch-size", type=int, default=TRAIN_BATCH_SIZE, nargs=1)
@click.option("-e", "--epochs", type=int, default=TRAIN_EPOCHS, nargs=1)
@click.option("-l", "--max-seq-len", type=int, default=MAX_SEQ_LEN, nargs=1)
@click.pass_context
def train(
    ctx, input_data, eval_data, export_path, model_name, batch_size, epochs, max_seq_len
):
    args = locals()
    for key in args:
        if key != "ctx":
            ctx.obj[key] = args[key]


@train.command("multiclass")
@click.option(
    "-x", "--text-column", required=True, type=str, help="Column header for the text"
)
@click.option(
    "-y", "--label-column", required=True, type=str, help="Column header for the labels"
)
@click.argument("label_names", nargs=-1, type=str)
@click.pass_obj
def multiclass(ctx, text_column, label_column, label_names):
    """Train Multiclass classification

    If labels are already encoded as integers,
    you can pass friendly [LABEL_NAMES] as the last argument.
    If labels are not encoded as integers,
    they will be auto encoded in alphanumeric order.
    """

    texts, labels, labelmap = load_classification_data(
        ctx["input_data"], text_column=text_column, label_column=label_column
    )
    num_labels = len(set(labels))
    n_train_examples = len(texts)

    if len(label_names) != 0 and num_labels != len(label_names):
        raise click.BadParameter(
            f"{num_labels} label classes detected, {len(label_names)} label names passed."
        )

    training_dataloader = create_training_dataloader(
        texts, labels, ctx["max_seq_len"], ctx["batch_size"]
    )

    bert_model = get_bert_multiclass_model(num_labels=num_labels)
    bert_opt = get_bert_opt(
        bert_model, n_train_examples, ctx["batch_size"], ctx["epochs"]
    )

    trained_model = run_model_training(
        bert_model, bert_opt, training_dataloader, ctx["epochs"]
    )

    if not ctx["model_name"]:
        ctx["model_name"] = "multiclass"
    if labelmap:
        label_names = [
            label for (label, _) in sorted(labelmap.items(), key=lambda x: x[1])
        ]

    training_parameters = create_training_parameters(
        num_labels=num_labels,
        problem_type="multiclass",
        max_seq_len=ctx["max_seq_len"],
        epochs=ctx["epochs"],
        label_names=label_names,
    )

    save_model(
        trained_model, training_parameters, ctx["export_path"], ctx["model_name"]
    )


@train.command("multilabel")
@click.option("-x", "--text-column", required=True, type=str)
@click.argument("label_names", required=True, nargs=-1, type=str)
@click.pass_obj
def multilabel(ctx, text_column, label_names):
    """Train Multilabel classification

    This will use columns passed as the last argument [LABEL_NAMES]
    Each column should be binary integer encoded
    """
    texts, labels = load_multilabel_data(
        ctx["input_data"], text_column=text_column, label_names=list(label_names)
    )
    num_labels = labels.shape[1]
    n_train_examples = len(texts)
    training_dataloader = create_training_dataloader(
        texts, labels, ctx["max_seq_len"], ctx["batch_size"]
    )
    bert_model = get_bert_multilabel_model(num_labels=num_labels)
    bert_opt = get_bert_opt(
        bert_model, n_train_examples, ctx["batch_size"], ctx["epochs"]
    )

    trained_model = run_model_training(
        bert_model, bert_opt, training_dataloader, ctx["epochs"]
    )

    if not ctx["model_name"]:
        ctx["model_name"] = "multilabel"

    training_parameters = create_training_parameters(
        num_labels=num_labels,
        problem_type="multilabel",
        max_seq_len=ctx["max_seq_len"],
        epochs=ctx["epochs"],
        label_names=label_names,
    )

    save_model(
        trained_model, training_parameters, ctx["export_path"], ctx["model_name"]
    )


@cli.command("predict")
@click.option(
    "-i",
    "--input-data",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, resolve_path=True),
    nargs=1,
)
@click.option(
    "-p",
    "--model-path",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, resolve_path=True),
    nargs=1,
)
@click.option("-n", "--model-name", type=str, nargs=1)
@click.option("-b", "--batch-size", type=int, default=TRAIN_BATCH_SIZE, nargs=1)
@click.option("-x", "--text-column", required=True, type=str)
def predict(input_data, model_path, model_name, batch_size, text_column):
    texts, df = load_evaluation_data(input_data, text_column)
    training_parameters = load_training_config(
        str(model_path + f"/{model_name}-training-parameters.json")
    )
    problem_type = training_parameters["problem_type"]
    num_labels = training_parameters["num_labels"]
    max_seq_len = training_parameters["max_seq_len"]
    label_names = training_parameters["label_names"]
    if problem_type in ("binary", "multiclass"):
        model = load_model_classification(model_path, model_name, num_labels=num_labels)
        prediction_loader = create_prediction_dataloader(texts, max_seq_len, batch_size)
        pred_softmax = run_prediction_softmax(model, prediction_loader)
        if problem_type == "binary":
            predictions_df = build_binary_predictions_df(df, pred_softmax)
            predictions_df.to_csv(model_path + "/preds.csv")
        elif problem_type == "multiclass":
            predictions_df = build_multi_predictions_df(df, pred_softmax, label_names)
            predictions_df.to_csv(model_path + "/preds.csv")

    elif problem_type == "multilabel":
        model = load_model_multilabel(
            str(model_path), model_name, num_labels=num_labels
        )
        prediction_loader = create_prediction_dataloader(texts, max_seq_len, batch_size)
        pred_sigmoid = run_prediction_sigmoid(model, prediction_loader)
        predictions_df = build_multi_predictions_df(df, pred_sigmoid, label_names)
        predictions_df.to_csv(model_path + "/preds.csv")


if __name__ == "__main__":
    cli(obj={})
