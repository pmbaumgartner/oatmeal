from typing import Optional

import fire
import torch

from models import (
    get_bert_binary_model,
    get_bert_multiclass_model,
    get_bert_multilabel_model,
    get_bert_opt,
    run_model_training,
    run_prediction_sigmoid,
    run_prediction_softmax,
)
from persistence import (
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
    create_training_dataloader_multilabel,
)

TRAIN_BATCH_SIZE = 16
TRAIN_EPOCHS = 3
MAX_SEQ_LEN = 64
PRED_BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device == "cuda":
    TRAIN_BATCH_SIZE = 32
    MAX_SEQ_LEN = 128


class TrainStage(object):
    def __init__(
        self,
        input_data: str,
        eval_data: Optional[str] = None,
        export_path: Optional[str] = None,
        model_name: Optional[str] = None,
        batch_size: int = TRAIN_BATCH_SIZE,
        epochs: int = TRAIN_EPOCHS,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        self.input_data = input_data
        self.eval_data = eval_data
        self.export_path = export_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_seq_len = max_seq_len

    def binary(self) -> None:
        texts, labels = load_classification_data(self.input_data)
        training_dataloader = create_training_dataloader(
            texts, labels, self.max_seq_len, self.batch_size
        )
        bert_model = get_bert_binary_model()
        n_train_examples = len(texts)
        bert_opt = get_bert_opt(
            bert_model, n_train_examples, self.batch_size, self.epochs
        )

        trained_model = run_model_training(
            bert_model, bert_opt, training_dataloader, self.epochs
        )

        if not self.model_name:
            self.model_name = "binary"

        training_parameters = create_training_parameters(
            num_labels=2,
            problem_type="binary",
            max_seq_len=self.max_seq_len,
            epochs=self.epochs,
        )

        save_model(
            trained_model, training_parameters, self.export_path, self.model_name
        )

    def multiclass(self) -> None:
        texts, labels = load_classification_data(self.input_data)
        training_loader = create_training_dataloader(
            texts, labels, self.max_seq_len, self.batch_size
        )
        num_labels = len(set(labels))
        bert_model = get_bert_multiclass_model(num_labels=num_labels)
        n_train_examples = len(texts)
        bert_opt = get_bert_opt(
            bert_model, n_train_examples, self.batch_size, self.epochs
        )

        trained_model = run_model_training(
            bert_model, bert_opt, training_loader, self.epochs
        )

        if not self.model_name:
            self.model_name = "multiclass"

        training_parameters = create_training_parameters(
            num_labels=2,
            problem_type="multiclass",
            max_seq_len=self.max_seq_len,
            epochs=self.epochs,
        )

        save_model(
            trained_model, training_parameters, self.export_path, self.model_name
        )

    def multilabel(self) -> None:
        texts, labels = load_multilabel_data(self.input_data)
        training_loader = create_training_dataloader_multilabel(
            texts, labels, self.max_seq_len, self.batch_size
        )
        num_labels = labels.shape[1]
        bert_model = get_bert_multilabel_model(num_labels=num_labels)
        n_train_examples = len(texts)
        bert_opt = get_bert_opt(
            bert_model, n_train_examples, self.batch_size, self.epochs
        )

        trained_model = run_model_training(
            bert_model, bert_opt, training_loader, self.epochs
        )

        if not self.model_name:
            self.model_name = "multilabel"

        training_parameters = create_training_parameters(
            num_labels=2,
            problem_type="binary",
            max_seq_len=self.max_seq_len,
            epochs=self.epochs,
        )

        save_model(
            trained_model, training_parameters, self.export_path, self.model_name
        )


class PredictStage(object):
    def __init__(
        self,
        input_data: str,
        model_path: str,
        model_name: str,
        batch_size=PRED_BATCH_SIZE,
    ):
        self.input_data = input_data
        self.model_path = model_path
        self.model_name = model_name
        self.batch_size = batch_size

        texts, df = load_evaluation_data(self.input_data)
        training_parameters = load_training_config(
            str(self.model_path + f"/{self.model_name}-training-parameters.json")
        )
        problem_type = training_parameters["problem_type"]
        num_labels = training_parameters["num_labels"]
        max_seq_len = training_parameters["max_seq_len"]
        if problem_type in ("binary", "multiclass"):
            model = load_model_classification(
                self.model_path, self.model_name, num_labels=num_labels
            )
            prediction_loader = create_prediction_dataloader(
                texts, max_seq_len, self.batch_size
            )
            pred_softmax = run_prediction_softmax(model, prediction_loader)
            if problem_type == "binary":
                predictions_df = build_binary_predictions_df(df, pred_softmax)
                predictions_df.to_csv(self.model_path + "/preds.csv")
            elif problem_type == "multiclass":
                predictions_df = build_multi_predictions_df(df, pred_softmax)
                predictions_df.to_csv(self.model_path + "/preds.csv")

        elif problem_type == "multilabel":
            model = load_model_multilabel(
                str(self.model_path), self.model_name, num_labels=num_labels
            )
            prediction_loader = create_prediction_dataloader(
                texts, max_seq_len, self.batch_size
            )
            pred_sigmoid = run_prediction_sigmoid(model, prediction_loader)
            predictions_df = build_multi_predictions_df(df, pred_sigmoid)
            predictions_df.to_csv(self.model_path + "/preds.csv")


class Pipeline(object):
    def __init__(self, input_data, model_path=None, model_name=None):
        self.train = TrainStage(input_data)
        self.predict = PredictStage(input_data, model_path, model_name)


if __name__ == "__main__":
    fire.Fire(Pipeline)
