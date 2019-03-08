import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List

import pandas as pd
from numpy import array
from pandas import DataFrame
import torch

from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from models import BertForMultiLabelSequenceClassification

bert_model_types = Union[
    BertForSequenceClassification, BertForMultiLabelSequenceClassification
]


def save_model(
    model: bert_model_types,
    training_parameters: Dict,
    export_dir: Optional[str],
    model_name: str,
) -> None:
    if not export_dir:
        now_timestamp = datetime.now().strftime("%y-%m-%dT%H-%M-%S")
        export_path = Path(f"./{now_timestamp}")
    else:
        export_path = Path(export_path)

    export_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), export_path / f"{model_name}-model.pt")
    json_config = export_path / f"{model_name}-config.json"
    json_config.write_text(model.config.to_json_string())
    training_config = export_path / f"{model_name}-training-parameters.json"
    training_config.write_text(json.dumps(training_parameters))


def load_model_classification(
    model_path: Path, model_name: str, num_labels: int = 2
) -> BertForSequenceClassification:
    model_path = Path(model_path)
    config = BertConfig(str(model_path / f"{model_name}-config.json"))
    model = BertForSequenceClassification(config, num_labels=num_labels)
    model.load_state_dict(torch.load(str(model_path / f"{model_name}-model.pt")))
    return model


def load_model_multilabel(
    model_path: Path, model_name: str, num_labels: int = 2
) -> BertForMultiLabelSequenceClassification:
    model_path = Path(model_path)
    config = BertConfig(str(model_path / f"{model_name}-config.json"))
    model = BertForMultiLabelSequenceClassification(  # type: ignore
        config, num_labels=num_labels
    )
    model.load_state_dict(torch.load(str(model_path / f"{model_name}-model.pt")))
    return model


def load_classification_data(
    input_csv: str, text_column: str, label_column: str
) -> Tuple[array, array]:
    df = pd.read_csv(input_csv)
    texts = df[text_column].values
    labels = df[label_column].values
    return texts, labels


def load_multilabel_data(
    input_csv: str, text_column: str, label_columns: List
) -> Tuple[array, array]:
    df = pd.read_csv(input_csv)
    texts = df["texts"].values
    labels = df[label_columns].values
    return texts, labels


def load_evaluation_data(input_csv: str) -> Tuple[array, DataFrame]:
    df = pd.read_csv(input_csv)
    texts = df["texts"].values
    return texts, df


def create_training_parameters(
    num_labels: int, problem_type: str, max_seq_len: int, epochs: int
) -> Dict:
    return dict(
        num_labels=num_labels,
        problem_type=problem_type,
        epochs=epochs,
        max_seq_len=max_seq_len,
    )


def load_training_config(json_path: str) -> Dict:
    with open(json_path, "r") as f:
        training_config = json.load(f)
    return training_config
