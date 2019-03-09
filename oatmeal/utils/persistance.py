import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List, Any

import pandas as pd
from numpy import array
from pandas import DataFrame
import torch

from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from .models import BertForMultiLabelSequenceClassification

bert_model_types = Union[
    BertForSequenceClassification, BertForMultiLabelSequenceClassification
]


def save_model(
    model: bert_model_types,
    training_parameters: Dict[str, str],
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
    model = BertForMultiLabelSequenceClassification(config, num_labels=num_labels)
    model.load_state_dict(torch.load(str(model_path / f"{model_name}-model.pt")))
    return model


def load_classification_data(
    input_csv: str, text_column: str, label_column: str
) -> Tuple[array, array, Optional[Dict[str, int]]]:
    df = pd.read_csv(input_csv, dtype={text_column: str}).dropna(
        subset=[text_column, label_column]
    )
    texts = df[text_column].values
    labelmap = None
    if df[label_column].dtype == "O":
        labelmap = {label: i for i, label in enumerate(df[label_column].unique())}
        labels = df[label_column].map(labelmap).values
    else:
        labels = df[label_column].values
    return texts, labels, labelmap


def load_multilabel_data(
    input_csv: str, text_column: str, label_names: List[str]
) -> Tuple[array, array]:
    dtype = {text_column: str}
    for label in label_names:
        dtype[label] = int
    df = pd.read_csv(input_csv, dtype=dtype)
    texts = df[text_column].values
    labels = df[label_names].values
    return texts, labels


def load_evaluation_data(input_csv: str) -> Tuple[array, DataFrame]:
    df = pd.read_csv(input_csv)
    texts = df["texts"].values
    return texts, df


def create_training_parameters(
    num_labels: int,
    problem_type: str,
    max_seq_len: int,
    epochs: int,
    label_names: List[str],
) -> Dict[str, Any]:
    return dict(
        num_labels=num_labels,
        problem_type=problem_type,
        epochs=epochs,
        max_seq_len=max_seq_len,
        label_names=label_names,
    )


def load_training_config(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r") as f:
        training_config = json.load(f)
    return training_config
