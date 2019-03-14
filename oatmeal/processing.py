from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from numpy import array
from pandas import DataFrame
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


def text_to_token_ids(
    text: str, max_seq_len: int
) -> Tuple[List[int], List[int], List[int]]:
    initial_tokens = tokenizer.tokenize(text)
    if len(initial_tokens) > max_seq_len - 2:
        initial_tokens = initial_tokens[: (max_seq_len - 2)]

    tokens = ["[CLS]"] + initial_tokens + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    padding = [0] * (max_seq_len - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_len
    assert len(input_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len

    return input_ids, input_mask, segment_ids


def tensorize_texts(
    raw_texts: List[str], max_seq_len: int
) -> Tuple[Tensor, Tensor, Tensor]:
    features = [text_to_token_ids(text, max_seq_len) for text in raw_texts]
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    return all_input_ids, all_input_mask, all_segment_ids


def labels_to_tensor(labels: array) -> Tensor:
    all_label_ids = torch.tensor(labels, dtype=torch.long)
    return all_label_ids


def build_train_dataloader(
    all_input_ids: Tensor,
    all_input_mask: Tensor,
    all_segment_ids: Tensor,
    all_label_ids: Tensor,
    batch_size: int,
) -> DataLoader:
    dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )
    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


def build_predict_dataloader(
    all_input_ids: Tensor,
    all_input_mask: Tensor,
    all_segment_ids: Tensor,
    batch_size: int,
) -> DataLoader:
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


def create_training_dataloader(
    texts: array, labels: array, max_seq_len: int, batch_size: int
) -> DataLoader:
    all_input_ids, all_input_mask, all_segment_ids = tensorize_texts(texts, max_seq_len)
    all_label_ids = labels_to_tensor(labels)
    dataloader = build_train_dataloader(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids, batch_size
    )
    return dataloader


def create_prediction_dataloader(
    texts: array, max_seq_len: int, batch_size: int
) -> DataLoader:
    all_input_ids, all_input_mask, all_segment_ids = tensorize_texts(texts, max_seq_len)
    dataloader = build_predict_dataloader(
        all_input_ids, all_input_mask, all_segment_ids, batch_size
    )
    return dataloader


def build_binary_predictions_df(input_df: DataFrame, predictions: array) -> DataFrame:
    assert len(input_df) == predictions.shape[0]
    output_df = input_df.copy()
    output_df["label"] = predictions[:, 1]
    return output_df


def build_multi_predictions_df(
    input_df: DataFrame, predictions: array, label_names: List[str]
) -> DataFrame:
    n_labels = predictions.shape[1]
    if label_names == []:
        label_names = [f"label{i}" for i in range(n_labels)]
    predictions_df = pd.DataFrame(predictions, columns=label_names)
    output_df = pd.concat([input_df, predictions_df], axis=1)
    return output_df
