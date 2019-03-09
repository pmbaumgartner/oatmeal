from oatmeal.processing import (
    text_to_token_ids,
    tensorize_texts,
    labels_to_tensor,
    create_training_dataloader,
    create_prediction_dataloader,
)
import torch
import numpy as np


def test_tokenization():
    sentence = "This sentence will be tokenized."
    MAX_SEQ_LEN = 32

    result = text_to_token_ids(sentence, max_seq_len=MAX_SEQ_LEN)
    result_ids, result_mask, result_segment_ids = result

    assert all(isinstance(x, int) for x in result_ids)
    assert len(result_ids) == MAX_SEQ_LEN
    assert len(result_mask) == MAX_SEQ_LEN
    assert len(result_segment_ids) == MAX_SEQ_LEN


def test_tensorize():
    sentences = [
        "The first sentence that will be transformed into a tensor",
        "Another sentence that will be converted into a tensor",
        "The final sentence to be turned into a PyTorch tensor",
    ]

    MAX_SEQ_LEN = 32

    result = tensorize_texts(sentences, max_seq_len=MAX_SEQ_LEN)
    result_ids, result_mask, result_segment_ids = result

    assert tuple(result_ids.size()) == (len(sentences), MAX_SEQ_LEN)
    assert tuple(result_mask.size()) == (len(sentences), MAX_SEQ_LEN)
    assert tuple(result_segment_ids.size()) == (len(sentences), MAX_SEQ_LEN)


def test_labels_to_tensor():
    labels = np.array([1, 2, 3, 1, 0, 1, 1, 3])

    result = labels_to_tensor(labels)

    assert torch.equal(result, torch.tensor([1, 2, 3, 1, 0, 1, 1, 3]))


def test_create_train_dataloader():
    sentences = [
        "The first sentence that will be transformed into a tensor",
        "Another sentence that will be converted into a tensor",
        "The final sentence to be turned into a PyTorch tensor",
    ]
    labels = [0, 1, 0]

    sentence_array = np.array(sentences)
    labels_array = np.array(labels)

    dataloader = create_training_dataloader(sentence_array, labels_array, 32, 1)

    assert len(dataloader) == len(sentences)


def test_create_predict_dataloader():
    sentences = [
        "The first sentence that will be transformed into a tensor",
        "Another sentence that will be converted into a tensor",
        "The final sentence to be turned into a PyTorch tensor",
    ]

    sentence_array = np.array(sentences)

    dataloader = create_prediction_dataloader(sentence_array, 32, 1)

    assert len(dataloader) == len(sentences)
