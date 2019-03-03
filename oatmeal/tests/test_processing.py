from oatmeal.processing import (
    text_to_token_ids,
    tensorize_texts,
    labels_to_tensor,
    multilabels_to_tensor,
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
    labels = np.array([1, 2, 3, 1, 1, 1, 1, 3])

    result = labels_to_tensor(labels)

    # np.unique will sort the unique input classes
    assert torch.equal(result, torch.tensor([0, 1, 2, 0, 0, 0, 0, 2]))


def test_multilabels_to_tensor():
    labels = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])

    result = multilabels_to_tensor(labels)

    assert torch.equal(result, torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1]]))

