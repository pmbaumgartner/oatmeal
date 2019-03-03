import pytest
import pandas as pd
from oatmeal.oatmeal import Pipeline
from tempfile import NamedTemporaryFile


@pytest.fixture(scope="module")
def binary_dataset():
    texts = [
        "Yay this is positive!",
        "Boo, this is negative I hate it.",
        "I love everything",
    ] * 100

    labels = [1, 0, 1] * 100

    with NamedTemporaryFile() as tf:
        df = pd.DataFrame(list(zip(texts, labels)), columns=["texts", "labels"])
        df.to_csv(tf.name)
        yield tf.name


def test_binary_model(binary_dataset):
    pipeline = Pipeline(input_data=binary_dataset).train.binary()

