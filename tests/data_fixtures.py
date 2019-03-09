import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import pandas as pd
import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="module")
def get_20newsgroups_csv():
    from sklearn.datasets import fetch_20newsgroups

    ng = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    labelmap = dict(enumerate(ng.target_names))
    df = pd.DataFrame({"texts": ng.data, "labels": ng.target}).assign(
        labels=lambda x: x["labels"].map(labelmap)
    )
    f = NamedTemporaryFile(suffix=".csv")
    df.to_csv(f.name)
    yield (f.name, labelmap)
    f.close()


@pytest.fixture(scope="module")
def get_kaggle_toxic_csv():
    load_dotenv()
    import kaggle

    kaggle.api.authenticate()

    file_name = "train.csv"
    path = TemporaryDirectory()
    kaggle.api.competition_download_file(
        competition="jigsaw-toxic-comment-classification-challenge",
        file_name=file_name,
        path=path.name,
    )
    with zipfile.ZipFile("train.csv.zip") as z:
        z.extractall(path.name)

    labelmap = dict(
        enumerate(
            ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        )
    )
    csv = str(Path(path.name) / file_name)
    yield csv, labelmap
    path.close()
