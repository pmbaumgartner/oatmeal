from click.testing import CliRunner
from oatmeal.cli import cli
from .data_fixtures import get_20newsgroups_csv
import pandas as pd
import pytest


@pytest.fixture
def runner():
    return CliRunner(echo_stdin=True)


def test_multiclass(runner, get_20newsgroups_csv):

    csv, labelmap = get_20newsgroups_csv
    print(csv)
    result = runner.invoke(
        cli, ["train", "--input-data", csv, "multiclass", "-x", "texts", "-y", "labels"]
    )
    print(result)
