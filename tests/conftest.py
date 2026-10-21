"""Here the numerical tests read the test data supplied with the assignment."""
import json
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_directory():
    return Path(__file__).resolve().parents[1] / "test_data"


@pytest.fixture(scope="session")
def test_data(test_data_directory):
    return {
        f"exercise{number}": json.loads(
            (test_data_directory / f"exercise{number}" / "cases.json").read_text(
                encoding="utf-8"))["cases"]
        for number in range(1, 5)
    }
