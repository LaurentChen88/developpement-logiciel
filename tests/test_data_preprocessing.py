import pandas as pd
import pytest
import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
)
from data_preprocessing import load_data, preprocess_data


# Fixtures pour les chemins de fichiers CSV
@pytest.fixture
def train_path(tmp_path):
    train_data = {
        "Pclass": [1, 2, 3],
        "Sex": ["male", "female", "female"],
        "SibSp": [1, 0, 0],
        "Parch": [0, 1, 2],
        "Survived": [1, 0, 1],
    }
    train_df = pd.DataFrame(train_data)
    train_csv = tmp_path / "train.csv"
    train_df.to_csv(train_csv, index=False)
    return train_csv


@pytest.fixture
def test_path(tmp_path):
    test_data = {
        "Pclass": [1, 2, 3],
        "Sex": ["male", "female", "female"],
        "SibSp": [1, 0, 0],
        "Parch": [0, 1, 2],
    }
    test_df = pd.DataFrame(test_data)
    test_csv = tmp_path / "test.csv"
    test_df.to_csv(test_csv, index=False)
    return test_csv


# Tests pour la fonction load_data
def test_load_data(train_path, test_path):
    train_data, test_data = load_data(train_path, test_path)
    assert not train_data.empty
    assert not test_data.empty
    assert "Survived" in train_data.columns
    assert "Survived" not in test_data.columns


# Tests pour la fonction preprocess_data
def test_preprocess_data(train_path, test_path):
    train_data, test_data = load_data(train_path, test_path)
    feature_columns = ["Pclass", "Sex", "SibSp", "Parch"]
    X, y, X_test = preprocess_data(train_data, test_data, feature_columns)
    assert not X.empty
    assert not y.empty
    assert not X_test.empty
    assert "Survived" not in X.columns
    assert "Survived" not in X_test.columns
    assert len(X.columns) == len(X_test.columns)


# Tests pour des cas extrêmes
def test_preprocess_data_empty(train_path, test_path):
    train_data, test_data = load_data(train_path, test_path)
    feature_columns = []
    X, y, X_test = preprocess_data(train_data, test_data, feature_columns)
    assert X.empty
    assert not y.empty
    assert X_test.empty


def test_preprocess_data_single_feature(train_path, test_path):
    train_data, test_data = load_data(train_path, test_path)
    feature_columns = ["Pclass"]
    X, y, X_test = preprocess_data(train_data, test_data, feature_columns)
    assert not X.empty
    assert not y.empty
    assert not X_test.empty
    assert "Survived" not in X.columns
    assert "Survived" not in X_test.columns
    assert len(X.columns) == len(X_test.columns)
