import pandas as pd
import pytest
import joblib
from model_training import train_model
from sklearn.ensemble import RandomForestClassifier


# Fixture pour les données d'entraînement
@pytest.fixture
def train_data():
    data = {
        "Pclass": [1, 2, 3],
        "Sex": ["male", "female", "female"],
        "SibSp": [1, 0, 0],
        "Parch": [0, 1, 2],
        "Survived": [1, 0, 1],
    }
    return pd.DataFrame(data)


# Fixture pour les données d'entraînement prétraitées
@pytest.fixture
def X_y(train_data):
    X = pd.get_dummies(train_data[["Pclass", "Sex", "SibSp", "Parch"]])
    y = train_data["Survived"]
    return X, y


# Fixture pour le chemin de sortie
@pytest.fixture
def output_path(tmp_path):
    return tmp_path / "model.pkl"


# Test pour la fonction train_model
def test_train_model(X_y, output_path):
    X, y = X_y
    model = train_model(X, y, output_path)
    assert model is not None
    assert output_path.exists()
    loaded_model = joblib.load(output_path)
    assert isinstance(loaded_model, RandomForestClassifier)
