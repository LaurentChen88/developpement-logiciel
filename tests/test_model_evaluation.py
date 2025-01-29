import pandas as pd
import pytest
import joblib
from prediction_utilities import make_predictions

# Fixture pour le chemin du modèle
@pytest.fixture
def model_path(tmp_path):
    model = joblib.dump(joblib.load("path/to/random_forest_model.pkl"), tmp_path / "model.pkl")
    return model

# Fixture pour les données de test
@pytest.fixture
def test_data():
    data = {
        "PassengerId": [1, 2, 3],
        "Pclass": [1, 2, 3],
        "Sex": ["male", "female", "female"],
        "SibSp": [1, 0, 0],
        "Parch": [0, 1, 2]
    }
    return pd.DataFrame(data)

# Fixture pour les données de test prétraitées
@pytest.fixture
def X_test(test_data):
    return pd.get_dummies(test_data[["Pclass", "Sex", "SibSp", "Parch"]])

# Fixture pour le chemin de sortie
@pytest.fixture
def output_path(tmp_path):
    return tmp_path / "output.csv"

# Test pour la fonction make_predictions
def test_make_predictions(model_path, X_test, test_data, output_path):
    make_predictions(model_path, X_test, test_data, output_path)
    output = pd.read_csv(output_path)
    assert not output.empty
    assert "PassengerId" in output.columns
    assert "Survived" in output.columns
    assert len(output) == len(test_data)
