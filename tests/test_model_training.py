import pandas as pd
import pytest
import joblib
from model_training import train_model
from sklearn.ensemble import RandomForestClassifier


# Fixture pour générer un jeu de données d'entraînement factice
@pytest.fixture
def train_data():
    """
    Génère un DataFrame pandas contenant des données d'entraînement factices
    pour tester le modèle.

    Returns:
        pd.DataFrame: Jeu de données contenant les caractéristiques et la cible.
    """
    data = {
        "Pclass": [1, 2, 3],
        "Sex": ["male", "female", "female"],
        "SibSp": [1, 0, 0],
        "Parch": [0, 1, 2],
        "Survived": [1, 0, 1],
    }
    return pd.DataFrame(data)


# Fixture pour prétraiter les données d'entraînement
@pytest.fixture
def X_y(train_data):
    """
    Prépare les caractéristiques (X) et la cible (y) à partir des données d'entraînement.

    Args:
        train_data (pd.DataFrame): Jeu de données d'entraînement factice.

    Returns:
        tuple: (X, y) où X est un DataFrame des caractéristiques et y est la série cible.
    """
    X = pd.get_dummies(train_data[["Pclass", "Sex", "SibSp", "Parch"]])
    y = train_data["Survived"]
    return X, y


# Fixture pour générer un chemin de fichier temporaire pour enregistrer le modèle
@pytest.fixture
def output_path(tmp_path):
    """
    Génère un chemin de fichier temporaire pour enregistrer le modèle entraîné.

    Args:
        tmp_path (pathlib.Path): Répertoire temporaire fourni par pytest.

    Returns:
        pathlib.Path: Chemin du fichier modèle.
    """
    return tmp_path / "model.pkl"


# Test unitaire pour la fonction train_model
def test_train_model(X_y, output_path):
    """
    Teste la fonction train_model pour s'assurer qu'elle entraîne et sauvegarde un modèle.

    Args:
        X_y (tuple): Données d'entraînement (X, y).
        output_path (pathlib.Path): Chemin de sauvegarde du modèle.
    """
    X, y = X_y
    model = train_model(X, y, output_path)

    # Vérifie que le modèle retourné n'est pas None
    assert model is not None

    # Vérifie que le fichier du modèle a bien été créé
    assert output_path.exists()

    # Charge le modèle sauvegardé et vérifie son type
    loaded_model = joblib.load(output_path)
    assert isinstance(loaded_model, RandomForestClassifier)
