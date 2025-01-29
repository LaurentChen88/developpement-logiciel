"""
Module pour charger et prétraiter les données avant analyse.
"""
import pandas as pd

def load_data(train_path, test_path):
    """
    Charge les données d'entraînement et de test à partir des fichiers CSV spécifiés.

    Parameters:
    train_path (str): Le chemin vers le fichier CSV contenant les données d'entraînement.
    test_path (str): Le chemin vers le fichier CSV contenant les données de test.

    Returns:
    tuple: Un tuple contenant les deux DataFrames chargés :
        - train_data (DataFrame): Les données d'entraînement.
        - test_data (DataFrame): Les données de test.
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def preprocess_data(train_data, test_data, features):
    """
    Prétraite les données d'entraînement et de test en appliquant les transformations nécessaires.

    Cette fonction effectue la conversion des variables catégorielles en variables numériques
    en utilisant la méthode `get_dummies` de pandas.

    Parameters:
    train_data (DataFrame): Les données d'entraînement.
    test_data (DataFrame): Les données de test.
    features (list of str): La liste des colonnes à utiliser pour le prétraitement.

    Returns:
    tuple: Un tuple contenant :
        - X (DataFrame): Les données d'entraînement après transformation.
        - y (Series): La variable cible d'entraînement.
        - X_test (DataFrame): Les données de test après transformation.
    """
    X = pd.get_dummies(train_data[features])
    X_test = pd.get_dummies(test_data[features])
    y = train_data["Survived"]
    return X, y, X_test

if __name__ == "__main__":
    train_path = "path/to/train.csv"
    test_path = "path/to/test.csv"
    features = ["Pclass", "Sex", "SibSp", "Parch"]

    train_data, test_data = load_data(train_path, test_path)
    X, y, X_test = preprocess_data(train_data, test_data, features)
