"""
Module pour l’entraînement et la sauvegarde des modèles.
"""
from sklearn.ensemble import RandomForestClassifier
import joblib


def train_model(X, y, output_path, n_estimators=100, 
                max_depth=5, random_state=1):
    """
    Entraîne un modèle RandomForestClassifier et l'enregistre dans un fichier.

    Paramètres :
    ----------
    X : pandas.DataFrame ou numpy.ndarray
        Les caractéristiques utilisées pour entraîner le modèle.
    y : pandas.Series ou numpy.ndarray
        Les étiquettes cibles associées aux données d'entraînement.
    output_path : str
        Chemin où enregistrer le modèle entraîné.
    n_estimators : int, optionnel (par défaut = 100)
        Nombre d'arbres dans la forêt aléatoire.
    max_depth : int, optionnel (par défaut = 5)
        Profondeur maximale des arbres de décision.
    random_state : int, optionnel (par défaut = 1)
        Graine aléatoire pour assurer la reproductibilité des résultats.

    Retourne :
    ----------
    model : RandomForestClassifier
        Le modèle entraîné.

    Enregistre également le modèle sous forme de fichier pour
    une utilisation ultérieure.
    """
    model = RandomForestClassifier(n_estimators=n_estimators, 
                                   max_depth=max_depth, random_state=random_state)
    model.fit(X, y)
    joblib.dump(model, output_path)
    return model


if __name__ == "__main__":
    import data_preprocessing

    # Charger et préparer les données
    train_data, test_data = data_preprocessing.load_data(
        "path/to/train.csv", "path/to/test.csv")
    X, y, X_test = data_preprocessing.preprocess_data(
        train_data, test_data, ["Pclass", "Sex", "SibSp", "Parch"])

    # Entraîner le modèle
    model_path = "random_forest_model.pkl"
    train_model(X, y, model_path)
