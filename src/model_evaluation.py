"""
Module pour l’évaluation des performances du modèle.
"""
import pandas as pd
import joblib

def make_predictions(model_path, X_test, test_data, output_path):
    """
    Charge un modèle depuis un fichier, effectue des prédictions et enregistre les résultats.

    Paramètres :
    ----------
    model_path : str
        Chemin du fichier contenant le modèle sauvegardé.
    X_test : pandas.DataFrame ou numpy.ndarray
        Données d'entrée utilisées pour effectuer les prédictions.
    test_data : pandas.DataFrame
        Chemin du fichier d'entraînement et de test.
    output_path : str
        Chemin où enregistrer le fichier CSV des prédictions.

    Résultats :
    ----------
    Crée un fichier CSV contenant les prédictions avec deux colonnes : 
    - 'PassengerId' : l'identifiant du passager issu de test_data.
    - 'Survived' : la prédiction du modèle (0 ou 1).

    Affiche un message indiquant l'emplacement du fichier sauvegardé.
    """
    model = joblib.load(model_path)
    predictions = model.predict(X_test)
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv(output_path, index=False)
    print("Predictions saved to", output_path)

if __name__ == "__main__":
    import data_preprocessing

    # Charger les données de test
    _, test_data = data_preprocessing.load_data("path/to/train.csv", "path/to/test.csv")
    _, _, X_test = data_preprocessing.preprocess_data(_, test_data, ["Pclass", "Sex", "SibSp", "Parch"])

    # Faire des prédictions
    model_path = "random_forest_model.pkl"
    output_path = "submission.csv"
    make_predictions(model_path, X_test, test_data, output_path)
