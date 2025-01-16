from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model
from src.model_evaluation import make_predictions

if __name__ == "__main__":
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    features = ["Pclass", "Sex", "SibSp", "Parch"]
    model_path = "random_forest_model.pkl"
    output_path = "submission.csv"

    # Prétraitement des données
    train_data, test_data = load_data(train_path, test_path)
    X, y, X_test = preprocess_data(train_data, test_data, features)

    # Entraînement du modèle
    train_model(X, y, model_path)

    # Prédictions et sauvegarde
    make_predictions(model_path, X_test, test_data, output_path)
