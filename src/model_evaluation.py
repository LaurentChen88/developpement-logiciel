import pandas as pd
import joblib

def make_predictions(model_path, X_test, test_data, output_path):
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
