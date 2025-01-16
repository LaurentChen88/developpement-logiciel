from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X, y, output_path, n_estimators=100, max_depth=5, random_state=1):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X, y)
    joblib.dump(model, output_path)
    return model

if __name__ == "__main__":
    import data_preprocessing
    
    # Charger et préparer les données
    train_data, test_data = data_preprocessing.load_data("path/to/train.csv", "path/to/test.csv")
    X, y, X_test = data_preprocessing.preprocess_data(train_data, test_data, ["Pclass", "Sex", "SibSp", "Parch"])

    # Entraîner le modèle
    model_path = "random_forest_model.pkl"
    train_model(X, y, model_path)
