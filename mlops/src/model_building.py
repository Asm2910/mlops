import pandas as pd
import yaml
import pickle
from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:
    def __init__(self, params_path: str, data_path: str, model_path: str):
        self.params_path = params_path
        self.data_path = data_path
        self.model_path = model_path
        self.n_estimators = self.load_params()

    def load_params(self) -> int:
        """
        Load model hyperparameters from a YAML file.

        """
        try:
            with open(self.params_path, "r") as file:
                params = yaml.safe_load(file)
            return params["model_building"]["n_estimators"]
        except Exception as e:
            raise Exception(f"Error loading parameters from {self.params_path}: {e}")

    def load_data(self) -> pd.DataFrame:
        """
        Load dataset from a CSV file.

        """
        try:
            return pd.read_csv(self.data_path)
        except Exception as e:
            raise Exception(f"Error loading data from {self.data_path}: {e}")

    def prepare_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variable for model training.

        """
        try:
            X = data.drop(columns=['Potability'], axis=1)
            y = data['Potability']
            return X, y
        except Exception as e:
            raise Exception(f"Error preparing data: {e}")

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        """
        Train a RandomForest model.

        """
        try:
            model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=42)
            model.fit(X, y)
            return model
        except Exception as e:
            raise Exception(f"Error training model: {e}")

    def save_model(self, model: RandomForestClassifier) -> None:
        """
        Save trained model as a pickle file.

        """
        try:
            with open(self.model_path, "wb") as file:
                pickle.dump(model, file)
        except Exception as e:
            raise Exception(f"Error saving model to {self.model_path}: {e}")

    def run(self):
        """
        Execute the end-to-end training pipeline.
        
        """
        try:
            print("Loading data...")
            data = self.load_data()

            print("Preparing data...")
            X_train, y_train = self.prepare_data(data)

            print(f"Training model with {self.n_estimators} estimators...")
            model = self.train_model(X_train, y_train)

            print("Saving model...")
            self.save_model(model)

            print("Model training and saving completed successfully!")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    trainer = ModelTrainer(
        params_path = r"F:\ml_ques_recommender\mlops\params.yaml",
        data_path = r"F:\ml_ques_recommender\mlops\data\processed_data\train_processed.csv",
        model_path = r"F:\ml_ques_recommender\mlops\models\model.pkl"
    )
    trainer.run()
