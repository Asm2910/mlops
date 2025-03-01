import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ModelEvaluator:
    def __init__(self, test_data_path: str, model_path: str, metrics_output_path: str = "metrics.json"):
        self.test_data_path = test_data_path
        self.model_path = model_path
        self.metrics_output_path = metrics_output_path
        self.model = None
        self.x_test = None
        self.y_test = None

    def load_data(self) -> None:
        """
        Load test data from a CSV file.

        """
        try:
            print("Loading test data...")
            test_data = pd.read_csv(self.test_data_path)
            self.x_test = test_data.iloc[:, :-1].values
            self.y_test = test_data.iloc[:, -1].values
        except Exception as e:
            raise Exception(f"Error loading test data from {self.test_data_path}: {e}")

    def load_model(self) -> None:
        """
        Load the trained model from a pickle file.

        """
        try:
            print("Loading trained model...")
            with open(self.model_path, "rb") as file:
                self.model = pickle.load(file)
        except Exception as e:
            raise Exception(f"Error loading model from {self.model_path}: {e}")

    def evaluate_model(self) -> dict:
        """
        Evaluate the model and return the computed metrics.

        """
        try:
            if self.model is None or self.x_test is None or self.y_test is None:
                raise Exception("Model or test data is not loaded properly.")

            print("Predicting on test data...")
            y_pred = self.model.predict(self.x_test)

            print("Calculating metrics...")
            metrics = {
                "accuracy": accuracy_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred),
                "recall": recall_score(self.y_test, y_pred),
                "f1_score": f1_score(self.y_test, y_pred)
            }
            return metrics
        except Exception as e:
            raise Exception(f"Error during model evaluation: {e}")

    def save_metrics(self, metrics: dict) -> None:
        """
        Save evaluation metrics to a JSON file.

        """
        try:
            print(f"Saving metrics to {self.metrics_output_path}...")
            with open(self.metrics_output_path, "w") as file:
                json.dump(metrics, file, indent=4)
        except Exception as e:
            raise Exception(f"Error saving metrics to {self.metrics_output_path}: {e}")

    def run(self) -> None:
        """
        Execute the complete evaluation pipeline.

        """
        try:
            self.load_data()
            self.load_model()
            metrics = self.evaluate_model()
            self.save_metrics(metrics)
            print("Model evaluation completed successfully!")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    evaluator = ModelEvaluator(
        test_data_path = r"F:\ml_ques_recommender\mlops\data\processed_data\test_processed.csv",
        model_path = r"F:\ml_ques_recommender\mlops\models\model_rf_v1.pkl"
    )
    evaluator.run()
