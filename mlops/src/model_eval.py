import numpy as np
import pandas as pd
import pickle
import json
import mlflow
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from mlflow import log_metric, log_param, log_artifact
import mlflow.sklearn
import dagshub
import mlflow
from mlflow.models import infer_signature

class MLflowPipeline:
    def __init__(self, repo_owner, repo_name, experiment_name, tracking_uri):
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._initialize_mlflow()
    
    def _initialize_mlflow(self):
        dagshub.init(repo_owner=self.repo_owner, repo_name=self.repo_name, mlflow=True)
        mlflow.set_experiment(self.experiment_name)
        mlflow.set_tracking_uri(self.tracking_uri)

    @staticmethod
    def load_data(filepath: str) -> pd.DataFrame:
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise Exception(f"Error loading data from {filepath}: {e}")

    @staticmethod
    def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        try:
            X = data.drop(columns=['Potability'], axis=1)
            y = data['Potability']
            return X, y
        except Exception as e:
            raise Exception(f"Error preparing data: {e}")

    @staticmethod
    def load_model(filepath: str):
        try:
            with open(filepath, "rb") as file:
                return pickle.load(file)
        except Exception as e:
            raise Exception(f"Error loading model from {filepath}: {e}")

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> dict:
        try:
            params = yaml.safe_load(open("params.yaml", "r"))
            test_size = params["data_collection"]["test_size"]
            n_estimators = params["model_building"]["n_estimators"]
            
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred)
            }
            
            mlflow.log_param("Test_size", test_size)
            mlflow.log_param("n_estimators", n_estimators) 
            
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(5, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix for {model_name}")
            cm_path = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            
            return metrics
        except Exception as e:
            raise Exception(f"Error evaluating model: {e}")

    @staticmethod
    def save_metrics(metrics: dict, metrics_path: str) -> None:
        try:
            with open(metrics_path, 'w') as file:
                json.dump(metrics, file, indent=4)
        except Exception as e:
            raise Exception(f"Error saving metrics to {metrics_path}: {e}")

    def run_pipeline(self, test_data_path, model_path, metrics_path, model_name):
        try:
            test_data = self.load_data(test_data_path)
            X_test, y_test = self.prepare_data(test_data)
            model = self.load_model(model_path)

            with mlflow.start_run() as run:
                metrics = self.evaluate_model(model, X_test, y_test, model_name)
                self.save_metrics(metrics, metrics_path)
                
                mlflow.log_artifact(model_path)
                mlflow.log_artifact(metrics_path)
                
                signature = infer_signature(X_test, model.predict(X_test))
                mlflow.sklearn.log_model(model, "Best Model", signature=signature)
                
                run_info = {'run_id': run.info.run_id, 'model_name': "Best Model"}
                reports_path = "reports/run_info.json"
                with open(reports_path, 'w') as file:
                    json.dump(run_info, file, indent=4)
        except Exception as e:
            raise Exception(f"An error occurred during the pipeline run: {e}")

if __name__ == "__main__":
    pipeline = MLflowPipeline(
        repo_owner='ASM2910',
        repo_name='mlops',
        experiment_name='DVC_PIPELINE_1',
        tracking_uri='https://dagshub.com/Asm2910/mlops.mlflow'
    )
    pipeline.run_pipeline(
        test_data_path = r"F:\ml_ques_recommender\mlops\data\processed_data\test_processed.csv",
        model_path = r"F:\ml_ques_recommender\mlops\models\model.pkl",
        metrics_path = "reports/metrics.json",
        model_name = "Best Model"
    )

