import os
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


class DataCollection:
    def __init__(self, data_filepath: str, params_filepath: str, raw_data_path: str):
        self.data_filepath = data_filepath
        self.params_filepath = params_filepath
        self.raw_data_path = raw_data_path
        self.test_size = self.load_params()
        
    def load_params(self) -> float:
        """
        Load test_size parameter from a YAML file.

        """
        try:
            with open(self.params_filepath, "r") as file:
                params = yaml.safe_load(file)
            return params["data_collection"]["test_size"]
        except Exception as e:
            raise Exception(f"Error loading parameters from {self.params_filepath}: {e}")

    def load_data(self) -> pd.DataFrame:
        """
        Load data from a CSV file.

        """
        try:
            return pd.read_csv(self.data_filepath)
        except Exception as e:
            raise Exception(f"Error loading data from {self.data_filepath}: {e}")

    def split_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.

        """
        try:
            return train_test_split(data, test_size=self.test_size, random_state=42)
        except Exception as e:
            raise Exception(f"Error splitting data: {e}")

    def save_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save DataFrame to a CSV file.

        """
        try:
            os.makedirs(self.raw_data_path, exist_ok=True)
            filepath = os.path.join(self.raw_data_path, filename)
            df.to_csv(filepath, index=False)
        except Exception as e:
            raise Exception(f"Error saving data to {filepath}: {e}")

    def process_data(self):
        """
        Main method to process data: load, split, and save it.

        """
        try:
            data = self.load_data()
            train_data, test_data = self.split_data(data)
            self.save_data(train_data, "train.csv")
            self.save_data(test_data, "test.csv")
        except Exception as e:
            raise Exception(f"An error occurred: {e}")


if __name__ == "__main__":
    data_filepath = r"C:\Users\HP\Downloads\water_potability.csv"
    params_filepath = r"F:\ml_ques_recommender\mlops\params.yaml"
    raw_data_path = os.path.join("data", "raw")
    
    processor = DataCollection(data_filepath, params_filepath, raw_data_path)
    processor.process_data()
