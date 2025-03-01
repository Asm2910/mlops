import os
import pandas as pd


class DataProcessor:
    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

    def load_data(self, filename: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        filepath = os.path.join(self.raw_data_path, filename)
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise Exception(f"Error loading data from {filepath}: {e}")

    def fill_missing_with_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in the DataFrame with the column mean."""
        try:
            for column in df.columns:
                if df[column].isnull().any():
                    mean_value = df[column].mean()
                    df[column].fillna(mean_value, inplace=True)
            return df
        except Exception as e:
            raise Exception(f"Error filling missing values with mean: {e}")

    def save_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to a CSV file."""
        filepath = os.path.join(self.processed_data_path, filename)
        try:
            os.makedirs(self.processed_data_path, exist_ok=True)
            df.to_csv(filepath, index=False)
        except Exception as e:
            raise Exception(f"Error saving data to {filepath}: {e}")

    def process_data(self):
        """Main method to process train and test datasets."""
        try:
            train_data = self.load_data("train.csv")
            test_data = self.load_data("test.csv")

            train_processed = self.fill_missing_with_mean(train_data)
            test_processed = self.fill_missing_with_mean(test_data)

            self.save_data(train_processed, "train_processed.csv")
            self.save_data(test_processed, "test_processed.csv")
        except Exception as e:
            raise Exception(f"An error occurred: {e}")


if __name__ == "__main__":
    raw_data_path = r"F:\ml_ques_recommender\mlops\data\raw"
    processed_data_path = r"F:\ml_ques_recommender\mlops\data\processed_data"
    
    processor = DataProcessor(raw_data_path, processed_data_path)
    processor.process_data()
