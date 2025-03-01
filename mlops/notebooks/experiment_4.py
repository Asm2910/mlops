import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
from mlflow.models import infer_signature
import dagshub
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


dagshub.init(repo_owner='Asm2910', repo_name='mlops', mlflow=True)

mlflow.set_experiment("Exp_4")

mlflow.set_tracking_uri("https://dagshub.com/Asm2910/mlops.mlflow")


filepath = r"C:\Users\HP\Downloads\water_potability.csv"
data = pd.read_csv(filepath)

train_data, test_data = train_test_split(data, test_size = 0.20, random_state=42)

def fill_missing_with_mean(df):
    for col in df.columns:
        if df[col].isnull().any():
            median_value = df[col].mean()
            df[col].fillna(median_value, inplace = True)
    return df


train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)

x_train = train_processed_data.drop(columns = ['Potability'], axis = 1)
y_train = train_processed_data['Potability']

rf = RandomForestClassifier(random_state=42)
param_dict = {
    'n_estimators': [100, 200, 300, 500, 1000],
    'max_depth': [None, 4, 5, 6, 10]
}

random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dict, n_iter=50, cv = 5, n_jobs=-1, verbose=2, random_state=42)

with mlflow.start_run(run_name="Random Forest Tuning") as parent_run:

    random_search.fit(x_train, y_train)

    for i in range(len(random_search.cv_results_['params'])):
        with mlflow.start_run(run_name=f"Combination{i + 1}", nested = True) as child_run:
            mlflow.log_params(random_search.cv_results_['params'][i])
            mlflow.log_metric("mean_test_score", random_search.cv_results_['mean_test_score'][i])

    print("Best parameters found:", random_search.best_params_)

    mlflow.log_params(random_search.best_params_)

    best_rf = random_search.best_estimator_
    best_rf.fit(x_train, y_train)

    # Save the trained model to a file for later use
    pickle.dump(best_rf, open("model.pkl", "wb"))

    # Prepare the test data by separating features and target variable
    X_test = test_processed_data.drop(columns=["Potability"], axis=1)  # Features
    y_test = test_processed_data["Potability"]  # Target variable

    # Load the saved model from the file
    model = pickle.load(open('model.pkl', "rb"))

    # Make predictions on the test set using the loaded model
    y_pred = model.predict(X_test)

    # Calculate and print performance metrics: accuracy, precision, recall, and F1-score
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log performance metrics into MLflow for tracking
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1 score", f1)

    # Log the training and testing data as inputs in MLflow
    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)
    
    mlflow.log_input(train_df, "train")  # Log training data
    mlflow.log_input(test_df, "test")  # Log test data

    # Log the current script file as an artifact in MLflow
    mlflow.log_artifact(__file__)

    # Infer the model signature using the test features and predictions
    sign = infer_signature(X_test, random_search.best_estimator_.predict(X_test))
    
    # Log the trained model in MLflow with its signature
    mlflow.sklearn.log_model(random_search.best_estimator_, "Best Model", signature=sign)

    # Print the calculated performance metrics to the console for review
    print("Accuracy: ", acc)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-score: ", f1)


