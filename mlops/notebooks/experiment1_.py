import pandas as pd
import numpy as np
import mlflow
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow.sklearn
import dagshub
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


dagshub.init(repo_owner='Asm2910', repo_name='mlops', mlflow=True)

mlflow.set_experiment("Exp_1")

mlflow.set_tracking_uri("https://dagshub.com/Asm2910/mlops.mlflow")


filepath = r"C:\Users\HP\Downloads\water_potability.csv"
data = pd.read_csv(filepath)

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data, test_size = 0.20, random_state=42)


def fill_missing_with_median(df):
    for col in df.columns:
        if df[col].isnull().any():
            median_value = df[col].median()
            df[col].fillna(median_value, inplace = True)
    return df


train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)

x_train = train_processed_data.drop(columns = ['Potability'], axis = 1)
y_train = train_processed_data['Potability']

n_estimators_ = 100

with mlflow.start_run():

    clf = RandomForestClassifier(n_estimators = n_estimators_)
    clf.fit(x_train, y_train)

    with open("model1.pkl", "wb") as model_file:
        pickle.dump(clf, model_file)

    x_test = test_processed_data.drop(columns = ['Potability'], axis = 1)
    y_test = test_processed_data['Potability']


    with open("model1.pkl", "rb") as model_file:
        model_1 = pickle.load(model_file)

    y_pred = model_1.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)

    mlflow.log_metric("acc", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1score", f1score)

    mlflow.log_param("n_estimators", n_estimators_)


    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (5, 5))
    sns.heatmap(cm, annot = True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")

    mlflow.log_artifact("confusion_matrix.png")

    mlflow.sklearn.log_model(clf, "RandomForestClassifier")

    mlflow.log_artifact(__file__)

    mlflow.set_tag("author", "ASM")
    mlflow.set_tag("model", "GB")

    print("Accuracy:", acc)
    print("Precision_score:", precision)
    print("Recall_score:", recall_score)
    print("f1_score:", f1score)


