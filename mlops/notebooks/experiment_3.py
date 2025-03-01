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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle


dagshub.init(repo_owner='Asm2910', repo_name='mlops', mlflow=True)

mlflow.set_experiment("Exp_3")

mlflow.set_tracking_uri("https://dagshub.com/Asm2910/mlops.mlflow")


filepath = r"C:\Users\HP\Downloads\water_potability.csv"
data = pd.read_csv(filepath)

from sklearn.model_selection import train_test_split
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

x_test = test_processed_data.drop(columns = ['Potability'], axis = 1)
y_test = test_processed_data['Potability']

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XB Boost": XGBClassifier()
}


with mlflow.start_run(run_name="Water Potability Models Experiment"):

    for model_name, model in models.items():
        with mlflow.start_run(run_name= model_name, nested=True):
            model.fit(x_train, y_train)

            model_filename = f"{model_name.replace(' ', '_')}.pkl"
            pickle.dump(model, open(model_filename, "wb"))

            y_pred = model.predict(x_test)


            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1score = f1_score(y_test, y_pred)

            mlflow.log_metric("acc", acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1score", f1score)



            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize = (5, 5))
            sns.heatmap(cm, annot = True)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix for {model_name}")
            plt.savefig(f"confusion_matrix_{model_name.replace(' ', '_')}.png")


            mlflow.log_artifact(f"confusion_matrix_{model_name.replace(' ', '_')}.png")

            mlflow.sklearn.log_model(model, model_name.replace(' ', '_'))

            mlflow.log_artifact(__file__)

            mlflow.set_tag("author", "ASM")

    print("All models have been trained and logged as child runs successfully.")

