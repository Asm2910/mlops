print('starting script ....')

import mlflow
import dagshub 

print(mlflow.__version__)

mlflow.set_tracking_uri("https://dagshub.com/Asm2910/mlops.mlflow")

dagshub.init(repo_owner='Asm2910', repo_name='mlops', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

print(mlflow.get_tracking_uri())
print(mlflow.search_runs)