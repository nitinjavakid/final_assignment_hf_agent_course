import dotenv
dotenv.load_dotenv()

import getpass
import os
from mlflow import MlflowClient
from mlflow.server import get_app_client

tracking_uri = os.environ["MLFLOW_ENDPOINT"]
auth_client = get_app_client("basic-auth", tracking_uri=tracking_uri)

username = input("Username: ")
password = getpass.getpass("Password: ")

try:
    auth_client.create_user(username=username, password=password)
except:
    pass

client = MlflowClient(tracking_uri=tracking_uri)

try:
    experiment_id = client.create_experiment(name=os.environ["MLFLOW_EXPERIMENT"])
except:
    experiment_id = client.get_experiment_by_name(name=os.environ["MLFLOW_EXPERIMENT"]).experiment_id

try:
    auth_client.create_experiment_permission(
        experiment_id=experiment_id, username=username, permission="MANAGE"
    )
except Exception as e:
    print(e)

try:
    auth_client.update_gateway_endpoint_permission(
        endpoint_id="qwen",
        username=username,
        permission="USE"
    )
except Exception as e:
    print(e)