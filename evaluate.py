import dotenv

dotenv.load_dotenv()

import mlflow
from agent import process_question
import pandas as pd
import os
import asyncio
import nest_asyncio

nest_asyncio.apply()

mlflow.set_tracking_uri(os.environ["MLFLOW_ENDPOINT"])
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT"])

def get_dataset():
    dataset_name = os.environ["MLFLOW_DATASET_NAME"]
    try:
        dataset = mlflow.genai.get_dataset("validation_questions")
    except:
        dataset = mlflow.genai.create_dataset("validation_questions")
    
    splits = {'test': '2023/test/metadata.level2.parquet', 'validation': '2023/validation/metadata.level2.parquet'}
    df = pd.read_parquet("hf://datasets/gaia-benchmark/GAIA/" + splits["validation"])
    
    dataset.merge_records([
        {
            "inputs": {
                "question": data["Question"]
            },
            "expectations": {
                "expected_response": data["Final answer"]
            }
        }

        for _, data in df.iterrows()
    ])
    dataset = mlflow.genai.get_dataset(dataset_name)   
    return dataset

@mlflow.genai.scorers.scorer
def exact_match(outputs: dict, expectations: dict) -> bool:
    return outputs == expectations["expected_response"]

def async_predict_fn(question):
    """Wrapper to run async process_question in the event loop"""
    return asyncio.run(process_question(question))

if __name__ == "__main__":
    with mlflow.genai.enable_git_model_versioning() as context:
        mlflow.genai.evaluate(
            data=get_dataset(),
            predict_fn=async_predict_fn,
            scorers=[
                exact_match
            ]
        )