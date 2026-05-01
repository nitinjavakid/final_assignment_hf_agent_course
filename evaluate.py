import dotenv

dotenv.load_dotenv()

import mlflow
from mlflow.genai.scorers import Equivalence
from agent import process_question
import pandas as pd
import os

mlflow.set_tracking_uri(os.environ["MLFLOW_ENDPOINT"])
mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT"])

mlflow.langchain.autolog()

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

if __name__ == "__main__":    
    mlflow.genai.evaluate(
        data=get_dataset(),
        predict_fn=process_question,
        scorers=[
            exact_match
        ]
    )