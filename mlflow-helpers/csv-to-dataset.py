import dotenv

dotenv.load_dotenv()

import mlflow
import os
import argparse
import sys
import pandas as pd

def process_row(inputs, expectations, outputs, row):
    inputs_dict = dict()
    expectations_dict = dict()
    outputs_dict = dict()

    for col in inputs:
        inputs_dict[col] = str(row[col])
    
    for col in expectations:
        expectations_dict[col] = str(row[col])

    for col in outputs:
        outputs_dict[col] = str(row[col])

    return {
        "inputs": inputs_dict,
        "outputs": outputs_dict,
        "expectations": expectations_dict
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", "-dsn", help="Dataset name", required=True)
    parser.add_argument("--csv", help="CSV file to import", required=True)
    parser.add_argument("--empty", help="Empty dataset", default=False, action='store_true')
    parser.add_argument("--inputs", help="Inputs", required=False)
    parser.add_argument("--outputs", help="Outputs", required=False)
    parser.add_argument("--expectations", help="Outputs", required=False)

    args, _ = parser.parse_known_args(sys.argv)

    mlflow.set_tracking_uri(os.environ["MLFLOW_ENDPOINT"])
    mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT"])

    dataset_name = args.dataset_name

    try:
        
        dataset = mlflow.genai.get_dataset(dataset_name)
        if args.empty:
            mlflow.genai.delete_dataset(dataset_id=dataset.dataset_id)
            dataset = mlflow.genai.create_dataset(dataset_name)
    except:
        dataset = mlflow.genai.create_dataset(dataset_name)

    inputs = args.inputs.split(",") if args.inputs else []
    expectations = args.expectations.split(",") if args.expectations else []
    outputs = args.outputs.split(",") if args.outputs else []
    df = pd.read_csv(args.csv)

    dataset.merge_records([
        process_row(inputs, expectations, outputs, row) for _, row in df.iterrows()
    ])

if __name__ == "__main__":
    main()