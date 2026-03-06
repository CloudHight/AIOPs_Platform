import boto3
import os
from sagemaker.huggingface import HuggingFace
from sagemaker.tuner import HyperparameterTuner
from sagemaker.parameter import ContinuousParameter, CategoricalParameter
from sagemaker import get_execution_role

print("[INFO] Current working directory:", os.getcwd())
print("[INFO] Files in current directory:", os.listdir("."))

def launch_sagemaker_tuning(
    train_uri, val_uri, train_script="train.py",
    bucket="YOUR_BUCKET_NAME", role=None
):
    # Check if train.py exists in current directory
    if not os.path.isfile(train_script):
        print(f"[WARNING] {train_script} not found in current directory.")
        print(f"[INFO] Make sure {train_script} is in the same directory as this notebook.")
        raise FileNotFoundError(f"[ERROR] {train_script} not found. Please provide your training script.")

    role = role or get_execution_role()
    
    estimator = HuggingFace(
        entry_point=train_script,
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        transformers_version="4.26",
        pytorch_version="1.13",
        py_version="py39",
        role=role,
        enable_network_isolation=False,
        hyperparameters={
            "epochs": 3,
            "model_name": "bert-base-uncased",
            "learning_rate": 3e-5,
            "per_device_train_batch_size": 16
        },
        metric_definitions=[
            {"Name": "eval_f1_macro", "Regex": r"'eval_f1_macro':\s*([0-9\.]+)"},
            {"Name": "eval_f1_micro", "Regex": r"'eval_f1_micro':\s*([0-9\.]+)"},
            {"Name": "eval_accuracy", "Regex": r"'eval_accuracy':\s*([0-9\.]+)"},
            {"Name": "eval_loss", "Regex": r"'eval_loss':\s*([0-9\.]+)"},
        ],
        base_job_name="hf-nginx-classifier",
    )

    tuner = HyperparameterTuner(
        estimator=estimator,
        objective_metric_name='eval_f1_macro',
        objective_type='Maximize',
        metric_definitions=estimator.metric_definitions,
        hyperparameter_ranges={
            'learning_rate': ContinuousParameter(1e-5, 5e-5),
            'per_device_train_batch_size': CategoricalParameter([8, 16, 32])
        },
        max_jobs=6,
        max_parallel_jobs=1
    )

    print("[INFO] Launching SageMaker tuning job ...")
    tuner.fit({'train': train_uri, 'validation': val_uri}, wait=False)
    print("[INFO] Tuning job launched.")
    
    return tuner

if __name__ == "__main__":
    BUCKET = "YOUR_BUCKET_NAME"
    PROCESSED_PREFIX = "YOUR_FOLDER_NAME/"
    
    # Check if train.py exists before proceeding
    if not os.path.isfile("train.py"):
        print("[ERROR] train.py not found in current directory.")
        print("[INFO] Please ensure train.py is in the same directory as this notebook.")
        exit(1)
    
    train_uri = f's3://{BUCKET}/{PROCESSED_PREFIX}train/'
    val_uri = f's3://{BUCKET}/{PROCESSED_PREFIX}validation/'
    
    tuner = launch_sagemaker_tuning(train_uri, val_uri, bucket=BUCKET)
    print(f"[INFO] Training job name: {tuner.latest_tuning_job.name}")