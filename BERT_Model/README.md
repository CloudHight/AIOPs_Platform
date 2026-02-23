# BERT ML Workflow for Nginx Log Anomaly Detection

This repository contains a complete ML workflow split into 6 stages, designed to run individually on SageMaker Jupyter notebooks.

## Workflow Stages

### 1. Data Creation (`01_create_data.py`)
- Generates synthetic nginx access and error logs
- Labels anomalies (5XX errors, critical errors)
- Splits data into train/validation/test sets
- Uploads data to S3

**Usage:**
```python
# Update these variables in the script:
BUCKET = "your-s3-bucket-name"
PROCESSED_PREFIX = "your-folder-name/"

# Run the script
%run 01_create_data.py
```

### 2. Model Training (`02_train_model.py`)
- Launches SageMaker hyperparameter tuning job
- Uses BERT-base-uncased for sequence classification
- Optimizes learning rate and batch size

**Usage:**
```python
# Update these variables:
BUCKET = "your-s3-bucket-name"
PROCESSED_PREFIX = "your-folder-name/"

# Run the script
%run 02_train_model.py
```

### 3. Model Validation (`03_validate_model.py`)
- Monitors hyperparameter tuning progress
- Identifies best performing model
- Retrieves model artifacts location

**Usage:**
```python
# Update with your tuning job name from step 2:
TUNING_JOB_NAME = "your-tuning-job-name"

# Run the script
%run 03_validate_model.py
```

### 4. Model Testing (`04_test_model.py`)
- Creates temporary endpoint with best model
- Evaluates model on test set
- Calculates final accuracy metrics
- Cleans up test endpoint

**Usage:**
```python
# Update these variables:
BUCKET = "your-s3-bucket-name"
PROCESSED_PREFIX = "your-folder-name/"
MODEL_ARTIFACTS_URI = "s3://path/to/best/model/artifacts"

# Run the script
%run 04_test_model.py
```

### 5. Model Deployment (`05_deploy_model.py`)
- Deploys model to production endpoint
- Tests endpoint with sample data
- Sets up persistent inference endpoint

**Usage:**
```python
# Update these variables:
MODEL_ARTIFACTS_URI = "s3://path/to/best/model/artifacts"
ENDPOINT_NAME = "nginx-anomaly-detector-prod"

# Run the script
%run 05_deploy_model.py
```

### 6. Model Monitoring (`06_monitor_model.py`)
- Monitors endpoint health and metrics
- Runs batch inference on new data
- Saves monitoring results to S3

**Usage:**
```python
# Update these variables:
ENDPOINT_NAME = "nginx-anomaly-detector-prod"
BUCKET = "your-s3-bucket-name"
MONITORING_PREFIX = "your-folder-name/monitoring/"

# Run the script
%run 06_monitor_model.py
```

## Prerequisites

1. **SageMaker Execution Role**: Ensure your notebook has proper IAM permissions
2. **S3 Bucket**: Create an S3 bucket for storing data and results
3. **Dependencies**: The `train.py` file must be present for training

## Configuration

Before running any script, update these common variables:
- `YOUR_BUCKET_NAME`: Your S3 bucket name
- `YOUR_FOLDER_NAME`: Prefix for organizing files in S3
- `YOUR_ROLE_NAME`: SageMaker execution role (if not using default)

## Execution Order

Run the scripts in numerical order (01 → 02 → 03 → 04 → 05 → 06), waiting for each stage to complete before proceeding to the next.

## Cost Considerations

- Training uses `ml.g4dn.xlarge` instances
- Deployment uses `ml.m5.xlarge` instances
- Remember to delete endpoints when not in use to avoid charges