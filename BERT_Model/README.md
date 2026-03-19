# BERT_UPDATE

This folder contains the updated BERT-based log anomaly detection pipeline for SageMaker.

## What Changed

- Better synthetic data with hard negatives and more realistic anomaly patterns
- Log normalization to reduce overfitting to timestamps, IPs, IDs, and numeric noise
- Context-window samples instead of only single-line classification
- Class-weighted training for imbalanced anomaly labels
- Threshold tuning during training, saved in `threshold_config.json`
- SageMaker-first workflow:
  - training artifacts stay in S3
  - validation reads SageMaker metadata first
  - deployment resolves the best model artifact directly from SageMaker

## Files

- `01_create_data.py`: generate train/validation/test datasets and optionally upload them to S3
- `train.py`: Hugging Face training entry point used by SageMaker
- `02_train_model.py`: launch SageMaker hyperparameter tuning and optionally monitor it until completion
- `03_validate_model.py`: report final results from a training job or tuning job using SageMaker metadata
- `04_test_model.py`: test a local saved model or deployed endpoint
- `05_deploy_model.py`: deploy from a model artifact S3 URI, training job, or tuning job
- `06_monitor_model.py`: run simple endpoint health and inference monitoring
- `monitor_training.py`: watch a SageMaker training job or tuning job until it finishes
- `inference.py`: custom SageMaker inference entry point that loads `threshold_config.json`

## Recommended Workflow

### 1. Create the dataset

Run from SageMaker Notebook:

```python
!python BERT_UPDATE/01_create_data.py \
  --output-dir BERT_UPDATE/data \
  --bucket <your-bucket> \
  --s3-prefix bert-update/data
```

This creates:

- `train/train.json`
- `validation/validation.json`
- `test/test.json`
- `dataset_summary.json`

and uploads them to:

- `s3://<your-bucket>/bert-update/data/train/`
- `s3://<your-bucket>/bert-update/data/validation/`
- `s3://<your-bucket>/bert-update/data/test/`

### 2. Launch and monitor training

```python
!python BERT_UPDATE/02_train_model.py \
  --bucket <your-bucket> \
  --prefix bert-update/data \
  --train-script BERT_UPDATE/train.py \
  --wait
```

What this does:

- launches a SageMaker hyperparameter tuning job
- prints the tuning job name
- if `--wait` is set, monitors the tuning job until it reaches `Completed`, `Failed`, or `Stopped`

Optional:

```python
!python BERT_UPDATE/02_train_model.py \
  --bucket <your-bucket> \
  --prefix bert-update/data \
  --train-script BERT_UPDATE/train.py \
  --wait \
  --poll-seconds 30
```

### 3. Validate the final result

For a tuning job:

```python
!python BERT_UPDATE/03_validate_model.py --tuning-job-name <your-tuning-job-name>
```

For a single training job:

```python
!python BERT_UPDATE/03_validate_model.py --training-job-name <your-training-job-name>
```

This script reports primarily from SageMaker metadata:

- best training job
- training/tuning status
- model artifact S3 URI
- final metrics reported by SageMaker
- tuned threshold if it was captured as a metric

### 4. Deploy the best model

Deploy directly from a tuning job:

```python
!python BERT_UPDATE/05_deploy_model.py \
  --tuning-job-name <your-tuning-job-name> \
  --endpoint-name nginx-anomaly-detector-update
```

Or deploy from a training job:

```python
!python BERT_UPDATE/05_deploy_model.py \
  --training-job-name <your-training-job-name> \
  --endpoint-name nginx-anomaly-detector-update
```

Or deploy from a direct S3 artifact URI:

```python
!python BERT_UPDATE/05_deploy_model.py \
  --model-artifacts-uri s3://<your-bucket>/<path>/model.tar.gz \
  --endpoint-name nginx-anomaly-detector-update
```

The deployed endpoint uses `inference.py`, which loads `threshold_config.json` from the model artifact automatically.

### 5. Watch training separately if needed

```python
!python BERT_UPDATE/monitor_training.py --tuning-job-name <your-tuning-job-name>
```

or:

```python
!python BERT_UPDATE/monitor_training.py --training-job-name <your-training-job-name>
```

### 6. Test and monitor the endpoint

Quick endpoint test:

```python
!python BERT_UPDATE/04_test_model.py --endpoint-name nginx-anomaly-detector-update
```

Simple monitoring script:

```python
!python BERT_UPDATE/06_monitor_model.py
```

## Important Outputs

- `threshold_config.json`: tuned anomaly threshold stored in the model artifact
- `training_summary.json`: detailed training summary stored in the model artifact
- SageMaker model artifact:
  - `s3://.../model.tar.gz`

## Notes

- `03_validate_model.py` is metadata-first. It does not need to download the model artifact for the normal SageMaker workflow.
- `05_deploy_model.py` deploys directly from the best SageMaker artifact in S3.
- If your SageMaker account has limited quota, keep `max_parallel_jobs=1` in `02_train_model.py`.
- The current runtime configuration uses:
  - `transformers_version="4.26"`
  - `pytorch_version="1.13"`
  - `py_version="py39"`
  because those are broadly compatible with common SageMaker notebook SDK setups.
