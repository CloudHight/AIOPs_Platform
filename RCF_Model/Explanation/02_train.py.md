# **02_train.py - Line-by-Line Explanation**

This script trains a **Random Cut Forest (RCF)** model on the CPU usage data for anomaly detection. Here's what each part does:

---

## **1. Import Libraries**
```python
import pandas as pd
from sagemaker import RandomCutForest, get_execution_role
import json
```
- **pandas (pd)**: For reading and manipulating the dataset
- **RandomCutForest**: Amazon SageMaker's implementation of the RCF algorithm
- **get_execution_role**: Gets AWS IAM permissions for SageMaker operations
- **json**: For saving model information as JSON file

---

## **2. Function Definition**
```python
def train_model(input_file="cpu_time_series_with_anomalies.csv",
                model_info_file="model_info.json"):
```
- Defines a function named `train_model`
- Takes two optional parameters:
  1. `input_file`: Path to the dataset CSV (default: our generated data)
  2. `model_info_file`: Where to save model metadata (default: `"model_info.json"`)

---

## **3. Load and Validate Data**
```python
    print("========== Training Random Cut Forest ==========")
    
    df = pd.read_csv(input_file)
    if df.empty or 'Average' not in df:
        print("No valid data for model training. Exiting.")
        return None
```
- Prints a header for clarity in logs
- Reads the CSV file into a pandas DataFrame
- **Safety check**: 
  - Ensures DataFrame isn't empty
  - Ensures 'Average' column exists (this is our feature for training)
- If checks fail, prints error and returns `None` (stopping the pipeline)

---

## **4. Prepare Training Data**
```python
    train_values = df['Average'].dropna().values.reshape(-1, 1)
    n_points = len(train_values)
```
- `df['Average']`: Extracts only the CPU usage column
- `.dropna()`: Removes any missing/NaN values (though our synthetic data shouldn't have any)
- `.values`: Converts to numpy array
- `.reshape(-1, 1)`: Reshapes to 2D array with 1 column
  - Example: `[45, 50, 55]` → `[[45], [50], [55]]`
  - RCF expects 2D input even for single feature
- `n_points`: Counts how many data points we have

---

## **5. Configure Model Hyperparameters**
```python
    num_trees = max(50, min(n_points, 1000)) if n_points >= 50 else n_points
```
- **Dynamic tree count** based on data size:
  - Minimum: 50 trees (or `n_points` if less than 50)
  - Maximum: 1000 trees (or `n_points` if less than 1000)
  - **Why?**: RCF performs better with more trees, but there's a computational cost
  - **Example**: With 10,080 points → `num_trees = 1000`

---

## **6. Print Training Info**
```python
    print(f"Training on {n_points} points with {num_trees} trees...")
```
- Shows progress information in logs
- Example: "Training on 10080 points with 1000 trees..."

---

## **7. Initialize Random Cut Forest Model**
```python
    rcf = RandomCutForest(
        role=get_execution_role(),
        instance_count=1,
        instance_type='ml.m5.large',
        num_samples_per_tree=200 if n_points > 200 else n_points,
        num_trees=num_trees,
        eval_metrics=["accuracy", "precision_recall_fscore"],
        feature_dim=1
    )
```
Configures the SageMaker RCF estimator with these parameters:

- `role`: AWS IAM role with SageMaker permissions
- `instance_count`: 1 training instance
- `instance_type`: 'ml.m5.large' (2 vCPUs, 8 GB RAM - cost-effective)
- `num_samples_per_tree`: 
  - Max 200 samples per tree (or `n_points` if less than 200)
  - Controls tree depth/size
- `num_trees`: Calculated earlier (e.g., 1000)
- `eval_metrics`: Tracks model performance metrics during training
- `feature_dim`: 1 (we only have 1 feature: CPU usage)

---

## **8. Train the Model**
```python
    rcf.fit(rcf.record_set(train_values))
```
- `rcf.record_set()`: Converts numpy array to SageMaker's proprietary RecordSet format
- `rcf.fit()`: Starts the training job on AWS SageMaker
  - This is **asynchronous** - it launches a cloud training job
  - SageMaker spins up the 'ml.m5.large' instance, trains, then shuts it down
  - The model is saved to S3 automatically

---

## **9. Save Model Information**
```python
    model_info = {
        'training_job_name': rcf.latest_training_job.name
    }
    with open(model_info_file, 'w') as f:
        json.dump(model_info, f)
```
- Creates a dictionary with the training job name
- The `training_job_name` is a unique identifier SageMaker assigns
- Saves this info to `model_info.json` (for later reference/deployment)
- **Example content**: `{"training_job_name": "randomcutforest-2025-01-01-12-00-00-123"}`

---

## **10. Return Model Object**
```python
    print(f"Model training complete. Info saved to {model_info_file}")
    return rcf
```
- Prints completion message
- Returns the `rcf` estimator object (which now has a trained model attached)

---

## **11. Main Execution Block**
```python
if __name__ == "__main__":
    train_model()
```
- When script runs directly, executes `train_model()`
- Allows script to be imported OR run standalone

---

## **SUMMARY: What This Script Does**

1. **Loads** the CPU usage data
2. **Configures** a Random Cut Forest model with appropriate settings
3. **Trains** the model on AWS SageMaker infrastructure
4. **Saves** metadata about the trained model
5. **Returns** the trained model object

**Key Concept**: Random Cut Forest is an **unsupervised** anomaly detection algorithm. It doesn't use the `anomaly` labels during training - it learns what "normal" CPU usage patterns look like, then flags deviations as potential anomalies.

**Cloud Aspect**: The actual training happens on AWS SageMaker, not your local machine. SageMaker handles:
- Provisioning compute resources
- Managing the training job
- Storing the model artifacts in S3
- Logging and monitoring

The training might take 5-15 minutes depending on data size and instance type.