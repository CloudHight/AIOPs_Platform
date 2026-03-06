# **02_train_model.py - Complete Line-by-Line Explanation**

## **Overview**
This script **launches a SageMaker hyperparameter tuning job** to train the model. Instead of training once with fixed settings, it trains multiple times with different configurations to find the best one automatically.

---

## **Section 1: Import Statements (Lines 1-7)**

```python
import boto3                # AWS SDK for interacting with AWS services
import os                   # Operating system functions (file checking)
from sagemaker.huggingface import HuggingFace  # SageMaker's HuggingFace integration
from sagemaker.tuner import HyperparameterTuner  # For automatic tuning
from sagemaker.parameter import ContinuousParameter, CategoricalParameter  # Parameter types
from sagemaker import get_execution_role  # Gets AWS role for permissions
```

**Translation:** "Get AWS tools to launch training jobs in the cloud, and SageMaker-specific tools for machine learning."

---

## **Section 2: Debug Prints (Lines 9-10)**

```python
print("[INFO] Current working directory:", os.getcwd())
print("[INFO] Files in current directory:", os.listdir("."))
```

**Translation:** "Print where we are in the file system and what files are available, to help debug if `train.py` is missing."

---

## **Section 3: Main Tuning Function (Lines 12-68)**

### **Part A: Function Definition and File Check**
```python
def launch_sagemaker_tuning(
    train_uri, val_uri, train_script="train.py",
    bucket="YOUR_BUCKET_NAME", role=None
):
    # Check if train.py exists in current directory
    if not os.path.isfile(train_script):
        print(f"[WARNING] {train_script} not found in current directory.")
        print(f"[INFO] Make sure {train_script} is in the same directory as this notebook.")
        raise FileNotFoundError(f"[ERROR] {train_script} not found. Please provide your training script.")
```

**Translation:** "Define the main function. First, check if `train.py` exists because we need it for training. Crash with a clear error if it's missing."

**Visual Check:**
```
Current directory:
├── 02_train_model.py  ← This script
└── train.py           ← Required training code (from your first file)
```

---

### **Part B: Get AWS Role**
```python
    role = role or get_execution_role()
```

**Translation:** "Get the AWS IAM role that gives SageMaker permission to access S3, create instances, etc. This is like getting the 'key' to use AWS services."

---

### **Part C: Create HuggingFace Estimator (Lines 16-36)**
```python
    estimator = HuggingFace(
        entry_point=train_script,          # Which script to run (train.py)
        instance_type="ml.g4dn.xlarge",    # GPU instance type (has NVIDIA T4 GPU)
        instance_count=1,                  # Use 1 machine
        transformers_version="4.26",       # HuggingFace library version
        pytorch_version="1.13",            # PyTorch version
        py_version="py39",                 # Python version
        role=role,                         # AWS permissions role
        enable_network_isolation=False,    # Allow internet access for downloading models
        
        # Default hyperparameters (can be overridden by tuner)
        hyperparameters={
            "epochs": 3,                          # Train for 3 cycles
            "model_name": "bert-base-uncased",    # Use BERT model
            "learning_rate": 3e-5,                # Default learning speed
            "per_device_train_batch_size": 16     # Default batch size
        },
        
        # How to extract metrics from training logs
        metric_definitions=[
            {"Name": "eval_f1_macro", "Regex": r"'eval_f1_macro':\s*([0-9\.]+)"},
            {"Name": "eval_f1_micro", "Regex": r"'eval_f1_micro':\s*([0-9\.]+)"},
            {"Name": "eval_accuracy", "Regex": r"'eval_accuracy':\s*([0-9\.]+)"},
            {"Name": "eval_loss", "Regex": r"'eval_loss':\s*([0-9\.]+)"},
        ],
        base_job_name="hf-nginx-classifier",  # Base name for training jobs
    )
```

**Translation:** "Create a configuration object that tells SageMaker:
1. What script to run (`train.py`)
2. What hardware to use (GPU instance for faster training)
3. What software versions to install
4. Default training settings
5. How to read performance metrics from the training output"

**What's an "Estimator"?**
Think of it as a **training job blueprint**. It's not training yet, just the plan.

---

### **Part D: Create Hyperparameter Tuner (Lines 38-55)**
```python
    tuner = HyperparameterTuner(
        estimator=estimator,                # The training blueprint
        objective_metric_name='eval_f1_macro',  # What metric to optimize
        objective_type='Maximize',          # We want higher F1 scores
        metric_definitions=estimator.metric_definitions,  # How to read metrics
        
        # Parameters to try different values for
        hyperparameter_ranges={
            'learning_rate': ContinuousParameter(1e-5, 5e-5),        # Try values between 0.00001 and 0.00005
            'per_device_train_batch_size': CategoricalParameter([8, 16, 32])  # Try these batch sizes
        },
        max_jobs=6,        # Run up to 6 training jobs
        max_parallel_jobs=1  # Run 1 job at a time (to save cost)
    )
```

**Translation:** "Create an **automatic tuner** that will:
1. Train the model multiple times with different settings
2. Try different learning rates (how fast to learn)
3. Try different batch sizes (how many examples to process at once)
4. Track which combination gives the best F1 score
5. Run 6 experiments total, one after another"

**Hyperparameter Tuning Explained:**
Imagine baking cookies and trying different temperatures and times:
- Job 1: 350°F for 10 minutes
- Job 2: 375°F for 12 minutes  
- Job 3: 325°F for 11 minutes
- etc.
Then pick the combination that makes the best cookies!

**Parameter Types:**
- `ContinuousParameter`: Any number in a range (like 0.00001 to 0.00005)
- `CategoricalParameter`: Pick from specific values (like 8, 16, or 32)

---

### **Part E: Launch Tuning Job (Lines 57-68)**
```python
    print("[INFO] Launching SageMaker tuning job ...")
    tuner.fit({'train': train_uri, 'validation': val_uri}, wait=False)
    print("[INFO] Tuning job launched.")
    
    return tuner
```

**Translation:** "Start the tuning job in AWS SageMaker. The `wait=False` means don't wait here for it to finish (it runs in the background)."

**What Actually Happens:**
```
SageMaker does:
1. Spins up ml.g4dn.xlarge instance (GPU machine)
2. Copies train.py and data to the instance
3. Runs train.py with first hyperparameter set
4. Saves results
5. Spins down instance
6. Repeats for next hyperparameter set
7. After 6 jobs, picks the best one
```

---

## **Section 4: Main Execution (Lines 70-88)**

### **Part A: Configuration and Checks**
```python
if __name__ == "__main__":
    BUCKET = "YOUR_BUCKET_NAME"
    PROCESSED_PREFIX = "YOUR_FOLDER_NAME/"
    
    # Check if train.py exists before proceeding
    if not os.path.isfile("train.py"):
        print("[ERROR] train.py not found in current directory.")
        print("[INFO] Please ensure train.py is in the same directory as this notebook.")
        exit(1)
```

**Translation:** "Run this code when script is executed directly. First, check again if `train.py` exists (double-check)."

---

### **Part B: Set Data Locations**
```python
    train_uri = f's3://{BUCKET}/{PROCESSED_PREFIX}train/'
    val_uri = f's3://{BUCKET}/{PROCESSED_PREFIX}validation/'
```

**Translation:** "Create S3 paths to where the training and validation data are stored (uploaded by `01_create_data.py`)."

**Example:** `s3://my-bucket/my-folder/train/` and `s3://my-bucket/my-folder/validation/`

---

### **Part C: Launch the Tuning**
```python
    tuner = launch_sagemaker_tuning(train_uri, val_uri, bucket=BUCKET)
    print(f"[INFO] Training job name: {tuner.latest_tuning_job.name}")
```

**Translation:** "Start the hyperparameter tuning job and print its name so we can monitor it later."

---

## **Key Concepts Simplified:**

### **1. What is SageMaker?**
AWS's **managed ML service** - like a "machine learning cloud kitchen":
- You provide the recipe (`train.py`)
- You provide ingredients (data in S3)
- SageMaker provides:
  - Kitchen (GPU instances)
  - Cooks (runs your code)
  - Cleanup (shuts down when done)
  - Storage (saves trained models)

### **2. Hyperparameter Tuning vs Regular Training**
- **Regular training**: Train once with fixed settings
- **Hyperparameter tuning**: Train multiple times with different settings, find the best automatically

### **3. What Gets Tuned?**
Two parameters in this case:
1. **Learning rate** (1e-5 to 5e-5): How big of steps to take when learning
   - Too high: Model might overshoot and learn poorly
   - Too low: Model learns too slowly
2. **Batch size** (8, 16, or 32): How many examples to process at once
   - Smaller: More precise but slower
   - Larger: Faster but needs more memory

### **4. The Training Process Flow:**
```
Your Computer (Jupyter Notebook)
    ↓ (sends request)
AWS SageMaker (Cloud)
    ↓ (spins up instance)
GPU Machine (ml.g4dn.xlarge)
    ↓ (downloads data)
S3 Bucket (your-bucket/train/)
    ↓ (runs training)
train.py (your training script)
    ↓ (saves model)
S3 Bucket (your-bucket/models/)
    ↓ (shuts down)
Instance terminated
```

### **5. Cost Implications:**
- `ml.g4dn.xlarge`: ~$0.70-$1.00 per hour
- 6 jobs × ~30 minutes each = ~3 hours = ~$2.10-$3.00
- **Important**: Instances auto-terminate when done!

### **6. Output:**
- **Multiple trained models** in S3 (one for each hyperparameter combination)
- **Metrics** for each training job (accuracy, F1 score, etc.)
- **Best model** identified automatically

### **7. Next Step:**
After this runs, you use `03_validate_model.py` to check which training job performed best and get the location of the best model.

---

## **Visual Summary:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Jupyter Notebook                    │
│                                                             │
│  02_train_model.py                                          │
│  ├── Checks: train.py exists? ✅                           │
│  ├── Creates: Estimator (training blueprint)               │
│  ├── Creates: Tuner (tries 6 combinations)                 │
│  └── Launches: → AWS SageMaker                             │
└───────────────────────────────┬─────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────┐
│                    AWS SageMaker (Cloud)                    │
│                                                             │
│  Job 1: learning_rate=0.00001, batch_size=8                 │
│  Job 2: learning_rate=0.00003, batch_size=16                │
│  Job 3: learning_rate=0.00005, batch_size=32                │
│  Job 4: learning_rate=0.00002, batch_size=16                │
│  Job 5: learning_rate=0.00004, batch_size=8                 │
│  Job 6: learning_rate=0.000025, batch_size=32               │
│                                                             │
│  ✅ Picks Job 4 as best (highest F1 score)                 │
│  📦 Saves best model to S3                                 │
└─────────────────────────────────────────────────────────────┘
```

**This script doesn't train the model itself - it tells AWS SageMaker to train it 6 different ways in the cloud and find the best version!**