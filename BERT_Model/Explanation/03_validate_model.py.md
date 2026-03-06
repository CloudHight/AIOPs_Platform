# **03_validate_model.py - Complete Line-by-Line Explanation**

## **Overview**
This script **monitors the hyperparameter tuning job** launched by `02_train_model.py` and **finds the best trained model**. It watches the training progress and identifies which hyperparameter combination performed best.

---

## **Section 1: Import Statements (Lines 1-4)**

```python
import boto3               # AWS SDK to talk to SageMaker
import time                # For adding delays between checks
from sagemaker.tuner import HyperparameterTuner  # For tuning job info
```

**Translation:** "Get AWS tools to check on our training jobs, and tools to wait between checks."

---

## **Section 2: Monitor Tuning Job Function (Lines 6-51)**

### **Part A: Function Definition**
```python
def monitor_tuning_job(tuning_job_name):
    sm_client = boto3.client('sagemaker')  # Connect to SageMaker service
    print(f"[INFO] Monitoring tuning job: {tuning_job_name}")
```

**Translation:** "Create a function to watch a tuning job. Connect to AWS SageMaker API."

---

### **Part B: Continuous Monitoring Loop**
```python
    while True:  # Keep checking until job finishes
        # Ask SageMaker for current job status
        response = sm_client.describe_hyper_parameter_tuning_job(
            HyperParameterTuningJobName=tuning_job_name
        )
        status = response['HyperParameterTuningJobStatus']  # Running, Completed, Failed
        counters = response['TrainingJobStatusCounters']    # How many jobs are in each state
        
        # Print progress
        print(f"[TUNER STATUS] {status} | Completed: {counters.get('Completed',0)}, InProgress: {counters.get('InProgress',0)}, Failed: {counters.get('Failed',0)}")
```

**Translation:** "Start a loop that keeps checking the tuning job status. Print how many training jobs have finished, are running, or failed."

**Example Output:**
```
[TUNER STATUS] InProgress | Completed: 2, InProgress: 1, Failed: 0
```
Meaning: 2 jobs finished, 1 is running, 0 failed

---

### **Part C: Check Best Job So Far**
```python
        best = response.get('BestTrainingJob', {})  # Get info about best performing job
        
        if best.get('TrainingJobName'):  # If we have a "best" job identified
            print(f"  [INFO] Current best: {best['TrainingJobName']}")
            
            # Get the metric that makes this job the best
            metric = best.get('FinalHyperParameterTuningJobObjectiveMetric', {})
            if metric:
                print(f"      Best {metric.get('MetricName','metric')}: {metric.get('Value','?')}")
            
            # Print the hyperparameters used in best job
            print(f"      Tuned hyperparameters: {best.get('TunedHyperParameters',{})}")
```

**Translation:** "Check if SageMaker has identified a best-performing training job yet. If yes, print:
1. Which job is best
2. What F1 score it achieved  
3. What hyperparameters it used"

**Example Output:**
```
  [INFO] Current best: hf-nginx-classifier-231101-1045-003-abc123
      Best eval_f1_macro: 0.942
      Tuned hyperparameters: {'learning_rate': '0.00003', 'per_device_train_batch_size': '16'}
```

---

### **Part D: Check If Job Finished**
```python
        # Check if job has reached a final state
        if status in ('Completed', 'Failed', 'Stopped'):
            print(f"[TUNER DONE] Final status: {status}")
            break  # Exit the loop
        
        time.sleep(60)  # Wait 1 minute before checking again
```

**Translation:** "If the tuning job finished (successfully or not), stop checking. Otherwise, wait 60 seconds and check again."

---

### **Part E: Return Results**
```python
    if status == "Completed":
        print(f"[INFO] Best training job: {best.get('TrainingJobName')}")
        return best  # Return info about best job
    else:
        print("[ERROR] Tuning failed or was stopped.")
        return None  # Return nothing if failed
```

**Translation:** "If tuning completed successfully, return information about the best job. If it failed, return nothing."

---

## **Section 3: Get Model Location Function (Lines 53-60)**

```python
def get_best_model_artifacts(best_training_job_name):
    sm_client = boto3.client('sagemaker')
    
    # Get detailed info about the best training job
    response = sm_client.describe_training_job(
        TrainingJobName=best_training_job_name
    )
    
    # Extract where the trained model is stored
    model_artifacts = response['ModelArtifacts']['S3ModelArtifacts']
    print(f"[INFO] Best model artifacts: {model_artifacts}")
    return model_artifacts
```

**Translation:** "Given the name of the best training job, find out where in S3 the trained model files are saved."

**Example Output:**
```
[INFO] Best model artifacts: s3://my-bucket/hf-nginx-classifier-231101-1045-003-abc123/output/model.tar.gz
```

---

## **Section 4: Main Execution (Lines 62-71)**

```python
if __name__ == "__main__":
    # Replace with your actual tuning job name from 02_train_model.py output
    TUNING_JOB_NAME = "YOUR_TUNING_JOB_NAME"
    
    # Monitor the tuning job until it completes
    best_job = monitor_tuning_job(TUNING_JOB_NAME)
    
    # If successful, get the model location
    if best_job:
        model_artifacts = get_best_model_artifacts(best_job['TrainingJobName'])
        print(f"[INFO] Validation complete. Best model ready for testing.")
```

**Translation:** "When script runs:
1. Use the tuning job name from step 2
2. Watch it until it finishes
3. Get the location of the best model
4. Report success"

---

## **Key Concepts Simplified:**

### **1. What is This Script Doing?**
It's like a **progress tracker and judge** for the tuning job:

```
BEFORE (after 02_train_model.py):
SageMaker is training 6 models with different settings in the cloud

THIS SCRIPT:
1. Watches the 6 training races
2. Checks which runner is ahead
3. Waits for all races to finish
4. Declares the winner
5. Tells you where the trophy (best model) is stored
```

### **2. What is a "Tuning Job"?**
Think of it as a **tournament** with 6 participants (training jobs):
- Each participant runs with different settings
- SageMaker tracks their scores (F1 metrics)
- This script watches the tournament progress

### **3. The Status Loop Explained:**
```python
while True:                          # Keep watching
    check_status()                   # Ask: "How's it going?"
    print_update()                   # Tell user what's happening
    if finished: break               # If done, stop watching
    time.sleep(60)                   # Otherwise, wait 1 minute
```

### **4. Where to Get TUNING_JOB_NAME?**
From the output of `02_train_model.py`:
```
[INFO] Training job name: hf-nginx-classifier-231101-1045
                                    ↑
                         Copy this name here
```

### **5. What's in the Model Artifacts?**
The `model.tar.gz` file contains:
- Trained PyTorch model weights
- Tokenizer files
- Configuration files
- Everything needed to run the model

**Location example:**
`s3://my-bucket/hf-nginx-classifier-231101-1045-003-abc123/output/model.tar.gz`

### **6. Timing & Cost:**
- Each training job: ~20-30 minutes
- 6 jobs sequentially: ~2-3 hours
- This script runs for that entire time, checking every minute
- **No extra cost** - just checking status, not running compute

### **7. Possible Statuses:**
- **InProgress**: Still running jobs
- **Completed**: All jobs finished successfully
- **Failed**: Something went wrong
- **Stopped**: Manually stopped by user

### **8. What Happens If a Job Fails?**
The script continues monitoring! If 1 job fails out of 6, it still finds the best among the 5 that succeeded.

---

## **Visual Walkthrough:**

### **Step 1: After Running 02_train_model.py**
```
You launched 6 training jobs in the cloud:
┌─────────────────────────────────────────────────┐
│ AWS SageMaker Tuning Job: hf-nginx-classifier-001 │
│                                                 │
│ Job 1: ████████░░░░ 80% (F1: 0.89)             │
│ Job 2: ████████████ 100% ✓ (F1: 0.92) ← Best!  │
│ Job 3: █░░░░░░░░░░░ 10%                        │
│ Job 4: Waiting...                              │
│ Job 5: Waiting...                              │
│ Job 6: Waiting...                              │
└─────────────────────────────────────────────────┘
```

### **Step 2: Running This Script**
```
You: "Hey AWS, how's job hf-nginx-classifier-001 doing?"
AWS: "2 completed, 1 in progress, 3 waiting"
You: "Which is best so far?"
AWS: "Job 2 with F1=0.92, learning_rate=0.00003, batch_size=16"
You: (waits 60 seconds)
You: "How about now?"
... (repeats every minute until all jobs finish)
```

### **Step 3: Final Result**
```
All 6 jobs complete! Final results:
┌─────────────────────────────────────────────────┐
│ Ranking:                                         │
│ 1. Job 4: F1=0.945 (learning_rate=0.00002, bs=16)│
│ 2. Job 2: F1=0.942 (learning_rate=0.00003, bs=16)│
│ 3. Job 6: F1=0.938 (learning_rate=0.000025, bs=32)│
│ ...                                              │
└─────────────────────────────────────────────────┘

Winner: Job 4!
Model saved at: s3://my-bucket/.../model.tar.gz
```

### **Step 4: Next Step**
The model location (`s3://.../model.tar.gz`) is passed to `04_test_model.py` for testing.

---

## **Common Issues & Solutions:**

### **1. "Job stuck on InProgress"**
- Normal: Training takes time
- Check CloudWatch logs if stuck for hours

### **2. "Failed jobs"**
- Might be out of memory (batch size too large)
- Check error messages in SageMaker console

### **3. "Where's my tuning job name?"**
- Check output of `02_train_model.py`
- Look in SageMaker Console → Hyperparameter tuning jobs

### **4. "Script running forever"**
- Jobs might have failed silently
- Check SageMaker console manually

---

## **Why This Step is Important:**

1. **Quality Control**: Ensures we get the best possible model
2. **Cost Control**: Stops watching when done (though jobs auto-terminate)
3. **Pipeline Automation**: Gets model location for next step automatically
4. **Transparency**: Shows what hyperparameters work best

**This script is like waiting for cookies to bake and checking which recipe made the best cookies, then saving that recipe for later!**