# **04_test_model.py - Complete Line-by-Line Explanation**

## **Overview**
This script **tests the best model** found in step 3 on unseen data. It creates a temporary endpoint, evaluates the model's accuracy, and saves the results.

---

## **Section 1: Import Statements (Lines 1-5)**

```python
import boto3                          # AWS SDK for S3 and SageMaker
import json                           # For working with JSON data
from sagemaker.huggingface import HuggingFaceModel  # To deploy model
from sagemaker import get_execution_role  # AWS permissions role
```

**Translation:** "Get tools to work with AWS (S3 for data, SageMaker for model), JSON for data format, and HuggingFace for deploying our trained model."

---

## **Section 2: Load Test Data Function (Lines 7-14)**

```python
def load_test_data_from_s3(bucket_name, object_key):
    s3 = boto3.client('s3')  # Connect to AWS S3 service
    
    # Download the test data file from S3
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    data = response['Body'].read().decode()  # Read file content
    
    test_data = json.loads(data)  # Convert JSON string to Python list
    print(f"[INFO] Loaded {len(test_data)} test entries from S3")
    return test_data
```

**Translation:** "Download the test dataset from AWS S3 cloud storage. This is data the model has NEVER seen before - its 'final exam'."

**Example:**
- `bucket_name`: "my-bucket"
- `object_key`: "my-folder/test/test.json"
- Downloads from: `s3://my-bucket/my-folder/test/test.json`

**What's in test.json?**
```json
[
  {"text": "192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] GET...", "label": 0},
  {"text": "2024/01/01 12:02:00 [error] upstream...", "label": 1},
  ... 218 more examples ...
]
```

---

## **Section 3: Create Temporary Endpoint Function (Lines 16-32)**

```python
def create_model_from_artifacts(model_artifacts_uri, role=None):
    role = role or get_execution_role()  # Get AWS permissions
    
    # Create a HuggingFace model object from saved model files
    huggingface_model = HuggingFaceModel(
        model_data=model_artifacts_uri,  # Location of trained model (from step 3)
        transformers_version='4.26',     # Same version used for training
        pytorch_version='1.13',          # Same PyTorch version
        py_version='py39',               # Same Python version
        role=role                        # AWS permissions
    )
    
    print("[INFO] Creating temporary endpoint for testing...")
    
    # Deploy model to a temporary endpoint
    predictor = huggingface_model.deploy(
        initial_instance_count=1,        # Use 1 server
        instance_type="ml.m5.xlarge",    # CPU instance (cheaper than GPU)
        endpoint_name=f"test-endpoint-{int(time.time())}"  # Unique name with timestamp
    )
    
    return predictor
```

**Translation:** "Take the trained model files and put them on a server where we can send it log entries and get predictions back."

**Key Points:**
- `model_artifacts_uri`: From step 3 output (e.g., `s3://my-bucket/.../model.tar.gz`)
- Creates a **temporary** endpoint (not permanent, to save money)
- Uses `ml.m5.xlarge` (CPU) instead of GPU - cheaper for testing
- Unique name with timestamp prevents conflicts

**What Happens in AWS:**
```
1. SageMaker spins up an ml.m5.xlarge instance
2. Downloads model.tar.gz from S3
3. Sets up the model to accept requests
4. Gives us a URL to send data to
```

---

## **Section 4: Evaluate Model Function (Lines 34-69)**

### **Part A: Setup**
```python
def evaluate_model_on_test_set(predictor, test_data):
    correct = 0      # Count correct predictions
    total = len(test_data)  # Total number of test examples
    predictions = []  # Store all prediction details
```

**Translation:** "Prepare to test the model. We'll count how many it gets right and store details of each prediction."

---

### **Part B: Test Each Example**
```python
    # Go through each test example
    for item in test_data:
        text = item['text']         # The log entry text
        true_label = item['label']  # Correct answer (0=normal, 1=anomaly)
```

**Translation:** "Get one test example - a log entry and its correct label."

---

### **Part C: Get Model Prediction**
```python
        try:
            # Send text to the model endpoint
            response = predictor.predict({"inputs": text})
            
            # Convert model output to 0 or 1
            # Model returns 'LABEL_1' for anomaly, 'LABEL_0' for normal
            predicted_label = 1 if response[0]['label'] == 'LABEL_1' else 0
```

**Translation:** "Ask the model: 'Is this log entry normal or anomalous?' Model responds with something like `{'label': 'LABEL_1', 'score': 0.95}`."

**Model Response Format:**
```json
[
  {
    "label": "LABEL_1",  # Prediction (LABEL_0=normal, LABEL_1=anomaly)
    "score": 0.954       # Confidence (0.0 to 1.0)
  }
]
```

---

### **Part D: Check If Correct**
```python
            # Store prediction details
            predictions.append({
                'text': text,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'correct': predicted_label == true_label  # True if correct
            })
            
            # Count if correct
            if predicted_label == true_label:
                correct += 1
                
        except Exception as e:
            print(f"[WARNING] Error predicting: {str(e)[:60]}")  # Show first 60 chars of error
```

**Translation:** "Record what happened and check if the model was right."

**Example Prediction Object:**
```json
{
  "text": "192.168.1.2 - - [01/Jan/2024:12:01:00 +0000] GET /api/data HTTP/1.1 500 0",
  "true_label": 1,
  "predicted_label": 1,
  "correct": true
}
```

---

### **Part E: Calculate Accuracy**
```python
    # Calculate overall accuracy
    accuracy = correct / total if total > 0 else 0
    print(f"[INFO] Test accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return predictions, accuracy
```

**Translation:** "Calculate percentage of correct predictions. Example: 198/220 = 0.90 = 90% accuracy."

---

## **Section 5: Save Results Function (Lines 71-84)**

```python
def save_test_results(predictions, accuracy, bucket, s3_key):
    # Create results object
    results = {
        'test_accuracy': accuracy,          # Overall accuracy
        'total_predictions': len(predictions),  # How many tested
        'predictions': predictions          # Detailed results
    }
    
    # Upload to S3
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket=bucket,           # Your S3 bucket
        Key=s3_key,              # Where to save (e.g., "my-folder/test_results.json")
        Body=json.dumps(results, indent=2)  # Convert to pretty JSON
    )
    
    print(f"[INFO] Test results saved to s3://{bucket}/{s3_key}")
```

**Translation:** "Save the test results to S3 so we have a record of how well the model performed."

**Example Output in S3:**
```json
{
  "test_accuracy": 0.9364,
  "total_predictions": 220,
  "predictions": [
    {
      "text": "192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] GET...",
      "true_label": 0,
      "predicted_label": 0,
      "correct": true
    },
    ... 219 more ...
  ]
}
```

---

## **Section 6: Main Execution (Lines 86-109)**

### **Part A: Configuration and Setup**
```python
if __name__ == "__main__":
    import time  # For timestamp in endpoint name
    
    BUCKET = "YOUR_BUCKET_NAME"
    PROCESSED_PREFIX = "YOUR_FOLDER_NAME/"
    MODEL_ARTIFACTS_URI = "s3://YOUR_BUCKET_NAME/path/to/model/artifacts"  # From validation step
```

**Translation:** "Set configuration variables. `MODEL_ARTIFACTS_URI` comes from step 3 output."

---

### **Part B: Load Test Data**
```python
    test_data = load_test_data_from_s3(BUCKET, f"{PROCESSED_PREFIX}test/test.json")
```

**Translation:** "Download the test data created in step 1."

---

### **Part C: Deploy and Test**
```python
    predictor = create_model_from_artifacts(MODEL_ARTIFACTS_URI)
    
    try:
        # Run the test
        predictions, accuracy = evaluate_model_on_test_set(predictor, test_data)
        
        # Save results
        save_test_results(predictions, accuracy, BUCKET, f"{PROCESSED_PREFIX}test_results.json")
        
        print(f"[INFO] Testing complete. Final accuracy: {accuracy:.4f}")
    
    finally:
        # Always clean up, even if there's an error
        predictor.delete_endpoint()
        print("[INFO] Test endpoint cleaned up")
```

**Translation:** "Deploy model, test it, save results, and ALWAYS delete the endpoint (to avoid paying for unused servers)."

**The `try/finally` pattern ensures cleanup:**
```
try:
    deploy → test → save results
finally:
    ALWAYS delete endpoint (even if error occurs)
```

---

## **Key Concepts Simplified:**

### **1. What is an "Endpoint"?**
Think of it as a **phone number for your model**:
- You call the number (send HTTP request)
- Model answers (returns prediction)
- After test, you hang up (delete endpoint)

**Real example:**
```
URL: https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/test-endpoint-1706891234/invocations
You send: {"inputs": "192.168.1.1 - - [01/Jan/2024...]"}
Model responds: [{"label": "LABEL_0", "score": 0.98}]
```

### **2. Why Temporary?**
- **Cost**: Endpoints cost money while running (~$0.12/hour for ml.m5.xlarge)
- **Test only**: We just need it for 5-10 minutes to test
- **Production later**: Step 5 creates a permanent endpoint

### **3. The Testing Process:**
```
Step 1: Get 220 test examples (model never saw these)
Step 2: For each example:
    - Show model: "Here's a log entry"
    - Model: "I think it's NORMAL with 98% confidence"
    - Check: Is model correct? ✓ or ✗
Step 3: Calculate: (Correct answers) / (Total) = Accuracy
```

### **4. Accuracy Interpretation:**
- **0.90**: 90% correct - Good!
- **0.50**: 50% correct - As good as guessing (bad!)
- **0.99**: 99% correct - Excellent!

### **5. Cost Breakdown:**
- **ml.m5.xlarge**: ~$0.12 per hour
- **Testing time**: ~5-10 minutes
- **Cost**: ~$0.01-$0.02 (pennies!)

### **6. Model Artifacts URI:**
From step 3 output:
```
s3://my-bucket/hf-nginx-classifier-231101-1045-003-abc123/output/model.tar.gz
```
Contains:
```
model.tar.gz
├── pytorch_model.bin  (Model weights)
├── config.json        (Model configuration)
├── tokenizer.json     (Tokenizer files)
└── special_tokens_map.json
```

### **7. The Cleanup is CRITICAL:**
**Without cleanup**: Endpoint keeps running → **$86.40/month!**
**With cleanup**: Endpoint runs 10 minutes → **$0.02**

---

## **Visual Walkthrough:**

### **Step 1: Setup**
```
Your Computer → AWS:
"Hey AWS, spin up a server and load this model: s3://my-bucket/.../model.tar.gz"

AWS:
✅ Creates ml.m5.xlarge instance
✅ Downloads model
✅ Sets up endpoint: test-endpoint-1706891234
✅ Returns: "Ready at https://..."
```

### **Step 2: Testing**
```
For each of 220 test log entries:
You → Endpoint: "Is this normal or anomaly?"
Endpoint → You: "LABEL_1 (anomaly) with 95% confidence"
You: Check if correct, record result

Progress: ███████████████████████░░ 198/220 (90%)
```

### **Step 3: Results**
```
Final Score: 198/220 = 90.0% accuracy

Detailed results saved to:
s3://my-bucket/my-folder/test_results.json

Example failures:
1. Log: "192.168.1.1 ... 404 ..." (normal)
   Model said: ANOMALY (wrong)
2. Log: "[warn] client closed connection" (normal)  
   Model said: ANOMALY (wrong)
```

### **Step 4: Cleanup**
```
You → AWS: "Delete test-endpoint-1706891234"
AWS: ✅ Instance terminated
AWS: ✅ No more charges
```

---

## **What Could Go Wrong:**

### **1. Low Accuracy (< 85%)**
- Model might be overfitting (memorized training data)
- Need more/better training data
- Try different model architecture

### **2. Endpoint Creation Fails**
- Check IAM permissions
- Model artifacts might be corrupted
- Instance type might not be available in your region

### **3. Slow Predictions**
- Normal: 100-200ms per prediction
- If >1 second: Check instance health

### **4. Memory Errors**
- Test data too large
- Model too big for ml.m5.xlarge
- Solution: Process in batches

---

## **Why This Step is Important:**

1. **Final Validation**: Ensures model works on real unseen data
2. **Performance Baseline**: Sets accuracy benchmark for future improvements
3. **Cost Verification**: Tests that deployment works before production
4. **Risk Mitigation**: Catches bad models before they go live

**This is like giving your newly trained employee (the model) a final job interview before hiring them full-time!**