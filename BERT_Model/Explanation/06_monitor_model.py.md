# **06_monitor_model.py - Complete Line-by-Line Explanation**

## **Overview**
This script **monitors your production model endpoint** - it checks health metrics, runs periodic inference tests, and saves monitoring results. Think of it as your model's "check-up doctor" that makes sure everything is running smoothly.

---

## **Section 1: Import Statements (Lines 1-5)**

```python
import boto3                 # AWS SDK for CloudWatch and SageMaker
import json                 # For working with JSON data
import time                 # For adding delays if needed
from datetime import datetime, timedelta  # For time calculations
```

**Translation:** "Get AWS tools to check metrics and run inferences, plus time tools to calculate time ranges."

---

## **Section 2: Get Endpoint Metrics Function (Lines 7-30)**

### **Part A: Function Definition**
```python
def get_endpoint_metrics(endpoint_name, start_time, end_time):
    cloudwatch = boto3.client('cloudwatch')  # AWS monitoring service
```

**Translation:** "Create a function to get monitoring metrics. CloudWatch is AWS's built-in monitoring system."

---

### **Part B: Define What Metrics to Collect**
```python
    # Important metrics to track
    metrics = [
        'Invocations',          # How many predictions made
        'ModelLatency',         # How fast predictions are (in milliseconds)
        'Invocation4XXErrors',  # Client errors (bad requests)
        'Invocation5XXErrors'   # Server errors (model crashes)
    ]
    
    endpoint_metrics = {}  # Dictionary to store all metrics
```

**Translation:** "We'll track 4 key metrics to understand endpoint health and performance."

**What each metric means:**
1. **Invocations**: Number of prediction requests (like "how many customers visited")
2. **ModelLatency**: Response time in ms (like "how fast is service")
3. **Invocation4XXErrors**: Client-side errors (like "customer gave wrong order")
4. **Invocation5XXErrors**: Server-side errors (like "kitchen burned down")

---

### **Part C: Get Each Metric from CloudWatch**
```python
    for metric_name in metrics:
        # Request metric data from CloudWatch
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/SageMaker',      # SageMaker metrics
            MetricName=metric_name,         # Which metric to get
            Dimensions=[                     # Which endpoint
                {'Name': 'EndpointName', 'Value': endpoint_name}
            ],
            StartTime=start_time,           # When to start looking
            EndTime=end_time,               # When to stop looking
            Period=300,                     # 5-minute intervals (300 seconds)
            Statistics=['Sum', 'Average']   # Get both total and average
        )
        endpoint_metrics[metric_name] = response['Datapoints']
```

**Translation:** "For each metric, ask CloudWatch for data from the specified time range in 5-minute chunks."

**Time parameters explained:**
- `StartTime` and `EndTime`: Like "check records from 2pm to 3pm"
- `Period=300`: Group data into 5-minute buckets
- `Statistics=['Sum', 'Average']`: Get both total count and average value

---

### **Part D: Return Metrics**
```python
    return endpoint_metrics
```

**Translation:** "Return all the collected metrics."

---

## **Section 3: Monitor Endpoint Health Function (Lines 32-51)**

```python
def monitor_endpoint_health(endpoint_name):
    # Look at last hour's data
    end_time = datetime.utcnow()  # Current time
    start_time = end_time - timedelta(hours=1)  # 1 hour ago
    
    print(f"[INFO] Monitoring endpoint: {endpoint_name}")
    print(f"[INFO] Time range: {start_time} to {end_time}")
```

**Translation:** "Create a function to check endpoint health. We'll look at the last hour of data."

---

```python
    # Get metrics for the last hour
    metrics = get_endpoint_metrics(endpoint_name, start_time, end_time)
    
    # Print each metric's latest value
    for metric_name, datapoints in metrics.items():
        if datapoints:  # If we have data
            # Get the most recent data point
            latest = max(datapoints, key=lambda x: x['Timestamp'])
            
            # Print either Sum or Average, whichever is available
            print(f"  {metric_name}: {latest.get('Sum', latest.get('Average', 0))}")
        else:
            print(f"  {metric_name}: No data")
```

**Translation:** "Get metrics and print the most recent value for each one."

**Example Output:**
```
[INFO] Monitoring endpoint: nginx-anomaly-detector-prod
[INFO] Time range: 2024-01-20 13:00:00 to 2024-01-20 14:00:00
  Invocations: 1250  # 1250 predictions in last hour
  ModelLatency: 85.2  # 85.2ms average response time
  Invocation4XXErrors: 2  # 2 bad requests
  Invocation5XXErrors: 0  # No server errors - good!
```

---

## **Section 4: Run Batch Inference Function (Lines 53-86)**

### **Part A: Function Setup**
```python
def run_inference_batch(endpoint_name, log_entries):
    runtime = boto3.client('sagemaker-runtime')  # Connect to endpoint
    results = []  # Store all prediction results
```

**Translation:** "Create a function to test the endpoint with sample logs."

---

### **Part B: Process Each Log Entry**
```python
    print(f"[INFO] Running batch inference on {len(log_entries)} entries")
    
    for i, entry in enumerate(log_entries):
        # Extract text from log entry
        text = entry if isinstance(entry, str) else entry.get('text', str(entry))
```

**Translation:** "Loop through each log entry. Handle both string entries and dictionary entries."

---

### **Part C: Send to Endpoint**
```python
        try:
            # Send to endpoint for prediction
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps({"inputs": text})
            )
            
            # Parse response
            result = json.loads(response['Body'].read().decode())
```

**Translation:** "Send the log text to our endpoint and get prediction back."

---

### **Part D: Store Results**
```python
            # Store prediction details
            results.append({
                'text': text,                      # Original log
                'prediction': result[0]['label'],  # LABEL_0 or LABEL_1
                'confidence': result[0]['score'],  # 0.0 to 1.0
                'timestamp': datetime.utcnow().isoformat()  # When predicted
            })
```

**Translation:** "Save prediction with metadata for tracking."

**Example result:**
```json
{
  "text": "192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] GET...",
  "prediction": "LABEL_0",
  "confidence": 0.987,
  "timestamp": "2024-01-20T14:00:00.123456"
}
```

---

### **Part E: Progress Updates and Error Handling**
```python
            # Show progress every 10 entries
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(log_entries)} entries")
                
        except Exception as e:
            print(f"  Error processing entry {i}: {str(e)}")
```

**Translation:** "Give progress feedback and handle any errors gracefully."

---

### **Part F: Return Results**
```python
    return results
```

**Translation:** "Return all prediction results."

---

## **Section 5: Save Monitoring Results Function (Lines 88-108)**

```python
def save_monitoring_results(results, bucket, s3_key):
    # Create monitoring report
    monitoring_data = {
        'timestamp': datetime.utcnow().isoformat(),  # When monitored
        'total_predictions': len(results),           # How many tested
        'anomaly_count': sum(1 for r in results if r['prediction'] == 'LABEL_1'),  # How many anomalies found
        'predictions': results                       # All prediction details
    }
```

**Translation:** "Create a structured report with monitoring results."

---

```python
    # Upload to S3 for permanent storage
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=json.dumps(monitoring_data, indent=2)  # Pretty-print JSON
    )
    
    print(f"[INFO] Monitoring results saved to s3://{bucket}/{s3_key}")
    return monitoring_data
```

**Translation:** "Save the monitoring report to S3 cloud storage."

**Example saved file:**
```json
{
  "timestamp": "2024-01-20T14:00:00.123456",
  "total_predictions": 5,
  "anomaly_count": 2,
  "predictions": [
    {
      "text": "192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] GET...",
      "prediction": "LABEL_0",
      "confidence": 0.987,
      "timestamp": "2024-01-20T14:00:00.123456"
    },
    ... 4 more ...
  ]
}
```

---

## **Section 6: Main Execution (Lines 110-148)**

### **Part A: Configuration**
```python
if __name__ == "__main__":
    ENDPOINT_NAME = "nginx-anomaly-detector-prod"
    BUCKET = "YOUR_BUCKET_NAME"
    MONITORING_PREFIX = "YOUR_FOLDER_NAME/monitoring/"
```

**Translation:** "Set up configuration - same endpoint name from step 5, and where to save monitoring results."

---

### **Part B: Monitor Health Metrics**
```python
    # Monitor endpoint health (check CloudWatch metrics)
    monitor_endpoint_health(ENDPOINT_NAME)
```

**Translation:** "Check how the endpoint is performing (health check)."

---

### **Part C: Create Sample Logs for Testing**
```python
    # Sample log entries for monitoring test
    sample_logs = [
        # Normal access log
        '192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] "GET /index.html HTTP/1.1" 200 1234',
        
        # Anomalous access log (500 error)
        '192.168.1.2 - - [01/Jan/2024:12:01:00 +0000] "GET /api/data HTTP/1.1" 500 0',
        
        # Anomalous error log
        '2024/01/01 12:02:00 [error] [client 192.168.1.3] upstream server temporarily disabled',
        
        # Normal access log
        '192.168.1.4 - - [01/Jan/2024:12:03:00 +0000] "POST /login HTTP/1.1" 200 567',
        
        # Anomalous critical error log
        '2024/01/01 12:04:00 [crit] [client 192.168.1.5] no live upstreams while connecting'
    ]
```

**Translation:** "Create sample logs to test the endpoint with. Includes both normal and anomalous examples."

**Why test with samples?**
- Verify endpoint still works
- Check model accuracy hasn't degraded
- Ensure no silent failures

---

### **Part D: Run Batch Inference**
```python
    # Run batch inference (test predictions)
    results = run_inference_batch(ENDPOINT_NAME, sample_logs)
```

**Translation:** "Send sample logs to endpoint and get predictions."

---

### **Part E: Save Results**
```python
    # Save monitoring results
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    monitoring_data = save_monitoring_results(
        results, 
        BUCKET, 
        f"{MONITORING_PREFIX}monitoring_{timestamp}.json"
    )
```

**Translation:** "Save results with timestamp in filename (e.g., `monitoring_20240120_140000.json`)."

**Filename pattern:** `monitoring_YYYYMMDD_HHMMSS.json`

---

### **Part F: Print Summary**
```python
    print(f"[INFO] Monitoring complete. Found {monitoring_data['anomaly_count']} anomalies out of {monitoring_data['total_predictions']} predictions")
```

**Translation:** "Print final summary of what was found."

**Example output:**
```
[INFO] Monitoring complete. Found 3 anomalies out of 5 predictions
```

---

## **Key Concepts Simplified:**

### **1. What is Monitoring?**
Think of it as your **model's annual physical check-up**:
- **Heart rate** = Invocations (how active is it?)
- **Blood pressure** = ModelLatency (is it stressed/slow?)
- **Temperature** = Errors (is it sick/broken?)
- **Reflex test** = Batch inference (does it still work correctly?)

### **2. CloudWatch Explained**
AWS's built-in **monitoring dashboard** that automatically tracks:
- How many requests your endpoint gets
- How fast it responds
- How many errors occur
- All stored for 15 months (by default)

### **3. Two Types of Monitoring:**

**A. Passive Monitoring (Lines 32-51)**
- Looks at existing metrics
- Checks: "How has it been performing?"
- No impact on endpoint (just reads data)

**B. Active Monitoring (Lines 53-86)**
- Sends test requests
- Checks: "Is it working right now?"
- Small impact (adds a few requests)

### **4. When to Run This Script:**

**Option 1: Manually** (as needed)
- When you want to check on things
- Before important events
- After making changes

**Option 2: Automatically** (cron job)
```
# Run every hour
0 * * * * python 06_monitor_model.py

# Run every day at 2 AM
0 2 * * * python 06_monitor_model.py
```

### **5. What to Look For in Results:**

**Healthy:**
- Invocations > 0 (people are using it)
- ModelLatency < 200ms (fast responses)
- 4XXErrors < 1% of Invocations (few bad requests)
- 5XXErrors = 0 (no server errors)

**Warning Signs:**
- Invocations = 0 for hours (no one using it?)
- ModelLatency > 1000ms (too slow)
- 5XXErrors > 0 (model crashing)
- Anomaly rate very different from expected

### **6. Storage of Results:**
```
S3 Bucket:
└── your-folder-name/
    └── monitoring/
        ├── monitoring_20240120_090000.json  # 9 AM check
        ├── monitoring_20240120_120000.json  # 12 PM check
        ├── monitoring_20240120_150000.json  # 3 PM check
        └── monitoring_20240120_180000.json  # 6 PM check
```

### **7. Cost Considerations:**
- **CloudWatch**: First 10 metrics free, then ~$0.30/metric/month
- **S3 storage**: ~$0.023/GB/month (very cheap for JSON files)
- **Inference calls**: ~$0.000004 per prediction (negligible for monitoring)

---

## **Visual Walkthrough:**

### **Step 1: Health Check**
```
You → CloudWatch: "How's nginx-anomaly-detector-prod doing?"
CloudWatch → You: 
    "In the last hour:
    - 1,250 predictions made
    - Average response: 85ms  
    - 2 client errors (typos in requests)
    - 0 server errors 👍"
```

### **Step 2: Active Test**
```
You → Endpoint: "Here are 5 sample logs, what do you think?"
Endpoint → You:
    1. Normal log → LABEL_0 (98.7% confidence)
    2. 500 error → LABEL_1 (95.4% confidence) 
    3. [error] log → LABEL_1 (89.2% confidence)
    4. Normal log → LABEL_0 (99.1% confidence)
    5. [crit] log → LABEL_1 (97.8% confidence)
```

### **Step 3: Save Results**
```
Saved to: s3://my-bucket/my-folder/monitoring/monitoring_20240120_140000.json

Report summary:
- Time: 2024-01-20 14:00:00
- Tests: 5 predictions
- Found: 3 anomalies (60%)
- All responses successful
```

### **Step 4: Alert if Problems**
**(Manual step - you check the output)**
If you see:
- 5XXErrors > 0 → "Endpoint is broken!"
- ModelLatency > 1000ms → "Endpoint is too slow!"
- 0 Invocations for 24h → "No one is using this!"
- Wrong predictions → "Model accuracy degraded!"

---

## **Common Issues Detected:**

### **1. High Latency (> 500ms)**
- **Cause**: Instance too small, model too complex
- **Fix**: Upgrade to larger instance (ml.m5.2xlarge)

### **2. 5XX Errors**
- **Cause**: Model crashed, out of memory
- **Fix**: Check logs, restart endpoint, increase memory

### **3. 0 Invocations**
- **Cause**: No traffic, endpoint down, DNS issues
- **Fix**: Check if endpoint is running, verify integration

### **4. Wrong Predictions**
- **Cause**: Data drift, model degraded
- **Fix**: Retrain model with new data

### **5. High 4XX Errors**
- **Cause**: Clients sending malformed requests
- **Fix**: Update client code, add request validation

---

## **Why This Step is Critical:**

1. **Proactive Maintenance**: Catch issues before users complain
2. **Performance Tracking**: Know if you need to scale up/down
3. **Cost Optimization**: Shut down unused endpoints
4. **Quality Assurance**: Ensure model still works correctly
5. **Audit Trail**: Historical record of model performance
6. **Business Insights**: How often are anomalies detected?

**This is like having a security guard, doctor, and accountant all checking on your model 24/7 to make sure it's healthy, secure, and cost-effective!**