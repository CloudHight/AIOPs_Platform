# **01_create_data.py - Complete Line-by-Line Explanation**

## **Overview**
This script **creates synthetic nginx log data** for training your anomaly detection model. Since real log data might be sensitive or unavailable, it generates fake but realistic-looking logs and labels them automatically.

---

## **Section 1: Import Statements (Lines 1-9)**

```python
import boto3          # AWS SDK for uploading to S3 (cloud storage)
import json          # For working with JSON format data
import random        # For generating random values
import datetime      # For working with dates and times
import re            # Regular expressions for pattern matching
from collections import Counter  # For counting label frequencies
from sklearn.model_selection import train_test_split  # For splitting data
```

**Translation:** "Get the tools we need: AWS tools for cloud storage, data manipulation tools, and a tool to split our data properly."

---

## **Section 2: Generate Access Log Entry (Lines 11-12)**

```python
def generate_access_log_entry(ip, timestamp, status_code):
    return f'{ip} - - [{timestamp}] "GET /index.html HTTP/1.1" {status_code} {random.randint(200, 5000)} "-" "Mozilla/5.0"'
```

**Translation:** "Create a function that makes a fake nginx **access log line** (like `192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] "GET /index.html HTTP/1.1" 200 1234 "-" "Mozilla/5.0"`)."

**Parts of an access log:**
- `ip`: Client IP address
- `timestamp`: When it happened
- `status_code`: HTTP status (200=OK, 404=Not Found, 500=Server Error)
- Random number: Size of response in bytes
- `"Mozilla/5.0"`: Fake user agent

---

## **Section 3: Generate Error Log Entry (Lines 14-15)**

```python
def generate_error_log_entry(timestamp, level, message):
    return f'{timestamp} [{level}] [client {random_ip()}] {message}'
```

**Translation:** "Create a function that makes a fake nginx **error log line** (like `2024/01/01 12:02:00 [error] [client 192.168.1.3] upstream server temporarily disabled`)."

**Parts of an error log:**
- `timestamp`: When error occurred
- `level`: Error severity (info, warn, error, crit)
- `message`: What went wrong

---

## **Section 4: Generate Random IP (Lines 17-18)**

```python
def random_ip():
    return '.'.join(str(random.randint(1, 255)) for _ in range(4))
```

**Translation:** "Make a random IP address like `192.168.1.100` by joining 4 random numbers between 1-255 with dots."

---

## **Section 5: Generate Random Timestamp (Lines 20-23)**

```python
def random_timestamp():
    now = datetime.datetime.now()  # Current time
    delta = datetime.timedelta(seconds=random.randint(0, 86400))  # Random offset up to 24 hours
    ts = now - delta  # Random time in last 24 hours
    return ts.strftime('%d/%b/%Y:%H:%M:%S +0000')  # Format as nginx timestamp
```

**Translation:** "Create a random timestamp from the last 24 hours in nginx format (like `01/Jan/2024:12:00:00 +0000`)."

---

## **Section 6: Main Dataset Generator (Lines 25-70)**

### **Part A: Setup**
```python
def generate_dataset(num_normal=1000, num_anomaly=100):
    access_logs = []   # Store access logs
    error_logs = []    # Store error logs
```

**Translation:** "Start the main data generation function. We'll make 1000 normal entries and 100 anomalies by default."

---

### **Part B: Normal Access Logs (Lines 29-36)**
```python
    # Normal access logs
    for _ in range(num_normal):
        ip = random_ip()  # Random IP
        ts = random_timestamp()  # Random time
        # Choose status code: 80% 200 (OK), 10% 301 (Redirect), 10% 404 (Not Found)
        status = random.choices([200, 301, 404], weights=[0.8, 0.1, 0.1])[0]
        access_logs.append(generate_access_log_entry(ip, ts, status))
```

**Translation:** "Create normal web server logs - mostly successful requests (200), some redirects (301), some not found (404)."

---

### **Part C: Anomalous Access Logs (Lines 38-43)**
```python
    # Anomalous access logs
    for _ in range(num_anomaly):
        ip = random_ip()
        ts = random_timestamp()
        status = random.choice([500, 502, 503, 504])  # Server errors = anomalies
        access_logs.append(generate_access_log_entry(ip, ts, status))
```

**Translation:** "Create anomalous access logs with server error codes (500, 502, 503, 504). These will be labeled as 'anomalies'."

---

### **Part D: Normal Error Logs (Lines 45-53)**
```python
    # Normal error logs
    for _ in range(num_normal):
        ts = random_timestamp()
        level = random.choice(['notice', 'info', 'warn'])  # Low severity
        msg = random.choice([
            'client sent HTTP/1.0 request without Host header',
            'connection closed while reading response',
            'request timed out'
        ])  # Minor, non-critical errors
        error_logs.append(generate_error_log_entry(ts, level, msg))
```

**Translation:** "Create normal error logs - minor warnings that aren't serious problems."

---

### **Part E: Anomalous Error Logs (Lines 55-64)**
```python
    # Anomalous error logs
    for _ in range(num_anomaly):
        ts = random_timestamp()
        level = random.choice(['error', 'crit'])  # High severity
        msg = random.choice([
            'upstream server temporarily disabled while connecting to upstream',
            'connect() failed (111: Connection refused) while connecting to upstream',
            'no live upstreams while connecting to upstream',
            'upstream timed out (110: Connection timed out) while reading response header from upstream'
        ])  # Serious server problems
        error_logs.append(generate_error_log_entry(ts, level, msg))
```

**Translation:** "Create serious error logs with 'error' or 'crit' level and server failure messages. These will be labeled as 'anomalies'."

---

### **Part F: Return Results (Lines 66-70)**
```python
    return access_logs, error_logs
```

**Translation:** "Return both lists of logs."

---

## **Section 7: Anomaly Detection Function (Lines 72-82)**

```python
def is_anomaly(line):
    # Look for HTTP status codes in the log line
    match = re.search(r'"\s*(\d{3})\s', line)
    if match:
        status = int(match.group(1))
        if 500 <= status < 600:  # Server errors (500-599)
            return 1  # Anomaly
    
    # Look for error keywords
    error_keywords = ['[error]', '[crit]', 'upstream', 'no live upstreams', 'connect() failed', 'timed out']
    if any(kw in line for kw in error_keywords):
        return 1  # Anomaly
    
    return 0  # Normal
```

**Translation:** "Check if a log line is anomalous by looking for:
1. HTTP status codes 500-599 (server errors)
2. Keywords like '[error]', '[crit]', or server failure messages
Return 1 for anomaly, 0 for normal."

---

## **Section 8: Parse Log Line (Lines 84-90)**

```python
def parse_nginx_log_line(line):
    anomaly_keywords = [
        "500", "502", "503", "504", "[error]", "[crit]",
        "upstream", "no live upstreams", "connect() failed", "request timed out"
    ]
    label = 1 if any(kw in line for kw in anomaly_keywords) else 0
    return {"text": line.strip(), "label": label}
```

**Translation:** "Convert a raw log line into a structured object with:
- `text`: The log message itself
- `label`: 1 if anomalous, 0 if normal (based on keywords)"

---

## **Section 9: Create and Split Data (Lines 92-126)**

### **Part A: Generate Data**
```python
def create_and_split_data(bucket, processed_prefix):
    # Generate synthetic data
    access_logs, error_logs = generate_dataset(num_normal=1000, num_anomaly=100)
    all_logs = access_logs + error_logs  # Combine both types
    
    # Parse and label data
    data = [parse_nginx_log_line(line) for line in all_logs if line.strip()]
    print(f"[INFO] Created {len(data)} log entries")
```

**Translation:** "Generate fake logs, then parse each line into the format needed for training."

---

### **Part B: Check Label Distribution**
```python
    label_counts = Counter(x['label'] for x in data)  # Count how many 0s and 1s
    print(f"[INFO] Label distribution: {label_counts}")
    
    # Check if we can do stratified splitting
    can_stratify = all(v > 1 for v in label_counts.values()) and len(label_counts) > 1
```

**Translation:** "Count how many normal (0) vs anomalous (1) examples we have. Check if we have enough of both types to split properly."

---

### **Part C: Split Data (Lines 106-117)**
```python
    # Stratified split (preserves label distribution in each split)
    if can_stratify:
        train, temp = train_test_split(
            data, test_size=0.2, stratify=[x['label'] for x in data], random_state=42
        )
        val, test = train_test_split(
            temp, test_size=0.5, stratify=[x['label'] for x in temp], random_state=42
        )
        print("[INFO] Successfully stratified splits!")
    else:
        # Fallback: random split if not enough samples
        train, temp = train_test_split(data, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
```

**Translation:** "Split data into 3 parts:
- **Train (80%)**: For teaching the model
- **Validation (10%)**: For tuning during training  
- **Test (10%)**: For final evaluation

Stratified means each split has same proportion of normal/anomalous logs."

**Visual:**
```
All Data (100%)
├── Train (80%) - Model learns from this
├── Validation (10%) - Model tunes itself on this
└── Test (10%) - Final exam for model
```

---

### **Part D: Upload to S3 (Lines 119-126)**
```python
    # Upload to S3 (AWS cloud storage)
    s3 = boto3.client('s3')
    for split_name, split_data in [('train', train), ('validation', val), ('test', test)]:
        key = f"{processed_prefix}{split_name}/{split_name}.json"
        s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(split_data))
        print(f"[INFO] Uploaded {len(split_data)} items to {key}")
    
    return len(train), len(val), len(test)
```

**Translation:** "Save each dataset to AWS S3 cloud storage so SageMaker can access it for training."

---

## **Section 10: Main Execution (Lines 128-133)**

```python
if __name__ == "__main__":
    BUCKET = "YOUR_BUCKET_NAME"  # Your S3 bucket name
    PROCESSED_PREFIX = "YOUR_FOLDER_NAME/"  # Folder path in bucket
    
    train_size, val_size, test_size = create_and_split_data(BUCKET, PROCESSED_PREFIX)
    print(f"[INFO] Data creation complete: Train={train_size}, Val={val_size}, Test={test_size}")
```

**Translation:** "Run the script when executed directly. Update the bucket name, then create and upload the data."

---

## **Key Concepts Simplified:**

### **1. Why Synthetic Data?**
- Real logs might contain sensitive info
- Ensures we have balanced dataset
- Can control anomaly ratio

### **2. Two Types of Logs:**
- **Access Logs**: `192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] "GET /index.html HTTP/1.1" 200 1234`
- **Error Logs**: `2024/01/01 12:02:00 [error] [client 192.168.1.3] upstream server temporarily disabled`

### **3. Labeling Rules:**
- **Normal (0)**: Status 200-404, 'info', 'warn' levels
- **Anomaly (1)**: Status 500+, 'error', 'crit' levels, server failure keywords

### **4. Data Splits:**
```
2200 total logs
├── 1760 training logs (80%)
├── 220 validation logs (10%) 
└── 220 test logs (10%)
```

### **5. End Result:**
Three JSON files in S3:
- `s3://your-bucket/your-folder/train/train.json`
- `s3://your-bucket/your-folder/validation/validation.json`  
- `s3://your-bucket/your-folder/test/test.json`

Each contains list of objects like:
```json
[
  {"text": "192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] GET...", "label": 0},
  {"text": "2024/01/01 12:02:00 [error] upstream...", "label": 1}
]
```

**This script creates the training data that `train.py` will use to teach the model what normal vs anomalous logs look like!**