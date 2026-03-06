import boto3
import json
import random
import datetime
import re
from collections import Counter
from sklearn.model_selection import train_test_split

def generate_access_log_entry(ip, timestamp, status_code):
    return f'{ip} - - [{timestamp}] "GET /index.html HTTP/1.1" {status_code} {random.randint(200, 5000)} "-" "Mozilla/5.0"'

def generate_error_log_entry(timestamp, level, message):
    return f'{timestamp} [{level}] [client {random_ip()}] {message}'

def random_ip():
    return '.'.join(str(random.randint(1, 255)) for _ in range(4))

def random_timestamp():
    now = datetime.datetime.now()
    delta = datetime.timedelta(seconds=random.randint(0, 86400))
    ts = now - delta
    return ts.strftime('%d/%b/%Y:%H:%M:%S +0000')

def generate_dataset(num_normal=1000, num_anomaly=100):
    access_logs = []
    error_logs = []

    # Normal access logs
    for _ in range(num_normal):
        ip = random_ip()
        ts = random_timestamp()
        status = random.choices([200, 301, 404], weights=[0.8, 0.1, 0.1])[0]
        access_logs.append(generate_access_log_entry(ip, ts, status))

    # Anomalous access logs
    for _ in range(num_anomaly):
        ip = random_ip()
        ts = random_timestamp()
        status = random.choice([500, 502, 503, 504])
        access_logs.append(generate_access_log_entry(ip, ts, status))

    # Normal error logs
    for _ in range(num_normal):
        ts = random_timestamp()
        level = random.choice(['notice', 'info', 'warn'])
        msg = random.choice([
            'client sent HTTP/1.0 request without Host header',
            'connection closed while reading response',
            'request timed out'
        ])
        error_logs.append(generate_error_log_entry(ts, level, msg))

    # Anomalous error logs
    for _ in range(num_anomaly):
        ts = random_timestamp()
        level = random.choice(['error', 'crit'])
        msg = random.choice([
            'upstream server temporarily disabled while connecting to upstream',
            'connect() failed (111: Connection refused) while connecting to upstream',
            'no live upstreams while connecting to upstream',
            'upstream timed out (110: Connection timed out) while reading response header from upstream'
        ])
        error_logs.append(generate_error_log_entry(ts, level, msg))

    return access_logs, error_logs

def is_anomaly(line):
    match = re.search(r'"\s*(\d{3})\s', line)
    if match:
        status = int(match.group(1))
        if 500 <= status < 600:
            return 1
    error_keywords = ['[error]', '[crit]', 'upstream', 'no live upstreams', 'connect() failed', 'timed out']
    if any(kw in line for kw in error_keywords):
        return 1
    return 0

def parse_nginx_log_line(line):
    anomaly_keywords = [
        "500", "502", "503", "504", "[error]", "[crit]",
        "upstream", "no live upstreams", "connect() failed", "request timed out"
    ]
    label = 1 if any(kw in line for kw in anomaly_keywords) else 0
    return {"text": line.strip(), "label": label}

def create_and_split_data(bucket, processed_prefix):
    # Generate synthetic data
    access_logs, error_logs = generate_dataset(num_normal=1000, num_anomaly=100)
    all_logs = access_logs + error_logs
    
    # Parse and label data
    data = [parse_nginx_log_line(line) for line in all_logs if line.strip()]
    print(f"[INFO] Created {len(data)} log entries")
    
    label_counts = Counter(x['label'] for x in data)
    print(f"[INFO] Label distribution: {label_counts}")
    
    # Stratified split
    can_stratify = all(v > 1 for v in label_counts.values()) and len(label_counts) > 1
    if can_stratify:
        train, temp = train_test_split(
            data, test_size=0.2, stratify=[x['label'] for x in data], random_state=42
        )
        val, test = train_test_split(
            temp, test_size=0.5, stratify=[x['label'] for x in temp], random_state=42
        )
        print("[INFO] Successfully stratified splits!")
    else:
        train, temp = train_test_split(data, test_size=0.2, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    # Upload to S3
    s3 = boto3.client('s3')
    for split_name, split_data in [('train', train), ('validation', val), ('test', test)]:
        key = f"{processed_prefix}{split_name}/{split_name}.json"
        s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(split_data))
        print(f"[INFO] Uploaded {len(split_data)} items to {key}")
    
    return len(train), len(val), len(test)

if __name__ == "__main__":
    BUCKET = "YOUR_BUCKET_NAME"
    PROCESSED_PREFIX = "YOUR_FOLDER_NAME/"
    
    train_size, val_size, test_size = create_and_split_data(BUCKET, PROCESSED_PREFIX)
    print(f"[INFO] Data creation complete: Train={train_size}, Val={val_size}, Test={test_size}")