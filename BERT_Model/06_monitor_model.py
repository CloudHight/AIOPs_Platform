import boto3
import json
import time
from datetime import datetime, timedelta

def get_endpoint_metrics(endpoint_name, start_time, end_time):
    cloudwatch = boto3.client('cloudwatch')
    
    metrics = ['Invocations', 'ModelLatency', 'Invocation4XXErrors', 'Invocation5XXErrors']
    endpoint_metrics = {}
    
    for metric_name in metrics:
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/SageMaker',
            MetricName=metric_name,
            Dimensions=[
                {'Name': 'EndpointName', 'Value': endpoint_name}
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=300,  # 5 minutes
            Statistics=['Sum', 'Average']
        )
        endpoint_metrics[metric_name] = response['Datapoints']
    
    return endpoint_metrics

def monitor_endpoint_health(endpoint_name):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)
    
    print(f"[INFO] Monitoring endpoint: {endpoint_name}")
    print(f"[INFO] Time range: {start_time} to {end_time}")
    
    metrics = get_endpoint_metrics(endpoint_name, start_time, end_time)
    
    for metric_name, datapoints in metrics.items():
        if datapoints:
            latest = max(datapoints, key=lambda x: x['Timestamp'])
            print(f"  {metric_name}: {latest.get('Sum', latest.get('Average', 0))}")
        else:
            print(f"  {metric_name}: No data")

def run_inference_batch(endpoint_name, log_entries):
    runtime = boto3.client('sagemaker-runtime')
    results = []
    
    print(f"[INFO] Running batch inference on {len(log_entries)} entries")
    
    for i, entry in enumerate(log_entries):
        text = entry if isinstance(entry, str) else entry.get('text', str(entry))
        
        try:
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps({"inputs": text})
            )
            result = json.loads(response['Body'].read().decode())
            
            results.append({
                'text': text,
                'prediction': result[0]['label'],
                'confidence': result[0]['score'],
                'timestamp': datetime.utcnow().isoformat()
            })
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(log_entries)} entries")
                
        except Exception as e:
            print(f"  Error processing entry {i}: {str(e)}")
    
    return results

def save_monitoring_results(results, bucket, s3_key):
    monitoring_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'total_predictions': len(results),
        'anomaly_count': sum(1 for r in results if r['prediction'] == 'LABEL_1'),
        'predictions': results
    }
    
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=json.dumps(monitoring_data, indent=2)
    )
    
    print(f"[INFO] Monitoring results saved to s3://{bucket}/{s3_key}")
    return monitoring_data

if __name__ == "__main__":
    ENDPOINT_NAME = "nginx-anomaly-detector-prod"
    BUCKET = "YOUR_BUCKET_NAME"
    MONITORING_PREFIX = "YOUR_FOLDER_NAME/monitoring/"
    
    # Monitor endpoint health
    monitor_endpoint_health(ENDPOINT_NAME)
    
    # Sample log entries for monitoring
    sample_logs = [
        '192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] "GET /index.html HTTP/1.1" 200 1234',
        '192.168.1.2 - - [01/Jan/2024:12:01:00 +0000] "GET /api/data HTTP/1.1" 500 0',
        '2024/01/01 12:02:00 [error] [client 192.168.1.3] upstream server temporarily disabled',
        '192.168.1.4 - - [01/Jan/2024:12:03:00 +0000] "POST /login HTTP/1.1" 200 567',
        '2024/01/01 12:04:00 [crit] [client 192.168.1.5] no live upstreams while connecting'
    ]
    
    # Run batch inference
    results = run_inference_batch(ENDPOINT_NAME, sample_logs)
    
    # Save monitoring results
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    monitoring_data = save_monitoring_results(
        results, 
        BUCKET, 
        f"{MONITORING_PREFIX}monitoring_{timestamp}.json"
    )
    
    print(f"[INFO] Monitoring complete. Found {monitoring_data['anomaly_count']} anomalies out of {monitoring_data['total_predictions']} predictions")