import boto3
import json
from sagemaker.huggingface import HuggingFaceModel
from sagemaker import get_execution_role

def load_test_data_from_s3(bucket_name, object_key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    data = response['Body'].read().decode()
    test_data = json.loads(data)
    print(f"[INFO] Loaded {len(test_data)} test entries from S3")
    return test_data

def create_model_from_artifacts(model_artifacts_uri, role=None):
    role = role or get_execution_role()
    
    huggingface_model = HuggingFaceModel(
        model_data=model_artifacts_uri,
        transformers_version='4.26',
        pytorch_version='1.13',
        py_version='py39',
        role=role
    )
    
    print("[INFO] Creating temporary endpoint for testing...")
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",
        endpoint_name=f"test-endpoint-{int(time.time())}"
    )
    
    return predictor

def evaluate_model_on_test_set(predictor, test_data):
    correct = 0
    total = len(test_data)
    predictions = []
    
    for item in test_data:
        text = item['text']
        true_label = item['label']
        
        try:
            response = predictor.predict({"inputs": text})
            predicted_label = 1 if response[0]['label'] == 'LABEL_1' else 0
            
            predictions.append({
                'text': text,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'correct': predicted_label == true_label
            })
            
            if predicted_label == true_label:
                correct += 1
                
        except Exception as e:
            print(f"[WARNING] Error predicting: {str(e)[:60]}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"[INFO] Test accuracy: {accuracy:.4f} ({correct}/{total})")
    
    return predictions, accuracy

def save_test_results(predictions, accuracy, bucket, s3_key):
    results = {
        'test_accuracy': accuracy,
        'total_predictions': len(predictions),
        'predictions': predictions
    }
    
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=json.dumps(results, indent=2)
    )
    print(f"[INFO] Test results saved to s3://{bucket}/{s3_key}")

if __name__ == "__main__":
    import time
    
    BUCKET = "YOUR_BUCKET_NAME"
    PROCESSED_PREFIX = "YOUR_FOLDER_NAME/"
    MODEL_ARTIFACTS_URI = "s3://YOUR_BUCKET_NAME/path/to/model/artifacts"  # From validation step
    
    test_data = load_test_data_from_s3(BUCKET, f"{PROCESSED_PREFIX}test/test.json")
    predictor = create_model_from_artifacts(MODEL_ARTIFACTS_URI)
    
    try:
        predictions, accuracy = evaluate_model_on_test_set(predictor, test_data)
        save_test_results(predictions, accuracy, BUCKET, f"{PROCESSED_PREFIX}test_results.json")
        print(f"[INFO] Testing complete. Final accuracy: {accuracy:.4f}")
    finally:
        predictor.delete_endpoint()
        print("[INFO] Test endpoint cleaned up")