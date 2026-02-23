import boto3
from sagemaker.huggingface import HuggingFaceModel
from sagemaker import get_execution_role

def deploy_production_model(model_artifacts_uri, endpoint_name, role=None):
    role = role or get_execution_role()
    
    huggingface_model = HuggingFaceModel(
        model_data=model_artifacts_uri,
        transformers_version='4.26',
        pytorch_version='1.13',
        py_version='py39',
        role=role
    )
    
    print(f"[INFO] Deploying production model to endpoint: {endpoint_name}")
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.xlarge",
        endpoint_name=endpoint_name
    )
    
    print(f"[INFO] Model deployed successfully to endpoint: {endpoint_name}")
    return predictor

def test_production_endpoint(endpoint_name, sample_texts):
    import boto3
    runtime = boto3.client('sagemaker-runtime')
    
    print("[INFO] Testing production endpoint...")
    for i, text in enumerate(sample_texts):
        try:
            response = runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps({"inputs": text})
            )
            result = json.loads(response['Body'].read().decode())
            print(f"  Sample {i+1}: {result[0]['label']} (confidence: {result[0]['score']:.3f})")
        except Exception as e:
            print(f"  Sample {i+1}: Error - {str(e)}")

if __name__ == "__main__":
    import json
    
    MODEL_ARTIFACTS_URI = "s3://YOUR_BUCKET_NAME/path/to/model/artifacts"  # From testing step
    ENDPOINT_NAME = "nginx-anomaly-detector-prod"
    
    # Sample test texts
    sample_texts = [
        '192.168.1.1 - - [01/Jan/2024:12:00:00 +0000] "GET /index.html HTTP/1.1" 200 1234 "-" "Mozilla/5.0"',
        '192.168.1.2 - - [01/Jan/2024:12:01:00 +0000] "GET /api/data HTTP/1.1" 500 0 "-" "curl/7.68.0"',
        '2024/01/01 12:02:00 [error] [client 192.168.1.3] upstream server temporarily disabled'
    ]
    
    predictor = deploy_production_model(MODEL_ARTIFACTS_URI, ENDPOINT_NAME)
    test_production_endpoint(ENDPOINT_NAME, sample_texts)
    
    print(f"[INFO] Production deployment complete. Endpoint: {ENDPOINT_NAME}")