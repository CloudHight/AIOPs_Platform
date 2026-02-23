import pandas as pd
import numpy as np
import re
import json
from sagemaker.predictor import Predictor

def extract_score_value(record):
    """Extract float value from SageMaker RCF JSON response."""
    try:
        if isinstance(record, dict):
            if 'score' in record:
                return float(record['score'])
            elif 'scores' in record:
                return float(record['scores'][0]['score'])
        elif isinstance(record, list) and len(record) > 0:
            return extract_score_value(record[0])
        return float(record) if isinstance(record, (int, float)) else 0.0
    except Exception:
        return 0.0

def test_model(data_file="cpu_time_series_with_anomalies.csv",
               endpoint_info_file="endpoint_info.json",
               scores_file="rcf_scores.csv"):
    """Test the deployed model with batch scoring."""
    print("========== Model Testing ==========")
    
    df = pd.read_csv(data_file)
    
    # Load endpoint info and create predictor
    with open(endpoint_info_file, 'r') as f:
        endpoint_info = json.load(f)
    
    predictor = Predictor(endpoint_name=endpoint_info['endpoint_name'])
    
    import io
    from sagemaker.serializers import CSVSerializer
    from sagemaker.deserializers import JSONDeserializer
    
    predictor.serializer = CSVSerializer()
    predictor.deserializer = JSONDeserializer()
    
    values_to_score = df['Average'].astype('float32').values.reshape(-1, 1)
    scores = predictor.predict(values_to_score)
    
    # Debug: print first few scores to understand format
    print(f"Response type: {type(scores)}")
    if hasattr(scores, '__iter__') and not isinstance(scores, str):
        sample = list(scores)[:3] if len(list(scores)) > 0 else []
        print(f"Sample scores: {sample}")
    else:
        print(f"Scores content: {scores}")
    
    # Handle dict response with 'scores' key
    if isinstance(scores, dict) and 'scores' in scores:
        score_list = scores['scores']
    else:
        score_list = scores
    
    rcf_scores = []
    for i, record in enumerate(score_list):
        score_value = extract_score_value(record)
        rcf_scores.append(score_value)
        if i < 3:  # Debug first few
            print(f"Record {i}: {record} -> {score_value}")
    
    # Ensure length matches DataFrame
    if len(rcf_scores) != len(df):
        if len(rcf_scores) > len(df):
            rcf_scores = rcf_scores[:len(df)]
        else:
            rcf_scores.extend([0.0] * (len(df) - len(rcf_scores)))
    
    # Save scores
    scores_df = pd.DataFrame({'RCFScore': rcf_scores})
    scores_df.to_csv(scores_file, index=False)
    
    print(f"Generated {len(rcf_scores)} anomaly scores. Saved to {scores_file}")
    return np.array(rcf_scores, dtype=float)

if __name__ == "__main__":
    test_model()