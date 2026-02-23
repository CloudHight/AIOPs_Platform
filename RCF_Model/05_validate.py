import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def validate_model(data_file="cpu_time_series_with_anomalies.csv",
                   scores_file="rcf_scores.csv",
                   threshold_percentile=95):
    """Validate model performance using ground truth labels."""
    print("========== Model Validation ==========")
    
    df = pd.read_csv(data_file)
    scores_df = pd.read_csv(scores_file)
    rcf_scores = scores_df['RCFScore'].values
    
    # Use percentile-based threshold for anomaly detection
    threshold = np.percentile(rcf_scores, threshold_percentile)
    predictions = (rcf_scores > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(df['anomaly'], predictions)
    precision = precision_score(df['anomaly'], predictions, zero_division=0)
    recall = recall_score(df['anomaly'], predictions, zero_division=0)
    f1 = f1_score(df['anomaly'], predictions, zero_division=0)
    
    print(f"Validation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Threshold used: {threshold:.4f}")
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'threshold': threshold
    }
    
    # Save validation results
    pd.DataFrame([results]).to_csv("validation_results.csv", index=False)
    return results

if __name__ == "__main__":
    validate_model()