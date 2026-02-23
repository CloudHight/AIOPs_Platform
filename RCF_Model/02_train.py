import pandas as pd
from sagemaker import RandomCutForest, get_execution_role
import json

def train_model(input_file="cpu_time_series_with_anomalies.csv",
                model_info_file="model_info.json"):
    """Train Random Cut Forest model on the dataset."""
    print("========== Training Random Cut Forest ==========")
    
    df = pd.read_csv(input_file)
    if df.empty or 'Average' not in df:
        print("No valid data for model training. Exiting.")
        return None
    
    train_values = df['Average'].dropna().values.reshape(-1, 1)
    n_points = len(train_values)
    num_trees = max(50, min(n_points, 1000)) if n_points >= 50 else n_points
    
    print(f"Training on {n_points} points with {num_trees} trees...")
    
    rcf = RandomCutForest(
        role=get_execution_role(),
        instance_count=1,
        instance_type='ml.m5.large',
        num_samples_per_tree=200 if n_points > 200 else n_points,
        num_trees=num_trees,
        eval_metrics=["accuracy", "precision_recall_fscore"],
        feature_dim=1
    )
    
    rcf.fit(rcf.record_set(train_values))
    
    # Save model info
    model_info = {
        'training_job_name': rcf.latest_training_job.name
    }
    with open(model_info_file, 'w') as f:
        json.dump(model_info, f)
    
    print(f"Model training complete. Info saved to {model_info_file}")
    return rcf

if __name__ == "__main__":
    train_model()