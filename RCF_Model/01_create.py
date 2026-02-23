import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_dataset(output_file="cpu_time_series_with_anomalies.csv"):
    """Create synthetic time series data with anomalies."""
    start_time = datetime(2025, 8, 19, 0, 0)
    num_minutes = 7 * 24 * 60  # 7 days
    anomaly_fraction = 0.01  # roughly 1% anomalous
    
    timestamps = [start_time + timedelta(minutes=i) for i in range(num_minutes)]
    
    # Generate normal CPU usage (Gaussian around 45 with std 10)
    cpu_usage = np.random.normal(loc=45, scale=10, size=num_minutes)
    cpu_usage = np.clip(cpu_usage, 0, 70)
    
    # Insert anomalies
    num_anomalies = int(anomaly_fraction * num_minutes)
    anomaly_indices = random.sample(range(num_minutes), num_anomalies)
    
    for idx in anomaly_indices:
        cpu_usage[idx] = np.random.uniform(70, 100)
    
    # Add anomaly labels
    labels = np.zeros(num_minutes, dtype=int)
    labels[anomaly_indices] = 1
    
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Average": cpu_usage,
        "anomaly": labels
    })
    
    df.to_csv(output_file, index=False)
    print(f"Dataset created with shape: {df.shape}")
    return df

if __name__ == "__main__":
    create_dataset()