import pandas as pd
import numpy as np

def analyze_anomalies(data_file="cpu_time_series_with_anomalies.csv",
                      scores_file="rcf_scores.csv",
                      threshold=70):
    """Analyze and report anomaly detection results."""
    print("========== Anomaly Analysis ==========")
    
    df = pd.read_csv(data_file)
    scores_df = pd.read_csv(scores_file)
    
    df['RCFScore'] = scores_df['RCFScore']
    df['AnomalyScore'] = (df['Average'] > threshold).astype(int)
    
    # Report statistics
    anomalies = df[df['AnomalyScore'] == 1]
    print(f"Detected anomalies (Average > {threshold}): {len(anomalies)}")
    
    if len(anomalies) > 0:
        print("\nTop anomalies:")
        print(anomalies[['Timestamp', 'Average', 'RCFScore']].head())
    
    # Statistical analysis
    rcf_threshold = df['RCFScore'].mean() + 3 * df['RCFScore'].std()
    stat_anomalies = df[df['RCFScore'] > rcf_threshold]
    print(f"\nStatistical anomalies (RCF > {rcf_threshold:.4f}): {len(stat_anomalies)}")
    
    # Save analyzed data
    df.to_csv("analyzed_results.csv", index=False)
    return df

def monitor_performance(df):
    """Monitor model performance metrics."""
    print("========== Performance Monitoring ==========")
    
    print(f"Dataset size: {len(df)}")
    print(f"Average CPU usage: {df['Average'].mean():.2f}")
    print(f"CPU usage std: {df['Average'].std():.2f}")
    print(f"RCF score range: {df['RCFScore'].min():.4f} - {df['RCFScore'].max():.4f}")
    
    # Basic drift detection
    recent_data = df.tail(len(df)//4)  # Last 25% of data
    historical_mean = df.head(len(df)//2)['Average'].mean()
    recent_mean = recent_data['Average'].mean()
    
    drift = abs(recent_mean - historical_mean) / historical_mean
    print(f"Data drift indicator: {drift:.4f}")
    
    if drift > 0.1:
        print("WARNING: Significant data drift detected")

if __name__ == "__main__":
    df_analyzed = analyze_anomalies()
    monitor_performance(df_analyzed)