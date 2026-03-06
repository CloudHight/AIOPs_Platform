# **06_monitor.py - Line-by-Line Explanation**

This script **monitors and analyzes** the anomaly detection results to provide insights, detect data drift, and generate summary reports.

---

## **1. Import Libraries**
```python
import pandas as pd
import numpy as np
```
- **pandas**: For reading/writing CSV files and data manipulation
- **numpy**: For mathematical operations and statistical calculations

---

## **2. Function Definition - analyze_anomalies()**
```python
def analyze_anomalies(data_file="cpu_time_series_with_anomalies.csv",
                      scores_file="rcf_scores.csv",
                      threshold=70):
```
Parameters:
- `data_file`: Original dataset with CPU usage data
- `scores_file`: Anomaly scores from model (from `04_test.py`)
- `threshold`: CPU usage threshold for simple rule-based detection (default: 70%)

---

## **3. Print Header and Load Data**
```python
    """Analyze and report anomaly detection results."""
    print("========== Anomaly Analysis ==========")
    
    df = pd.read_csv(data_file)
    scores_df = pd.read_csv(scores_file)
```
1. Function documentation string
2. Visual separator for logs
3. Load both datasets into DataFrames

---

## **4. Combine Data with RCF Scores**
```python
    df['RCFScore'] = scores_df['RCFScore']
```
- Adds RCF anomaly scores as a new column to the original dataframe
- **Result**: `df` now has columns: `Timestamp`, `Average`, `anomaly`, `RCFScore`

---

## **5. Create Rule-Based Anomaly Detection**
```python
    df['AnomalyScore'] = (df['Average'] > threshold).astype(int)
```
- **Simple rule**: CPU usage > 70% = anomaly (1), else normal (0)
- Creates a binary column based on the threshold
- **Purpose**: Compare ML model (RCF) against simple baseline rule

---

## **6. Report Statistics for Rule-Based Detection**
```python
    # Report statistics
    anomalies = df[df['AnomalyScore'] == 1]
    print(f"Detected anomalies (Average > {threshold}): {len(anomalies)}")
```
1. Filters rows where simple rule detected anomalies
2. Prints count of rule-based anomalies
3. **Example**: "Detected anomalies (Average > 70): 150"

---

## **7. Display Top Anomalies**
```python
    if len(anomalies) > 0:
        print("\nTop anomalies:")
        print(anomalies[['Timestamp', 'Average', 'RCFScore']].head())
```
- If anomalies were found, show first 5 with:
  - Timestamp
  - CPU usage (Average)
  - RCF anomaly score
- **Purpose**: Quick visual inspection of worst anomalies

---

## **8. Statistical Anomaly Detection**
```python
    # Statistical analysis
    rcf_threshold = df['RCFScore'].mean() + 3 * df['RCFScore'].std()
    stat_anomalies = df[df['RCFScore'] > rcf_threshold]
    print(f"\nStatistical anomalies (RCF > {rcf_threshold:.4f}): {len(stat_anomalies)}")
```
**Statistical Method**: Uses "3-sigma rule" (standard deviation method):
- `df['RCFScore'].mean()`: Average RCF score
- `df['RCFScore'].std()`: Standard deviation of RCF scores
- `rcf_threshold`: Mean + 3 × Standard Deviation
- **Why 3?**: In normal distribution, 99.7% of data within mean ± 3σ
- Anything above threshold = statistically significant anomaly

---

## **9. Save Analyzed Data**
```python
    # Save analyzed data
    df.to_csv("analyzed_results.csv", index=False)
    return df
```
1. Saves enriched dataframe with all columns to CSV
2. Returns dataframe for use in next function

---

## **10. Function Definition - monitor_performance()**
```python
def monitor_performance(df):
```
- Takes the analyzed dataframe as input
- Monitors key performance metrics and detects data drift

---

## **11. Print Header and Basic Stats**
```python
    """Monitor model performance metrics."""
    print("========== Performance Monitoring ==========")
    
    print(f"Dataset size: {len(df)}")
    print(f"Average CPU usage: {df['Average'].mean():.2f}")
    print(f"CPU usage std: {df['Average'].std():.2f}")
    print(f"RCF score range: {df['RCFScore'].min():.4f} - {df['RCFScore'].max():.4f}")
```
**Reports**:
1. Total data points
2. Mean CPU usage (e.g., "45.23%")
3. CPU usage standard deviation (variability)
4. Range of RCF scores (min to max)

---

## **12. Data Drift Detection**
```python
    # Basic drift detection
    recent_data = df.tail(len(df)//4)  # Last 25% of data
    historical_mean = df.head(len(df)//2)['Average'].mean()
    recent_mean = recent_data['Average'].mean()
    
    drift = abs(recent_mean - historical_mean) / historical_mean
    print(f"Data drift indicator: {drift:.4f}")
```

## **DRIFT DETECTION EXPLAINED:**

### **What is Data Drift?**
When new data patterns differ from what the model was trained on.

### **Method Used:**
1. **Historical data**: First 50% of data (training/early period)
2. **Recent data**: Last 25% of data (newest observations)
3. **Compare means**: Calculate percentage change

### **Calculation:**
```
historical_mean = mean of first 50% of CPU usage
recent_mean = mean of last 25% of CPU usage
drift = |recent_mean - historical_mean| / historical_mean
```

### **Example:**
- Historical CPU mean: 45.0%
- Recent CPU mean: 49.5%
- Drift = |49.5 - 45.0| / 45.0 = 0.10 (10% drift)

---

## **13. Drift Alert**
```python
    if drift > 0.1:
        print("WARNING: Significant data drift detected")
```
- **Threshold**: 10% change in mean CPU usage
- **If drift > 0.1**: Prints warning
- **Why important**: Model may need retraining if data patterns change

---

## **14. Main Execution Block**
```python
if __name__ == "__main__":
    df_analyzed = analyze_anomalies()
    monitor_performance(df_analyzed)
```
When script runs directly:
1. First analyze anomalies
2. Then monitor performance on analyzed data

---

## **SUMMARY: What This Script Does**

1. **Combines** original data with model scores
2. **Compares** ML model vs simple rule-based detection
3. **Identifies** statistical anomalies using 3-sigma rule
4. **Monitors** data drift over time
5. **Generates** alerts for significant pattern changes

---

## **KEY CONCEPTS EXPLAINED**

### **Three Types of Anomaly Detection in This Script:**

1. **Rule-Based (Simple)**
   ```python
   df['Average'] > 70  # CPU usage threshold
   ```
   - **Pros**: Simple, interpretable
   - **Cons**: Misses subtle patterns, can't adapt

2. **ML-Based (RCF)**
   ```python
   df['RCFScore']  # Learned patterns
   ```
   - **Pros**: Detects complex patterns, adapts to normal behavior
   - **Cons**: "Black box", needs training

3. **Statistical (3-Sigma Rule)**
   ```python
   mean + 3 * std  # Statistical threshold
   ```
   - **Pros**: Data-driven, objective
   - **Cons**: Assumes normal distribution

### **Data Drift - Why It Matters**

**Scenario**: Model trained in summer (AC usage = high CPU), used in winter (heating = different CPU patterns)

```
Summer training data: CPU ~40-60%
Winter new data: CPU ~30-50%

Result: Model sees "normal" winter patterns as "anomalous"
         ↓
False alarms! (Model degradation)
```

### **Expected Output Example**
```
========== Anomaly Analysis ==========
Detected anomalies (Average > 70): 101

Top anomalies:
Timestamp           Average  RCFScore
2025-08-19 03:15:00  85.7    3.8921
2025-08-19 07:30:00  92.1    4.1567

Statistical anomalies (RCF > 1.8563): 89

========== Performance Monitoring ==========
Dataset size: 10080
Average CPU usage: 45.23
CPU usage std: 12.45
RCF score range: 0.1234 - 4.5678
Data drift indicator: 0.0342  # No warning (3.4% drift)
```

---

## **PRACTICAL INSIGHTS**

### **Comparing Detection Methods**
If results differ significantly:
- **Rule-based finds 101, RCF finds 89**: RCF might be filtering false positives
- **Rule-based finds 101, RCF finds 150**: RCF detecting subtle anomalies rule misses
- **Statistical finds similar to RCF**: Good sign - model aligns with statistics

### **Drift Interpretation**
- **< 0.05 (5%)**: Normal variation
- **0.05-0.10 (5-10%)**: Monitor closely
- **> 0.10 (10%)**: Investigate and consider retraining

### **When to Retrain Model**
1. **High drift (>10%)** + **decreasing accuracy**
2. **Seasonal changes** (quarterly/yearly)
3. **Infrastructure changes** (new servers, software updates)
4. **Business changes** (more users, new features)

---

## **USING THE RESULTS**

The `analyzed_results.csv` file contains everything needed for:
- **Dashboards**: Visualize anomalies over time
- **Alerting**: Set up monitoring alerts
- **Reporting**: Generate weekly/monthly performance reports
- **Root cause analysis**: Investigate specific anomalies

This script turns raw model outputs into **actionable insights** for system monitoring and maintenance!