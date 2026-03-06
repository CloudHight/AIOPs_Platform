# **01_create.py - Line-by-Line Explanation**

This script generates **synthetic time-series data** for CPU usage with injected anomalies. Here's what each part does:

---

## **1. Import Libraries**
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
```
- **pandas (pd)**: For creating and saving the dataset as a CSV file
- **numpy (np)**: For generating numerical data and statistical operations
- **datetime/timedelta**: For creating timestamp sequences
- **random**: For randomly selecting anomaly positions

---

## **2. Function Definition**
```python
def create_dataset(output_file="cpu_time_series_with_anomalies.csv"):
```
- Defines a function named `create_dataset`
- Has one optional parameter `output_file` (default: `"cpu_time_series_with_anomalies.csv"`)
- This function will create and save the dataset

---

## **3. Setup Time Parameters**
```python
    start_time = datetime(2025, 8, 19, 0, 0)
    num_minutes = 7 * 24 * 60  # 7 days
    anomaly_fraction = 0.01  # roughly 1% anomalous
```
- `start_time`: Sets the starting timestamp to August 19, 2025, at midnight
- `num_minutes`: Calculates total minutes in 7 days (7 × 24 × 60 = 10,080 minutes)
- `anomaly_fraction`: 1% of data points will be anomalies

---

## **4. Create Timestamps**
```python
    timestamps = [start_time + timedelta(minutes=i) for i in range(num_minutes)]
```
- Creates a list of 10,080 timestamps
- Each timestamp increases by 1 minute from the previous one
- Example: 00:00, 00:01, 00:02, ... up to 7 days later

---

## **5. Generate Normal CPU Usage**
```python
    cpu_usage = np.random.normal(loc=45, scale=10, size=num_minutes)
    cpu_usage = np.clip(cpu_usage, 0, 70)
```
- `np.random.normal()`: Creates normally distributed random numbers
  - `loc=45`: Mean (average) CPU usage is 45%
  - `scale=10`: Standard deviation is 10%
  - `size=num_minutes`: Creates 10,080 values
- `np.clip()`: Ensures all values stay between 0% and 70%
  - This represents "normal" CPU usage range

---

## **6. Insert Anomalies**
```python
    num_anomalies = int(anomaly_fraction * num_minutes)
    anomaly_indices = random.sample(range(num_minutes), num_anomalies)
    
    for idx in anomaly_indices:
        cpu_usage[idx] = np.random.uniform(70, 100)
```
- `num_anomalies`: Calculates 1% of 10,080 ≈ 101 anomalies
- `anomaly_indices`: Randomly selects 101 positions from the 10,080 indices
- For each anomaly position:
  - Replaces the normal CPU value with a random value between 70% and 100%
  - This simulates abnormal CPU spikes

---

## **7. Create Anomaly Labels**
```python
    labels = np.zeros(num_minutes, dtype=int)
    labels[anomaly_indices] = 1
```
- Creates an array of 10,080 zeros (all points initially labeled "normal")
- Sets the anomaly positions to 1 (labeled "anomalous")
- This creates "ground truth" labels for model validation

---

## **8. Create DataFrame**
```python
    df = pd.DataFrame({
        "Timestamp": timestamps,
        "Average": cpu_usage,
        "anomaly": labels
    })
```
- Combines all data into a pandas DataFrame with 3 columns:
  1. **Timestamp**: The datetime for each reading
  2. **Average**: CPU usage percentage (normal values: 0-70%, anomalies: 70-100%)
  3. **anomaly**: 0 = normal, 1 = anomaly (ground truth label)

---

## **9. Save to CSV**
```python
    df.to_csv(output_file, index=False)
    print(f"Dataset created with shape: {df.shape}")
    return df
```
- Saves the DataFrame to a CSV file
- Prints the dataset dimensions (should be 10,080 rows × 3 columns)
- Returns the DataFrame for use in other scripts

---

## **10. Main Execution Block**
```python
if __name__ == "__main__":
    create_dataset()
```
- Standard Python pattern: when script is run directly, execute `create_dataset()`
- This allows the script to be both imported as a module AND run standalone

---

## **SUMMARY: What This Script Creates**

A **synthetic dataset** that looks like this:

| Timestamp           | Average | anomaly |
|---------------------|---------|---------|
| 2025-08-19 00:00:00 | 47.2    | 0       |
| 2025-08-19 00:01:00 | 52.1    | 0       |
| 2025-08-19 00:02:00 | 38.5    | 0       |
| 2025-08-19 00:03:00 | **85.7**| **1**   | ← Anomaly!
| 2025-08-19 00:04:00 | 46.8    | 0       |
| ... (10,080 rows)   | ...     | ...     |

**Purpose**: This synthetic data mimics real server CPU monitoring data, allowing you to test your anomaly detection pipeline without needing real production data.