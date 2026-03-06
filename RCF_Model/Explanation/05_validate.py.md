# **05_validate.py - Line-by-Line Explanation**

This script **evaluates the model's performance** by comparing its anomaly predictions against the ground truth labels (which we created in `01_create.py`). It calculates standard ML metrics to see how well the model detected anomalies.

---

## **1. Import Libraries**
```python
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
```
- **pandas/numpy**: Data manipulation
- **scikit-learn metrics**: Functions to calculate performance metrics
  - `accuracy_score`: Overall correctness
  - `precision_score`: Quality of positive predictions
  - `recall_score`: Ability to find all positives
  - `f1_score`: Balance of precision and recall

---

## **2. Function Definition - validate_model()**
```python
def validate_model(data_file="cpu_time_series_with_anomalies.csv",
                   scores_file="rcf_scores.csv",
                   threshold_percentile=95):
```
Parameters:
- `data_file`: Original dataset with ground truth labels
- `scores_file`: Anomaly scores from model (created by `04_test.py`)
- `threshold_percentile`: What percentile to use as anomaly threshold (default: 95th percentile)

---

## **3. Print Header and Load Data**
```python
    print("========== Model Validation ==========")
    
    df = pd.read_csv(data_file)
    scores_df = pd.read_csv(scores_file)
    rcf_scores = scores_df['RCFScore'].values
```
1. Visual separator for logs
2. Load original dataset (has `Timestamp`, `Average`, and `anomaly` columns)
3. Load anomaly scores from `rcf_scores.csv`
4. Extract scores as numpy array for calculations

---

## **4. Determine Anomaly Threshold**
```python
    # Use percentile-based threshold for anomaly detection
    threshold = np.percentile(rcf_scores, threshold_percentile)
```
**Key Concept**: RCF gives continuous scores. We need a **threshold** to decide "anomaly vs normal".

- `np.percentile(array, 95)`: Finds the score value where 95% of scores are below it
- **Example**: If scores are `[0.1, 0.2, 0.3, 0.4, 2.0]`
  - 95th percentile ≈ 2.0 (top 5% highest scores)
  - Anything above 2.0 = anomaly
- **Why 95%?**: Assumes ~5% of data are anomalies (matches our 1% synthetic + buffer)

---

## **5. Convert Scores to Predictions**
```python
    predictions = (rcf_scores > threshold).astype(int)
```
- **`rcf_scores > threshold`**: Creates boolean array (True/False)
  - Example: `[0.1, 2.1, 0.3, 2.5] > 2.0` → `[False, True, False, True]`
- `.astype(int)`: Converts to 0/1
  - False → 0 (normal)
  - True → 1 (anomaly)
- **Result**: Binary predictions matching ground truth format

---

## **6. Calculate Performance Metrics**
```python
    # Calculate metrics
    accuracy = accuracy_score(df['anomaly'], predictions)
    precision = precision_score(df['anomaly'], predictions, zero_division=0)
    recall = recall_score(df['anomaly'], predictions, zero_division=0)
    f1 = f1_score(df['anomaly'], predictions, zero_division=0)
```

## **METRICS EXPLAINED:**

### **Accuracy**
```python
accuracy_score(truth, predictions)
```
- **Formula**: `(Correct Predictions) / (Total Predictions)`
- **Example**: 95% accuracy = 95% of predictions were correct
- **Limitation**: Can be misleading with imbalanced data (we have 99% normal, 1% anomalies)

### **Precision**
```python
precision_score(truth, predictions, zero_division=0)
```
- **Formula**: `(True Anomalies Found) / (All Predicted Anomalies)`
- **Question**: When model says "anomaly", how often is it right?
- **High precision**: Few false alarms
- `zero_division=0`: If no anomalies predicted, return 0 instead of error

### **Recall**
```python
recall_score(truth, predictions, zero_division=0)
```
- **Formula**: `(True Anomalies Found) / (All Actual Anomalies)`
- **Question**: Of all real anomalies, how many did we find?
- **High recall**: Misses few real anomalies
- Trade-off: Can't have both high precision AND high recall

### **F1-Score**
```python
f1_score(truth, predictions, zero_division=0)
```
- **Formula**: `2 × (Precision × Recall) / (Precision + Recall)`
- **Balance**: Harmonic mean of precision and recall
- **Good for**: Imbalanced datasets (like ours with few anomalies)
- **Best value**: 1.0 (perfect), **Worst**: 0.0

---

## **7. Print Results**
```python
    print(f"Validation Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Threshold used: {threshold:.4f}")
```
Formats metrics to 4 decimal places for easy reading.
Example output:
```
Accuracy: 0.9832
Precision: 0.7500
Recall: 0.8000
F1-Score: 0.7742
Threshold used: 1.8563
```

---

## **8. Save Results to Dictionary**
```python
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
```
1. Creates dictionary with all metrics
2. Saves as CSV (single row with columns for each metric)
3. Returns dictionary for use in other scripts

---

## **9. Main Execution Block**
```python
if __name__ == "__main__":
    validate_model()
```
- When script runs directly, execute `validate_model()`

---

## **SUMMARY: What This Script Does**

1. **Loads** ground truth labels and model scores
2. **Determines** anomaly threshold (95th percentile of scores)
3. **Converts** continuous scores to binary predictions
4. **Calculates** 4 key ML metrics
5. **Saves** results for reporting

---

## **KEY CONCEPTS EXPLAINED**

### **Confusion Matrix (Visualizing Performance)**
Based on our predictions vs ground truth:

|                    | **Predicted Normal** | **Predicted Anomaly** |
|--------------------|---------------------|---------------------|
| **Actual Normal**  | True Negative (TN)  | False Positive (FP) |
| **Actual Anomaly** | False Negative (FN) | True Positive (TP)  |

Our metrics use these counts:
- **Accuracy** = `(TN + TP) / (TN + TP + FN + FP)`
- **Precision** = `TP / (TP + FP)` (How many predicted anomalies were real?)
- **Recall** = `TP / (TP + FN)` (How many real anomalies did we catch?)

### **Threshold Selection Trade-offs**
```python
# Higher threshold (e.g., 98th percentile):
threshold = np.percentile(scores, 98)
# Result: Fewer predictions as anomalies
# → Higher precision, lower recall

# Lower threshold (e.g., 90th percentile):
threshold = np.percentile(scores, 90)
# Result: More predictions as anomalies  
# → Lower precision, higher recall
```

### **Example Scenario**
Our synthetic data has 101 real anomalies. Model predictions:

- **If threshold too high**: Only catch 20 anomalies (low recall)
- **If threshold too low**: Catch 100 anomalies but include 500 false alarms (low precision)
- **Goal**: Find sweet spot (catch most real anomalies with few false alarms)

### **Expected Results with Synthetic Data**
Since we control the data generation:
- **High accuracy** (~98%): Easy to distinguish normal vs anomaly
- **Good F1-score** (~0.7-0.8): Model should perform reasonably well
- **Perfect scores unlikely**: Anomalies near boundary (70% CPU) are hard to detect

---

## **INTERPRETING RESULTS**

### **Good Model (Example)**
```
Accuracy: 0.9832    # 98% of all predictions correct
Precision: 0.7500   # When it says anomaly, 75% chance it's right
Recall: 0.8000      # Catches 80% of real anomalies
F1-Score: 0.7742    # Good balance between precision/recall
```

### **Bad Model (Problems)**
- **Low precision (<0.5)**: Too many false alarms
- **Low recall (<0.5)**: Missing too many real anomalies  
- **F1-score < 0.6**: Model needs improvement

### **Improving the Model**
If results are poor:
1. Adjust `threshold_percentile` parameter
2. Retrain with more data
3. Use different algorithm
4. Add more features (not just CPU average)

This validation tells us **how trustworthy** our anomaly detection system is before using it in production!