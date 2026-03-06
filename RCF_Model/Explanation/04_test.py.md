# **04_test.py - Line-by-Line Explanation**

This script **tests the deployed model** by sending data to the SageMaker endpoint and getting anomaly scores back. It handles the complex JSON response format from SageMaker's Random Cut Forest algorithm.

---

## **1. Import Libraries**
```python
import pandas as pd
import numpy as np
import re
import json
from sagemaker.predictor import Predictor
```
- **pandas/numpy**: Data manipulation
- **re**: Regular expressions (for parsing responses, though not used here)
- **json**: For reading endpoint info file
- **Predictor**: SageMaker class to interact with deployed endpoints

---

## **2. Helper Function - extract_score_value()**
```python
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
```
**Purpose**: SageMaker RCF returns JSON in different formats. This function handles all cases:

1. **Dictionary with 'score' key**: `{"score": 0.75}`
2. **Dictionary with 'scores' list**: `{"scores": [{"score": 0.75}]}`
3. **List of dictionaries**: `[{"score": 0.75}]`
4. **Direct number**: `0.75`

**Why needed?**: Response format varies based on how data is sent. This function ensures we always get a float score.

---

## **3. Main Function - test_model()**
```python
def test_model(data_file="cpu_time_series_with_anomalies.csv",
               endpoint_info_file="endpoint_info.json",
               scores_file="rcf_scores.csv"):
```
Parameters:
- `data_file`: Dataset to score (default: our synthetic data)
- `endpoint_info_file`: JSON with endpoint name (from `03_deploy.py`)
- `scores_file`: Where to save anomaly scores

---

## **4. Print Header and Load Data**
```python
    print("========== Model Testing ==========")
    
    df = pd.read_csv(data_file)
```
- Visual separator for logs
- Loads the dataset into DataFrame

---

## **5. Load Endpoint Info and Create Predictor**
```python
    # Load endpoint info and create predictor
    with open(endpoint_info_file, 'r') as f:
        endpoint_info = json.load(f)
    
    predictor = Predictor(endpoint_name=endpoint_info['endpoint_name'])
```
1. Reads `endpoint_info.json` (contains `{"endpoint_name": "your-endpoint"}`)
2. Creates a `Predictor` object connected to that endpoint
3. **Note**: Doesn't deploy new endpoint - connects to existing one

---

## **6. Configure Serializers**
```python
    import io
    from sagemaker.serializers import CSVSerializer
    from sagemaker.deserializers import JSONDeserializer
    
    predictor.serializer = CSVSerializer()
    predictor.deserializer = JSONDeserializer()
```
**Critical for communication with endpoint**:
- `CSVSerializer()`: Converts our data to CSV format before sending
- `JSONDeserializer()`: Converts endpoint's JSON response to Python dict/list
- **Why?**: Endpoints expect specific data formats. These handle the translation.

---

## **7. Prepare Data for Scoring**
```python
    values_to_score = df['Average'].astype('float32').values.reshape(-1, 1)
```
1. `df['Average']`: Get CPU usage column
2. `.astype('float32')`: Convert to 32-bit float (memory efficient)
3. `.values`: Convert to numpy array
4. `.reshape(-1, 1)`: Make 2D with 1 column
   - Input: `[45, 50, 55]`
   - Output: `[[45], [50], [55]]`
   - Required format for RCF endpoint

---

## **8. Send Data to Endpoint for Scoring**
```python
    scores = predictor.predict(values_to_score)
```
**The actual API call**:
- Sends all 10,080 data points to the endpoint
- Endpoint processes each point, returns anomaly scores
- This is a **batch prediction** (all at once, not one-by-one)
- Takes a few seconds to complete

---

## **9. Debug: Examine Response Format**
```python
    # Debug: print first few scores to understand format
    print(f"Response type: {type(scores)}")
    if hasattr(scores, '__iter__') and not isinstance(scores, str):
        sample = list(scores)[:3] if len(list(scores)) > 0 else []
        print(f"Sample scores: {sample}")
    else:
        print(f"Scores content: {scores}")
```
**Why debug?**: SageMaker responses vary. This helps understand what we received.
Example outputs:
- `Response type: <class 'dict'>`
- `Sample scores: [{'score': 0.75}, {'score': 0.25}, {'score': 1.2}]`

---

## **10. Handle Different Response Formats**
```python
    # Handle dict response with 'scores' key
    if isinstance(scores, dict) and 'scores' in scores:
        score_list = scores['scores']
    else:
        score_list = scores
```
- Case 1: Response is `{"scores": [{"score": 0.75}, ...]}` → Extract the list
- Case 2: Response is already a list `[{"score": 0.75}, ...]` → Use as-is

---

## **11. Extract Scores Using Helper Function**
```python
    rcf_scores = []
    for i, record in enumerate(score_list):
        score_value = extract_score_value(record)
        rcf_scores.append(score_value)
        if i < 3:  # Debug first few
            print(f"Record {i}: {record} -> {score_value}")
```
1. Loop through each response record
2. Use `extract_score_value()` to get numeric score from JSON
3. Store in list
4. Print first 3 for verification
   - Example: `Record 0: {'score': 0.75} -> 0.75`

---

## **12. Handle Length Mismatch**
```python
    # Ensure length matches DataFrame
    if len(rcf_scores) != len(df):
        if len(rcf_scores) > len(df):
            rcf_scores = rcf_scores[:len(df)]
        else:
            rcf_scores.extend([0.0] * (len(df) - len(rcf_scores)))
```
**Safety check**: Sometimes endpoint returns different number of scores than data points sent.
- If **more scores**: Trim to match data length
- If **fewer scores**: Pad with zeros
- Ensures we always have one score per data point

---

## **13. Save Scores to CSV**
```python
    # Save scores
    scores_df = pd.DataFrame({'RCFScore': rcf_scores})
    scores_df.to_csv(scores_file, index=False)
    
    print(f"Generated {len(rcf_scores)} anomaly scores. Saved to {scores_file}")
    return np.array(rcf_scores, dtype=float)
```
1. Create DataFrame with one column `RCFScore`
2. Save to `rcf_scores.csv`
3. Print confirmation message
4. Return scores as numpy array for use in other scripts

---

## **14. Main Execution Block**
```python
if __name__ == "__main__":
    test_model()
```
- When script runs directly, execute `test_model()`
- Can also be imported as module

---

## **SUMMARY: What This Script Does**

1. **Connects** to the deployed SageMaker endpoint
2. **Sends** all CPU usage data for scoring
3. **Processes** the JSON response (handles different formats)
4. **Extracts** anomaly scores (higher score = more anomalous)
5. **Saves** scores to CSV for later analysis

---

## **KEY CONCEPTS EXPLAINED**

### **What are RCF Scores?**
- **0.0-1.0**: "Normal" behavior
- **1.0-3.0**: "Slightly anomalous"
- **>3.0**: "Highly anomalous"
- **Interpretation**: Higher score = more different from normal patterns

### **Batch vs Real-Time Scoring**
- **This script**: Batch scoring (all data at once)
- **Production**: Could score real-time (one data point at a time)
- Example real-time: `predictor.predict([[current_cpu_usage]])`

### **Endpoint Communication Flow**
```
Your Code → CSVSerializer → HTTP Request → SageMaker Endpoint
         ← JSONDeserializer ← HTTP Response ←
```

### **Example Output File (`rcf_scores.csv`)**
```
RCFScore
0.25
0.18
0.31
3.75    ← Anomaly!
0.22
...
```

---

## **TROUBLESHOOTING TIPS**

If scores are all 0.0:
1. Check endpoint is running (AWS Console → SageMaker → Endpoints)
2. Check `endpoint_info.json` has correct endpoint name
3. Check IAM permissions allow invoking endpoint
4. Check data format (must be 2D array: `[[value1], [value2], ...]`)

If response format unexpected:
1. The debug prints (lines 36-43) show what you actually received
2. May need to adjust `extract_score_value()` function
3. Check SageMaker documentation for RCF response format

**Remember**: The endpoint must be deployed (`03_deploy.py`) before testing!