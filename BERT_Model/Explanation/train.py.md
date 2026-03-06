# **train.py - Complete Line-by-Line Explanation**

## **Overview**
This is a **PyTorch/Hugging Face training script** for a text classification model. It trains a BERT-like model to classify log entries as "normal" or "anomalous" based on nginx log data.

---

## **Section 1: Import Statements (Lines 1-21)**

```python
import argparse          # For handling command-line arguments
import os               # For operating system operations (file paths, etc.)
import json             # For reading/writing JSON files
import numpy as np      # For numerical operations
import pandas as pd     # For data manipulation (DataFrames)
import torch            # PyTorch deep learning framework
import logging          # For logging progress and errors

from typing import List, Dict, Any  # Type hints for better code clarity
from transformers import (           # Hugging Face Transformers library
    AutoTokenizer,                   # Automatically loads tokenizer
    AutoModelForSequenceClassification,  # Pre-trained model for classification
    TrainingArguments,               # Configuration for training
    Trainer,                         # Main training class
    EarlyStoppingCallback            # Stops training if no improvement
)
from sklearn.metrics import (        # For calculating evaluation metrics
    f1_score, precision_score, recall_score, accuracy_score
)
```

**Translation:** "Get all the tools we need: data handling tools (pandas), deep learning tools (PyTorch, Transformers), and evaluation tools (sklearn)."

---

## **Section 2: Logging Setup (Lines 23-24)**

```python
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
```

**Translation:** "Set up a logging system that will print messages with timestamps and importance levels (INFO, WARNING, etc.)."

---

## **Section 3: Command-line Arguments (Lines 26-46)**

```python
def parse_args():
    parser = argparse.ArgumentParser()  # Create argument parser
    parser.add_argument("--epochs", type=int, default=3)  # Training cycles
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)  # Batch size for training
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)  # Batch size for evaluation
    parser.add_argument("--learning_rate", type=float, default=2e-5)  # How fast model learns
    parser.add_argument("--weight_decay", type=float, default=0.01)  # Regularization to prevent overfitting
    parser.add_argument("--max_seq_length", type=int, default=128)  # Max text length (in tokens)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")  # Which pre-trained model to use
    parser.add_argument("--num_labels", type=int, default=2)  # Binary classification (normal/anomaly)
    parser.add_argument("--output_data_dir", type=str, default="./outputs")  # Where to save outputs
    parser.add_argument("--model_dir", type=str, default="./model")  # Where to save trained model
    parser.add_argument("--train_data", type=str, default="./train")  # Training data location
    parser.add_argument("--validation_data", type=str, default="./val")  # Validation data location
    parser.add_argument("--fp16", action="store_true")  # Use 16-bit precision (faster)
    parser.add_argument("--bf16", action="store_true")  # Use bfloat16 precision (newer GPUs)
    return parser.parse_args()
```

**Translation:** "Create a configuration system so users can change settings without editing the code (like how many training cycles, what model to use, etc.)."

---

## **Section 4: JSON Cleaner (Lines 48-60)**

```python
def clean_json_data(data: Any) -> List[Dict[str, Any]]:
    """Recursively clean null fields out of loaded JSON for robust parsing."""
    if data is None:
        return []
    if isinstance(data, list):
        out = []
        for item in data:
            cleaned = clean_json_data(item)
            if cleaned is not None:
                out.append(cleaned)
        return out
    if isinstance(data, dict):
        return {k: v for k, v in ((k, clean_json_data(v)) for k, v in data.items()) if v is not None}
    return data
```

**Translation:** "Create a cleaning function that removes empty/null values from JSON data to prevent crashes when loading messy data."

---

## **Section 5: Data Loader (Lines 62-100)**

```python
def load_dataset(path: str) -> pd.DataFrame:
    """Loads and cleans data from a directory or file into a pandas DataFrame."""
    logger.info(f"Loading data from: {path}")
    
    # Check if path exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data path not found: {path}")

    # Get all JSON/JSONL files
    files = [path] if os.path.isfile(path) else [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith((".json", ".jsonl"))
    ]

    records = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.endswith(".jsonl"):  # JSON Lines format
                    data = [json.loads(line) for line in f if line.strip()]
                else:  # Regular JSON
                    data = json.load(f)
            cleaned = clean_json_data(data)  # Clean the data
            records.extend(cleaned)  # Add to records list
        except Exception as e:
            logger.warning(f"Failed reading {file_path}: {e}")

    # Create DataFrame
    df = pd.DataFrame(records)
    # Ensure required columns exist
    df["text"] = df.get("text", "").astype(str).str.strip()  # Log text
    df["label"] = df.get("label", 0)  # Label (0=normal, 1=anomaly)
    df = df.dropna(subset=["text"])  # Remove rows without text
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df
```

**Translation:** "Load training data from files, clean it up, and put it in a nice table format. Handle both single files and folders of files."

---

## **Section 6: Evaluation Metrics (Lines 102-117)**

```python
def compute_metrics(pred):
    """Calculate standard metrics for classification."""
    labels = pred.label_ids  # True labels
    preds = pred.predictions.argmax(-1)  # Predicted labels (pick highest probability)
    
    metrics = {
        "eval_f1_micro": f1_score(labels, preds, average="micro"),  # Overall F1 score
        "eval_f1_macro": f1_score(labels, preds, average="macro"),  # Average F1 across classes
        "eval_precision": precision_score(labels, preds, average="weighted"),  # Precision
        "eval_recall": recall_score(labels, preds, average="weighted"),  # Recall
        "eval_accuracy": accuracy_score(labels, preds),  # Simple accuracy
    }
    metrics["eval_f1"] = metrics["eval_f1_macro"]  # Main metric for comparison
    
    # Log and print results
    logger.info(f"Eval metrics: {metrics}")
    for k, v in metrics.items():
        print(f"'{k}': {v}")
    return metrics
```

**Translation:** "Calculate how well the model is doing using several different measures (accuracy, F1 score, etc.)."

---

## **Section 7: Dataset Wrapper (Lines 119-130)**

```python
class LogDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings  # Tokenized text data
        self.labels = labels  # Corresponding labels

    def __getitem__(self, idx):
        # Convert numpy arrays to PyTorch tensors for one data point
        item = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)  # Total number of examples
```

**Translation:** "Create a custom data container that PyTorch can use for training. It converts text into the format the model expects."

---

## **Section 8: Main Training Function (Lines 132-226)**

### **Part A: Setup and Configuration**
```python
def main():
    args = parse_args()  # Get command-line arguments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
```

**Translation:** "Start the main program. Get settings from user and check if we have a GPU."

---

### **Part B: Mixed Precision Setup**
```python
    # Handle mixed precision (auto or manual)
    if args.fp16 and args.bf16:
        raise ValueError("Only one of fp16/bf16 can be enabled, not both.")
    if not args.fp16 and not args.bf16 and torch.cuda.is_available():
        try:
            if torch.cuda.get_device_capability()[0] >= 8:  # Newer GPUs
                args.bf16 = True
            else:  # Older GPUs
                args.fp16 = True
        except Exception:
            pass
```

**Translation:** "Configure precision settings to make training faster on GPUs. New GPUs use bfloat16, older ones use float16."

---

### **Part C: Load Data**
```python
    train_df = load_dataset(args.train_data)  # Load training data
    val_df = load_dataset(args.validation_data)  # Load validation data

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Empty train or validation dataset after loading.")
```

**Translation:** "Load the training and validation datasets. Crash if either is empty."

---

### **Part D: Tokenize Text**
```python
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # Load tokenizer

    def tokenize_texts(texts: List[str]) -> Dict[str, List[int]]:
        return tokenizer(
            texts,
            padding="max_length",  # Pad all texts to same length
            truncation=True,  # Cut texts that are too long
            max_length=args.max_seq_length,  # Maximum length
            return_tensors="np",  # Return as numpy arrays
        )

    # Create PyTorch datasets
    train_ds = LogDataset(tokenize_texts(train_df["text"].tolist()), train_df["label"].values)
    val_ds = LogDataset(tokenize_texts(val_df["text"].tolist()), val_df["label"].values)
```

**Translation:** "Convert text into numbers (tokens) that the model can understand. Make all texts the same length by padding or truncating."

---

### **Part E: Load Model**
```python
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=args.num_labels
    ).to(device)  # Move model to GPU if available

    grad_accum = max(1, 32 // args.per_device_train_batch_size)  # Accumulate gradients for larger effective batch size
```

**Translation:** "Load the pre-trained BERT model and configure it for our task (2-class classification)."

---

### **Part F: Training Configuration**
```python
    training_args = TrainingArguments(
        output_dir=args.model_dir,  # Where to save model
        evaluation_strategy="epoch",  # Evaluate after each epoch
        save_strategy="epoch",  # Save model after each epoch
        learning_rate=args.learning_rate,  # How fast to learn
        per_device_train_batch_size=args.per_device_train_batch_size,  # Batch size
        per_device_eval_batch_size=args.per_device_eval_batch_size,  # Eval batch size
        num_train_epochs=args.epochs,  # How many training cycles
        weight_decay=args.weight_decay,  # Regularization
        load_best_model_at_end=True,  # Keep the best model
        metric_for_best_model="eval_f1_macro",  # Use F1 score to pick best
        greater_is_better=True,  # Higher F1 is better
        logging_dir=f"{args.output_data_dir}/logs",  # Where to save logs
        logging_steps=10,  # Log every 10 steps
        report_to="none",  # Don't report to external services
        fp16=args.fp16,  # Mixed precision settings
        bf16=args.bf16,
        no_cuda=not torch.cuda.is_available(),  # Use CPU if no GPU
        gradient_accumulation_steps=grad_accum,  # Accumulate gradients
        optim="adamw_torch",  # Optimization algorithm
        save_total_limit=2,  # Keep only 2 best models
        seed=42,  # Random seed for reproducibility
    )
```

**Translation:** "Set up all training parameters in one place: how to train, when to save, what metrics to track."

---

### **Part G: Create Trainer**
```python
    trainer = Trainer(
        model=model,  # The model to train
        args=training_args,  # Training configuration
        train_dataset=train_ds,  # Training data
        eval_dataset=val_ds,  # Validation data
        compute_metrics=compute_metrics,  # How to calculate metrics
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Stop early if no improvement
    )
```

**Translation:** "Create the trainer object that will handle the actual training loop."

---

### **Part H: Training Loop**
```python
    train_result = trainer.train()  # Start training!
    trainer.save_model(args.model_dir)  # Save trained model
    tokenizer.save_pretrained(args.model_dir)  # Save tokenizer

    # Print training metrics
    print("\nTraining metrics:")
    for k, v in train_result.metrics.items():
        print(f"'{k}': {v}")

    print("\nValidation metrics:")
    eval_metrics = trainer.evaluate()  # Final evaluation
    for k, v in eval_metrics.items():
        print(f"'{k}': {v}")
```

**Translation:** "Train the model, save it, then print out how well it performed."

---

### **Part I: Main Guard**
```python
if __name__ == "__main__":
    main()
```

**Translation:** "Only run the main() function if this script is executed directly (not imported)."

---

## **Key Concepts Simplified:**

1. **Tokenizer**: Converts text → numbers (like "error" → [123, 456])
2. **Model**: BERT neural network that learns patterns in log data
3. **Training**: Adjusts model weights to minimize mistakes
4. **Validation**: Tests model on unseen data to prevent overfitting
5. **Metrics**: Numbers that tell us how good the model is

## **The Training Process:**
1. Load log data with labels (0=normal, 1=anomaly)
2. Convert text to tokens
3. Train BERT model to recognize patterns
4. Save the best model for later use
5. Report performance metrics

This script is the **core** of your ML pipeline - it's what actually trains the model to distinguish between normal and anomalous nginx logs!