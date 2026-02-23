import argparse
import os
import json
import numpy as np
import pandas as pd
import torch
import logging
from typing import List, Dict, Any
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

# --- 1. Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- 2. Command-line Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "./outputs"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--train_data", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "./train"))
    parser.add_argument("--validation_data", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "./val"))
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()

# --- 3. Utility: Recursive JSON Cleaner ---
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

# --- 4. Data Loader for JSON/JSONL Datasets ---
def load_dataset(path: str) -> pd.DataFrame:
    """Loads and cleans data from a directory or file into a pandas DataFrame."""
    logger.info(f"Loading data from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data path not found: {path}")

    # Accept either a directory of files or a single file
    files = [path] if os.path.isfile(path) else [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith((".json", ".jsonl"))
    ]

    records = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.endswith(".jsonl"):
                    data = [json.loads(line) for line in f if line.strip()]
                else:
                    data = json.load(f)
            cleaned = clean_json_data(data)
            if isinstance(cleaned, list):
                records.extend(cleaned)
            elif isinstance(cleaned, dict):
                records.append(cleaned)
        except Exception as e:
            logger.warning(f"Failed reading {file_path}: {e}")

    if not records:
        return pd.DataFrame(columns=["text", "label"])

    df = pd.DataFrame(records)
    # Ensure columns present and consistent
    df["text"] = df.get("text", "").astype(str).str.strip()
    df["label"] = df.get("label", 0)
    df = df.dropna(subset=["text"])
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df

# --- 5. Metrics for Model Evaluation ---
def compute_metrics(pred):
    """Calculate standard metrics for classification."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    metrics = {
        "eval_f1_micro": f1_score(labels, preds, average="micro"),
        "eval_f1_macro": f1_score(labels, preds, average="macro"),
        "eval_precision": precision_score(labels, preds, average="weighted"),
        "eval_recall": recall_score(labels, preds, average="weighted"),
        "eval_accuracy": accuracy_score(labels, preds),
    }
    metrics["eval_f1"] = metrics["eval_f1_macro"]
    logger.info(f"Eval metrics: {metrics}")
    for k, v in metrics.items():
        print(f"'{k}': {v}")
    return metrics

# --- 6. PyTorch Dataset Wrapper for HuggingFace ---
class LogDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx], dtype=torch.long) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# --- 7. Main Training Function ---
def main():
    # ---- Parse training/evaluation config ----
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Handle mixed precision (auto or manual)
    if args.fp16 and args.bf16:
        raise ValueError("Only one of fp16/bf16 can be enabled, not both.")
    if not args.fp16 and not args.bf16 and torch.cuda.is_available():
        try:
            if torch.cuda.get_device_capability()[0] >= 8:
                args.bf16 = True
            else:
                args.fp16 = True
        except Exception:
            pass

    # ---- Load datasets ----
    train_df = load_dataset(args.train_data)
    val_df = load_dataset(args.validation_data)

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Empty train or validation dataset after loading.")

    # ---- Tokenizer & Data Encoding ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_texts(texts: List[str]) -> Dict[str, List[int]]:
        return tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_length,
            return_tensors="np",
        )

    train_ds = LogDataset(tokenize_texts(train_df["text"].tolist()), train_df["label"].values)
    val_ds = LogDataset(tokenize_texts(val_df["text"].tolist()), val_df["label"].values)

    # ---- Model Initialization ----
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=args.num_labels
    ).to(device)

    grad_accum = max(1, 32 // args.per_device_train_batch_size)

    # ---- Trainer Setup ----
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        logging_dir=f"{args.output_data_dir}/logs",
        logging_steps=10,
        report_to="none",
        fp16=args.fp16,
        bf16=args.bf16,
        no_cuda=not torch.cuda.is_available(),
        gradient_accumulation_steps=grad_accum,
        torch_compile=False,  # safe to leave False for PT<2
        optim="adamw_torch",
        save_total_limit=2,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ---- Training Loop ----
    train_result = trainer.train()
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    # ---- Print training metrics ----
    print("\nTraining metrics:")
    for k, v in train_result.metrics.items():
        print(f"'{k}': {v}")

    print("\nValidation metrics:")
    eval_metrics = trainer.evaluate()
    for k, v in eval_metrics.items():
        print(f"'{k}': {v}")

if __name__ == "__main__":
    main()
