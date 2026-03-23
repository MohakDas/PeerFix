import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "roberta-base"

# Load dataset
df = pd.read_csv("data/raw/processed_data.csv")

train_df = df[df["set"] == "train"].copy()
test_df = df[df["set"] == "test"].copy()

label2id = {
    "non_actionable": 0,
    "actionable": 1
}
id2label = {v: k for k, v in label2id.items()}

train_df["label"] = train_df["Label1"].map(label2id)
test_df["label"] = test_df["Label1"].map(label2id)

train_dataset = Dataset.from_pandas(train_df[["Text", "label"]])
test_dataset = Dataset.from_pandas(test_df[["Text", "label"]])

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example):
    return tokenizer(
        example["Text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(["Text"])
test_dataset = test_dataset.remove_columns(["Text"])

train_dataset.set_format("torch")
test_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

training_args = TrainingArguments(
    output_dir="training/actionability_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

metrics = trainer.evaluate()
print(metrics)

trainer.save_model("training/actionability_model/final")
tokenizer.save_pretrained("training/actionability_model/final")