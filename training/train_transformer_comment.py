import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

MODEL_NAME = "roberta-base"

# Load actionable-only dataset
df = pd.read_csv("data/processed/actionable_comments.csv")

train_df = df[df["set"] == "train"].copy()
test_df = df[df["set"] == "test"].copy()

# Build label mappings from training labels
label_list = sorted(train_df["Label2"].unique())
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

train_df["label"] = train_df["Label2"].map(label2id)
test_df["label"] = test_df["Label2"].map(label2id)

# Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df[["Text", "label"]], preserve_index=False)
test_dataset = Dataset.from_pandas(test_df[["Text", "label"]], preserve_index=False)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(example):
    return tokenizer(
        example["Text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.remove_columns(["Text"])
test_dataset = test_dataset.remove_columns(["Text"])

train_dataset.set_format("torch")
test_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

training_args = TrainingArguments(
    output_dir="training/comment_type_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    save_total_limit=1,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

metrics = trainer.evaluate()
print(metrics)

trainer.save_model("training/comment_type_model/final")
tokenizer.save_pretrained("training/comment_type_model/final")