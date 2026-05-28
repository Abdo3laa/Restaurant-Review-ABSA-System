import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

import evaluate

# --- 1. Load Data ---
train_df = pd.read_csv(
    r"C:\Users\Abdo\Desktop\ABSA_Project\data\processed\train_sentiment.csv"
)

test_df = pd.read_csv(
    r"C:\Users\Abdo\Desktop\ABSA_Project\data\processed\test_sentiment.csv"
)

# --- 2. Setup Configuration & Labels ---
checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

label_list = ["negative", "neutral", "positive"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# --- 3. Tokenize Sentence + Aspect (IMPROVED FORMAT) ---
def tokenize_dataset(df):
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for _, row in df.iterrows():
        sentence = str(row["sentence"])
        aspect = str(row["aspect"])
        sentiment = str(row["sentiment"])

        # Improved aspect-aware format (stronger signal)
        text = f"{sentence} [SEP] aspect: {aspect}"

        encoding = tokenizer(
            text,
            truncation=True,
            padding=False
        )

        all_input_ids.append(encoding["input_ids"])
        all_attention_masks.append(encoding["attention_mask"])
        all_labels.append(label2id[sentiment])

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels
    })

tokenized_train = tokenize_dataset(train_df)
tokenized_test = tokenize_dataset(test_df)

# --- 4. Initialize Model & Metrics ---
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_metric.compute(
            predictions=predictions, references=labels
        )["accuracy"],
        "precision": precision_metric.compute(
            predictions=predictions, references=labels, average="macro"
        )["precision"],
        "recall": recall_metric.compute(
            predictions=predictions, references=labels, average="macro"
        )["recall"],
        "f1": f1_metric.compute(
            predictions=predictions, references=labels, average="macro"
        )["f1"],
    }

# --- 5. Configure Trainer and Train Model ---
training_args = TrainingArguments(
    output_dir=r"C:\Users\Abdo\Desktop\ABSA_Project\models\sentiment_model",
    learning_rate=1.5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# --- 6. Evaluate and Save ---
metrics = trainer.evaluate()

print("\nEvaluation Metrics:")
print(f"Accuracy:  {metrics['eval_accuracy']:.4f}")
print(f"Precision: {metrics['eval_precision']:.4f}")
print(f"Recall:    {metrics['eval_recall']:.4f}")
print(f"F1 Score:  {metrics['eval_f1']:.4f}")

model.save_pretrained(
    r"C:\Users\Abdo\Desktop\ABSA_Project\models\sentiment_model"
)

tokenizer.save_pretrained(
    r"C:\Users\Abdo\Desktop\ABSA_Project\models\sentiment_model"
)