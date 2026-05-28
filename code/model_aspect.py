import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
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

label_list = ["O", "B-ASP", "I-ASP"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# --- 3. Tokenize & Align Labels Using Offset Mapping ---
def tokenize_and_align_labels(df):
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for _, row in df.iterrows():
        text = str(row["sentence"])
        aspect = str(row["aspect"])

        encoding = tokenizer(
            text,
            truncation=True,
            padding=False,
            return_offsets_mapping=True
        )

        offsets = encoding["offset_mapping"]
        label_ids = []

        asp_start = int(row["from"])
        asp_end = int(row["to"])

        first_asp_token = True
        for idx, (s, e) in enumerate(offsets):
            # Special tokens ([CLS], [SEP]) have offset (0, 0)
            if s == 0 and e == 0:
                label_ids.append(-100)
            elif s >= asp_start and e <= asp_end:
                if first_asp_token:
                    label_ids.append(label2id["B-ASP"])
                    first_asp_token = False
                else:
                    label_ids.append(label2id["I-ASP"])
            else:
                label_ids.append(label2id["O"])

        all_input_ids.append(encoding["input_ids"])
        all_attention_masks.append(encoding["attention_mask"])
        all_labels.append(label_ids)

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels
    })

tokenized_train = tokenize_and_align_labels(train_df)
tokenized_test = tokenize_and_align_labels(test_df)

# --- 4. Compute Class Weights to Handle O/B-ASP/I-ASP Imbalance ---
def compute_class_weights(df):
    all_labels = []

    for _, row in df.iterrows():
        text = str(row["sentence"])
        encoding = tokenizer(text, truncation=True, return_offsets_mapping=True)
        offsets = encoding["offset_mapping"]

        asp_start = int(row["from"])
        asp_end = int(row["to"])
        first_asp_token = True

        for s, e in offsets:
            if s == 0 and e == 0:
                continue
            elif s >= asp_start and e <= asp_end:
                all_labels.append(label2id["B-ASP"] if first_asp_token else label2id["I-ASP"])
                first_asp_token = False
            else:
                all_labels.append(label2id["O"])

    counts = np.bincount(all_labels, minlength=len(label_list))
    total = counts.sum()
    # Inverse frequency weighting
    weights = total / (len(label_list) * counts)
    return torch.tensor(weights, dtype=torch.float)

class_weights = compute_class_weights(train_df)
print(f"\nClass weights -> O: {class_weights[0]:.2f} | B-ASP: {class_weights[1]:.2f} | I-ASP: {class_weights[2]:.2f}\n")

# --- 5. Custom Trainer with Weighted Loss ---
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = nn.CrossEntropyLoss(
            weight=class_weights.to(logits.device),
            ignore_index=-100
        )
        # logits: (batch, seq_len, num_labels) -> (batch * seq_len, num_labels)
        loss = loss_fn(
            logits.view(-1, len(label_list)),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss

# --- 6. Initialize Model & Metrics ---
model = AutoModelForTokenClassification.from_pretrained(
    checkpoint,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)

seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(
        predictions=true_predictions,
        references=true_labels
    )

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# --- 7. Configure Trainer and Train Model ---
training_args = TrainingArguments(
    output_dir="./aspect_model",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# --- 8. Evaluate and Save ---
metrics = trainer.evaluate()

print("\nEvaluation Metrics:")
print(f"Precision: {metrics['eval_precision']:.4f}")
print(f"Recall:    {metrics['eval_recall']:.4f}")
print(f"F1 Score:  {metrics['eval_f1']:.4f}")
print(f"Accuracy:  {metrics['eval_accuracy']:.4f}")

model.save_pretrained(
    r"C:\Users\Abdo\Desktop\ABSA_Project\models\aspect_model"
)

tokenizer.save_pretrained(
    r"C:\Users\Abdo\Desktop\ABSA_Project\models\aspect_model"
)