import os
import numpy as np
from dataclasses import dataclass

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Можно заменить на ruBERT-совместимую модель, если знаешь подходящую под твой язык/задачу.
MODEL_NAME = "distilbert-base-multilingual-cased"  # мультиязычная, нормально для RU/EN

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}

def main():
    # Ожидаем файлы train.csv и val.csv в текущей папке
    dataset = load_dataset(
        "csv",
        data_files={"train": "train.csv", "validation": "val.csv"},
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(batch):
        # Переводим строковые метки в id
        labels = [LABEL2ID[x] for x in batch["label"]]
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
        )
        enc["labels"] = labels
        return enc

    tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir="sentiment_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        fp16=False,  # поставь True, если есть GPU с поддержкой
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("sentiment_model")
    tokenizer.save_pretrained("sentiment_model")

    print("✅ Saved to: sentiment_model")

if __name__ == "__main__":
    main()
