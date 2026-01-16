import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "sentiment_model"

def predict(text: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    pred_id = int(torch.argmax(probs).item())
    pred_label = model.config.id2label[str(pred_id)] if isinstance(model.config.id2label, dict) else model.config.id2label[pred_id]
    return pred_label, probs.tolist()

if __name__ == "__main__":
    txt = "Сервис был нормальный, ничего особенного"
    label, probs = predict(txt)
    print("Text:", txt)
    print("Pred:", label)
    print("Probs [neg, neutral, pos]:", probs)
