import re
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification
)

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load models ---
aspect_path = r"C:\Users\Abdo\Desktop\ABSA_Project\models\aspect_model"
sentiment_path = r"C:\Users\Abdo\Desktop\ABSA_Project\models\sentiment_model"

aspect_tokenizer = AutoTokenizer.from_pretrained(aspect_path)
aspect_model = AutoModelForTokenClassification.from_pretrained(aspect_path).to(device)

sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_path)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_path).to(device)

aspect_model.eval()
sentiment_model.eval()

label_list = ["O", "B-ASP", "I-ASP"]
sentiment_labels = ["negative", "neutral", "positive"]


# --- Aspect extraction ---
def extract_aspects(text):

    enc = aspect_tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        out = aspect_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    preds = torch.argmax(out.logits, dim=2)[0].cpu().tolist()
    tokens = aspect_tokenizer.convert_ids_to_tokens(input_ids[0])

    aspects = []
    current = []

    for tok, pred in zip(tokens, preds):

        label = label_list[pred]

        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        if label == "B-ASP":

            if current:
                aspects.append(" ".join(current))

            current = [tok]

        elif label == "I-ASP":

            current.append(tok)

        else:

            if current:
                aspects.append(" ".join(current))
                current = []

    if current:
        aspects.append(" ".join(current))

    cleaned = []

    for a in aspects:

        a = a.replace(" ##", "")
        a = a.replace("##", "")
        a = a.strip()

        if len(a) > 1:
            cleaned.append(a)

    return list(set(cleaned))


# --- 🔥 Smart context splitter ---
def build_focused_context(text, aspect):

    # split sentence smarter using punctuation + connectors
    chunks = re.split(
        r'\s*(?:,|but|and|while|although|however)\s*',
        text,
        flags=re.IGNORECASE
    )

    selected = []

    for chunk in chunks:

        if aspect.lower() in chunk.lower():
            selected.append(chunk.strip())

    # fallback
    if not selected:
        selected.append(text)

    focused_text = " ".join(selected)

    return focused_text


# --- 🔥 Smarter sentiment prediction ---
def predict_sentiment(text, aspect):

    focused_text = build_focused_context(text, aspect)

    # aspect-aware input
    enc = sentiment_tokenizer(
        focused_text,
        aspect,
        return_tensors="pt",
        truncation=True
    )

    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():

        out = sentiment_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    probs = torch.softmax(out.logits, dim=1)[0]
    pred = torch.argmax(probs).item()

    return sentiment_labels[pred]


# --- Full pipeline ---
def analyze(text):

    aspects = extract_aspects(text)

    results = []

    for asp in aspects:

        sent = predict_sentiment(text, asp)

        results.append({
            "aspect": asp,
            "sentiment": sent
        })

    return {
        "text": text,
        "results": results
    }


# --- TEST CASES ---
if __name__ == "__main__":

    tests = [

        'the food is amazing and the waiter is bad',
        'the food was bad, the waiter was rude, and the table was very good and clean',
        'the pizza was delicious but the drinks were terrible and the staff were friendly',
        'the pasta tasted amazing, the soup was cold, and the service was extremely slow',
        'the burgers were excellent but the chairs were uncomfortable and the waiter ignored us',
        'the dessert was fantastic, the music was too loud, but the atmosphere was beautiful'
    ]

    for t in tests:

        out = analyze(t)

        print("\n====================")
        print("TEXT:", out["text"])

        for r in out["results"]:

            print(f"- Aspect: {r['aspect']} | Sentiment: {r['sentiment']}")