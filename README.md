# 🍽️ Restaurant Review ABSA System

> **Aspect-Based Sentiment Analysis** for restaurant reviews — built with DistilBERT, PyTorch, and Flask.

---

## 📁 Project Structure

```
ABSA_Project/
│
├── data/
│   ├── raw/                        # Original XML restaurant review datasets
│   │   ├── train.xml
│   │   └── test.xml
│   └── processed/                  # Cleaned & structured CSV files
│       ├── aspect_train.csv
│       ├── aspect_test.csv
│       ├── sentiment_train.csv
│       └── sentiment_test.csv
│
├── notebooks/
│   └── data_prep.ipynb             # Data parsing, cleaning, and EDA
│
├── models/
│   ├── aspect_model/               # Saved HuggingFace TokenClassification model
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── tokenizer files...
│   └── sentiment_model/            # Saved HuggingFace SequenceClassification model
│       ├── config.json
│       ├── model.safetensors
│       └── tokenizer files...
│
├── code/
│   ├── model_aspect.py             # Aspect extraction training script (BIO tagging)
│   ├── model_sentiment.py          # Sentiment classification training script
│   └── inference.py                # Core ABSA pipeline (aspect extract + sentiment predict)
│
├── app/                            # Flask web application
│   ├── app.py                      # Flask server — 3 API endpoints
│   ├── absa.py                     # ← Copy of code/inference.py
│   └── templates/
│       └── index.html              # Full UI: single review + batch upload + charts
│
├── requirements.txt                # All dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo and install dependencies
```bash
git clone https://github.com/your-username/ABSA_Project.git
cd ABSA_Project
pip install -r requirements.txt
```

### 2. Copy the inference pipeline into the app folder
```bash
cp code/inference.py app/absa.py
```
> Make sure the model paths inside `app/absa.py` still point correctly to your `models/` folder.

### 3. Run the Flask app
```bash
cd app
python app.py
```
Visit → **http://localhost:5000**

---

## 🧠 Pipeline Overview

```
Raw Review Text
      │
      ▼
┌─────────────────────┐
│  Aspect Extraction  │  DistilBERT Token Classification (BIO tags)
│  model_aspect.py    │  B-ASP / I-ASP / O
└────────┬────────────┘
         │  ["food", "waiter", "table"]
         ▼
┌─────────────────────┐
│ Sentiment Prediction│  DistilBERT Sequence Classification
│ model_sentiment.py  │  Input: (focused_context, aspect)
└────────┬────────────┘
         │  positive / negative / neutral
         ▼
   Structured Output
   [{ aspect: "food", sentiment: "positive" }, ...]
```

---

## 📊 Model Results

### Aspect Extraction (Token Classification)
| Metric    | Score  |
|-----------|--------|
| Precision | 0.3410 |
| Recall    | 0.8121 |
| F1 Score  | 0.4803 |
| Accuracy  | 0.8632 |

> High recall ensures maximum aspect coverage from reviews.

### Sentiment Classification (Sequence Classification)
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 0.7980 |
| Precision | 0.7340 |
| Recall    | 0.7206 |
| F1 Score  | 0.7267 |

---

## 🌐 Flask API Endpoints

| Method | Endpoint         | Description                              |
|--------|------------------|------------------------------------------|
| `GET`  | `/`              | Serves the web UI                        |
| `POST` | `/analyze/text`  | Analyse a single review text             |
| `POST` | `/analyze/file`  | Batch analyse a `.csv` or `.txt` file    |
| `POST` | `/download/csv`  | Download results as a CSV file           |

### `POST /analyze/text`
```json
// Request
{ "text": "The pizza was amazing but the service was terrible." }

// Response
{
  "text": "The pizza was amazing but the service was terrible.",
  "results": [
    { "aspect": "pizza",   "sentiment": "positive" },
    { "aspect": "service", "sentiment": "negative" }
  ]
}
```

### `POST /analyze/file`
Form fields:
- `file` — `.csv` or `.txt` upload
- `has_header` — `"true"` / `"false"` *(CSV only)*
- `column` — column name or 0-based index *(CSV only, default `"0"`)*

```json
// Response
{
  "records": [
    { "sentence": "...", "aspect": "food",    "sentiment": "positive" },
    { "sentence": "...", "aspect": "service", "sentiment": "negative" }
  ],
  "aspect_counts":         { "food": 12, "service": 8, ... },
  "aspect_sentiment_data": { "aspects": [...], "positive": [...], "negative": [...], "neutral": [...] },
  "total_sentences": 42,
  "total_aspects":   97
}
```

---

## 🖥️ Web UI Features

| Feature | Details |
|---|---|
| **Single Review Tab** | Paste any review → get color-coded aspect pills |
| **Batch File Tab** | Upload `.csv` or `.txt` with drag-and-drop |
| **CSV Options** | Toggle header row, pick column by name or index, live preview |
| **Summary Stats** | Sentences, total aspects, positive / negative / neutral counts |
| **Chart 1** | Aspect frequency bar chart (top 20) |
| **Chart 2** | Grouped sentiment breakdown per aspect |
| **Results Table** | Full sentence → aspect → sentiment table |
| **Download CSV** | Export results as `absa_results.csv` |

---

## 💡 Example Predictions

```
Input:  "the food is good but the table is bad"
Output: food → positive | table → negative

Input:  "the waiter was rude but the food was amazing"
Output: waiter → negative | food → positive

Input:  "the pizza was delicious but the drinks were terrible and the staff were friendly"
Output: pizza → positive | drinks → negative | staff → positive
```

---

## ⚙️ Training Configuration

### Aspect Model
```python
epochs         = 10
learning_rate  = 1e-5
batch_size     = 8
weight_decay   = 0.01
warmup_ratio   = 0.1
loss           = weighted CrossEntropyLoss  # handles BIO class imbalance
metric         = seqeval F1
```

### Sentiment Model
```python
epochs         = 5
learning_rate  = 2e-5
batch_size     = 16
weight_decay   = 0.01
labels         = ["negative", "neutral", "positive"]
```

---

## 🔬 Advanced Testing

Tested on:
- Mixed sentiments in the same sentence
- Negation handling (`"not bad"`, `"wasn't good"`)
- Comparative sentiment (`"better than expected"`)
- Conflicting clauses
- Long-range context dependency

**Known limitations:**
- Sarcasm is still difficult
- Implicit sentiment is challenging
- Conflicting sentiment within one aspect span can confuse the model

---

## 🔮 Future Improvements

- [ ] Use **DeBERTa** or **RoBERTa** as backbone for better performance
- [ ] Combine restaurant + laptop ABSA datasets for more training data
- [ ] Add **confidence scores** per prediction
- [ ] Integrate real-time web scraping of restaurant reviews
- [ ] Add filtering and search to the results table UI

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Models | DistilBERT, HuggingFace Transformers |
| Training | PyTorch, Datasets, Evaluate, seqeval |
| Data | NumPy, Pandas, XML parsing |
| Web App | Flask, Plotly.js |
| UI | HTML / CSS / Vanilla JS |

---

## 📄 License

MIT
