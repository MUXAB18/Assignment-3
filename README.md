# Emotion Classification System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A text-based emotion classification system built with a **Naive Bayes** algorithm and a **Flask** web interface. Upload or type any text and get instant emotion predictions with confidence scores and detailed model performance metrics.

> **Assignment 3** — [GitHub Repository](https://github.com/musab-18/Assignment-3.git)

---

## Demo

| Home Page | Results Page | Metrics Page |
|---|---|---|
| Enter any text | See predicted emotion + confidence | View accuracy & per-class metrics |

---

## Features

- **Naive Bayes Classifier** — trained on 16,000 labelled samples, achieving **80.45% accuracy**
- **6 Emotion Classes** — joy, sadness, anger, fear, love, surprise
- **Confidence Scores** — ranked probabilities for all emotion classes
- **Flask Web UI** — clean, responsive interface with example texts
- **Performance Metrics** — accuracy, precision, recall, F1-score per class with visual charts
- **REST API** — `/api/predict` and `/api/metrics` endpoints
- **Parquet Dataset Support** — handles both split and unsplit `.parquet` files

---

## Project Structure

```
Assignment-3/
├── src/
│   ├── api/
│   │   └── web_app.py          # Flask application & routes
│   ├── data/
│   │   └── data_loader.py      # Parquet dataset loader & validator
│   ├── ml/
│   │   ├── emotion_classifier.py  # Naive Bayes classifier
│   │   ├── text_processor.py      # Text preprocessing & feature extraction
│   │   ├── model_evaluator.py     # Accuracy, F1, confusion matrix
│   │   └── prediction_engine.py   # Prediction orchestrator
│   └── utils/
│       ├── config.py           # App configuration
│       └── logging.py          # Logging setup
├── templates/                  # Jinja2 HTML templates
│   ├── base.html
│   ├── index.html
│   ├── results.html
│   ├── metrics.html
│   └── error.html
├── static/
│   ├── css/style.css
│   └── js/app.js
├── data/
│   └── models/                 # Trained model artifacts
├── tests/                      # Unit & integration tests
├── split/                      # Split parquet datasets (train/val/test)
├── unsplit/                    # Unsplit parquet dataset
├── run_app.py                  # App entry point
└── requirements.txt
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/musab-18/Assignment-3.git
cd Assignment-3
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python run_app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## Dataset

The system uses the [Emotions dataset](https://huggingface.co/datasets/dair-ai/emotion) in Parquet format.

| Split | File | Samples |
|---|---|---|
| Train | `split/train-00000-of-00001.parquet` | 16,000 |
| Validation | `split/validation-00000-of-00001.parquet` | 2,000 |
| Test | `split/test-00000-of-00001.parquet` | 2,000 |
| Unsplit | `unsplit/train-00000-of-00001.parquet` | 416,000 |

**Schema:** `text` (string), `label` (int 0–5)

| Label | Emotion |
|---|---|
| 0 | sadness |
| 1 | joy |
| 2 | love |
| 3 | anger |
| 4 | fear |
| 5 | surprise |

---

## Pre-trained Model

This project uses a **pre-trained Naive Bayes model** that is included directly in the repository — no training step required.

| File | Description |
|---|---|
| `data/models/real_data_model.json` | Learned model parameters (class priors, word likelihoods) |
| `data/models/real_data_model.preprocessor.pkl` | Fitted vocabulary and CountVectorizer |

The model was trained on the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset (16,000 samples) using Multinomial Naive Bayes with Laplace smoothing (α=1.0). Both files are committed to this repository so the app runs immediately without any retraining.

> If you want to retrain from scratch, see the [Training the Model](#training-the-model-optional) section below.

---

## Model Performance

Evaluated on 2,000 held-out test samples:

| Metric | Score |
|---|---|
| **Overall Accuracy** | **80.45%** |

| Emotion | Precision | Recall | F1-Score |
|---|---|---|---|
| anger | 0.887 | 0.716 | 0.793 |
| fear | 0.824 | 0.670 | 0.739 |
| joy | 0.785 | 0.950 | 0.859 |
| love | 0.827 | 0.390 | 0.530 |
| sadness | 0.794 | 0.921 | 0.853 |
| surprise | 0.833 | 0.076 | 0.139 |

---

## API Reference

### `POST /api/predict`

Predict emotion from text.

**Request**
```json
{ "text": "I am so happy today!" }
```

**Response**
```json
{
  "emotion": "joy",
  "confidence": 0.923,
  "all_predictions": {
    "joy": 0.923,
    "sadness": 0.031,
    "anger": 0.021,
    "fear": 0.014,
    "love": 0.008,
    "surprise": 0.003
  },
  "processing_time": 0.002,
  "is_valid": true
}
```

### `GET /api/metrics`

Returns the latest model evaluation metrics as JSON.

---

## Configuration

Customise via environment variables:

```bash
export MODEL_ALPHA=1.0            # Laplace smoothing
export FLASK_HOST=127.0.0.1
export FLASK_PORT=5000
export FLASK_DEBUG=false
export LOG_LEVEL=INFO
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Training the Model (Optional)

The pre-trained model is already included — you only need this if you want to retrain from scratch.

```python
import sys
sys.path.insert(0, 'src')
from data.data_loader import DataLoader
from ml.emotion_classifier import EmotionClassifier

loader = DataLoader()
train_df, _, _ = loader.load_split_data()

classifier = EmotionClassifier(alpha=1.0)
classifier.train(train_df['text'].tolist(), train_df['emotion'].tolist())
classifier.save_model('data/models/real_data_model')
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Web Framework | Flask 3.1 |
| ML Algorithm | Multinomial Naive Bayes (scikit-learn) |
| Data Processing | pandas, pyarrow |
| Numerical | numpy, scipy |
| Frontend | Bootstrap 5, Font Awesome |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Musab** — [GitHub](https://github.com/musab-18)
