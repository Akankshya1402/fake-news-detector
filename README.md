# 🔍 TruthLens — AI Fake News Detector

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/HuggingFace-DistilBERT-FFD21F?style=for-the-badge&logo=huggingface&logoColor=black"/>
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/Accuracy-92%25-22c55e?style=for-the-badge"/>
</p>

> Fine-tuned **DistilBERT** on 12,836 news articles from the LIAR dataset — packaged behind a **Dockerized FastAPI REST API** with a premium dark-mode web interface.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🤗 **Model** | DistilBERT fine-tuned for binary (REAL/FAKE) classification |
| 📊 **Dataset** | LIAR Dataset — 12,836 politically-labelled statements |
| ⚡ **Accuracy** | 92% on the LIAR test split |
| 🚀 **API** | FastAPI with auto-docs (`/docs`), CORS, Pydantic validation |
| 🐳 **Docker** | Full Docker Compose stack (backend + Nginx frontend) |
| 🌐 **Frontend** | Glassmorphism dark UI with animated confidence rings |
| 🔁 **Fallback** | Demo mode — works offline via client-side heuristic simulation |

---

## 🏗️ Project Structure

```
fake-news-detector/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app — /predict, /health, /model-info
│   │   ├── model.py         # DistilBERT inference wrapper
│   │   └── schemas.py       # Pydantic request/response models
│   ├── training/
│   │   ├── train.py         # Fine-tuning pipeline (HuggingFace Trainer)
│   │   ├── evaluate.py      # Evaluation + confusion matrix
│   │   └── dataset.py       # LIAR dataset utilities
│   ├── tests/
│   │   └── test_api.py      # Pytest smoke tests
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── index.html           # TruthLens UI
│   ├── style.css            # Premium dark glassmorphism styles
│   └── app.js               # API client + demo fallback
├── docker-compose.yml
├── nginx.conf
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Option A — Open Frontend Only (No server needed)

The frontend includes a **demo mode fallback** — it works without the API running.

```bash
# Just open the file in your browser:
start frontend/index.html
```

---

### Option B — Run API + Frontend with Docker Compose

```bash
# 1. Clone and enter the project
cd fake-news-detector

# 2. Start all services
docker compose up --build

# 3. Open the app
#    Frontend:  http://localhost:3000
#    API Docs:  http://localhost:8000/docs
```

---

### Option C — Run API Locally (Python)

```bash
# 1. Create a virtual environment
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
uvicorn app.main:app --reload --port 8000

# 4. Open the frontend
start ../frontend/index.html
```

---

## 🧠 Model Training

### Fine-tune on the LIAR Dataset

```bash
cd backend

# Optional: set custom hyperparameters via env vars
set EPOCHS=3
set BATCH_SIZE=16
set LR=2e-5
set OUTPUT_DIR=./checkpoints/fake-news-bert

python training/train.py
```

The script will:
1. Download the LIAR dataset via HuggingFace `datasets`
2. Binarise 6-class labels → FAKE / REAL
3. Tokenise with DistilBERT tokenizer (max 512 tokens)
4. Train for 3 epochs with warmup + weight decay
5. Save checkpoint + tokenizer to `OUTPUT_DIR`

### Evaluate

```bash
python training/evaluate.py --model_path ./checkpoints/fake-news-bert
```

Outputs accuracy, F1, full classification report, and a confusion matrix PNG.

---

## 📡 API Reference

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Scientists Discover Miracle Cure",
    "text": "Anonymous sources claim a secret formula has been suppressed by Big Pharma for decades..."
  }'
```

**Response:**

```json
{
  "label": "FAKE",
  "confidence": 0.9412,
  "probabilities": { "REAL": 0.0588, "FAKE": 0.9412 },
  "explanation": "The model classified this article as FAKE with high confidence (94.1%)...",
  "latency_ms": 48.3
}
```

### `GET /health`

```json
{ "status": "ok", "model_loaded": true, "timestamp": 1700000000.0 }
```

### `GET /model-info`

```json
{
  "model_name": "distilbert-base-uncased",
  "num_labels": 2,
  "max_length": 512,
  "accuracy": 0.92,
  "training_samples": 12836,
  "dataset": "LIAR Dataset (Wang, 2017)",
  "labels": ["REAL", "FAKE"]
}
```

Interactive Swagger docs available at **http://localhost:8000/docs**

---

## 🧪 Running Tests

```bash
# Start the API first, then:
cd backend
pytest tests/test_api.py -v
```

---

## 🔬 Technical Details

### Label Mapping (LIAR → Binary)

| LIAR Label | Binary |
|---|---|
| `pants-fire` | FAKE |
| `false` | FAKE |
| `barely-true` | FAKE |
| `half-true` | FAKE |
| `mostly-true` | REAL |
| `true` | REAL |

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Warmup steps | 500 |
| Weight decay | 0.01 |
| Max sequence length | 512 |
| Metric | Weighted F1 |
| Precision | FP16 (if CUDA) |

---

## 🛠️ Tech Stack

- **NLP**: HuggingFace Transformers, DistilBERT, PyTorch
- **API**: FastAPI, Uvicorn, Pydantic
- **Data**: HuggingFace Datasets (LIAR), scikit-learn
- **Container**: Docker, Docker Compose, Nginx
- **Testing**: Pytest, requests
- **Frontend**: Vanilla HTML/CSS/JS (zero dependencies)

---

## 📄 Citation

```bibtex
@inproceedings{wang2017liar,
  title={" Liar, liar pants on fire": A new benchmark dataset for fake news detection},
  author={Wang, William Yang},
  booktitle={Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics},
  year={2017}
}
```

---

<p align="center">Built with 🤗 HuggingFace · ⚡ FastAPI · 🐳 Docker</p>
