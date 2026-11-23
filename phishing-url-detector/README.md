# Phishing URL Detector

Web application that tells you if a URL is **safe or phishing**, using several layers that work together.

> Final-year project (PFE, Licence), University M'Hamed Bougara of Boumerdès (UMBB), 2026.

## How it works
| Layer | Tool | Role | Weight |
|---|---|---|---|
| 1 | Whitelist | Trusted domains pass directly (unless the path looks suspicious, e.g. `/verify-account`) | - |
| 2 | **XGBoost** | Analyzes 35 structural features of the URL | 35% |
| 3 | **SecureBERT** (fine-tuned) | Understands the meaning of the words in the URL | 45% |
| 4 | Zero-Day rules | 15 heuristic checks for new, unknown threats | 20% |
| 5 | HTML analysis (optional) | Fetches the page: password fields, external forms, hidden iframes, urgent language | +29% blend |



## Results
Each model was evaluated separately on its own test set:

| Model | Test samples | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| XGBoost (36 URL features) | 119,315 | 98.82% | 99.46% | 96.99% | 98.21% |
| SecureBERT (fine-tuned) | 16,000 | 94.84% | 94.60% | 95.11% | 94.86% |

The full ensemble (with zero-day rules and HTML analysis) has not been evaluated as a whole yet.

## Project structure
```
app.py               Flask server + API (/api/predict, batch scan, history)
features.py          URL feature extraction (used by app and training)
train_xgboost.py     Train the XGBoost model
templates/           Web interface (index.html)
icon/                Interface icons
models/              Trained models (see models/README.md)
notebooks/           SecureBERT fine-tuning (Kaggle and Colab versions)
docs/                Full documentation (English/Arabic)
datasets/            Training data and scan log (not uploaded)
```

## Installation
```bash
git clone https://github.com/YOUR_USERNAME/phishing-url-detector.git
cd phishing-url-detector
pip install -r requirements.txt
# download SecureBERT into models/securebert_phishing/ (see models/README.md)
python app.py
```
Open http://localhost:5000

## Training
- XGBoost: `python train_xgboost.py --data datasets/prepared_urls.csv`
- SecureBERT: run a notebook from `notebooks/` on Kaggle or Google Colab (GPU).

## Dataset
[Phishing and Legitimate URLs](https://www.kaggle.com/datasets/harisudhan411/phishing-and-legitimate-urls) (Kaggle).

## Limitations
No detector is perfect: it can give false positives or miss brand-new attacks. Use it as a helper, not as the only protection.

## Author
MERAH Assia, LAISSA Chahrazed, DERICHE Besma, DOUMAZ Hadjer
Supervisor: SUPERVISOR NAME
