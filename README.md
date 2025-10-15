# 🕵️ Fraud Detection Batch System

Un sistema di analisi batch per la rilevazione automatica di frodi basato su pipeline ML (Scikit-Learn / XGBoost).

## 🚀 Funzionalità
- Predizione frodi da file CSV batch
- Generazione automatica di report PDF e grafici SHAP
- Logging dettagliato e storico dei batch
- Threshold calibrabile con `scripts/calibrate_threshold.py`
- Export dei risultati e reportistica per auditing

## 📁 Struttura del progetto
Vedi la sezione nel codice o nella documentazione.

## 🧩 Setup locale
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python predict_batch_advanced.py
