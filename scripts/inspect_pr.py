# scripts/inspect_pr.py
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT.joinpath("src")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
val_csv = PROJECT_ROOT.joinpath("data","validation.csv")
if not val_csv.exists():
    raise SystemExit("data/validation.csv non trovato.")

# importa pipeline
try:
    from pipeline_module import get_pipeline
except Exception:
    # fallback: prova import da src.pipeline_module
    from src.pipeline_module import get_pipeline

df = pd.read_csv(val_csv)
target = "Class"
if target not in df.columns:
    raise SystemExit(f"Colonna target '{target}' non trovata. Colonne: {list(df.columns)}")

X_val = df.drop(columns=[target])
y_val = df[target]

pipe = get_pipeline()
if not hasattr(pipe, "predict_proba"):
    raise SystemExit("La pipeline non espone predict_proba.")

proba = pipe.predict_proba(X_val)[:,1]
precisions, recalls, thresholds = precision_recall_curve(y_val, proba)
pr_auc = auc(recalls, precisions)
roc = roc_auc_score(y_val, proba)
print("PR AUC:", pr_auc, "ROC AUC:", roc)

# Plot
plt.figure(figsize=(6,6))
plt.plot(recalls, precisions, label=f"PR curve (AUC={pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve")
plt.grid(True)
plt.legend()
plt.savefig("reports/pr_curve_validation.png", dpi=150, bbox_inches="tight")
print("Saved PR curve -> reports/pr_curve_validation.png")

# create table of candidate thresholds and metrics
# thresholds array has len = len(precisions)-1; align by skipping last precision point
import pandas as pd
df_thr = pd.DataFrame({
    "threshold": np.append(thresholds, np.nan),
    "precision": precisions,
    "recall": recalls
})
# drop last row without threshold or keep for info
df_thr.dropna(subset=["threshold"], inplace=True)
df_thr = df_thr.sort_values(by="threshold", ascending=False).reset_index(drop=True)
df_thr.to_csv("reports/pr_thresholds.csv", index=False)
print("Saved thresholds table -> reports/pr_thresholds.csv")

# show top candidate rows with precision >= some values
for p in [0.99, 0.95, 0.9, 0.8, 0.7]:
    cands = df_thr[df_thr["precision"] >= p]
    if not cands.empty:
        print(f"\nFirst threshold with precision >= {p}:")
        print(cands.iloc[-1].to_dict())   # pick one with high recall among those
    else:
        print(f"\nNo threshold with precision >= {p}")
