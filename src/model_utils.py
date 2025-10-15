import json
from pathlib import Path
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

def save_model_card(dest_path, info: dict):
    """
    Salva un model_card.json con le informazioni passate.
    Esempio info: {"version":"1.0.0","threshold":0.42,"pr_auc":0.78, "notes":"..."}
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2)

def calibrate_and_choose_threshold(model, X_val, y_val,
                                   method='isotonic', cv=5,
                                   min_precision=0.90):
    # Calibrazione
    calibrated_model = model
    if hasattr(model, "predict_proba"):
        try:
            calibrated_model = CalibratedClassifierCV(model, cv=cv, method=method)
            calibrated_model.fit(X_val, y_val)
        except Exception as e:
            # fallback: log e usa modello originale
            print(f"[Warning] Calibrazione fallita: {e}")

    if hasattr(calibrated_model, "predict_proba"):
        y_prob = calibrated_model.predict_proba(X_val)[:, 1]
    elif hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:, 1]
    else:
        raise ValueError("Model non espone predict_proba; impossibile calibrare")

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
    pr_auc = auc(recalls, precisions)
    roc_auc = roc_auc_score(y_val, y_prob)

    # scelta threshold: last threshold con precision >= min_precision
    idxs = np.where(precisions >= min_precision)[0]
    if len(idxs):
        chosen_idx = idxs[-1]
    else:
        # fallback: massimizza F1
        f1 = 2 * precisions * recalls / (precisions + recalls + 1e-12)
        chosen_idx = int(np.nanargmax(f1))

    # assicurati che chosen_idx sia valido per thresholds
    chosen_idx = np.clip(chosen_idx, 0, len(thresholds)-1)
    chosen_thr = float(thresholds[chosen_idx])

    return {
        "threshold": chosen_thr,
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "chosen_idx": int(chosen_idx),
        "precisions": precisions.tolist(),
        "recalls": recalls.tolist(),
        "thresholds": thresholds.tolist(),
        "calibrated_model": calibrated_model
    }
