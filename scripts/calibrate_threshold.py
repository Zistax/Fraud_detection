# scripts/calibrate_threshold.py
"""
Script per calibrare il modello ed esportare model_card.json con la soglia operativa.
Usa:
  - pipeline salvata accessibile tramite src.pipeline_module.get_pipeline() o pipeline_module.get_pipeline()
  - file di validazione CSV in data/validation.csv con colonna target 'Class'

Esempio:
  python scripts/calibrate_threshold.py
"""

import json
from pathlib import Path
import pandas as pd
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT.joinpath("src")
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# prova import relativo (metti model_utils in src/ o nella root)
try:
    # preferisci src.model_utils se esiste
    from src.model_utils import calibrate_and_choose_threshold
except Exception:
    try:
        from model_utils import calibrate_and_choose_threshold
    except Exception as e:
        print("Impossibile importare model_utils. Assicurati di avere src/model_utils.py o model_utils.py nel project root.")
        print("Errore:", e)
        sys.exit(1)

# import pipeline loader
try:
    from src.pipeline_module import get_pipeline
except Exception:
    try:
        from pipeline_module import get_pipeline
    except Exception as e:
        print("Impossibile importare pipeline_module. Assicurati che pipeline_module.py sia in src/ o project root.")
        print("Errore:", e)
        sys.exit(1)

def main(validation_csv: Path, out_model_card: Path, min_precision: float = 0.90, method: str = "isotonic", cv: int = 5):
    print("Caricamento pipeline...")
    pipe = get_pipeline()
    if pipe is None:
        print("Pipeline non disponibile. Esci.")
        return

    if not validation_csv.exists():
        print(f"Validation CSV non trovato: {validation_csv}")
        return

    print("Caricamento validation CSV:", validation_csv)
    df = pd.read_csv(validation_csv)
    # default target name: 'Class' (modifica se necessario)
    target_col = "Class"
    if target_col not in df.columns:
        print(f"Colonna target '{target_col}' non trovata in {validation_csv}. Colonne disponibili: {list(df.columns)}")
        return

    y_val = df[target_col]
    X_val = df.drop(columns=[target_col])

    print("Eseguo calibrazione e scelta soglia (min_precision=%s, method=%s, cv=%s)..." % (min_precision, method, cv))
    res = calibrate_and_choose_threshold(pipe, X_val, y_val, method=method, cv=cv, min_precision=min_precision)

    thr = res.get("threshold")
    pr_auc = res.get("pr_auc")
    roc_auc = res.get("roc_auc")
    print("Threshold scelto:", thr)
    print("PR-AUC:", pr_auc, "ROC-AUC:", roc_auc)

    model_card = {
        "version": "1.0.0",
        "threshold": thr,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "created_at": pd.Timestamp.now().isoformat()
    }
    out_model_card.parent.mkdir(parents=True, exist_ok=True)
    with open(out_model_card, "w", encoding="utf-8") as fh:
        json.dump(model_card, fh, indent=2)
    print("Model card salvata in", out_model_card)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibra modello e salva model_card.json (threshold).")
    parser.add_argument("--validation-csv", "-v", default="data/validation.csv",
                        help="Path al CSV di validazione (default: data/validation.csv). Deve contenere colonna 'Class'.")
    parser.add_argument("--out", "-o", default="model/model_card.json",
                        help="Path di output per model_card.json (default: model/model_card.json)")
    parser.add_argument("--min-precision", "-p", type=float, default=0.90,
                        help="Precision minima richiesta quando si sceglie la soglia (default: 0.90).")
    parser.add_argument("--method", "-m", default="isotonic",
                        help="Metodo di calibrazione per CalibratedClassifierCV (isotonic o sigmoid).")
    parser.add_argument("--cv", type=int, default=5, help="Numero di fold CV per CalibratedClassifierCV (default: 5).")

    args = parser.parse_args()
    main(validation_csv=Path(args.validation_csv), out_model_card=Path(args.out),
         min_precision=args.min_precision, method=args.method, cv=args.cv)
