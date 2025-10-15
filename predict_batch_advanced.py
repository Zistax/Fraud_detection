import os
import sys
import time
import logging
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# optional heavy import: seaborn only used if correlation plot needed
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

# -----------------------
# PATH / CONFIG bootstrap
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT.joinpath("src")
DATA_DIR = PROJECT_ROOT.joinpath("data", "new")
PREDICTIONS_DIR = PROJECT_ROOT.joinpath("data", "predictions")
REPORT_DIR = PROJECT_ROOT.joinpath("reports")
LOGS_DIR = PROJECT_ROOT.joinpath("logs")
SUMMARY_CSV = REPORT_DIR.joinpath("batch_summary.csv")

for d in (PREDICTIONS_DIR, REPORT_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ensure project and src are on path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# -----------------------
# local imports (after sys.path patch)
# -----------------------
from pipeline_module import get_pipeline
from shap_utils import generate_shap_report, save_detected_frauds_to_excel

# -----------------------
# Config
# -----------------------
DO_SHAP = True
MAX_ROWS_EXCEL = 50

# -----------------------
# Logger
# -----------------------
def setup_logger(name: str = "fraud_batch") -> Tuple[logging.Logger, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = LOGS_DIR.joinpath(f"batch_run_{ts}.log")
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(logfile) for h in logger.handlers):
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger, logfile

logger, LOGFILE_PATH = setup_logger()
logger.info("Logger inizializzato. Log file: %s", LOGFILE_PATH)

# -----------------------
# Summary CSV helper
# -----------------------
SUMMARY_FIELDS = [
    "run_time", "input_file", "batch_name", "n_rows", "n_frauds",
    "duration_s", "shap_generated", "excel_path",
    "error_message", "fraud_mean_proba", "top_fraud_features", "model_version"
]

def append_summary_record(record: Dict[str, Any]):
    try:
        write_header = not SUMMARY_CSV.exists()
        with SUMMARY_CSV.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(record)
        logger.info("Appended summary record to %s", SUMMARY_CSV)
    except Exception as e:
        logger.exception("Impossibile scrivere summary CSV: %s", e)

# -----------------------
# Load pipeline + model_card
# -----------------------
pipeline = None
model_version = None
model_card: Dict[str, Any] = {}
try:
    pipeline = get_pipeline()
    logger.info("Pipeline caricata")
    mc_path = PROJECT_ROOT.joinpath("model", "model_card.json")
    if mc_path.exists():
        with mc_path.open("r", encoding="utf-8") as fh:
            model_card = json.load(fh)
            model_version = model_card.get("version")
except Exception as e:
    pipeline = None
    logger.exception("Impossibile caricare pipeline: %s", e)

# -----------------------
# Safe CSV reader
# -----------------------
def _read_csv_safe(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except MemoryError:
        logger.warning("MemoryError reading %s; consider running with chunks or more RAM.", path)
        return None
    except Exception:
        logger.exception("Errore lettura CSV %s", path)
        return None

# -----------------------
# Main batch function
# -----------------------
def predict_batch(file_path: str, do_shap: bool = DO_SHAP):
    start_time = time.time()
    run_ts = datetime.now().isoformat()
    input_file = str(file_path)
    batch_name = Path(file_path).stem
    logger.info("Processing file: %s", input_file)

    n_rows = 0
    n_frauds = 0
    shap_generated = False
    shap_summary_png = None
    shap_bar_png = None
    shap_csv = None
    excel_path = ""
    error_message = ""

    df_new = _read_csv_safe(Path(file_path))
    if df_new is None:
        error_message = "read_error"
        duration = time.time() - start_time
        append_summary_record({
            "run_time": run_ts, "input_file": input_file, "batch_name": batch_name,
            "n_rows": n_rows, "n_frauds": n_frauds, "duration_s": round(duration, 2),
            "shap_generated": shap_generated, "excel_path": excel_path,
            "error_message": error_message, "fraud_mean_proba": None, "top_fraud_features": None, "model_version": model_version
        })
        return

    n_rows = len(df_new)
    logger.info("Loaded CSV: %s rows", n_rows)

    # --- predictions ---
    proba = None
    try:
        threshold = model_card.get("threshold") if isinstance(model_card, dict) else None
        if pipeline is None:
            raise RuntimeError("Pipeline non disponibile")

        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(df_new)[:, 1]
            df_new["fraud_proba"] = proba
            if threshold is not None:
                df_new["predicted_fraud"] = (proba >= float(threshold)).astype(int)
            else:
                df_new["predicted_fraud"] = pipeline.predict(df_new)
        else:
            df_new["predicted_fraud"] = pipeline.predict(df_new)

        n_frauds = int(df_new["predicted_fraud"].sum())
        logger.info("Predizioni completate. Frodi predette: %d", n_frauds)
    except Exception as e:
        logger.exception("Errore durante predizione: %s", e)
        error_message = f"predict_error: {e}"
        duration = time.time() - start_time
        append_summary_record({
            "run_time": run_ts, "input_file": input_file, "batch_name": batch_name,
            "n_rows": n_rows, "n_frauds": n_frauds, "duration_s": round(duration, 2),
            "shap_generated": shap_generated, "excel_path": excel_path,
            "error_message": error_message, "fraud_mean_proba": None, "top_fraud_features": None, "model_version": model_version
        })
        return

    # save predictions CSV
    try:
        out_name = f"{batch_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_pred.csv"
        out_path = PREDICTIONS_DIR.joinpath(out_name)
        df_new.to_csv(out_path, index=False)
        logger.info("Saved predictions -> %s", out_path)
    except Exception as e:
        logger.exception("Errore salvataggio predizioni: %s", e)
        error_message = (error_message + " | save_pred_error: " + str(e)) if error_message else f"save_pred_error: {e}"

    # skip SHAP/Excel if no frauds
    if n_frauds == 0:
        logger.info("Nessuna frode predetta per %s: salto SHAP/Excel", input_file)
        duration = time.time() - start_time
        append_summary_record({
            "run_time": run_ts, "input_file": input_file, "batch_name": batch_name,
            "n_rows": n_rows, "n_frauds": n_frauds, "duration_s": round(duration, 2),
            "shap_generated": shap_generated, "excel_path": excel_path,
            "error_message": error_message, "fraud_mean_proba": None, "top_fraud_features": None, "model_version": model_version
        })
        return

    # --- generate SHAP report ---
    try:
        if do_shap:
            logger.info("Generating SHAP report...")
            s_png, b_png, s_csv = generate_shap_report(
                pipeline=pipeline, df=df_new, output_dir=str(REPORT_DIR), batch_name=batch_name
            )
            shap_summary_png, shap_bar_png, shap_csv = s_png, b_png, s_csv
            if shap_summary_png:
                shap_generated = True
                logger.info("SHAP report saved: %s", shap_summary_png)
    except Exception as e:
        logger.exception("Errore generazione SHAP: %s", e)
        error_message = (error_message + " | shap_error: " + str(e)) if error_message else f"shap_error: {e}"

    # --- save Detected Fraud Samples to Excel ---
    try:
        if df_new["predicted_fraud"].sum() > 0:
            excel_path = save_detected_frauds_to_excel(
                df=df_new,
                batch_name=batch_name,
                output_dir=str(REPORT_DIR),
                max_rows=MAX_ROWS_EXCEL
            )
            if excel_path:
                logger.info("Detected fraud samples saved to Excel: %s", excel_path)
    except Exception as e:
        logger.exception("Errore salvataggio fraud samples in Excel: %s", e)
        error_message = (error_message + " | excel_error: " + str(e)) if error_message else f"excel_error: {e}"

    # --- compute fraud_mean_proba ---
    fraud_mean_proba = None
    try:
        if "fraud_proba" in df_new.columns:
            fraud_mean_proba = float(df_new.loc[df_new['predicted_fraud'] == 1, 'fraud_proba'].mean()) if df_new['predicted_fraud'].sum() > 0 else None
    except Exception as e:
        logger.warning("Non posso calcolare fraud_mean_proba: %s", e)
        fraud_mean_proba = None

    # --- top features from SHAP ---
    top_fraud_features = None
    if shap_csv:
        try:
            shap_vals = pd.read_csv(shap_csv, index_col=0)
            shap_vals_fraud = shap_vals.loc[shap_vals.index.intersection(df_new.loc[df_new['predicted_fraud'] == 1].index)]
            if not shap_vals_fraud.empty:
                top_fraud_features = ",".join(shap_vals_fraud.abs().mean().sort_values(ascending=False).head(5).index)
        except Exception as e:
            logger.warning("Non posso calcolare top_fraud_features: %s", e)

    duration = time.time() - start_time
    summary_record = {
        "run_time": run_ts, "input_file": input_file, "batch_name": batch_name,
        "n_rows": n_rows, "n_frauds": n_frauds, "duration_s": round(duration, 2),
        "shap_generated": shap_generated, "excel_path": str(excel_path) if excel_path else "",
        "error_message": error_message,
        "fraud_mean_proba": fraud_mean_proba, "top_fraud_features": top_fraud_features, "model_version": model_version
    }
    append_summary_record(summary_record)
    logger.info("Batch processing finished for %s in %.1fs", input_file, duration)

# -----------------------
# Entrypoint
# -----------------------
if __name__ == "__main__":
    files = sorted([str(p) for p in DATA_DIR.glob("*.csv")])
    if not files:
        logger.info("Nessun file CSV trovato in %s", DATA_DIR)
    for file_path in files:
        try:
            predict_batch(file_path, do_shap=DO_SHAP)
        except Exception as e:
            logger.exception("Errore non gestito durante l'elaborazione di %s: %s", file_path, e)
