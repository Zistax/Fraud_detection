import os
from pathlib import Path
import joblib
import logging
from threading import Lock

logger = logging.getLogger("fraud_batch.pipeline")
logger.addHandler(logging.NullHandler())

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "pipeline.pkl"
MODEL_PATH = Path(os.environ.get("MODEL_PATH") or os.environ.get("MODEL_FILE") or DEFAULT_MODEL_PATH)

_pipeline = None
_pipeline_lock = Lock()

def _load_pipeline(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model file non trovato: {path}")
    try:
        pipeline = joblib.load(path)
        logger.info("Pipeline caricata da: %s", path)
        return pipeline
    except Exception:
        logger.exception("Errore caricamento pipeline")
        raise

def get_pipeline(force_reload: bool = False):
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None or force_reload:
            try:
                if MODEL_PATH.exists():
                    _pipeline = _load_pipeline(MODEL_PATH)
                elif DEFAULT_MODEL_PATH.exists():
                    _pipeline = _load_pipeline(DEFAULT_MODEL_PATH)
                else:
                    raise FileNotFoundError(f"Nessun modello trovato in {MODEL_PATH} o {DEFAULT_MODEL_PATH}")
            except Exception:
                logger.exception("Non sono riuscito a caricare la pipeline")
                raise
        else:
            logger.debug("Usando pipeline già caricata")
    return _pipeline

if __name__ == "__main__":
    print(f"pipeline_module file: {__file__}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DEFAULT_MODEL_PATH: {DEFAULT_MODEL_PATH}")
    print(f"MODEL_PATH (effettivo): {MODEL_PATH}")
    print(f"MODEL EXISTS?: {MODEL_PATH.exists()}")
