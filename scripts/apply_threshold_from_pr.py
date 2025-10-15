# scripts/apply_threshold_from_pr.py
import argparse, json
from pathlib import Path
import pandas as pd

def choose_threshold_from_csv(csv_path: Path, min_precision:float):
    df = pd.read_csv(csv_path)
    # assicuriamoci che ci sia colonna 'threshold','precision','recall'
    df = df.dropna(subset=['threshold'])
    # ordina per threshold decrescente per coerenza con inspect_pr
    df = df.sort_values(by='threshold', ascending=False).reset_index(drop=True)
    cand = df[df['precision'] >= min_precision]
    if cand.empty:
        return None
    # tra i candidati scegli quello con max recall
    best = cand.loc[cand['recall'].idxmax()]
    return float(best['threshold']), float(best['precision']), float(best['recall'])

def update_model_card(model_card_path: Path, threshold: float, extra: dict = None):
    model_card_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if model_card_path.exists():
        try:
            data = json.load(open(model_card_path, 'r', encoding='utf-8'))
        except Exception:
            data = {}
    data['threshold'] = threshold
    if extra:
        data.update(extra)
    json.dump(data, open(model_card_path, 'w', encoding='utf-8'), indent=2)
    return data

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pr-csv", default="reports/pr_thresholds.csv")
    p.add_argument("--min-precision", type=float, default=0.80)
    p.add_argument("--model-card", default="model/model_card.json")
    args = p.parse_args()

    csv_path = Path(args.pr_csv)
    if not csv_path.exists():
        print("CSV thresholds non trovato:", csv_path)
        raise SystemExit(1)

    res = choose_threshold_from_csv(csv_path, args.min_precision)
    if res is None:
        print(f"Nessuna soglia trovata con precision >= {args.min_precision}")
        raise SystemExit(2)

    thr, prec, rec = res
    print(f"Scelta soglia: {thr} (precision={prec:.3f}, recall={rec:.3f})")
    mc = update_model_card(Path(args.model_card), thr, extra={"chosen_by": f"precision>={args.min_precision}"})
    print("Model card aggiornata in", args.model_card)
