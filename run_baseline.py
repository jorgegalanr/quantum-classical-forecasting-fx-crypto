from pathlib import Path

import numpy as np
import pandas as pd

from src.temporal import chronological_return_split, regression_metrics


ASSETS = ("eurusd", "gbpusd", "usdjpy", "btc", "eth")
DATA_DIR = Path("data")
OUTPUT = Path("results/zero_baseline_resultados.csv")


def run() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for asset in ASSETS:
        frame = pd.read_csv(
            DATA_DIR / f"{asset}_processed.csv", index_col=0, parse_dates=True
        )
        split = chronological_return_split(frame)
        predictions = np.zeros(len(split.test))
        metrics = regression_metrics(split.test.to_numpy(), predictions)
        row: dict[str, float | str] = {
            "Activo": asset.upper(),
            "MAE_global": round(metrics["MAE"], 6),
            "RMSE_global": round(metrics["RMSE"], 6),
        }
        for regime, suffix in (("Baja", "baja_vol"), ("Alta", "alta_vol")):
            mask = split.test_regime == regime
            row[f"MAE_{suffix}"] = round(
                regression_metrics(split.test.to_numpy()[mask], predictions[mask])["MAE"],
                6,
            )
        rows.append(row)

    result = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    run()
