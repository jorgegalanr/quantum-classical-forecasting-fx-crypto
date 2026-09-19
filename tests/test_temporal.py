import numpy as np
import pandas as pd

from src.temporal import chronological_return_split, prepare_scaled_sequences


def sample_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "Retorno": np.arange(100, dtype=float),
            "Volatilidad_20d": np.arange(100, dtype=float),
        },
        index=index,
    )


def test_scaler_is_fitted_only_with_training_values():
    prepared = prepare_scaled_sequences(sample_frame(), window=5, train_fraction=0.8)

    assert prepared.scaler.mean_[0] == np.mean(np.arange(80, dtype=float))
    assert prepared.test_index.min() == pd.Timestamp("2024-03-21")
    assert len(prepared.y_test) == 20


def test_regime_threshold_uses_only_training_period():
    split = chronological_return_split(sample_frame(), train_fraction=0.8)

    assert split.volatility_threshold == np.quantile(np.arange(80), 0.70)
    assert len(split.test_regime) == len(split.test)

