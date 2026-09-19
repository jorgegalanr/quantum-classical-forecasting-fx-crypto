from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ReturnSplit:
    train: pd.Series
    test: pd.Series
    test_regime: np.ndarray
    volatility_threshold: float


@dataclass(frozen=True)
class SequenceSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    test_regime: np.ndarray
    test_index: pd.DatetimeIndex
    scaler: StandardScaler
    volatility_threshold: float


def _aligned_returns(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"Retorno", "Volatilidad_20d"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")

    aligned = frame.loc[:, ["Retorno", "Volatilidad_20d"]].dropna().copy()
    if not isinstance(aligned.index, pd.DatetimeIndex):
        raise TypeError("El índice debe ser DatetimeIndex")
    if not aligned.index.is_monotonic_increasing:
        aligned = aligned.sort_index()
    return aligned


def chronological_return_split(
    frame: pd.DataFrame, train_fraction: float = 0.8
) -> ReturnSplit:
    """Divide retornos en orden temporal y define regímenes solo con train."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction debe estar entre 0 y 1")

    aligned = _aligned_returns(frame)
    split = int(len(aligned) * train_fraction)
    if split < 2 or split >= len(aligned):
        raise ValueError("La división no deja observaciones suficientes")

    threshold = float(aligned["Volatilidad_20d"].iloc[:split].quantile(0.70))
    test_regime = np.where(
        aligned["Volatilidad_20d"].iloc[split:].to_numpy() > threshold,
        "Alta",
        "Baja",
    )
    return ReturnSplit(
        train=aligned["Retorno"].iloc[:split],
        test=aligned["Retorno"].iloc[split:],
        test_regime=test_regime,
        volatility_threshold=threshold,
    )


def prepare_scaled_sequences(
    frame: pd.DataFrame, window: int = 20, train_fraction: float = 0.8
) -> SequenceSplit:
    """Crea secuencias y ajusta el escalador exclusivamente con train."""
    if window < 1:
        raise ValueError("window debe ser positivo")

    aligned = _aligned_returns(frame)
    split = int(len(aligned) * train_fraction)
    if split <= window or split >= len(aligned):
        raise ValueError("La división no es compatible con la ventana")

    values = aligned["Retorno"].to_numpy(dtype=float).reshape(-1, 1)
    scaler = StandardScaler().fit(values[:split])
    scaled = scaler.transform(values).ravel()

    x = np.asarray([scaled[i - window : i] for i in range(window, len(scaled))])
    y = scaled[window:]
    target_positions = np.arange(window, len(scaled))
    train_mask = target_positions < split

    threshold = float(aligned["Volatilidad_20d"].iloc[:split].quantile(0.70))
    test_volatility = aligned["Volatilidad_20d"].iloc[target_positions[~train_mask]]
    test_regime = np.where(test_volatility.to_numpy() > threshold, "Alta", "Baja")

    return SequenceSplit(
        x_train=x[train_mask],
        y_train=y[train_mask],
        x_test=x[~train_mask],
        y_test=y[~train_mask],
        test_regime=test_regime,
        test_index=aligned.index[target_positions[~train_mask]],
        scaler=scaler,
        volatility_threshold=threshold,
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(np.mean(np.abs(true - pred))),
        "RMSE": float(np.sqrt(np.mean((true - pred) ** 2))),
    }

