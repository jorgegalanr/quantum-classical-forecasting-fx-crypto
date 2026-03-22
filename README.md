# Financial Time Series Forecasting — Classical, Deep Learning & Quantum Models

> **TFG · Grado en Ciencia de Datos e IA · Universidad Alfonso X el Sabio · 2026**

Comparativa exhaustiva de cinco familias de modelos de forecasting sobre retornos diarios de divisas FX y criptomonedas, con detección de régimen de volatilidad e interpretabilidad por régimen de mercado.

---

## Motivación

Los mercados financieros plantean uno de los problemas de forecasting más difíciles: series no estacionarias, alta eficiencia de mercado y comportamiento radicalmente distinto según el régimen de volatilidad. Este proyecto evalúa si los modelos de deep learning y quantum ML ofrecen ventaja real sobre los modelos clásicos en este contexto, o si la hipótesis de mercado eficiente limita el techo de cualquier modelo.

---

## Activos analizados

| Activo | Registros | Tipo |
|--------|-----------|------|
| EUR/USD | 2.134 días | Forex |
| GBP/USD | 2.134 días | Forex |
| USD/JPY | 2.135 días | Forex |
| BTC/USD | 2.995 días | Crypto |
| ETH/USD | 2.995 días | Crypto |

Fuente: Yahoo Finance · Período: 2016–2025

---

## Pipeline

```
01_data_exploration.ipynb   →  Descarga, limpieza, detección de régimen
02_arima.ipynb              →  Baseline clásico (ARIMA auto-order)
03_Prophet.ipynb            →  Modelo aditivo Meta Prophet
04_lstm.ipynb               →  Red recurrente LSTM (PyTorch)
05_tft.ipynb                →  Temporal Fusion Transformer (PyTorch)
06_qnn.ipynb                →  Quantum Neural Network (PennyLane + PyTorch)
```

---

## Detección de régimen de volatilidad

Etiquetado no supervisado basado en volatilidad rolling anualizada de 20 días con umbral en el percentil 70:

| Activo | Umbral vol. anualizada | % tiempo alta vol. |
|--------|------------------------|-------------------|
| EUR/USD | 7.58% | 30% |
| GBP/USD | 8.75% | 30% |
| USD/JPY | 9.45% | 30% |
| BTC | 55.0% | 30% |
| ETH | ~65% | 30% |

> El umbral de BTC es ~7× mayor que el de EUR/USD, reflejando la diferencia estructural entre mercados regulados y cripto.

---

## Modelos implementados

### Clásicos
- **ARIMA** — selección automática de orden (p,d,q) por AIC. Test ADF previo de estacionariedad.
- **Prophet** — modelo aditivo de Meta con componentes de tendencia y estacionalidad.

### Deep Learning
- **LSTM** — red recurrente con dependencias temporales no lineales de largo plazo (PyTorch 2.10).
- **TFT** — Temporal Fusion Transformer con mecanismo de atención temporal y Gated Residual Networks.

### Quantum / Híbrido
- **QNN** — circuito cuántico variacional (VQC) con encoding angular (RX, RY), 6 qubits, 2 capas variacionales, combinado con capas clásicas (PennyLane 0.44 + PyTorch).

---

## Resultados — MAE global por modelo y activo

| Activo | ARIMA | Prophet | LSTM | TFT | QNN | **Mejor** |
|--------|-------|---------|------|-----|-----|-----------|
| EUR/USD | 0.003458 | 0.003465 | 0.003466 | **0.003445** | 0.003579 | TFT |
| GBP/USD | **0.003538** | 0.003540 | 0.003557 | 0.003539 | 0.003519 | ARIMA |
| USD/JPY | **0.004960** | 0.004976 | 0.005211 | 0.005016 | 0.005092 | ARIMA |
| BTC | 0.017535 | 0.017641 | 0.017445 | **0.017387** | 0.017420 | TFT |
| ETH | 0.027143 | **0.027143** | 0.027200 | 0.027300 | 0.027500 | Prophet |

### MAE en régimen de alta volatilidad

| Activo | ARIMA | Prophet | LSTM | TFT | QNN |
|--------|-------|---------|------|-----|-----|
| EUR/USD | 0.004339 | 0.004344 | 0.004225 | **0.004186** | 0.004277 |
| GBP/USD | 0.004753 | 0.004742 | 0.004482 | **0.004445** | — |
| USD/JPY | 0.005787 | 0.005813 | 0.006149 | **0.005881** | — |
| BTC | 0.026445 | 0.026649 | 0.029049 | 0.029601 | — |

---

## Conclusiones principales

**1. Eficiencia de mercado Forex**
ARIMA encuentra orden óptimo (0,0,0) en los tres pares: los retornos diarios son estadísticamente indistinguibles del ruido blanco (ADF p-valor = 0.000 en todos los activos). Ningún modelo encuentra estructura explotable consistente.

**2. TFT gana en EUR/USD y BTC**
El mecanismo de atención temporal de TFT encuentra patrones que LSTM no captura, especialmente en USD/JPY (+3.82% de mejora sobre LSTM) y en regímenes de alta volatilidad en Forex.

**3. Deep learning no supera a clásicos en Forex**
LSTM obtiene resultados ligeramente peores que ARIMA en las tres divisas. La complejidad añadida no compensa en mercados eficientes.

**4. Cripto vs Forex — diferencia estructural**
El MAE de cripto es 5–8× superior al de Forex (BTC MAE ≈ 0.0174 vs EUR/USD ≈ 0.0035). La mayor predictibilidad relativa de BTC/ETH permite que TFT y LSTM muestren ventaja sobre los clásicos en periodos de baja volatilidad.

**5. QNN — rendimiento competitivo con limitaciones**
La QNN obtiene el mejor resultado global en GBP/USD (MAE=0.003519), superando a todos los modelos clásicos y de DL. Sin embargo, el coste computacional (~10–20 min vs segundos para ARIMA) y la limitación a simulación clásica de circuitos cuánticos (PennyLane `default.qubit`) hacen que la ventaja práctica sea actualmente marginal.

---

## Stack tecnológico

```
Python 3.11
PyTorch 2.10
PennyLane 0.44
Prophet (Meta)
statsmodels (ARIMA)
pandas · numpy · matplotlib · scikit-learn
```

---

## Estructura del repositorio

```
├── data/                        # Datos procesados (no incluidos en repo)
├── results/                     # Gráficas y comparativas exportadas
│   ├── 01_series_precio.png
│   ├── 02_retornos_volatilidad.png
│   ├── 03_regimenes_volatilidad.png
│   ├── 04_arima_predicciones.png
│   ├── 05_comparativa_clasicos_lstm.png
│   ├── 06_comparativa_todos_modelos.png
│   └── 07_comparativa_final_todos_modelos.png
├── 01_data_exploration.ipynb
├── 02_arima.ipynb
├── 03_Prophet.ipynb
├── 04_lstm.ipynb
├── 05_tft.ipynb
├── 06_qnn.ipynb
└── README.md
```

---

## Cómo ejecutar

```bash
# Clonar el repositorio
git clone https://github.com/jorgegalanr/quantum-classical-forecasting-fx-crypto
cd quantum-classical-forecasting-fx-crypto

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar en orden
jupyter notebook 01_data_exploration.ipynb
# ... continuar con 02, 03, 04, 05, 06
```

> Los notebooks deben ejecutarse en orden: el pipeline de datos (01) genera los CSV procesados que usan los modelos.

---

## Autor

**Jorge Galán Rodríguez**
Junior Data Scientist | Financial Analytics & Risk Modeling
[LinkedIn](https://linkedin.com/in/jorgegalanrodriguez) · [GitHub](https://github.com/jorgegalanr)
