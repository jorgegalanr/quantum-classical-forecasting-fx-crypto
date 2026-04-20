# Financial Time Series Forecasting — Classical, Deep Learning & Quantum Models

> **TFG · Grado en Computación e Inteligencia Artificial · Universidad Alfonso X el Sabio · 2026**

Comparativa de cinco familias de modelos de predicción de series temporales sobre retornos diarios de divisas Forex y criptomonedas, con detección de régimen de volatilidad y análisis segmentado por condiciones de mercado.

---

## Motivación

Los mercados financieros plantean uno de los problemas de forecasting más difíciles: series no estacionarias, alta eficiencia de mercado y comportamiento radicalmente distinto según el régimen de volatilidad. Este proyecto evalúa si los modelos de deep learning y machine learning cuántico ofrecen ventaja real sobre los modelos estadísticos clásicos en este contexto, o si la hipótesis de mercado eficiente limita el techo de cualquier modelo.

---

## Activos analizados

| Activo | Registros | Tipo |
|--------|-----------|------|
| EUR/USD | 2.125 días | Forex |
| GBP/USD | 2.125 días | Forex |
| USD/JPY | 2.125 días | Forex |
| BTC/USD | 2.991 días | Cripto |
| ETH/USD | 2.991 días | Cripto |

**Fuente:** Yahoo Finance (vía `yfinance`) · **Período:** enero 2018 – marzo 2026

La diferencia entre el número de registros de Forex y cripto (~866 días) corresponde a los fines de semana y festivos bancarios durante el período de estudio: Forex opera de lunes a viernes; las criptomonedas cotizan los 365 días del año.

---

## Pipeline

```
01_data_exploration.ipynb   →  Descarga, limpieza y detección de régimen
02_arima.ipynb              →  Baseline clásico (ARIMA con selección por AIC)
03_Prophet.ipynb            →  Modelo aditivo de Meta
04_lstm.ipynb               →  Red recurrente LSTM (PyTorch)
05_tft.ipynb                →  Temporal Fusion Transformer (PyTorch)
06_qnn.ipynb                →  Quantum Neural Network (PennyLane + PyTorch)
```

---

## Detección de régimen de volatilidad

Etiquetado no supervisado basado en la volatilidad rolling anualizada de 20 días, con umbral en el percentil 70 de la volatilidad histórica de cada activo:

| Activo | Umbral vol. anualizada | % tiempo alta vol. |
|--------|------------------------|-------------------|
| EUR/USD | ~7,6% | 30% |
| BTC | ~55,2% | 30% |
| ETH | ~73,7% | 30% |

> El umbral de BTC es aproximadamente 7× mayor que el de EUR/USD, reflejando la diferencia estructural entre mercados regulados maduros y el ecosistema cripto.

---

## Modelos implementados

### Clásicos
- **ARIMA** — selección automática de orden (p,d,q) mediante minimización del criterio de Akaike (AIC). Test ADF previo de estacionariedad sobre retornos.
- **Prophet** — modelo aditivo de Meta con componentes de tendencia, estacionalidad y efectos puntuales.

### Deep Learning
- **LSTM** — 2 capas apiladas, 64 unidades ocultas, dropout 0,2, ventana deslizante de 20 días (~50.000 parámetros).
- **TFT** — Temporal Fusion Transformer con LSTM encoder, atención multi-cabeza (4 heads) y Gated Residual Networks (~80.000 parámetros).

### Cuántico / Híbrido
- **QNN** — Circuito cuántico variacional con 6 qubits, 2 capas variacionales (rotaciones RX/RY/RZ + CNOT en topología circular), encoding RY y medida Pauli-Z sobre el primer qubit. Integración PennyLane + PyTorch (<100 parámetros).

---

## Resultados — MAE global por modelo y activo

| Activo | ARIMA | Prophet | LSTM | TFT | QNN | **Mejor** |
|--------|-------|---------|------|-----|-----|-----------|
| EUR/USD | 0,003498 | 0,003505 | 0,003478 | **0,003451** | 0,003528 | TFT |
| GBP/USD | 0,003611 | 0,003611 | 0,003576 | **0,003555** | 0,003659 | TFT |
| USD/JPY | **0,004947** | 0,004954 | 0,005065 | 0,005004 | 0,004976 | ARIMA |
| BTC | 0,017434 | 0,017453 | **0,017241** | 0,017321 | 0,017890 | LSTM |
| ETH | 0,026968 | 0,027030 | **0,026938** | 0,026956 | 0,028034 | LSTM |

### MAE en régimen de alta volatilidad

| Activo | ARIMA | Prophet | LSTM | TFT | QNN | **Mejor** |
|--------|-------|---------|------|-----|-----|-----------|
| EUR/USD | 0,004317 | 0,004324 | **0,004142** | 0,004162 | 0,004246 | LSTM |
| GBP/USD | 0,004707 | 0,004705 | 0,004381 | **0,004352** | 0,004445 | TFT |
| USD/JPY | **0,005722** | 0,005740 | 0,006040 | 0,005913 | 0,005827 | ARIMA |
| BTC | 0,029341 | 0,029372 | **0,027258** | 0,027899 | 0,027809 | LSTM |
| ETH | 0,034948 | 0,035288 | **0,033984** | 0,035446 | 0,036527 | LSTM |

---

## Conclusiones principales

**1. No existe un modelo universalmente superior**
El mejor modelo depende tanto del activo como del régimen de mercado. Resultado consistente con el *no free lunch theorem* y con la experiencia acumulada en competiciones de forecasting (M5, M6).

**2. Eficiencia del mercado Forex**
ARIMA encuentra orden óptimo (0,0,0) en los tres pares de divisas, lo que equivale a predecir la media histórica (cero para retornos centrados). Este resultado es una manifestación empírica directa de la hipótesis de mercado eficiente en forma débil: los retornos diarios son estadísticamente indistinguibles del ruido blanco. Las diferencias entre modelos quedan relegadas al quinto decimal.

**3. TFT se impone en Forex de baja volatilidad; ARIMA resiste en USD/JPY**
El mecanismo de atención temporal del TFT detecta pequeñas señales sutiles en EUR/USD y GBP/USD. En USD/JPY, el cambio estructural de régimen a partir de 2022 favorece la simplicidad de ARIMA frente a modelos más complejos.

**4. LSTM lidera en criptomonedas**
En BTC y ETH, la LSTM supera tanto a los modelos clásicos como al TFT, especialmente en régimen de alta volatilidad en BTC (+7% de mejora sobre ARIMA). La no-linealidad pronunciada del mercado cripto es donde el deep learning aporta valor real. El TFT muestra cierta propensión al sobreajuste en series univariantes cortas.

**5. QNN — paridad con muchos menos parámetros**
La QNN no gana en ningún activo, pero se mantiene dentro del 3-4% del mejor modelo operando con menos de 100 parámetros entrenables frente a los ~80.000 de LSTM o TFT. El resultado sugiere una posible eficiencia estructural, aunque con tres limitaciones relevantes: mayor coste computacional (10-20 min por cada 10 epochs de cada activo vs 2-5 min para LSTM), dependencia de simulador ideal (sin ruido de hardware NISQ real) y menor expresividad efectiva con 6 qubits.

**6. Diferencia estructural Forex vs Cripto**
El MAE de cripto es 5-8× superior al de Forex (BTC ≈ 0,0174 vs EUR/USD ≈ 0,0035), reflejo directo de la diferencia de volatilidad entre ambos mercados.

**7. Degradación universal en alta volatilidad**
Todos los modelos deterioran su rendimiento en el régimen de alta volatilidad, con incrementos de MAE que oscilan entre el 25% y el 80% según el activo. Ningún modelo es inmune a los shocks de mercado.

---

## Stack tecnológico

```
Python 3
PyTorch 2.x
PennyLane 0.44.1
Prophet (Meta)
statsmodels       → ARIMA
scikit-learn      → preprocesamiento y métricas
yfinance          → descarga de datos
pandas · numpy · matplotlib
```

---

## Estructura del repositorio

```
├── data/                                      # Datos procesados (generados por notebook 01)
│   ├── eurusd_raw.csv
│   ├── eurusd_processed.csv
│   └── (...) resto de activos
├── notebooks/
│   ├── 01_data_exploration.ipynb              # Descarga, limpieza, regímenes
│   ├── 02_arima.ipynb
│   ├── 03_Prophet.ipynb
│   ├── 04_lstm.ipynb
│   ├── 05_tft.ipynb                           # Implementación PyTorch desde cero
│   └── 06_qnn.ipynb                           # PennyLane + PyTorch
├── results/
│   ├── 01_series_historicas.png
│   ├── 02_retornos_volatilidad.png
│   ├── 03_regimenes_volatilidad.png
│   ├── 04_comparativa_arima_prophet.png
│   ├── 05_comparativa_clasicos_lstm.png
│   ├── 06_comparativa_todos_modelos.png
│   ├── 07_comparativa_final_todos_modelos.png
│   ├── arima_resultados.csv
│   ├── prophet_resultados.csv
│   ├── lstm_resultados.csv
│   ├── tft_resultados.csv
│   ├── qnn_resultados.csv
│   └── comparativa_final_todos_modelos.csv
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

# Ejecutar los notebooks en orden
jupyter notebook
```

> Los notebooks deben ejecutarse en orden secuencial. El notebook 01 genera los CSV procesados en `data/` que consumen los notebooks 02 a 06. El notebook 06 (QNN) es el más costoso computacionalmente: aproximadamente 10-20 minutos cada 10 epochs por activo al ejecutarse sobre simulador.

---

## Reproducibilidad

Todos los experimentos se ejecutan con semillas fijadas (`torch.manual_seed(42)`, `np.random.seed(42)`) para garantizar la reproducibilidad de los resultados entre ejecuciones en la misma máquina.

---

## Autor

**Jorge Galán Rodríguez**
[LinkedIn](https://linkedin.com/in/jorgegalanrodriguez) · [GitHub](https://github.com/jorgegalanr)
