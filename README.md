# Financial Time Series Forecasting — Classical, Deep Learning & Hybrid QNN

Comparación reproducible de modelos para anticipar el retorno diario de tres pares Forex y dos criptomonedas. El proyecto estudia una pregunta deliberadamente exigente: si arquitecturas cada vez más complejas aportan valor fuera de muestra frente a una referencia tan sencilla como predecir retorno cero.

> Proyecto académico del Grado en Computación e Inteligencia Artificial. No constituye asesoramiento financiero ni un sistema de trading.

## Pregunta de investigación

> ¿Mejoran ARIMA, Prophet, una LSTM, una red temporal inspirada en TFT y una QNN híbrida el error de una predicción de retorno cero, y cambia el resultado durante periodos de alta volatilidad?

La referencia de retorno cero es imprescindible: en series de retornos próximas a ruido, un modelo complejo puede parecer competitivo frente a otros modelos y aun así no superar una regla trivial.

## Activos y datos

| Activo | Mercado | Frecuencia |
|---|---|---|
| EUR/USD | Forex | Días de cotización |
| GBP/USD | Forex | Días de cotización |
| USD/JPY | Forex | Días de cotización |
| BTC/USD | Cripto | Diaria |
| ETH/USD | Cripto | Diaria |

- Fuente: Yahoo Finance mediante `yfinance`.
- Periodo versionado: enero de 2018 — marzo de 2026.
- Variable objetivo: variación porcentual diaria calculada con `pct_change()`.
- La distinta cantidad de observaciones responde a que las criptomonedas cotizan también en fines de semana.

Los CSV versionados permiten reproducir el análisis sin depender de una nueva descarga. El notebook 01 permite actualizar los datos de origen.

## Protocolo temporal

La evaluación corregida utiliza las mismas reglas para todos los modelos:

1. Se eliminan únicamente las filas iniciales sin retorno o volatilidad de 20 días.
2. Se reserva cronológicamente el último 20 % de cada activo como test.
3. El escalador de las redes se ajusta exclusivamente con el 80 % de entrenamiento.
4. El umbral de alta volatilidad es el percentil 70 calculado solo sobre entrenamiento.
5. La ventana de las redes contiene 20 retornos anteriores y predice el día siguiente.
6. Los parámetros se congelan durante test; las ventanas rolling sí incorporan cada retorno real una vez observado.
7. MAE y RMSE sustituyen a MAPE, que no es adecuada para retornos cercanos a cero o con cambio de signo.

`src/temporal.py` centraliza estas reglas y las pruebas verifican que ni el escalador ni el umbral consulten el periodo de test.

## Modelos

| Modelo | Papel en el experimento |
|---|---|
| Retorno cero | Baseline principal |
| ARIMA | Modelo estadístico con selección `(p,0,q)` por AIC dentro de train |
| Prophet | Modelo aditivo entrenado únicamente con train |
| LSTM | Red recurrente de dos capas y ventana de 20 días |
| Red inspirada en TFT | LSTM encoder, atención temporal y GRN simplificada |
| QNN híbrida | Compresión clásica, circuito variacional y salida clásica |

La red denominada anteriormente “TFT” es una implementación educativa inspirada en algunos componentes del Temporal Fusion Transformer. No incluye la arquitectura TFT completa —por ejemplo, selección de variables, covariables conocidas y variables estáticas— y por eso se presenta con un nombre más preciso.

La QNN utiliza:

- 6 qubits y 2 capas variacionales;
- rotaciones `RX`, `RY` y `RZ` y entrelazamiento circular con `CNOT`;
- medición `PauliZ` de los 6 qubits;
- 169 parámetros entrenables: 36 cuánticos y 133 clásicos;
- simulador ideal `default.qubit`, no hardware cuántico real.

El objetivo es comparar un modelo híbrido pequeño, no demostrar ventaja cuántica.

## Resultados y regeneración

El baseline corregido puede regenerarse rápidamente:

```bash
python run_baseline.py
```

| Activo | MAE retorno cero | RMSE retorno cero |
|---|---:|---:|
| EUR/USD | 0,003498 | 0,004751 |
| GBP/USD | 0,003603 | 0,004639 |
| USD/JPY | 0,004869 | 0,006302 |
| BTC | 0,017193 | 0,024301 |
| ETH | 0,026714 | 0,038019 |

Las métricas comparativas anteriores se generaron con un escalador ajustado antes de separar train y test. Los notebooks 02–06 ya contienen el protocolo corregido, por lo que esos experimentos deben ejecutarse de nuevo antes de publicar una tabla definitiva de modelos. Esta decisión evita conservar resultados que ya no corresponden al código actual.

## Notebooks

```text
01_data_exploration.ipynb  Descarga, retornos, volatilidad y análisis descriptivo
02_arima.ipynb             ARIMA rolling y baseline de retorno cero
03_Prophet.ipynb           Prophet
04_lstm.ipynb              LSTM
05_tft.ipynb               Red temporal inspirada en TFT
06_qnn.ipynb               QNN híbrida y comparación final
```

Los notebooks deben ejecutarse en orden. Los modelos neuronales y, especialmente, la QNN pueden requerir bastante tiempo en CPU.

## Instalación

Probado originalmente con Python 3.11 y PyTorch CPU.

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m jupyter notebook
```

## Pruebas rápidas

Las pruebas de prevención de fuga no requieren instalar PyTorch, Prophet o PennyLane:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
python run_baseline.py
```

GitHub Actions valida las funciones temporales, la sintaxis de los notebooks y la compilación de los módulos en cada Pull Request.

## Limitaciones

- Una única división 80/20 no cuantifica la variabilidad entre diferentes periodos de mercado.
- Las redes se entrenan con una sola semilla y sin búsqueda sistemática de hiperparámetros.
- No se incluyen costes de transacción, *slippage* ni una regla de trading; menor MAE no equivale a rentabilidad.
- El régimen es una segmentación descriptiva basada en volatilidad, no un estado de mercado observable con certeza anticipada.
- La red inspirada en TFT no es una reproducción completa de la arquitectura original.
- La QNN se ejecuta en simulador ideal y no demuestra ventaja cuántica.
- Diferencias pequeñas de MAE no permiten afirmar superioridad sin validación temporal repetida e intervalos de incertidumbre.

## Estructura

```text
.
├── data/                    Datos descargados y procesados
├── notebooks/               Experimentos secuenciales
├── results/                 Métricas y figuras generadas
├── src/temporal.py          Split, escalado y regímenes sin fuga
├── tests/                   Controles metodológicos
├── run_baseline.py          Baseline reproducible
└── requirements*.txt        Dependencias completas y de pruebas
```

## Autor

Jorge Galán Rodríguez — [GitHub](https://github.com/jorgegalanr) · [LinkedIn](https://linkedin.com/in/jorgegalanrodriguez)

## Licencia

MIT.
