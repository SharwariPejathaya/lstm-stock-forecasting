# Forecasting U.S. Stock Returns Using LSTM

> **Course Project — Machine Learning (BITS F464)**  
> Instructor: Prof. Paresh Saxena | April 2025

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

This project replicates and extends the paper *"Regression and Forecasting of U.S. Stock Returns Based on LSTM"*, applying LSTM networks to forecast U.S. stock returns using **Fama-French factor data** spanning **60 years (1965–2024)** across **49 industry portfolios**.

The improved model achieves a **60% RMSE reduction** over the paper's baseline (0.0216 vs 0.0549), outperforming both the original LSTM and an ARIMA benchmark through targeted architectural and feature engineering improvements.

---

## Results Summary

| Model Variant | RMSE | MAE | Notes |
|---|---|---|---|
| Fama-French Regression (baseline) | 0.0108 | — | Strong linear baseline |
| Paper LSTM (baseline) | 0.0549 | — | Weak fit on smoothed returns |
| **Improved LSTM** | **0.0216** | **~0.018** | ✅ Best overall performance |
| Improved LSTM on Real Returns | 0.0491 | 0.0379 | Generalized to raw data |
| ARIMA (1,0,1) | 0.0252 | 0.0198 | R² = -0.16, regresses to mean |
| Time-Split LSTM (2006–24 test) | 0.0226 | 0.0174 | Proved temporal robustness |

---

## Key Improvements Over Paper

| Component | Paper | This Work |
|---|---|---|
| Dataset range | 2004–present | 1965–2024 (full history) |
| Features | All FF5 factors | Mkt-RF, SMB, HML only |
| Return smoothing | None | 3-month moving average |
| Lookback window | Default | 18 months |
| Architecture | Basic LSTM | LSTM + Dropout + Dense |

---

## Project Structure

```
lstm-stock-forecasting/
│
├── notebooks/
│   └── US_Stock_LSTM.ipynb          # Main notebook (LSTM + ARIMA + experiments)
│
├── src/
│   ├── model.py                     # LSTM model architecture
│   ├── data_loader.py               # Data loading & preprocessing
│   ├── train.py                     # Training pipeline
│   └── evaluate.py                  # Evaluation metrics & plots
│
├── data/
│   └── README.md                    # Instructions to download Fama-French data
│
├── results/
│   └── figures/                     # Output plots
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/SharwariPejathaya/lstm-stock-forecasting.git
cd lstm-stock-forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the data

Download the following datasets from [Kenneth French's Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html):

- **Fama/French 5 Factors (2x3)** — monthly
- **49 Industry Portfolios** — monthly

Place the CSV files in the `data/` directory. See `data/README.md` for exact filenames expected.

### 4. Run the notebook

```bash
jupyter notebook notebooks/US_Stock_LSTM.ipynb
```

Or open directly in **Google Colab**:

- [Paper Implementation Notebook](https://colab.research.google.com/drive/1RLCB7vwLWXVXzJ9cQVCdZDz1NKvUIXJu?usp=sharing) — Fama-French regression & baseline LSTM replication
- [Innovation & Experiments Notebook](https://colab.research.google.com/drive/1NsdkXYhtxPpuo8cfkm0IqU-V7fkJwWXK?usp=sharing) — Improved LSTM, ARIMA comparison, all experiments

---

## Methodology

### Data
- **Source**: Kenneth French's Data Library
- **Period**: 1965–2024 (monthly)
- **Target**: High-Tech sector returns (average of LabEq, Chips, Softw, Telcm)
- **Factors**: Mkt-RF, SMB, HML (Fama-French 3-factor subset)

### Preprocessing
- Returns converted from % to decimal
- 3-month moving average smoothing applied to reduce noise
- MinMax scaling for LSTM input
- 18-month lookback window for sequence construction

### Model Architecture

```
LSTM(50, relu, return_sequences=True)
Dropout(0.2)
LSTM(60, relu, return_sequences=True)
Dropout(0.3)
LSTM(80, relu, return_sequences=True)
Dropout(0.4)
LSTM(120, relu)
Dropout(0.5)
Dense(1)
```

Optimizer: Adam | Loss: MSE | Epochs: 100

### Validation Strategy
- **Standard split**: 80/20 train-test on full 1965–2024 data
- **Time-based split**: Train 1965–2005, Test 2006–2024 (forward-only, no data leakage)

---

## Experiments

**Experiment 1 — Real Return Generalization**: Model trained on smoothed returns tested against raw actual returns → RMSE 0.0491, demonstrating real-world applicability.

**Experiment 2 — ARIMA Comparison**: ARIMA(1,0,1) benchmarked against improved LSTM. ARIMA regressed to the mean in volatile periods (R² = -0.16), while LSTM maintained lower RMSE.

**Experiment 3 — Time-Based Generalization**: Strict forward-looking test (train 1965–2005, test 2006–2024) confirmed temporal robustness with RMSE 0.0226.

---

## Requirements

```
tensorflow>=2.10
keras
numpy
pandas
scikit-learn
matplotlib
statsmodels
pmdarima
yfinance
scipy
seaborn
jupyter
```

---

## Team

| Name | ID | Contribution |
|---|---|---|
| Veda Agrawal | 2022B3A70586H | Overall Implementation |
| Kumar Shivansh Sinha | 2022B1AA1227H | Paper Implementation (Section 2) |
| Siddhi Gadodia | 2022A7PS1652H | Innovation (Section 3) |
| Priten Rathore | 2022B3AA0690H | Experiments (Section 4) |
| Sharwari Pejathaya | 2022B3AA0792H | Experiments (Section 4) |

---

## Citation

If you use this work, please cite:

```
@misc{lstm-stock-forecasting-2025,
  title   = {Forecasting U.S. Stock Returns Using LSTM},
  author  = {Agrawal, Veda and Sinha, Kumar Shivansh and Gadodia, Siddhi 
             and Rathore, Priten and Pejathaya, Sharwari},
  year    = {2025},
  note    = {Course Project, BITS F464 Machine Learning, BITS Pilani}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
