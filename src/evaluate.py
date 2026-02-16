"""
evaluate.py
-----------
Evaluation metrics and visualization for the LSTM stock return forecasting model.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "Model") -> dict:
    """
    Compute RMSE, MAE, and R² for a set of predictions.

    Args:
        y_true: Array of actual values
        y_pred: Array of predicted values
        label:  Name to print in the report

    Returns:
        Dictionary with metric names and values
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print(f"\n{'='*40}")
    print(f" {label}")
    print(f"{'='*40}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")

    return {'rmse': rmse, 'mae': mae, 'r2': r2}


def inverse_scale_predictions(scaler, y_pred_scaled: np.ndarray, n_features: int) -> np.ndarray:
    """
    Inverse transform scaled predictions back to original return scale.
    Handles the case where scaler was fit on multiple columns.

    Args:
        scaler:         Fitted MinMaxScaler
        y_pred_scaled:  1D array of scaled predictions
        n_features:     Total number of features the scaler was fit on

    Returns:
        1D array of predictions in original scale
    """
    dummy = np.zeros((len(y_pred_scaled), n_features))
    dummy[:, 0] = y_pred_scaled.flatten()
    return scaler.inverse_transform(dummy)[:, 0]


def run_arima_benchmark(train_series: pd.Series, n_forecast: int, order: tuple = (1, 0, 1)) -> np.ndarray:
    """
    Fit ARIMA on training data and forecast n_forecast steps.

    Args:
        train_series: Pandas Series of training returns
        n_forecast:   Number of steps to forecast
        order:        ARIMA (p, d, q) order. Default (1,0,1)

    Returns:
        Array of ARIMA forecasts
    """
    model = ARIMA(train_series, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=n_forecast)
    return forecast.values


def plot_predictions(
    dates,
    y_actual: np.ndarray,
    y_lstm: np.ndarray,
    y_arima: np.ndarray = None,
    title: str = "LSTM: Actual vs Predicted Returns",
    save_path: str = None
):
    """
    Plot actual vs predicted returns, optionally including ARIMA.

    Args:
        dates:     Date index for x-axis
        y_actual:  Ground truth returns
        y_lstm:    LSTM predictions
        y_arima:   ARIMA predictions (optional)
        title:     Plot title
        save_path: If provided, saves figure to this path
    """
    plt.figure(figsize=(14, 6))
    plt.plot(dates, y_actual, label='Actual Returns (Smoothed)', color='steelblue', linewidth=1.2)
    plt.plot(dates, y_lstm,   label='LSTM Predicted', color='darkorange', linestyle='--', linewidth=1.2)

    if y_arima is not None:
        plt.plot(dates, y_arima, label='ARIMA Predicted', color='green', linestyle=':', linewidth=1.2)

    plt.title(title, fontsize=13)
    plt.xlabel('Date')
    plt.ylabel('Monthly Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    plt.show()


def plot_training_history(history, save_path: str = None):
    """
    Plot training and validation loss over epochs.

    Args:
        history:   Keras History object
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['loss'], label='Train Loss', color='steelblue')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Val Loss', color='darkorange')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def print_results_table(results: list):
    """
    Print a formatted comparison table of model results.

    Args:
        results: List of dicts with keys: name, rmse, mae, r2
    """
    print(f"\n{'Model':<35} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-" * 62)
    for r in results:
        r2_str = f"{r['r2']:.4f}" if r.get('r2') is not None else "  —"
        print(f"{r['name']:<35} {r['rmse']:>8.4f} {r['mae']:>8.4f} {r2_str:>8}")
