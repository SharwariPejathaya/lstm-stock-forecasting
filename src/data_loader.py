"""
data_loader.py
--------------
Data loading and preprocessing for Fama-French factor data
and 49 Industry Portfolios.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple


# Industries used to construct the High-Tech sector return
HIGHTECH_INDUSTRIES = ['LabEq', 'Chips', 'Softw', 'Telcm']

# Fama-French factors used as features
FF_FEATURES = ['Mkt-RF', 'SMB', 'HML']


def load_fama_french_factors(filepath: str, skiprows: int = 3) -> pd.DataFrame:
    """
    Load Fama-French 5-factor data from CSV.

    Args:
        filepath:  Path to the Fama-French factors CSV file
        skiprows:  Number of header rows to skip (default 3)

    Returns:
        DataFrame with Date index and factor columns
    """
    df = pd.read_csv(filepath, skiprows=skiprows, index_col=0)
    df.index = pd.to_datetime(df.index, format='%Y%m')
    df = df / 100  # Convert from % to decimal
    df = df[~df.index.duplicated()]
    return df


def load_industry_portfolios(filepath: str, skiprows: int = 11) -> pd.DataFrame:
    """
    Load 49 Industry Portfolio returns from CSV.

    Args:
        filepath:  Path to the 49 Industry Portfolios CSV file
        skiprows:  Number of header rows to skip

    Returns:
        DataFrame with Date index and industry return columns
    """
    df = pd.read_csv(filepath, skiprows=skiprows, index_col=0)
    df.index = pd.to_datetime(df.index, format='%Y%m')
    df = df / 100  # Convert from % to decimal
    df.replace(-99.99, np.nan, inplace=True)
    df.replace(-999, np.nan, inplace=True)
    df = df[~df.index.duplicated()]
    return df


def build_hightech_return(industry_df: pd.DataFrame) -> pd.Series:
    """
    Construct the High-Tech sector return by averaging
    LabEq, Chips, Softw, and Telcm industry returns.

    Args:
        industry_df: DataFrame of industry portfolio returns

    Returns:
        Series of High-Tech monthly returns
    """
    available = [col for col in HIGHTECH_INDUSTRIES if col in industry_df.columns]
    if not available:
        raise ValueError(f"None of {HIGHTECH_INDUSTRIES} found in industry DataFrame columns.")
    return industry_df[available].mean(axis=1)


def apply_smoothing(series: pd.Series, window: int = 3) -> pd.Series:
    """
    Apply rolling mean smoothing to a return series.

    Args:
        series: Monthly return series
        window: Rolling window size (default 3 months)

    Returns:
        Smoothed return series
    """
    return series.rolling(window=window, min_periods=1).mean()


def create_sequences(data: np.ndarray, lookback: int = 18) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create (X, y) sequences for LSTM training.

    Args:
        data:     Scaled 2D array of shape (n_timesteps, n_features)
        lookback: Number of months to look back (default 18)

    Returns:
        X: shape (n_samples, lookback, n_features)
        y: shape (n_samples,)
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, 0])  # Predicting the first column (target return)
    return np.array(X), np.array(y)


def prepare_dataset(
    ff_path: str,
    industry_path: str,
    start_date: str = '1965-01',
    end_date: str = '2024-12',
    lookback: int = 18,
    smoothing_window: int = 3,
    train_ratio: float = 0.80
) -> dict:
    """
    Full data pipeline: load → align → smooth → scale → sequence → split.

    Args:
        ff_path:         Path to Fama-French 5 Factors CSV
        industry_path:   Path to 49 Industry Portfolios CSV
        start_date:      Start of data range (YYYY-MM)
        end_date:        End of data range (YYYY-MM)
        lookback:        LSTM lookback window in months
        smoothing_window: Rolling average window for return smoothing
        train_ratio:     Fraction of data for training

    Returns:
        Dictionary with keys:
            X_train, X_test, y_train, y_test,
            scaler, dates_test, raw_returns
    """
    # Load
    ff = load_fama_french_factors(ff_path)
    industries = load_industry_portfolios(industry_path)

    # Align date range
    ff = ff.loc[start_date:end_date]
    industries = industries.loc[start_date:end_date]

    # Build target
    hightech = build_hightech_return(industries)
    hightech_smoothed = apply_smoothing(hightech, window=smoothing_window)

    # Align factors and target on common dates
    common_idx = ff.index.intersection(hightech_smoothed.index).dropna()
    factors = ff.loc[common_idx, FF_FEATURES]
    target = hightech_smoothed.loc[common_idx]

    # Combine into single array: [target, factor1, factor2, factor3]
    combined = pd.concat([target.rename('Return'), factors], axis=1).dropna()

    # Scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(combined.values)

    # Sequences
    X, y = create_sequences(scaled, lookback=lookback)

    # Train/test split
    split = int(len(X) * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'dates_test': combined.index[split + lookback:],
        'raw_returns': hightech.loc[common_idx],
        'combined': combined
    }
