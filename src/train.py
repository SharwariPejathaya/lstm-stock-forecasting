"""
train.py
--------
Training pipeline for the improved LSTM model.
"""

import numpy as np
from src.model import build_lstm_model
from src.data_loader import prepare_dataset


def train(
    ff_path: str,
    industry_path: str,
    lookback: int = 18,
    epochs: int = 100,
    batch_size: int = 32,
    train_ratio: float = 0.80,
    save_path: str = None
):
    """
    Full training run: load data → build model → train → return results.

    Args:
        ff_path:        Path to Fama-French 5 Factors CSV
        industry_path:  Path to 49 Industry Portfolios CSV
        lookback:       LSTM lookback window in months
        epochs:         Number of training epochs
        batch_size:     Batch size
        train_ratio:    Fraction of data for training
        save_path:      Optional path to save trained model weights (.h5)

    Returns:
        dict with model, history, and dataset splits
    """
    print("Loading and preprocessing data...")
    data = prepare_dataset(
        ff_path=ff_path,
        industry_path=industry_path,
        lookback=lookback,
        train_ratio=train_ratio
    )

    X_train = data['X_train']
    y_train = data['y_train']
    input_shape = (X_train.shape[1], X_train.shape[2])

    print(f"Training samples: {len(X_train)} | Test samples: {len(data['X_test'])}")
    print(f"Input shape: {input_shape}")

    print("\nBuilding model...")
    model = build_lstm_model(input_shape=input_shape)
    model.summary()

    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )

    if save_path:
        model.save(save_path)
        print(f"\nModel saved to {save_path}")

    return {
        'model': model,
        'history': history,
        'data': data
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LSTM stock return forecaster")
    parser.add_argument('--ff',        required=True, help="Path to Fama-French 5 Factors CSV")
    parser.add_argument('--industry',  required=True, help="Path to 49 Industry Portfolios CSV")
    parser.add_argument('--lookback',  type=int, default=18)
    parser.add_argument('--epochs',    type=int, default=100)
    parser.add_argument('--save',      default=None, help="Path to save model (.h5)")
    args = parser.parse_args()

    results = train(
        ff_path=args.ff,
        industry_path=args.industry,
        lookback=args.lookback,
        epochs=args.epochs,
        save_path=args.save
    )
    print("\nTraining complete.")
