"""
model.py
--------
LSTM model architecture for U.S. stock return forecasting.
Based on: "Regression and Forecasting of U.S. Stock Returns Based on LSTM"
Enhanced with dropout regularization and deeper architecture.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm_model(input_shape: tuple, units: list = None, dropouts: list = None) -> Sequential:
    """
    Build the improved LSTM model.

    Args:
        input_shape: (window_size, n_features) — shape of input sequences
        units:       List of LSTM unit sizes per layer. Default: [50, 60, 80, 120]
        dropouts:    List of dropout rates per layer.  Default: [0.2, 0.3, 0.4, 0.5]

    Returns:
        Compiled Keras Sequential model
    """
    if units is None:
        units = [50, 60, 80, 120]
    if dropouts is None:
        dropouts = [0.2, 0.3, 0.4, 0.5]

    assert len(units) == len(dropouts), "units and dropouts must have the same length"

    model = Sequential()

    # First LSTM layer — needs input_shape
    model.add(LSTM(
        units=units[0],
        activation='relu',
        return_sequences=(len(units) > 1),
        input_shape=input_shape
    ))
    model.add(Dropout(dropouts[0]))

    # Intermediate LSTM layers
    for i in range(1, len(units) - 1):
        model.add(LSTM(units=units[i], activation='relu', return_sequences=True))
        model.add(Dropout(dropouts[i]))

    # Final LSTM layer — no return_sequences
    if len(units) > 1:
        model.add(LSTM(units=units[-1], activation='relu'))
        model.add(Dropout(dropouts[-1]))

    # Output layer
    model.add(Dense(units=1))

    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

    return model


def build_baseline_lstm(input_shape: tuple) -> Sequential:
    """
    Baseline LSTM replicating the original paper architecture.

    Args:
        input_shape: (window_size, n_features)

    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        LSTM(units=50, activation='relu', input_shape=input_shape),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


if __name__ == "__main__":
    # Quick sanity check
    model = build_lstm_model(input_shape=(18, 3))
    model.summary()
