"""
Deep Learning Time Series Forecasting - Main Analysis
Author: Jose Orlando Bobadilla Fuentes | CQF
"""

import numpy as np
import pandas as pd
import sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from models.feature_engineer import FeatureEngineer
from models.dl_models import build_lstm_model, build_gru_model, build_transformer_model
from visualization.forecast_plots import plot_predictions_comparison, plot_training_history


def header(t):
    print(f"\n{'='*70}\n  {t}\n{'='*70}")

def main():
    header("DEEP LEARNING TIME SERIES FORECASTING")

    # --- 1. Data ---
    header("1. DATA PREPARATION")
    try:
        import yfinance as yf
        df = yf.download("SPY", start="2018-01-01", end="2024-12-31",
                          progress=False)
        print(f"  Downloaded SPY: {len(df)} observations")
    except Exception:
        print("  yfinance unavailable, generating synthetic data")
        np.random.seed(42)
        n = 1500
        dates = pd.date_range("2018-01-01", periods=n, freq="B")
        price = 270 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, n)))
        df = pd.DataFrame({"Open": price*0.999, "High": price*1.005,
                           "Low": price*0.995, "Close": price,
                           "Volume": np.random.randint(5e7, 2e8, n)}, index=dates)

    # --- 2. Feature Engineering ---
    header("2. FEATURE ENGINEERING")
    lookback = 60
    fe = FeatureEngineer(lookback=lookback)
    X_train, y_train, X_test, y_test = fe.prepare(df, train_split=0.8)
    input_shape = (X_train.shape[1], X_train.shape[2])

    print(f"  Features: {len(fe.feature_names)}")
    print(f"  Lookback: {lookback} days")
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # --- 3. Models ---
    header("3. MODEL TRAINING")
    epochs = 30
    batch = 32
    models = {
        "LSTM": build_lstm_model(input_shape),
        "GRU": build_gru_model(input_shape),
        "Transformer": build_transformer_model(input_shape),
    }

    histories = {}
    predictions = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.summary(print_fn=lambda x: None)  # suppress
        hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch,
                         validation_split=0.15, verbose=0)
        histories[name] = hist.history

        preds = model.predict(X_test, verbose=0).flatten()
        predictions[name] = preds

        mse = np.mean((preds - y_test)**2)
        mae = np.mean(np.abs(preds - y_test))
        # Directional accuracy
        dir_acc = np.mean(np.sign(preds) == np.sign(y_test))
        # Sharpe of predictions as signal
        strat_ret = np.sign(preds) * y_test
        sharpe = np.mean(strat_ret) / (np.std(strat_ret) + 1e-10) * np.sqrt(252)

        print(f"    MSE:    {mse:.8f}")
        print(f"    MAE:    {mae:.6f}")
        print(f"    Dir Acc: {dir_acc:.2%}")
        print(f"    Sharpe:  {sharpe:.3f}")

    # --- 4. Visualizations ---
    header("4. VISUALIZATIONS")
    out = "outputs/figures"
    plot_predictions_comparison(y_test, predictions, out)
    plot_training_history(histories, out)

    header("ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()
