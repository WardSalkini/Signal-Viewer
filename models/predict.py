"""
predict.py — Price prediction using Amazon Chronos (pretrained transformer).

Chronos is a pretrained time-series foundation model from Amazon (2024).
It was trained on a massive corpus of diverse real-world time series and
performs zero-shot forecasting — no retraining on your data ever needed.

Model: amazon/chronos-t5-base  (200M params, ~400MB, good balance of speed and accuracy)

Install:
    pip install git+https://github.com/amazon-science/chronos-forecasting.git
    pip install yfinance torch transformers accelerate

Register in Flask:
    from models.predict import predict_bp
    app.register_blueprint(predict_bp)
"""

from flask import Blueprint, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

predict_bp = Blueprint("predict", __name__)


# ── Model singleton — loaded once on first request, reused forever ────────────

_pipeline = None


def _get_pipeline():
    """
    Lazy-load the Chronos pipeline the first time a prediction is requested.
    Subsequent calls reuse the already-loaded model (no reload penalty).
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    try:
        from chronos import ChronosPipeline
    except ImportError:
        raise ImportError(
            "chronos is not installed.\n"
            "Run: pip install git+https://github.com/amazon-science/chronos-forecasting.git"
        )

    model_id = "amazon/chronos-t5-base"   # 200M params, ~400MB, good balance of speed and accuracy
    device   = "cpu"
    dtype    = torch.float32

    logger.info(f"Loading Chronos: {model_id} on {device} …")

    _pipeline = ChronosPipeline.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=dtype,
    )

    logger.info("Chronos ready.")
    return _pipeline


# ── Data fetching ─────────────────────────────────────────────────────────────

def _get_history(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Download daily close prices. Returns DataFrame with [date, close]."""
    ticker = yf.Ticker(symbol)
    hist   = ticker.history(period=period)

    if hist.empty:
        raise ValueError(f"No price data found for symbol: {symbol}")

    hist = hist[["Close"]].dropna().reset_index()
    hist.columns = ["date", "close"]
    hist["date"] = pd.to_datetime(hist["date"]).dt.tz_localize(None)
    return hist


# ── Core prediction ───────────────────────────────────────────────────────────

def _predict_chronos(symbol: str, days_ahead: int = 30) -> dict:
    """
    Run Chronos zero-shot forecast.

    How Chronos works:
    ------------------
    1. The pretrained transformer reads the historical price sequence as
       a context (just like a language model reads text tokens).
    2. It outputs 100 probabilistic sample paths — different plausible
       futures drawn from its learned distribution.
    3. We summarise those samples into:
         median → point forecast  (yhat)
         P10    → lower bound     (yhat_lower)  \  80% confidence band
         P90    → upper bound     (yhat_upper)  /

    No model training or fine-tuning happens here — Chronos already
    "knows" how financial time series behave from pretraining.
    """
    pipe = _get_pipeline()

    # --- 1. Historical data (2 years gives a solid context window) -----------
    hist       = _get_history(symbol, period="2y")
    prices     = hist["close"].values.astype(np.float32)
    last_date  = hist["date"].iloc[-1]
    last_price = float(prices[-1])

    # --- 2. Context tensor: shape [1, context_length] -------------------------
    context = torch.tensor(prices, dtype=torch.float32).unsqueeze(0)

    # --- 3. Inference (no gradients needed) -----------------------------------
    with torch.no_grad():
        forecast_samples = pipe.predict(
            context,                      # positional in newer Chronos versions
            prediction_length=days_ahead,
            num_samples=100,              # 100 sample paths → smooth distribution
            limit_prediction_length=False,
        )

    # forecast_samples: [batch=1, num_samples=100, prediction_length]
    samples = forecast_samples[0].cpu().numpy()   # shape: (100, days_ahead)

    # --- 4. Summarise sample distribution ------------------------------------
    yhat       = np.median(samples, axis=0)
    yhat_lower = np.percentile(samples, 10, axis=0)   # 80% band
    yhat_upper = np.percentile(samples, 90, axis=0)

    # Prices can't go negative
    yhat       = np.maximum(yhat, 0)
    yhat_lower = np.maximum(yhat_lower, 0)
    yhat_upper = np.maximum(yhat_upper, 0)

    # --- 5. Forecast date labels (skip weekends) -----------------------------
    forecast_dates = []
    cursor = last_date
    while len(forecast_dates) < days_ahead:
        cursor += timedelta(days=1)
        if cursor.weekday() < 5:    # 0=Mon … 4=Fri
            forecast_dates.append(cursor)

    # --- 6. Historical tail for chart (last 60 data points) ------------------
    tail = hist.tail(60)

    # --- 7. Pack result -------------------------------------------------------
    trend = "up" if float(yhat[-1]) > float(yhat[0]) else "down"

    model_name = "amazon/chronos-t5-base"

    return {
        "symbol":          symbol,
        "days_ahead":      days_ahead,
        "model":           model_name,
        "current_price":   round(last_price, 4),
        "predicted_price": round(float(yhat[-1]), 4),
        "trend":           trend,
        "history": {
            "labels": tail["date"].dt.strftime("%Y-%m-%d").tolist(),
            "prices": tail["close"].round(4).tolist(),
        },
        "forecast": {
            "labels":     [d.strftime("%Y-%m-%d") for d in forecast_dates],
            "yhat":       [round(float(v), 4) for v in yhat],
            "yhat_lower": [round(float(v), 4) for v in yhat_lower],
            "yhat_upper": [round(float(v), 4) for v in yhat_upper],
        },
    }


# ── Flask route ───────────────────────────────────────────────────────────────

@predict_bp.route("/api/stocks/predict")
def predict():
    """
    GET /api/stocks/predict?symbol=AAPL&days=30

    Response shape is identical to the old Prophet endpoint —
    the frontend requires zero changes.

    {
        success:         true,
        symbol:          "AAPL",
        days_ahead:      30,
        model:           "amazon/chronos-t5-base",
        trend:           "up" | "down",
        current_price:   182.63,
        predicted_price: 191.40,
        history:  { labels: [...], prices: [...] },
        forecast: { labels: [...], yhat: [...], yhat_lower: [...], yhat_upper: [...] }
    }
    """
    symbol = request.args.get("symbol", "").upper().strip()
    if not symbol:
        return jsonify({"success": False, "error": "symbol is required"}), 400

    try:
        days = int(request.args.get("days", 30))
        days = max(7, min(days, 90))
    except ValueError:
        days = 30

    try:
        result = _predict_chronos(symbol, days_ahead=days)
        return jsonify({"success": True, **result})

    except ImportError as e:
        return jsonify({"success": False, "error": str(e)}), 500
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        logger.exception(f"Chronos prediction failed for {symbol}")
        return jsonify({"success": False, "error": f"Prediction failed: {str(e)}"}), 500
