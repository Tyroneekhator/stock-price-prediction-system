from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent

MODEL_CANDIDATES = [
    Path(os.getenv("MODEL_PATH", "")) if os.getenv("MODEL_PATH") else None,
    BASE_DIR / "keras_model.keras",
    PROJECT_DIR / "keras_model.keras",
    PROJECT_DIR.parent / "keras_model.keras",
]

app = FastAPI(
    title="Stock Price Prediction API",
    description="FastAPI backend for the React + Tailwind stock prediction dashboard.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    ticker: str = Field(default="GOOG", min_length=1, max_length=15)
    start_date: str = Field(default="2017-01-01")
    end_date: str | None = None
    sequence_length: int = Field(default=100, ge=30, le=250)
    train_ratio: float = Field(default=0.70, ge=0.50, le=0.90)


class PredictionPoint(BaseModel):
    date: str
    actual: float
    predicted: float
    difference: float
    error_percent: float


class PredictResponse(BaseModel):
    ticker: str
    start_date: str
    end_date: str
    total_records: int
    training_records: int
    test_records: int
    current_close: float
    previous_close: float | None
    percent_change: float | None
    rmse: float
    mae: float
    mape: float
    model_confidence: float
    data_range_label: str
    chart_data: list[PredictionPoint]
    results: list[PredictionPoint]


def _find_model_path() -> Path:
    for candidate in MODEL_CANDIDATES:
        if candidate and candidate.exists():
            return candidate
    searched = "\n".join(str(p) for p in MODEL_CANDIDATES if p)
    raise FileNotFoundError(
        "Could not find keras_model.keras. Copy it into the backend folder or project root.\n"
        f"Searched:\n{searched}"
    )


@lru_cache(maxsize=1)
def get_model():
    model_path = _find_model_path()
    return load_model(model_path, compile=False)


def _download_stock_data(ticker: str, start_date: str, end_date: str | None) -> pd.DataFrame:
    ticker = ticker.strip().upper()
    end = end_date or date.today().isoformat()
    df = yf.download(ticker, start=start_date, end=end, progress=False, auto_adjust=False)

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No Yahoo Finance data found for ticker '{ticker}'.")

    # yfinance can sometimes return MultiIndex columns. Flatten them safely.
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            close = df[("Close", ticker)]
        else:
            close = df["Close"].iloc[:, 0]
    else:
        close = df["Close"]

    out = pd.DataFrame({"Close": close}).dropna()
    if len(out) < 160:
        raise HTTPException(
            status_code=400,
            detail="Not enough closing-price records to run the LSTM prediction. Try an earlier start date.",
        )
    return out


def _confidence_from_mape(mape: float) -> float:
    # A simple dashboard-friendly score, not a financial guarantee.
    return round(max(0.0, min(99.9, 100.0 - mape)), 1)


def run_prediction(req: PredictRequest) -> PredictResponse:
    ticker = req.ticker.strip().upper()
    df = _download_stock_data(ticker, req.start_date, req.end_date)

    close_df = pd.DataFrame(df["Close"])
    train_size = int(len(close_df) * req.train_ratio)
    data_training = close_df.iloc[:train_size]
    data_testing = close_df.iloc[train_size:]

    if len(data_training) < req.sequence_length or data_testing.empty:
        raise HTTPException(
            status_code=400,
            detail="Not enough data after train/test split. Use a smaller sequence length or earlier start date.",
        )

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_training)

    past_sequence = data_training.tail(req.sequence_length)
    final_df = pd.concat([past_sequence, data_testing], ignore_index=False)
    input_data = scaler.transform(final_df)

    x_test: list[np.ndarray] = []
    y_test_scaled: list[float] = []
    prediction_dates = data_testing.index

    for i in range(req.sequence_length, input_data.shape[0]):
        x_test.append(input_data[i - req.sequence_length : i])
        y_test_scaled.append(input_data[i, 0])

    if not x_test:
        raise HTTPException(status_code=400, detail="No test sequences could be created for prediction.")

    x_test_array = np.array(x_test)
    y_test_array = np.array(y_test_scaled).reshape(-1, 1)

    model = get_model()
    y_pred_scaled = model.predict(x_test_array, verbose=0)

    y_actual = scaler.inverse_transform(y_test_array).reshape(-1)
    y_pred = scaler.inverse_transform(y_pred_scaled).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
    mae = float(mean_absolute_error(y_actual, y_pred))
    mape = float(np.mean(np.abs((y_actual - y_pred) / np.maximum(np.abs(y_actual), 1e-8))) * 100)

    points: list[PredictionPoint] = []
    for dt, actual, predicted in zip(prediction_dates, y_actual, y_pred):
        diff = float(predicted - actual)
        error_percent = float(abs(diff) / max(abs(actual), 1e-8) * 100)
        points.append(
            PredictionPoint(
                date=pd.Timestamp(dt).strftime("%Y-%m-%d"),
                actual=round(float(actual), 2),
                predicted=round(float(predicted), 2),
                difference=round(diff, 2),
                error_percent=round(error_percent, 2),
            )
        )

    current_close = float(close_df["Close"].iloc[-1])
    previous_close = float(close_df["Close"].iloc[-2]) if len(close_df) > 1 else None
    percent_change = None
    if previous_close and previous_close != 0:
        percent_change = ((current_close - previous_close) / previous_close) * 100

    chart_limit = 120
    results_limit = 50

    return PredictResponse(
        ticker=ticker,
        start_date=req.start_date,
        end_date=req.end_date or date.today().isoformat(),
        total_records=len(close_df),
        training_records=len(data_training),
        test_records=len(data_testing),
        current_close=round(current_close, 2),
        previous_close=round(previous_close, 2) if previous_close is not None else None,
        percent_change=round(percent_change, 2) if percent_change is not None else None,
        rmse=round(rmse, 4),
        mae=round(mae, 4),
        mape=round(mape, 4),
        model_confidence=_confidence_from_mape(mape),
        data_range_label=f"{pd.Timestamp(close_df.index.min()).strftime('%Y-%m-%d')} to {pd.Timestamp(close_df.index.max()).strftime('%Y-%m-%d')}",
        chart_data=points[-chart_limit:],
        results=points[-results_limit:],
    )


@app.get("/health")
def health() -> dict[str, Any]:
    model_exists = False
    model_path = None
    try:
        path = _find_model_path()
        model_exists = True
        model_path = str(path)
    except FileNotFoundError:
        pass
    return {"status": "ok", "model_exists": model_exists, "model_path": model_path}


@app.post("/api/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        return run_prediction(req)
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
