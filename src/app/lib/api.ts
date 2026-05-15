export type PredictionPoint = {
  date: string;
  actual: number;
  predicted: number;
  difference: number;
  error_percent: number;
};

export type PredictionResponse = {
  ticker: string;
  start_date: string;
  end_date: string;
  total_records: number;
  training_records: number;
  test_records: number;
  current_close: number;
  previous_close: number | null;
  percent_change: number | null;
  rmse: number;
  mae: number;
  mape: number;
  model_confidence: number;
  data_range_label: string;
  chart_data: PredictionPoint[];
  results: PredictionPoint[];
};

export type PredictRequest = {
  ticker: string;
  start_date?: string;
  end_date?: string | null;
  sequence_length?: number;
  train_ratio?: number;
};

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const STORAGE_KEY = "stockPrediction.latestResult";

export function savePredictionResult(result: PredictionResponse) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(result));
  window.dispatchEvent(new CustomEvent("prediction-result-updated", { detail: result }));
}

export function loadPredictionResult(): PredictionResponse | null {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as PredictionResponse;
  } catch {
    return null;
  }
}

export async function runPrediction(payload: PredictRequest): Promise<PredictionResponse> {
  const response = await fetch(`${API_BASE_URL}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      ticker: payload.ticker,
      start_date: payload.start_date || "2017-01-01",
      end_date: payload.end_date || null,
      sequence_length: payload.sequence_length || 100,
      train_ratio: payload.train_ratio || 0.7,
    }),
  });

  if (!response.ok) {
    let message = "Prediction request failed.";
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch {
      message = await response.text();
    }
    throw new Error(message);
  }

  const result = (await response.json()) as PredictionResponse;
  savePredictionResult(result);
  return result;
}

export const fallbackPrediction: PredictionResponse = {
  ticker: "AAPL",
  start_date: "2017-01-01",
  end_date: "2026-05-13",
  total_records: 365,
  training_records: 255,
  test_records: 110,
  current_close: 165.8,
  previous_close: 161.92,
  percent_change: 2.4,
  rmse: 2.35,
  mae: 0.72,
  mape: 1.18,
  model_confidence: 94.2,
  data_range_label: "Demo data",
  chart_data: [
    { date: "2026-01-01", actual: 150.2, predicted: 148.5, difference: -1.7, error_percent: 1.13 },
    { date: "2026-01-08", actual: 152.8, predicted: 151.2, difference: -1.6, error_percent: 1.05 },
    { date: "2026-01-15", actual: 148.5, predicted: 149.8, difference: 1.3, error_percent: 0.88 },
    { date: "2026-01-22", actual: 155.3, predicted: 154.1, difference: -1.2, error_percent: 0.77 },
    { date: "2026-01-29", actual: 158.7, predicted: 157.9, difference: -0.8, error_percent: 0.5 },
    { date: "2026-02-05", actual: 161.2, predicted: 160.5, difference: -0.7, error_percent: 0.43 },
    { date: "2026-02-12", actual: 159.8, predicted: 161.2, difference: 1.4, error_percent: 0.88 },
    { date: "2026-02-19", actual: 163.5, predicted: 162.8, difference: -0.7, error_percent: 0.43 },
    { date: "2026-02-26", actual: 167.2, predicted: 166.5, difference: -0.7, error_percent: 0.42 },
    { date: "2026-03-05", actual: 165.8, predicted: 167.1, difference: 1.3, error_percent: 0.78 },
  ],
  results: [
    { date: "2026-05-01", actual: 165.8, predicted: 167.15, difference: 1.35, error_percent: 0.81 },
    { date: "2026-05-02", actual: 167.2, predicted: 166.85, difference: -0.35, error_percent: 0.21 },
    { date: "2026-05-05", actual: 168.5, predicted: 169.2, difference: 0.7, error_percent: 0.42 },
    { date: "2026-05-06", actual: 166.9, predicted: 167.55, difference: 0.65, error_percent: 0.39 },
    { date: "2026-05-07", actual: 169.3, predicted: 168.9, difference: -0.4, error_percent: 0.24 },
    { date: "2026-05-08", actual: 171.4, predicted: 170.8, difference: -0.6, error_percent: 0.35 },
    { date: "2026-05-09", actual: 170.2, predicted: 171.1, difference: 0.9, error_percent: 0.53 },
    { date: "2026-05-12", actual: 172.8, predicted: 172.35, difference: -0.45, error_percent: 0.26 },
    { date: "2026-05-13", actual: 174.5, predicted: 173.9, difference: -0.6, error_percent: 0.34 },
    { date: "2026-05-14", actual: 173.2, predicted: 174.25, difference: 1.05, error_percent: 0.61 },
  ],
};
