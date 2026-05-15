import { useEffect, useState } from "react";
import {
  Brain,
  Database,
  Settings,
  Play,
  Loader2,
  CheckCircle2,
  HelpCircle,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  fallbackPrediction,
  PredictionResponse,
  runPrediction,
} from "../lib/api";

const SELECTED_TICKER_KEY = "stockPrediction.selectedTicker";

const trainingData = [
  { epoch: 1, loss: 0.085 },
  { epoch: 5, loss: 0.062 },
  { epoch: 10, loss: 0.045 },
  { epoch: 15, loss: 0.032 },
  { epoch: 20, loss: 0.024 },
  { epoch: 25, loss: 0.019 },
  { epoch: 30, loss: 0.015 },
];

export function PredictionWorkspace() {
  const [ticker, setTicker] = useState(
    () => localStorage.getItem(SELECTED_TICKER_KEY) || "TSLA",
  );
  const [startDate, setStartDate] = useState("2017-01-01");
  const [sequenceLength, setSequenceLength] = useState(100);
  const [prediction, setPrediction] =
    useState<PredictionResponse>(fallbackPrediction);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const selectedTicker = localStorage.getItem(SELECTED_TICKER_KEY);
    if (selectedTicker) {
      setTicker(selectedTicker);
    }
  }, []);

  async function handleRunPrediction() {
    setLoading(true);
    setError("");
    try {
      const result = await runPrediction({
        ticker,
        start_date: startDate,
        sequence_length: sequenceLength,
        train_ratio: 0.7,
      });
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <div className="mb-8">
        <h2 className="text-3xl text-white mb-2">Prediction Workspace</h2>
        <p className="text-gray-400">Configure and run LSTM prediction model</p>
      </div>

      <div className="grid grid-cols-3 gap-6 mb-6">
        <div className="col-span-1 space-y-6">
          <div className="bg-[#0d1117] rounded-xl p-6 border border-gray-800">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-blue-600/20 rounded-lg">
                <Database className="w-5 h-5 text-blue-400" />
              </div>
              <h3 className="text-lg text-white">Input Configuration</h3>
              <InfoTooltip text="This card controls the stock symbol, start date, and sequence length sent to the FastAPI backend when running a prediction." />
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-2">
                  Stock Ticker
                </label>
                <input
                  type="text"
                  value={ticker}
                  onChange={(e) => {
                    const nextTicker = e.target.value.toUpperCase();
                    setTicker(nextTicker);
                    localStorage.setItem(SELECTED_TICKER_KEY, nextTicker);
                  }}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">
                  Start Date
                </label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">
                  Sequence Length
                </label>
                <input
                  type="number"
                  value={sequenceLength}
                  min={30}
                  max={250}
                  onChange={(e) => setSequenceLength(Number(e.target.value))}
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
                />
              </div>

              <button
                onClick={handleRunPrediction}
                disabled={loading || !ticker.trim()}
                className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 disabled:opacity-60 disabled:cursor-not-allowed text-white px-4 py-3 rounded-lg transition-all flex items-center justify-center gap-2 mt-4"
              >
                {loading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Play className="w-5 h-5" />
                )}
                {loading ? "Running Prediction..." : "Run Prediction"}
              </button>

              {error && (
                <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-300">
                  {error}
                </div>
              )}
            </div>
          </div>

          <div className="bg-[#0d1117] rounded-xl p-6 border border-gray-800">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-purple-600/20 rounded-lg">
                <Settings className="w-5 h-5 text-purple-400" />
              </div>
              <h3 className="text-lg text-white">Model Settings</h3>
              <InfoTooltip text="These are the fixed machine learning settings used by the backend: an LSTM model, 70% training split, Close price feature, and MinMax scaling." />
            </div>

            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Model Type</span>
                <span className="text-white">LSTM</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Train Split</span>
                <span className="text-white">70%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Feature</span>
                <span className="text-white">Close Price</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Scaler</span>
                <span className="text-white">MinMaxScaler</span>
              </div>
            </div>
          </div>
        </div>

        <div className="col-span-2 space-y-6">
          <div className="bg-[#0d1117] rounded-xl p-6 border border-gray-800">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-green-600/20 rounded-lg">
                <Database className="w-5 h-5 text-green-400" />
              </div>
              <h3 className="text-lg text-white">Dataset Summary</h3>
              <InfoTooltip text="This card summarizes the downloaded stock data after prediction, including total records, training records, and testing records." />
            </div>

            <div className="grid grid-cols-3 gap-4">
              <SummaryCard
                label="Total Records"
                value={prediction.total_records.toLocaleString()}
              />
              <SummaryCard
                label="Training Set"
                value={`${prediction.training_records.toLocaleString()} (70%)`}
              />
              <SummaryCard
                label="Test Set"
                value={prediction.test_records.toLocaleString()}
              />
            </div>
          </div>

          <div className="bg-[#0d1117] rounded-xl p-6 border border-gray-800">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-blue-600/20 rounded-lg">
                <Brain className="w-5 h-5 text-blue-400" />
              </div>
              <h3 className="text-lg text-white">Preprocessing Pipeline</h3>
              <InfoTooltip text="This card shows the steps the backend follows before prediction: loading Yahoo Finance data, selecting Close prices, scaling values, creating sequences, and using the LSTM model." />
            </div>

            <div className="space-y-3">
              {[
                "Data Loading from Yahoo Finance",
                "Feature Selection (Close)",
                "MinMax Scaling (0-1)",
                `Sequence Creation (${sequenceLength} days)`,
                "Prediction with saved Keras LSTM model",
              ].map((step, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between bg-gray-800/30 rounded-lg p-3"
                >
                  <span className="text-gray-300">{step}</span>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-green-500" />
                    <span className="text-sm text-gray-400">Ready</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-[#0d1117] rounded-xl p-6 border border-gray-800">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <h3 className="text-lg text-white">Training Loss Preview</h3>
                <InfoTooltip text="This chart is a visual preview of how training loss usually decreases over epochs. The real prediction results are shown on the Dashboard and Results pages." />
              </div>
              <span className="text-sm text-gray-400">
                Real predictions appear on Dashboard and Results
              </span>
            </div>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={trainingData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="epoch" stroke="#6b7280" />
                <YAxis stroke="#6b7280" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#1f2937",
                    border: "1px solid #374151",
                    borderRadius: "8px",
                    color: "#fff",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#8b5cf6"
                  strokeWidth={2}
                  dot={{ fill: "#8b5cf6", r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

function SummaryCard({ label, value }: { label: string; value: string }) {
  const explanations: Record<string, string> = {
    "Total Records":
      "The total number of rows downloaded from Yahoo Finance for the selected ticker and date range.",
    "Training Set":
      "The first 70% of the downloaded records used to fit the scaler and prepare the historical pattern context.",
    "Test Set":
      "The remaining records used to compare actual closing prices with the model's predicted prices.",
  };

  return (
    <div className="bg-gray-800/50 rounded-lg p-4">
      <div className="mb-1 flex items-center gap-2">
        <p className="text-sm text-gray-400">{label}</p>
        <InfoTooltip text={explanations[label]} />
      </div>
      <p className="text-2xl text-white">{value}</p>
    </div>
  );
}

function InfoTooltip({ text }: { text: string }) {
  return (
    <span className="relative inline-flex group">
      <HelpCircle className="w-4 h-4 text-gray-500 hover:text-blue-400 cursor-help" />
      <span className="pointer-events-none absolute left-1/2 top-6 z-50 hidden w-64 -translate-x-1/2 rounded-lg border border-gray-700 bg-[#111827] px-3 py-2 text-xs leading-relaxed text-gray-200 shadow-xl group-hover:block">
        {text}
      </span>
    </span>
  );
}
