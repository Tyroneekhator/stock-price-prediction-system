import { useEffect, useState } from "react";
import {
  Search,
  TrendingUp,
  TrendingDown,
  Activity,
  Calendar,
  BarChart3,
  Loader2,
  HelpCircle,
} from "lucide-react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  fallbackPrediction,
  loadPredictionResult,
  PredictionResponse,
  runPrediction,
} from "../lib/api";

const AUTO_ANALYZE_KEY = "stockPrediction.autoAnalyzeTicker";

export function Dashboard() {
  const [ticker, setTicker] = useState("AAPL");
  const [prediction, setPrediction] = useState<PredictionResponse>(
    () => loadPredictionResult() || fallbackPrediction,
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const handler = () => {
      const latest = loadPredictionResult();
      if (latest) {
        setPrediction(latest);
        setTicker(latest.ticker);
      }
    };
    window.addEventListener("prediction-result-updated", handler);
    return () =>
      window.removeEventListener("prediction-result-updated", handler);
  }, []);

  useEffect(() => {
    const selectedTicker = localStorage.getItem(AUTO_ANALYZE_KEY);
    if (!selectedTicker) return;

    localStorage.removeItem(AUTO_ANALYZE_KEY);
    void runDashboardPrediction(selectedTicker);
  }, []);

  async function runDashboardPrediction(selectedTicker: string) {
    const cleanTicker = selectedTicker.trim().toUpperCase();
    if (!cleanTicker) return;

    setTicker(cleanTicker);
    setLoading(true);
    setError("");

    try {
      const result = await runPrediction({
        ticker: cleanTicker,
        start_date: "2017-01-01",
        sequence_length: 100,
        train_ratio: 0.7,
      });
      setPrediction(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed.");
    } finally {
      setLoading(false);
    }
  }

  function handleRunPrediction() {
    void runDashboardPrediction(ticker);
  }

  const percentChange = prediction.percent_change ?? 0;
  const chartData = prediction.chart_data.map((item) => ({
    ...item,
    label: new Date(item.date).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    }),
  }));

  return (
    <div className="p-8 max-w-7xl mx-auto">
      <div className="mb-8">
        <h2 className="text-3xl text-white mb-2">
          Stock Price Prediction Dashboard
        </h2>
        <p className="text-gray-400">
          AI-powered LSTM model for accurate stock price forecasting
        </p>
      </div>

      <div className="bg-gradient-to-br from-[#1e3a5f] to-[#0d1117] rounded-xl p-6 mb-6 border border-blue-900/50">
        <div className="flex gap-4 items-end">
          <div className="flex-1">
            <label className="block text-sm text-gray-300 mb-2">
              Enter Stock Ticker
            </label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                value={ticker}
                onChange={(e) => setTicker(e.target.value.toUpperCase())}
                placeholder="e.g., AAPL, GOOG, TSLA"
                className="w-full bg-[#0d1117] border border-gray-700 rounded-lg pl-10 pr-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
              />
            </div>
          </div>
          <button
            onClick={handleRunPrediction}
            disabled={loading || !ticker.trim()}
            className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-500 hover:to-blue-600 disabled:opacity-60 disabled:cursor-not-allowed text-white px-8 py-3 rounded-lg transition-all flex items-center gap-2 shadow-lg shadow-blue-900/50"
          >
            {loading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Activity className="w-5 h-5" />
            )}
            {loading ? "Running..." : "Run Prediction"}
          </button>
        </div>
        {error && (
          <div className="mt-4 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-300">
            {error}
          </div>
        )}
      </div>

      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard
          title="Current Close"
          value={`$${prediction.current_close.toFixed(2)}`}
          change={`${percentChange >= 0 ? "+" : ""}${percentChange.toFixed(2)}%`}
          positive={percentChange >= 0}
          icon={TrendingUp}
          tooltip="The most recent closing price returned from Yahoo Finance for the selected ticker. The percentage shows how the latest close changed compared with the previous close."
        />
        <MetricCard
          title="Model Confidence"
          value={`${prediction.model_confidence.toFixed(1)}%`}
          change="API result"
          positive={true}
          icon={Activity}
          tooltip="A simplified confidence score calculated from the prediction error. A higher value means the predicted prices were closer to the real closing prices."
        />
        <MetricCard
          title="RMSE"
          value={prediction.rmse.toFixed(2)}
          change={`MAE ${prediction.mae.toFixed(2)}`}
          positive={true}
          icon={BarChart3}
          tooltip="Root Mean Squared Error. It measures the average prediction error, with bigger mistakes counted more strongly. Lower RMSE means better prediction performance."
        />
        <MetricCard
          title="Data Range"
          value={`${prediction.total_records} rows`}
          change={prediction.ticker}
          positive={null}
          icon={Calendar}
          tooltip="The number of historical stock records downloaded and used for the prediction workflow for the selected ticker."
        />
      </div>

      <div className="bg-[#0d1117] rounded-xl p-6 border border-gray-800">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-xl text-white">
            Actual vs Predicted Closing Price
          </h3>
          <p className="text-sm text-gray-400">{prediction.data_range_label}</p>
        </div>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
            <XAxis dataKey="label" stroke="#6b7280" />
            <YAxis stroke="#6b7280" />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1f2937",
                border: "1px solid #374151",
                borderRadius: "8px",
                color: "#fff",
              }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="actual"
              stroke="#10b981"
              strokeWidth={2}
              dot={false}
              name="Actual Price"
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="#60a5fa"
              strokeWidth={2}
              dot={false}
              name="Predicted Price"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function MetricCard({
  title,
  value,
  change,
  positive,
  icon: Icon,
  tooltip,
}: any) {
  return (
    <div className="bg-[#0d1117] rounded-xl p-5 border border-gray-800">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <p className="text-sm text-gray-400">{title}</p>
          <InfoTooltip text={tooltip} />
        </div>
        <Icon className="w-4 h-4 text-gray-500" />
      </div>
      <p className="text-2xl text-white mb-1">{value}</p>
      <div className="flex items-center gap-1">
        {positive === true && <TrendingUp className="w-4 h-4 text-green-500" />}
        {positive === false && (
          <TrendingDown className="w-4 h-4 text-red-500" />
        )}
        <p
          className={`text-sm ${positive === true ? "text-green-500" : positive === false ? "text-red-500" : "text-gray-400"}`}
        >
          {change}
        </p>
      </div>
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
