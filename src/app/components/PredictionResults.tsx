import { useEffect, useMemo, useState } from "react";
import { Download, TrendingUp, TrendingDown, HelpCircle } from "lucide-react";
import {
  fallbackPrediction,
  loadPredictionResult,
  PredictionResponse,
} from "../lib/api";

export function PredictionResults() {
  const [prediction, setPrediction] = useState<PredictionResponse>(
    () => loadPredictionResult() || fallbackPrediction,
  );

  useEffect(() => {
    const handler = () => {
      const latest = loadPredictionResult();
      if (latest) setPrediction(latest);
    };
    window.addEventListener("prediction-result-updated", handler);
    return () =>
      window.removeEventListener("prediction-result-updated", handler);
  }, []);

  const results = prediction.results;

  const csv = useMemo(() => {
    const header =
      "date,original_close,predicted_price,difference,error_percent";
    const rows = results.map(
      (r) =>
        `${r.date},${r.actual},${r.predicted},${r.difference},${r.error_percent}`,
    );
    return [header, ...rows].join("\n");
  }, [results]);

  function exportCsv() {
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${prediction.ticker}_prediction_results.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <div className="mb-8">
        <h2 className="text-3xl text-white mb-2">Prediction Results</h2>
        <p className="text-gray-400">
          Compare actual vs predicted closing prices for {prediction.ticker}
        </p>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-6">
        <div className="bg-[#0d1117] rounded-xl p-5 border border-gray-800">
          <div className="mb-2 flex items-center gap-2">
            <p className="text-sm text-gray-400">Model Confidence</p>
            <InfoTooltip text="A simplified confidence score based on prediction error. A higher value means the predicted closing prices are closer to the actual closing prices." />
          </div>
          <p className="text-3xl text-green-400 mb-1">
            {prediction.model_confidence.toFixed(1)}%
          </p>
          <div className="flex items-center gap-1 text-sm text-gray-400">
            <TrendingUp className="w-4 h-4" />
            <span>Based on MAPE</span>
          </div>
        </div>

        <div className="bg-[#0d1117] rounded-xl p-5 border border-gray-800">
          <div className="mb-2 flex items-center gap-2">
            <p className="text-sm text-gray-400">Avg. Error (MAE)</p>
            <InfoTooltip text="Mean Absolute Error. It shows the average dollar difference between the actual closing prices and the predicted prices. Lower is better." />
          </div>
          <p className="text-3xl text-blue-400 mb-1">
            ${prediction.mae.toFixed(2)}
          </p>
          <p className="text-sm text-gray-400">
            RMSE: {prediction.rmse.toFixed(2)}
          </p>
        </div>

        <div className="bg-[#0d1117] rounded-xl p-5 border border-gray-800">
          <div className="mb-2 flex items-center gap-2">
            <p className="text-sm text-gray-400">Predictions</p>
            <InfoTooltip text="The number of recent test samples displayed in the results table after the backend runs the prediction." />
          </div>
          <p className="text-3xl text-white mb-1">{results.length}</p>
          <p className="text-sm text-gray-400">Recent test samples</p>
        </div>
      </div>

      <div className="bg-[#0d1117] rounded-xl border border-gray-800 overflow-hidden">
        <div className="flex items-center justify-between p-6 border-b border-gray-800">
          <div>
            <h3 className="text-lg text-white">Detailed Predictions</h3>
            <p className="text-sm text-gray-400 mt-1">
              {prediction.data_range_label}
            </p>
          </div>
          <button
            onClick={exportCsv}
            className="flex items-center gap-2 bg-blue-600/20 hover:bg-blue-600/30 text-blue-400 px-4 py-2 rounded-lg transition-colors border border-blue-600/30"
          >
            <Download className="w-4 h-4" />
            Export CSV
          </button>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-800/50">
              <tr>
                <th className="text-left px-6 py-4 text-sm text-gray-400">
                  Date
                </th>
                <th className="text-right px-6 py-4 text-sm text-gray-400">
                  Original Close
                </th>
                <th className="text-right px-6 py-4 text-sm text-gray-400">
                  Predicted Price
                </th>
                <th className="text-right px-6 py-4 text-sm text-gray-400">
                  Difference
                </th>
                <th className="text-right px-6 py-4 text-sm text-gray-400">
                  Error %
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {results.map((result, i) => {
                const isPositive = result.difference > 0;

                return (
                  <tr
                    key={`${result.date}-${i}`}
                    className="hover:bg-gray-800/30 transition-colors"
                  >
                    <td className="px-6 py-4 text-gray-300">{result.date}</td>
                    <td className="px-6 py-4 text-right text-white font-mono">
                      ${result.actual.toFixed(2)}
                    </td>
                    <td className="px-6 py-4 text-right text-blue-400 font-mono">
                      ${result.predicted.toFixed(2)}
                    </td>
                    <td className="px-6 py-4 text-right">
                      <div className="flex items-center justify-end gap-1">
                        {isPositive ? (
                          <TrendingUp className="w-4 h-4 text-green-500" />
                        ) : (
                          <TrendingDown className="w-4 h-4 text-red-500" />
                        )}
                        <span
                          className={`font-mono ${
                            isPositive ? "text-green-500" : "text-red-500"
                          }`}
                        >
                          {isPositive ? "+" : ""}${result.difference.toFixed(2)}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-right text-gray-400 font-mono">
                      {result.error_percent.toFixed(2)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
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
