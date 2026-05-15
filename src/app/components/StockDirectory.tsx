import { useState } from "react";
import { useNavigate } from "react-router";
import { Search, TrendingUp, TrendingDown, Loader2 } from "lucide-react";

const AUTO_ANALYZE_KEY = "stockPrediction.autoAnalyzeTicker";

const stocks = [
  {
    ticker: "AAPL",
    name: "Apple Inc.",
    price: "$165.80",
    change: "+2.4%",
    positive: true,
  },
  {
    ticker: "MSFT",
    name: "Microsoft Corporation",
    price: "$378.91",
    change: "+1.8%",
    positive: true,
  },
  {
    ticker: "GOOG",
    name: "Alphabet Inc.",
    price: "$139.45",
    change: "-0.5%",
    positive: false,
  },
  {
    ticker: "TSLA",
    name: "Tesla Inc.",
    price: "$242.38",
    change: "+3.2%",
    positive: true,
  },
  {
    ticker: "AMZN",
    name: "Amazon.com Inc.",
    price: "$178.25",
    change: "+1.1%",
    positive: true,
  },
  {
    ticker: "NVDA",
    name: "NVIDIA Corporation",
    price: "$895.62",
    change: "+5.7%",
    positive: true,
  },
  {
    ticker: "META",
    name: "Meta Platforms Inc.",
    price: "$485.33",
    change: "-1.2%",
    positive: false,
  },
  {
    ticker: "NFLX",
    name: "Netflix Inc.",
    price: "$612.45",
    change: "+2.9%",
    positive: true,
  },
];

export function StockDirectory() {
  const navigate = useNavigate();
  const [search, setSearch] = useState("");
  const [analyzingTicker, setAnalyzingTicker] = useState<string | null>(null);

  function handleAnalyze(ticker: string) {
    const cleanTicker = ticker.trim().toUpperCase();
    setAnalyzingTicker(cleanTicker);
    localStorage.setItem(AUTO_ANALYZE_KEY, cleanTicker);
    navigate("/");
  }

  const filteredStocks = stocks.filter(
    (stock) =>
      stock.ticker.toLowerCase().includes(search.toLowerCase()) ||
      stock.name.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <div className="p-8 max-w-6xl mx-auto">
      <div className="mb-8">
        <h2 className="text-3xl text-white mb-2">Stock Directory</h2>
        <p className="text-gray-400">
          Browse available stock symbols for analysis
        </p>
      </div>

      <div className="mb-6">
        <div className="relative">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search by ticker or company name..."
            className="w-full bg-[#0d1117] border border-gray-700 rounded-lg pl-12 pr-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          />
        </div>
      </div>

      <div className="bg-[#0d1117] rounded-xl border border-gray-800 overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-800/50">
            <tr>
              <th className="text-left px-6 py-4 text-sm text-gray-400">
                Ticker
              </th>
              <th className="text-left px-6 py-4 text-sm text-gray-400">
                Company Name
              </th>
              <th className="text-right px-6 py-4 text-sm text-gray-400">
                Current Price
              </th>
              <th className="text-right px-6 py-4 text-sm text-gray-400">
                Change
              </th>
              <th className="text-right px-6 py-4 text-sm text-gray-400">
                Action
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {filteredStocks.length > 0 ? (
              filteredStocks.map((stock) => (
                <tr
                  key={stock.ticker}
                  className="hover:bg-gray-800/30 transition-colors"
                >
                  <td className="px-6 py-4">
                    <span className="text-blue-400 font-mono">
                      {stock.ticker}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-gray-300">{stock.name}</td>
                  <td className="px-6 py-4 text-right text-white">
                    {stock.price}
                  </td>
                  <td className="px-6 py-4 text-right">
                    <div className="flex items-center justify-end gap-1">
                      {stock.positive ? (
                        <TrendingUp className="w-4 h-4 text-green-500" />
                      ) : (
                        <TrendingDown className="w-4 h-4 text-red-500" />
                      )}
                      <span
                        className={
                          stock.positive ? "text-green-500" : "text-red-500"
                        }
                      >
                        {stock.change}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 text-right">
                    <button
                      onClick={() => handleAnalyze(stock.ticker)}
                      disabled={analyzingTicker === stock.ticker}
                      className="inline-flex items-center justify-center gap-2 text-sm bg-blue-600/20 hover:bg-blue-600/30 disabled:opacity-70 disabled:cursor-not-allowed text-blue-400 px-4 py-2 rounded-lg transition-colors border border-blue-600/30"
                    >
                      {analyzingTicker === stock.ticker && (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      )}
                      {analyzingTicker === stock.ticker
                        ? "Opening..."
                        : "Analyze"}
                    </button>
                  </td>
                </tr>
              ))
            ) : (
              <tr>
                <td
                  colSpan={5}
                  className="px-6 py-12 text-center text-gray-500"
                >
                  No stocks found matching "{search}"
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
