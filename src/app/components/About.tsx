import { Download, SlidersHorizontal, Layers, Cpu, BarChart3 } from "lucide-react";

export function About() {
  const steps = [
    {
      icon: Download,
      title: "Download Stock Data",
      description: "Fetch historical stock prices from Yahoo Finance using the yfinance Python library.",
      color: "blue",
    },
    {
      icon: SlidersHorizontal,
      title: "Select Close Column",
      description: "Extract the closing price column as the primary feature for prediction.",
      color: "green",
    },
    {
      icon: Layers,
      title: "Scale Data (0-1)",
      description: "Normalize values using MinMaxScaler to improve model training efficiency.",
      color: "purple",
    },
    {
      icon: Cpu,
      title: "Create Sequences",
      description: "Generate 100-day time sequences to capture temporal patterns in stock movement.",
      color: "cyan",
    },
    {
      icon: BarChart3,
      title: "Train LSTM Model",
      description: "Run the Long Short-Term Memory neural network to learn price patterns.",
      color: "orange",
    },
  ];

  return (
    <div className="p-8 max-w-5xl mx-auto">
      <div className="mb-8">
        <h2 className="text-3xl text-white mb-2">About This System</h2>
        <p className="text-gray-400">
          Understanding how the LSTM stock prediction model works
        </p>
      </div>

      <div className="bg-gradient-to-br from-[#1e3a5f] to-[#0d1117] rounded-xl p-8 mb-8 border border-blue-900/50">
        <h3 className="text-2xl text-white mb-4">Stock Price Prediction System</h3>
        <p className="text-gray-300 leading-relaxed mb-4">
          This is an academic machine learning project that demonstrates the application of
          deep learning to financial time series forecasting. The system uses a Long Short-Term
          Memory (LSTM) neural network to analyze historical stock data and predict future
          closing prices.
        </p>
        <p className="text-gray-300 leading-relaxed">
          Built with Python, Streamlit, and TensorFlow/Keras, this tool is designed for
          educational purposes and provides insight into how AI can be applied to financial
          markets. The predictions should not be used for actual trading decisions.
        </p>
      </div>

      <div className="mb-8">
        <h3 className="text-2xl text-white mb-6">How It Works</h3>
        <div className="space-y-4">
          {steps.map((step, i) => (
            <div
              key={i}
              className="bg-[#0d1117] rounded-xl p-6 border border-gray-800 hover:border-gray-700 transition-colors"
            >
              <div className="flex items-start gap-4">
                <div className={`p-3 bg-${step.color}-600/20 rounded-lg shrink-0`}>
                  <step.icon className={`w-6 h-6 text-${step.color}-400`} />
                </div>
                <div>
                  <div className="flex items-center gap-3 mb-2">
                    <span className="text-sm text-gray-500 font-mono">Step {i + 1}</span>
                    <h4 className="text-lg text-white">{step.title}</h4>
                  </div>
                  <p className="text-gray-400">{step.description}</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="bg-[#0d1117] rounded-xl p-6 border border-gray-800">
        <h3 className="text-lg text-white mb-4">Model Architecture</h3>
        <div className="space-y-3 text-sm">
          <div className="flex justify-between py-2 border-b border-gray-800">
            <span className="text-gray-400">Input Layer</span>
            <span className="text-white font-mono">(100, 1)</span>
          </div>
          <div className="flex justify-between py-2 border-b border-gray-800">
            <span className="text-gray-400">LSTM Layer 1</span>
            <span className="text-white font-mono">50 units, return_sequences=True</span>
          </div>
          <div className="flex justify-between py-2 border-b border-gray-800">
            <span className="text-gray-400">LSTM Layer 2</span>
            <span className="text-white font-mono">50 units</span>
          </div>
          <div className="flex justify-between py-2 border-b border-gray-800">
            <span className="text-gray-400">Dense Layer</span>
            <span className="text-white font-mono">25 units</span>
          </div>
          <div className="flex justify-between py-2">
            <span className="text-gray-400">Output Layer</span>
            <span className="text-white font-mono">1 unit</span>
          </div>
        </div>
      </div>

      <div className="mt-8 bg-yellow-900/20 border border-yellow-700/50 rounded-xl p-6">
        <h4 className="text-yellow-400 mb-2">⚠️ Disclaimer</h4>
        <p className="text-gray-300 text-sm">
          This is an educational project. Stock market predictions are inherently uncertain,
          and this model should not be used as the sole basis for investment decisions.
          Always conduct thorough research and consult financial advisors before trading.
        </p>
      </div>
    </div>
  );
}
