# Stock Price Prediction System

A modern stock price prediction web application built with a **React + Tailwind CSS frontend** and a **FastAPI Python backend**. The system uses **Yahoo Finance**, **TensorFlow/Keras**, and a trained **Long Short-Term Memory (LSTM)** neural network model to compare real stock closing prices with predicted closing prices.

The project started as a Python/Streamlit machine learning application, but it has now been redesigned into a more professional full-stack web application so the user interface can closely match the Figma AI design.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Purpose of the Project](#purpose-of-the-project)
- [Main Features](#main-features)
- [Technologies Used](#technologies-used)
- [How the Project Works](#how-the-project-works)
- [Project Structure](#project-structure)
- [Frontend Overview](#frontend-overview)
- [Backend Overview](#backend-overview)
- [Machine Learning Model](#machine-learning-model)
- [Installation Guide](#installation-guide)
- [How to Run the Project](#how-to-run-the-project)
- [How to Use the App](#how-to-use-the-app)
- [API Endpoints](#api-endpoints)
- [Stock Ticker Examples](#stock-ticker-examples)
- [Model Training](#model-training)
- [Important Notes](#important-notes)
- [Common Errors and Fixes](#common-errors-and-fixes)
- [Future Improvements](#future-improvements)
- [Disclaimer](#disclaimer)
- [Author](#author)
- [License](#license)

---

## Project Overview

The **Stock Price Prediction System** is a full-stack machine learning web application that allows users to enter a stock ticker symbol, download historical stock data, run a trained LSTM prediction model, and compare actual closing prices with predicted closing prices.

The application has two main parts:

```text
React + Tailwind CSS frontend
        ↓
FastAPI Python backend
        ↓
Yahoo Finance data + trained Keras LSTM model
```

The frontend provides a modern dashboard interface based on the Figma AI design. The backend performs the actual stock data download, preprocessing, prediction, and metric calculation.

---

## Purpose of the Project

The purpose of this project is to demonstrate how machine learning can be used with financial time-series data in a full-stack web application.

This project helps show:

- How stock market data can be collected from Yahoo Finance.
- How closing prices can be prepared for machine learning.
- How an LSTM neural network can be used for time-series prediction.
- How a Python machine learning model can be exposed through an API.
- How a React frontend can communicate with a FastAPI backend.
- How actual and predicted stock prices can be displayed using a modern dashboard UI.

This project is mainly for **educational and academic purposes**.

---

## Main Features

### 1. Modern Dashboard UI

The frontend has a Figma-inspired financial dashboard design with:

- Dark finance-themed layout.
- Sidebar navigation.
- Metric cards.
- Stock ticker input.
- Prediction action buttons.
- Clean charts and tables.
- Responsive layout.

### 2. Dashboard Page

The dashboard page gives users a high-level overview of the system. It includes:

- Project title and description.
- Ticker input.
- Run prediction button.
- Current close price card.
- Model confidence card.
- RMSE card.
- Data range card.
- Actual vs predicted price chart.

### 3. Stock Directory Page

The stock directory page provides example ticker symbols and company names.

Example:

| Ticker | Company |
|---|---|
| AAPL | Apple Inc. |
| MSFT | Microsoft Corporation |
| GOOG | Alphabet Inc. |
| TSLA | Tesla Inc. |
| AMZN | Amazon.com Inc. |
| NVDA | NVIDIA Corporation |

### 4. Prediction Workspace Page

The prediction workspace is where users can enter a stock ticker and run the model.

It shows:

- Selected ticker symbol.
- Dataset summary.
- Preprocessing workflow.
- Prediction chart.
- Model output from the backend.

### 5. Prediction Results Page

The results page displays a table comparing:

- Date.
- Actual closing price.
- Predicted closing price.
- Difference between predicted and actual values.
- Error percentage.

### 6. About / How It Works Page

The about page explains the project workflow in simple terms:

```text
Download stock data
        ↓
Select Close price column
        ↓
Scale values using MinMaxScaler
        ↓
Create 100-day LSTM sequences
        ↓
Load trained Keras model
        ↓
Predict closing prices
        ↓
Display charts, metrics, and results
```

### 7. Real Backend Prediction

The FastAPI backend connects the React UI to the Python machine learning model.

The backend:

- Downloads stock data using `yfinance`.
- Extracts the `Close` price column.
- Splits the data into training and testing sections.
- Scales the values using `MinMaxScaler`.
- Builds LSTM input sequences.
- Loads `keras_model.keras`.
- Runs prediction.
- Returns chart data, result rows, and accuracy metrics to the frontend.

---

## Technologies Used

### Frontend

| Technology | Purpose |
|---|---|
| React | Frontend user interface |
| Vite | Frontend development/build tool |
| TypeScript | Type-safe frontend code |
| Tailwind CSS | Styling and layout |
| Recharts | Charts and data visualization |
| Lucide React | Icons |
| React Router | Page routing/navigation |
| shadcn/ui and Radix UI | UI component foundation |

### Backend

| Technology | Purpose |
|---|---|
| Python | Backend programming language |
| FastAPI | API framework |
| Uvicorn | Runs the FastAPI server |
| yfinance | Downloads stock data from Yahoo Finance |
| Pandas | Data handling and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Scaling and metrics |
| TensorFlow / Keras | Loads and runs the LSTM model |

### Machine Learning

| Technology | Purpose |
|---|---|
| LSTM | Time-series prediction model |
| MinMaxScaler | Scales stock prices between 0 and 1 |
| RMSE | Measures prediction error |
| MAE | Measures average absolute error |
| MAPE | Measures percentage error |

---

## How the Project Works

The system follows this workflow:

```text
User opens the React app
        ↓
User enters a stock ticker symbol
        ↓
React sends the ticker to FastAPI
        ↓
FastAPI downloads stock data from Yahoo Finance
        ↓
The backend extracts the Close price column
        ↓
The data is split into training and testing data
        ↓
The data is scaled using MinMaxScaler
        ↓
The backend creates 100-day LSTM sequences
        ↓
The trained keras_model.keras model is loaded
        ↓
The model predicts closing prices
        ↓
The backend calculates RMSE, MAE, MAPE, and confidence score
        ↓
React receives the prediction response
        ↓
Charts, metric cards, and tables are displayed to the user
```

---

## Project Structure

A typical project structure is shown below:

```text
stock_prediction_react_tailwind_app/
│
├── backend/
│   ├── app/
│   │   └── main.py
│   ├── keras_model.keras
│   ├── train_model.py
│   ├── requirements.txt
│   └── README.md
│
├── src/
│   ├── app/
│   │   ├── App.tsx
│   │   ├── routes.tsx
│   │   ├── lib/
│   │   │   └── api.ts
│   │   └── components/
│   │       ├── Layout.tsx
│   │       ├── Dashboard.tsx
│   │       ├── StockDirectory.tsx
│   │       ├── PredictionWorkspace.tsx
│   │       ├── PredictionResults.tsx
│   │       └── About.tsx
│   │
│   ├── main.tsx
│   └── styles/
│       ├── globals.css
│       ├── theme.css
│       ├── tailwind.css
│       ├── fonts.css
│       └── index.css
│
├── .env.example
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
├── postcss.config.mjs
├── pnpm-workspace.yaml
├── Machine Learning.docx
└── README.md
```

### File Explanation

| File / Folder | Description |
|---|---|
| `src/app/App.tsx` | Main React app entry component |
| `src/app/routes.tsx` | Defines frontend routes/pages |
| `src/app/components/Layout.tsx` | Main dashboard layout and navigation |
| `src/app/components/Dashboard.tsx` | Dashboard/home page |
| `src/app/components/StockDirectory.tsx` | Stock ticker directory page |
| `src/app/components/PredictionWorkspace.tsx` | Main prediction page |
| `src/app/components/PredictionResults.tsx` | Prediction table/results page |
| `src/app/components/About.tsx` | Explanation/how-it-works page |
| `src/app/lib/api.ts` | Connects React frontend to FastAPI backend |
| `backend/app/main.py` | FastAPI backend and prediction logic |
| `backend/keras_model.keras` | Trained Keras LSTM model |
| `backend/train_model.py` | Script used to train the model |
| `backend/requirements.txt` | Python backend dependencies |
| `package.json` | React frontend dependencies and scripts |
| `.env.example` | Example frontend environment configuration |

---

## Frontend Overview

The frontend is a React application generated from the Figma AI design and adapted for this project.

It uses:

- React for UI components.
- Vite for running and building the frontend.
- Tailwind CSS for styling.
- Recharts for stock prediction charts.
- React Router for navigation.
- Local storage to temporarily keep the latest prediction result between pages.

The frontend calls the backend through:

```text
POST http://127.0.0.1:8000/api/predict
```

The API base URL is configured in:

```text
.env.example
```

Example:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

## Backend Overview

The backend is built with FastAPI.

The main backend file is:

```text
backend/app/main.py
```

The backend exposes two main endpoints:

```text
GET /health
POST /api/predict
```

The backend is responsible for:

- Checking that the model file exists.
- Downloading stock data.
- Preparing data for the LSTM model.
- Running predictions.
- Calculating metrics.
- Returning data to the React frontend.

The backend also enables CORS for local frontend development on ports such as:

```text
http://localhost:5173
http://127.0.0.1:5173
```

---

## Machine Learning Model

The project uses a trained LSTM model saved as:

```text
keras_model.keras
```

LSTM stands for **Long Short-Term Memory**. It is a type of recurrent neural network that is useful for sequence and time-series prediction.

In this project, the LSTM model uses previous closing prices to predict closing-price patterns in the test data.

The model workflow is:

```text
Historical closing prices
        ↓
Scale values between 0 and 1
        ↓
Create 100-day sequences
        ↓
Send sequences into LSTM model
        ↓
Predict closing prices
        ↓
Convert predicted values back to real prices
```

---

## Installation Guide

You need to run **two parts** of the project:

1. The Python FastAPI backend.
2. The React + Tailwind frontend.

Open two separate terminals in VS Code or PowerShell.

---

## How to Run the Project

### Step 1: Extract the Project

Extract the project zip file and open the folder in VS Code.

Example folder:

```text
stock_prediction_react_tailwind_app
```

---

### Step 2: Start the Backend

Open a terminal in the project folder and run:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

After the backend starts, test it in your browser:

```text
http://127.0.0.1:8000/health
```

You should see a response similar to:

```json
{
  "status": "ok",
  "model_exists": true,
  "model_path": ".../backend/keras_model.keras"
}
```

If `model_exists` is `false`, make sure this file exists:

```text
backend/keras_model.keras
```

---

### Step 3: Start the Frontend

Open a second terminal in the main project folder and run:

```powershell
npm install
npm run dev
```

The frontend should start at:

```text
http://localhost:5173
```

Open that address in your browser.

---

## How to Use the App

### Step 1: Start Both Servers

Make sure both are running:

```text
Backend:  http://127.0.0.1:8000
Frontend: http://localhost:5173
```

### Step 2: Open the Dashboard

Go to:

```text
http://localhost:5173
```

### Step 3: Enter a Ticker Symbol

Enter a ticker symbol such as:

```text
GOOG
```

or:

```text
AAPL
```

### Step 4: Run Prediction

Click the prediction button on the dashboard or prediction workspace page.

### Step 5: View the Chart

The app will display an actual vs predicted closing price chart.

### Step 6: View the Results Page

Open the results page to view the prediction table.

---

## API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "model_exists": true,
  "model_path": "backend/keras_model.keras"
}
```

### Run Prediction

```http
POST /api/predict
```

Example request:

```json
{
  "ticker": "GOOG",
  "start_date": "2017-01-01",
  "end_date": null,
  "sequence_length": 100,
  "train_ratio": 0.7
}
```

Example response fields:

```json
{
  "ticker": "GOOG",
  "total_records": 1800,
  "training_records": 1260,
  "test_records": 540,
  "current_close": 172.45,
  "percent_change": 2.4,
  "rmse": 4.82,
  "mae": 3.15,
  "mape": 2.8,
  "model_confidence": 97.2,
  "chart_data": [],
  "results": []
}
```

---

## Stock Ticker Examples

You can test the app using these stock tickers:

| Ticker | Company Name |
|---|---|
| AAPL | Apple Inc. |
| MSFT | Microsoft Corporation |
| GOOG | Alphabet Inc. |
| GOOGL | Alphabet Inc. |
| AMZN | Amazon.com Inc. |
| TSLA | Tesla Inc. |
| META | Meta Platforms Inc. |
| NVDA | NVIDIA Corporation |
| NFLX | Netflix Inc. |
| JPM | JPMorgan Chase & Co. |
| V | Visa Inc. |
| MA | Mastercard Incorporated |
| WMT | Walmart Inc. |
| DIS | The Walt Disney Company |
| KO | The Coca-Cola Company |
| PEP | PepsiCo Inc. |
| MCD | McDonald's Corporation |
| NKE | Nike Inc. |
| INTC | Intel Corporation |
| AMD | Advanced Micro Devices Inc. |
| ORCL | Oracle Corporation |

---

## Model Training

The model can be trained using:

```text
backend/train_model.py
```

To retrain the model:

```powershell
cd backend
.\.venv\Scripts\activate
python train_model.py
```

After training, the model should be saved as:

```text
backend/keras_model.keras
```

The backend loads the model using Keras:

```python
load_model("keras_model.keras", compile=False)
```

---

## Important Notes

### 1. Internet Connection Is Required

The backend uses Yahoo Finance through `yfinance`, so the project needs an internet connection to download stock data.

### 2. Backend Must Be Running

The React app depends on the backend for real predictions. If the backend is not running, the frontend may show an error when you click the prediction button.

### 3. The Model File Must Exist

Make sure this file exists:

```text
backend/keras_model.keras
```

If it is missing, the prediction API will fail.

### 4. Predictions Are Not Guaranteed

The prediction is based on historical data. Stock prices are affected by many real-world factors, including:

- Company performance.
- Market news.
- Inflation.
- Interest rates.
- Politics.
- Global events.
- Investor behavior.

Because of this, predictions may not always be accurate.

### 5. This Is Not Financial Advice

This system is for academic and educational use only. It should not be used as the only basis for investment decisions.

---

## Common Errors and Fixes

### Error 1: Frontend Cannot Connect to Backend

If the frontend shows an API connection error, make sure the backend is running:

```powershell
cd backend
.\.venv\Scripts\activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Also check that the frontend API URL is correct:

```text
VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

### Error 2: `model_exists` Is False

If this page:

```text
http://127.0.0.1:8000/health
```

returns:

```json
"model_exists": false
```

copy the model file into:

```text
backend/keras_model.keras
```

---

### Error 3: `ModuleNotFoundError`

Install the backend requirements:

```powershell
cd backend
.\.venv\Scripts\activate
pip install -r requirements.txt
```

---

### Error 4: `npm` Command Not Found

Install Node.js from the official Node.js website, then close and reopen VS Code or PowerShell.

After installation, check:

```powershell
node -v
npm -v
```

---

### Error 5: Frontend Dependencies Not Installed

Run:

```powershell
npm install
```

Then start the frontend again:

```powershell
npm run dev
```

---

### Error 6: TensorFlow Installation Takes Long

TensorFlow is a large package. If installation takes time, wait for it to finish. Make sure your virtual environment is active before installing requirements.

---

## Future Improvements

Possible future improvements include:

- Add user-selectable date ranges in the UI.
- Add a loading animation while the model predicts.
- Add downloadable prediction reports.
- Add CSV export for prediction results.
- Add more financial indicators.
- Add moving averages.
- Add candlestick charts.
- Add model comparison with Linear Regression, Random Forest, or GRU.
- Add authentication for users.
- Add a database to store previous searches.
- Add deployment using Render, Railway, Vercel, or similar platforms.
- Add Nigerian Stock Exchange ticker support.
- Add a page explaining model accuracy in more detail.

---

## Disclaimer

This project is for educational purposes only.

The stock price predictions generated by this system are not financial advice. Users should not use this project as the only basis for making investment decisions.

Stock market investments involve risk, and actual prices may be different from predicted prices.

---

## Author

Developed by **Tyrone Ekhator**.

---

## License

This project is for academic and educational use.

The generated Figma Make files may include open-source UI components and assets. Check `ATTRIBUTIONS.md` for attribution information.
