# Swiss Grocery Prices

An exploratory and predictive analysis of two years of Swiss grocery pricing data covering 35 product categories. The analysis identifies seasonal price patterns, removes outliers using IQR, and produces a 30-day autoregressive price forecast for a selected product.

---

## Architecture

```
data/SwissBills.csv   (2,307 rows × 3 columns: Description, Price, Date)
        │
        ▼  Pandas — load, type casting, shape/dtype validation
        │
        ▼  EDA
        │   per-category time series (Matplotlib, Plotly interactive)
        │   per-product descriptive statistics
        │   boxplots: bread, cherry tomatoes, mozzarella
        │
        ▼  Outlier removal
        │   IQR method: discard prices outside [Q1 − 1.5×IQR, Q3 + 1.5×IQR]
        │
        ▼  Time series preparation (cherry tomatoes)
        │   resample to daily frequency, forward-fill missing dates
        │   train/test split: all − 30 days / last 30 days
        │
        ▼  skforecast ForecasterAutoreg
        │   regressor=LinearRegression, lags=10
        │
        ▼  Evaluation: MSE on 30-day test window
```

---

## Core Technical Stack

Python, pandas, NumPy, SciPy, Matplotlib, Seaborn, Plotly Express, scikit-learn, skforecast

---

## Key Methodologies

- **IQR outlier removal** — rather than z-score clipping (which assumes normality), IQR-based bounds are used because grocery price distributions are right-skewed. The method is applied per-product, not globally, so a high-priced item (e.g., mozzarella) is not penalised by the price range of staples (e.g., bread).

- **Forward-fill for daily resampling** — the raw dataset contains irregular observation dates. Before fitting the autoregressive model, the series is resampled to daily frequency using forward-fill, which preserves the last known price rather than interpolating a value that never existed.

- **Autoregressive forecasting with skforecast** — `ForecasterAutoreg` wraps a `LinearRegression` estimator using 10 lagged values as features (i.e., the model predicts `price[t]` from `price[t-1], ..., price[t-10]`). This is appropriate for a series with strong autocorrelation like a seasonal food item, and keeps the model interpretable.

- **Plotly for interactive exploration** — alongside static Matplotlib/Seaborn charts, a Plotly Express line chart renders all 35 categories together with colour-coded traces, making it practical to identify which categories drive price spikes.

---

## Production Metrics & Validation

- Dataset: 2,307 price observations across 35 product categories over two years (source: [Kaggle — Real Swiss Grocery Prices](https://www.kaggle.com/datasets/timurmaksutov/real-swiss-grocery-prices)).
- Cherry tomatoes selected for forecasting due to the strongest observable seasonality (winter–summer price cycle clearly visible in the time series).
- Model evaluated on a held-out 30-day window; MSE reported in-notebook.
- Key finding: cherry tomatoes show pronounced seasonal price variation; bread prices are near-flat over the same period, confirming different demand elasticity between seasonal and staple goods.

---

## Local Replication

Prerequisites: Python 3.8+, Jupyter.

```bash
git clone https://github.com/tharrmeehan/Swiss-Grocery-Prices.git
cd Swiss-Grocery-Prices

pip install pandas numpy scipy matplotlib seaborn plotly scikit-learn skforecast jupyter

jupyter notebook notebook.ipynb
```

The dataset is included in the `data/` directory; no external downloads are required.
