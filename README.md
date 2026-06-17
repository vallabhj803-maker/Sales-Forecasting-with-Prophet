# Sales Forecasting with Prophet

Forecasting daily retail sales 90 days into the future using Meta's Prophet library, applied to the Superstore Sales dataset (1,461 daily records across 4 years).

## What this project does

1. **Aggregates and cleans** raw order-level data into a daily sales time series using `pandas`.
2. **Decomposes the series** — isolates trend, weekly seasonality, yearly seasonality, and holiday effects.
3. **Models holidays explicitly** — Black Friday, Christmas, and New Year windows are passed to Prophet as named events.
4. **Trains Prophet** with tuned `changepoint_prior_scale` to balance flexibility vs. overfitting.
5. **Evaluates on a strict 90-day holdout** — the model never sees test data during training.
6. **Forecasts 90 days beyond** the historical window with a 95% confidence interval.
7. **Extracts business insights** — peak days, seasonal patterns, and category/region breakdowns.

## Key Results

| Metric | Value |
|---|---|
| MAE | $107.20 |
| RMSE | $165.30 |
| MAPE | **6.2%** |
| Actuals within 95% CI | 71.1% |
| Mean daily sales (test period) | $1,669 |

- **MAPE of 6.2%** means the model's daily forecasts are off by just 6.2% on average — well within actionable business tolerance.
- **Wednesday is the strongest sales day** ($1,569 avg), while Saturday underperforms ($1,285 avg) — a 22% gap useful for staffing and promotions.
- **July is the peak revenue month** ($52,475 avg/year), driven by mid-year demand and holiday adjacency.
- **Office Supplies account for 40.3%** of total revenue, making it the highest-revenue category.
- **Projected next-90-day revenue: $146,969** (avg $1,633/day), based on extrapolated trend and seasonality.

## Charts

### Forecast Overview (with confidence interval)
![Forecast Overview](charts/forecast_overview.png)

### Seasonality Decomposition
![Seasonality](charts/seasonality.png)

### Category & Region Breakdown
![Category and Region](charts/category_region.png)

### Monthly Sales Heatmap
![Heatmap](charts/monthly_heatmap.png)

## How to Run

```bash
pip install -r requirements.txt
python generate_data.py      # creates superstore_sales.csv
python sales_forecasting.py  # trains model, generates charts, writes results.json
```

Outputs: 4 charts in `charts/` and a `results.json` with all metrics and insights.

## Tools Used

Python, pandas, Prophet (Meta), scikit-learn, matplotlib, seaborn

## Dataset

Superstore Sales — synthetic dataset modelled on the widely-used Sample Superstore retail dataset, with realistic trend, weekly/yearly seasonality, and holiday spikes.
