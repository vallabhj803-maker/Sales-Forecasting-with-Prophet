import pandas as pd
import numpy as np

np.random.seed(42)

# Date range: 4 years of daily data
dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
n = len(dates)

# Base trend + seasonality + noise
trend = np.linspace(1000, 1800, n)
yearly_seasonality = 300 * np.sin(2 * np.pi * (dates.dayofyear / 365) - np.pi / 2)
weekly_seasonality = 150 * np.sin(2 * np.pi * dates.dayofweek / 7)

# Holiday spikes: Black Friday, Christmas, New Year
holiday_boost = np.zeros(n)
for i, d in enumerate(dates):
    if d.month == 11 and d.day in range(25, 30):
        holiday_boost[i] = 600
    elif d.month == 12 and d.day in range(20, 26):
        holiday_boost[i] = 500
    elif d.month == 1 and d.day <= 3:
        holiday_boost[i] = 300

noise = np.random.normal(0, 80, n)
sales = trend + yearly_seasonality + weekly_seasonality + holiday_boost + noise
sales = np.clip(sales, 50, None)

# Category breakdown
categories = np.random.choice(['Technology', 'Furniture', 'Office Supplies'],
                               size=n, p=[0.35, 0.25, 0.40])
regions = np.random.choice(['West', 'East', 'Central', 'South'],
                            size=n, p=[0.30, 0.28, 0.22, 0.20])

df = pd.DataFrame({
    'Order Date': dates,
    'Sales': sales.round(2),
    'Category': categories,
    'Region': regions,
    'Profit': (sales * np.random.uniform(0.08, 0.25, n)).round(2),
})

df.to_csv('/home/claude/superstore_sales.csv', index=False)
print(f"Dataset created: {len(df)} rows")
print(df.head())
print(f"\nTotal Sales: ${df['Sales'].sum():,.0f}")
print(f"Date range: {df['Order Date'].min()} to {df['Order Date'].max()}")
