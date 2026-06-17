"""
Sales Forecasting with Prophet
Superstore Sales Dataset — 4 years (2020–2023), forecasting 90 days ahead
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings, json, os
warnings.filterwarnings('ignore')

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d27',
    'axes.edgecolor':   '#2e3148',
    'axes.labelcolor':  '#c8cfe0',
    'xtick.color':      '#c8cfe0',
    'ytick.color':      '#c8cfe0',
    'text.color':       '#c8cfe0',
    'grid.color':       '#2e3148',
    'grid.linewidth':   0.6,
    'font.family':      'DejaVu Sans',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})
ACCENT   = '#7c83fd'
GREEN    = '#4ecca3'
ORANGE   = '#f4a261'
RED_SOFT = '#e07070'

os.makedirs('charts', exist_ok=True)

# ── 1. Load & aggregate ────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv('superstore_sales.csv', parse_dates=['Order Date'])
daily = df.groupby('Order Date')['Sales'].sum().reset_index()
daily.columns = ['ds', 'y']

# ── 2. Train / test split — last 90 days as holdout ───────────────────────────
cutoff   = daily['ds'].max() - pd.Timedelta(days=90)
train    = daily[daily['ds'] <= cutoff].copy()
test     = daily[daily['ds'] >  cutoff].copy()
print(f"Train: {len(train)} days | Test (holdout): {len(test)} days")

# ── 3. Define holidays ────────────────────────────────────────────────────────
holidays = pd.DataFrame({
    'holiday':    ['black_friday', 'christmas', 'new_year', 'black_friday', 'christmas', 'new_year',
                   'black_friday', 'christmas', 'new_year', 'black_friday', 'christmas', 'new_year'],
    'ds': pd.to_datetime([
        '2020-11-27','2020-12-25','2020-01-01',
        '2021-11-26','2021-12-25','2021-01-01',
        '2022-11-25','2022-12-25','2022-01-01',
        '2023-11-24','2023-12-25','2023-01-01',
    ]),
    'lower_window': [0]*12,
    'upper_window': [3, 5, 2, 3, 5, 2, 3, 5, 2, 3, 5, 2],
})

# ── 4. Fit Prophet ────────────────────────────────────────────────────────────
print("Training Prophet model...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    holidays=holidays,
    seasonality_mode='additive',
    changepoint_prior_scale=0.05,
)
model.fit(train)

# ── 5. Forecast 90-day holdout + 90 days beyond ───────────────────────────────
future_all = model.make_future_dataframe(periods=90 + 90)   # covers test + extra forecast
forecast   = model.predict(future_all)

# Align test predictions
test_preds = forecast[forecast['ds'].isin(test['ds'])][['ds','yhat','yhat_lower','yhat_upper']]
test_merged = test.merge(test_preds, on='ds')

# ── 6. Metrics ────────────────────────────────────────────────────────────────
mae   = mean_absolute_error(test_merged['y'], test_merged['yhat'])
rmse  = np.sqrt(mean_squared_error(test_merged['y'], test_merged['yhat']))
mape  = (np.abs((test_merged['y'] - test_merged['yhat']) / test_merged['y'])).mean() * 100
mean_sales = test_merged['y'].mean()
within_ci  = ((test_merged['y'] >= test_merged['yhat_lower']) &
              (test_merged['y'] <= test_merged['yhat_upper'])).mean() * 100

print(f"\n── Forecast Metrics (90-day holdout) ──")
print(f"  MAE  : ${mae:,.1f}")
print(f"  RMSE : ${rmse:,.1f}")
print(f"  MAPE : {mape:.1f}%")
print(f"  Mean daily sales: ${mean_sales:,.1f}")
print(f"  Actuals within 95% CI: {within_ci:.1f}%")

# ── 7. Business insights ──────────────────────────────────────────────────────
weekly = df.copy()
weekly['DayOfWeek'] = weekly['Order Date'].dt.day_name()
dow_sales = weekly.groupby('DayOfWeek')['Sales'].mean().reindex(
    ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

best_day  = dow_sales.idxmax()
best_val  = dow_sales.max()
worst_day = dow_sales.idxmin()
worst_val = dow_sales.min()

monthly = df.copy()
monthly['Month'] = monthly['Order Date'].dt.month_name()
month_order = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
monthly_avg = monthly.groupby('Month')['Sales'].sum() / 4  # avg over 4 years
monthly_avg = monthly_avg.reindex(month_order)
peak_month  = monthly_avg.idxmax()
peak_val    = monthly_avg.max()

cat_sales = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
top_cat   = cat_sales.index[0]
top_pct   = (cat_sales.iloc[0] / cat_sales.sum()) * 100

future_90 = forecast[forecast['ds'] > daily['ds'].max()].head(90)
projected_total = future_90['yhat'].sum()
projected_daily = future_90['yhat'].mean()

print(f"\n── Business Insights ──")
print(f"  Best sales day   : {best_day} (avg ${best_val:,.0f})")
print(f"  Worst sales day  : {worst_day} (avg ${worst_val:,.0f})")
print(f"  Peak month       : {peak_month} (avg ${peak_val:,.0f}/year)")
print(f"  Top category     : {top_cat} ({top_pct:.1f}% of total sales)")
print(f"  Projected 90-day : ${projected_total:,.0f} (avg ${projected_daily:,.0f}/day)")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Full forecast with confidence interval
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5.5))

# Historical (weekly smoothed for readability)
hist_smooth = daily.set_index('ds')['y'].rolling(7).mean()
ax.plot(hist_smooth.index, hist_smooth.values,
        color='#555b7a', linewidth=1.2, label='Historical (7-day avg)', zorder=2)

# Test actuals
ax.scatter(test_merged['ds'], test_merged['y'],
           color=GREEN, s=12, zorder=4, alpha=0.7, label='Actual (holdout)')

# Forecast line
fc_plot = forecast[forecast['ds'] > cutoff]
ax.plot(fc_plot['ds'], fc_plot['yhat'],
        color=ACCENT, linewidth=2.2, label='Prophet Forecast', zorder=3)
ax.fill_between(fc_plot['ds'], fc_plot['yhat_lower'], fc_plot['yhat_upper'],
                color=ACCENT, alpha=0.15, label='95% Confidence Interval')

# Cutoff line
ax.axvline(cutoff, color=ORANGE, linewidth=1.4, linestyle='--', alpha=0.8)
ax.text(cutoff + pd.Timedelta(days=2), ax.get_ylim()[0]*1.02 + ax.get_ylim()[1]*0.02,
        'Train / Test split', color=ORANGE, fontsize=8)

ax.set_title('Superstore Daily Sales — Prophet Forecast (90-Day Horizon)',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Date', fontsize=10)
ax.set_ylabel('Daily Sales ($)', fontsize=10)
ax.legend(fontsize=9, framealpha=0.15)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=30, ha='right')
ax.grid(True, axis='y')
plt.tight_layout()
plt.savefig('charts/forecast_overview.png', dpi=160, bbox_inches='tight')
plt.close()
print("✓ Chart 1 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Seasonality decomposition (2 panels)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

# Weekly seasonality
week_comp = forecast[['ds','weekly']].copy()
week_comp['dow'] = week_comp['ds'].dt.dayofweek
week_avg  = week_comp.groupby('dow')['weekly'].mean()
days_lbl  = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
colors_w  = [GREEN if v == week_avg.max() else ACCENT for v in week_avg.values]
axes[0].bar(days_lbl, week_avg.values, color=colors_w, edgecolor='none', width=0.6)
axes[0].set_title('Weekly Seasonality Effect', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Sales Lift ($)', fontsize=10)
axes[0].axhline(0, color='#555b7a', linewidth=0.8)
axes[0].grid(True, axis='y')

# Yearly seasonality
year_comp = forecast[['ds','yearly']].copy()
year_comp['doy'] = year_comp['ds'].dt.dayofyear
year_avg  = year_comp.groupby('doy')['yearly'].mean()
axes[1].plot(year_avg.index, year_avg.values, color=ACCENT, linewidth=2)
axes[1].fill_between(year_avg.index, 0, year_avg.values,
                     where=(year_avg.values > 0), color=GREEN, alpha=0.25)
axes[1].fill_between(year_avg.index, 0, year_avg.values,
                     where=(year_avg.values < 0), color=RED_SOFT, alpha=0.2)
axes[1].set_title('Yearly Seasonality Effect', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Sales Lift ($)', fontsize=10)
axes[1].set_xlabel('Day of Year', fontsize=10)
axes[1].axhline(0, color='#555b7a', linewidth=0.8)
axes[1].grid(True, axis='y')

plt.suptitle('Prophet Seasonality Decomposition', fontsize=12,
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('charts/seasonality.png', dpi=160, bbox_inches='tight')
plt.close()
print("✓ Chart 2 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Category & region breakdown
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Category bar
cat_vals  = cat_sales.values
cat_lbls  = cat_sales.index.tolist()
cat_clrs  = [GREEN, ACCENT, ORANGE]
bars = axes[0].bar(cat_lbls, cat_vals, color=cat_clrs, width=0.55, edgecolor='none')
for bar, val in zip(bars, cat_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8000,
                 f'${val/1e6:.2f}M', ha='center', fontsize=9, color='#c8cfe0')
axes[0].set_title('Total Sales by Category (4 Years)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Total Sales ($)', fontsize=10)
axes[0].grid(True, axis='y')

# Region pie
region_sales = df.groupby('Region')['Sales'].sum()
wedge_colors = [ACCENT, GREEN, ORANGE, RED_SOFT]
axes[1].pie(region_sales.values, labels=region_sales.index,
            colors=wedge_colors, autopct='%1.1f%%',
            startangle=140, pctdistance=0.78,
            wedgeprops={'edgecolor': '#0f1117', 'linewidth': 1.5},
            textprops={'color': '#c8cfe0', 'fontsize': 9})
axes[1].set_title('Sales Distribution by Region', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('charts/category_region.png', dpi=160, bbox_inches='tight')
plt.close()
print("✓ Chart 3 saved")

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Monthly seasonality heatmap (year × month)
# ═══════════════════════════════════════════════════════════════════════════════
df['Year']  = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
pivot = df.pivot_table(values='Sales', index='Year', columns='Month', aggfunc='sum')
pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun',
                 'Jul','Aug','Sep','Oct','Nov','Dec']

fig, ax = plt.subplots(figsize=(13, 3.5))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='Blues',
            linewidths=0.5, linecolor='#0f1117',
            ax=ax, cbar_kws={'shrink': 0.7},
            annot_kws={'size': 8})
ax.set_title('Monthly Sales Heatmap by Year ($)', fontsize=11,
             fontweight='bold', pad=10)
ax.set_xlabel('Month', fontsize=10)
ax.set_ylabel('Year', fontsize=10)
plt.tight_layout()
plt.savefig('charts/monthly_heatmap.png', dpi=160, bbox_inches='tight')
plt.close()
print("✓ Chart 4 saved")

# ── Save results ──────────────────────────────────────────────────────────────
results = {
    "model": "Prophet (Meta)",
    "dataset": "Superstore Sales (2020–2023, 1,461 daily records)",
    "train_days": int(len(train)),
    "test_days":  int(len(test)),
    "metrics": {
        "MAE":  round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "actuals_within_95pct_CI": round(within_ci, 1),
    },
    "business_insights": {
        "best_sales_day":     best_day,
        "peak_month":         peak_month,
        "top_category":       top_cat,
        "top_category_share": round(top_pct, 1),
        "projected_90d_revenue": round(projected_total, 0),
        "projected_daily_avg":   round(projected_daily, 0),
    }
}
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n── All done! ──")
print(f"  results.json saved")
print(f"  4 charts saved to charts/")
