import pandas as pd
import numpy as np
from pathlib import Path

print(" Starting Retail ETL Pipeline...")

# ==========================================================
# PATH SETUP
# ==========================================================

BASE_PATH = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_PATH / "data" / "raw"
CURATED_PATH = BASE_PATH / "data" / "curated"
CURATED_PATH.mkdir(parents=True, exist_ok=True)

# ==========================================================
# LOAD RAW FILES
# ==========================================================

sales = pd.read_csv(RAW_PATH / "sales_daily.csv")
inventory = pd.read_csv(RAW_PATH / "inventory_daily.csv")
products = pd.read_json(RAW_PATH / "products.json")
calendar = pd.read_csv(RAW_PATH / "calendar.csv")
purchase_orders = pd.read_csv(RAW_PATH / "purchase_orders.csv")

print(" Raw files loaded")

# ==========================================================
# STANDARDIZE COLUMN NAMES
# ==========================================================

for df in [sales, inventory, products, calendar, purchase_orders]:
    df.columns = df.columns.str.strip().str.lower()

# ==========================================================
# DATE CONVERSION
# ==========================================================

sales['date'] = pd.to_datetime(sales.get('date'), errors='coerce')
inventory['date'] = pd.to_datetime(inventory.get('date'), errors='coerce')
calendar['date'] = pd.to_datetime(calendar.get('date'), errors='coerce')

sales = sales.dropna(subset=['date'])
inventory = inventory.dropna(subset=['date'])
calendar = calendar.dropna(subset=['date'])

# ==========================================================
# CLEANING
# ==========================================================

sales['units_sold'] = pd.to_numeric(
    sales.get('units_sold'), errors='coerce'
).fillna(0)

inventory = inventory.sort_values(['store_id', 'sku_id', 'date'])
inventory['on_hand_close'] = inventory.groupby(
    ['store_id', 'sku_id']
)['on_hand_close'].ffill().fillna(0)

if 'category' in products.columns:
    products['category'] = products['category'].str.lower().str.strip()

# ==========================================================
# FACT SALES
# ==========================================================

products.rename(columns={
    'unit_price': 'price',
    'unit_cost': 'cost'
}, inplace=True)

sales = sales.merge(
    products[['sku_id', 'price', 'cost']],
    on='sku_id',
    how='left'
)

sales[['price','cost']] = sales[['price','cost']].fillna(0)

calendar_cols = ['date']
if 'promo_flag' in calendar.columns:
    calendar_cols.append('promo_flag')
if 'holiday_flag' in calendar.columns:
    calendar_cols.append('holiday_flag')

sales = sales.merge(
    calendar[calendar_cols].drop_duplicates(),
    on='date',
    how='left'
)

sales['promo_flag'] = sales.get('promo_flag', 0)
sales['holiday_flag'] = sales.get('holiday_flag', 0)

sales['promo_flag'] = sales['promo_flag'].fillna(0).astype(int)
sales['holiday_flag'] = sales['holiday_flag'].fillna(0).astype(int)

sales['day_of_week'] = sales['date'].dt.day_name()
sales['revenue'] = sales['units_sold'] * sales['price']
sales['margin_proxy'] = sales['units_sold'] * (sales['price'] - sales['cost'])

fact_sales = sales[[
    'date','store_id','sku_id','units_sold',
    'revenue','margin_proxy','promo_flag',
    'holiday_flag','day_of_week'
]]

fact_sales.to_csv(CURATED_PATH / "fact_sales_store_sku_daily.csv", index=False)
print(" fact_sales created")

# ==========================================================
# 4 WEEK WINDOW
# ==========================================================

latest_date = sales['date'].max()
cutoff_date = latest_date - pd.Timedelta(days=28)
sales_4w = sales[sales['date'] >= cutoff_date].copy()

# ==========================================================
# FACT INVENTORY
# ==========================================================

avg_demand_4w = sales_4w.groupby(
    ['store_id','sku_id']
)['units_sold'].mean().reset_index()

avg_demand_4w.rename(
    columns={'units_sold':'avg_daily_demand_4w'},
    inplace=True
)

inventory = inventory.merge(
    avg_demand_4w,
    on=['store_id','sku_id'],
    how='left'
)

inventory['avg_daily_demand_4w'] = inventory['avg_daily_demand_4w'].fillna(0)
inventory['stockout_flag'] = inventory['on_hand_close'] == 0

inventory['days_of_cover'] = np.where(
    inventory['avg_daily_demand_4w'] > 0,
    inventory['on_hand_close'] / inventory['avg_daily_demand_4w'],
    0
)

fact_inventory = inventory[[
    'date','store_id','sku_id',
    'on_hand_close','stockout_flag','days_of_cover'
]].rename(columns={'on_hand_close':'on_hand_units'})

fact_inventory.to_csv(CURATED_PATH / "fact_inventory_store_sku_daily.csv", index=False)
print(" fact_inventory created")

# ==========================================================
# REPLENISHMENT INPUTS
# ==========================================================

demand_stats = sales_4w.groupby(
    ['store_id','sku_id']
)['units_sold'].agg(['mean','std']).reset_index()

demand_stats.rename(columns={
    'mean':'avg_daily_demand',
    'std':'demand_std_dev'
}, inplace=True)

demand_stats['demand_std_dev'] = demand_stats['demand_std_dev'].fillna(0)

# ==========================================================
# SAFE LEAD TIME (SILENT)
# ==========================================================

lead_time_df = demand_stats[['store_id','sku_id']].copy()
lead_time_df['lead_time_days'] = 7   # default

if not purchase_orders.empty:

    purchase_orders.columns = purchase_orders.columns.str.strip().str.lower()

    order_col = next((c for c in purchase_orders.columns
                      if 'order' in c and 'date' in c), None)

    delivery_col = next((c for c in purchase_orders.columns
                         if ('deliver' in c or 'receive' in c) and 'date' in c), None)

    if order_col and delivery_col:

        purchase_orders[order_col] = pd.to_datetime(
            purchase_orders[order_col], errors='coerce'
        )
        purchase_orders[delivery_col] = pd.to_datetime(
            purchase_orders[delivery_col], errors='coerce'
        )

        purchase_orders['lead_time'] = (
            purchase_orders[delivery_col] -
            purchase_orders[order_col]
        ).dt.days.clip(lower=0)

        real_lead = purchase_orders.groupby(
            ['store_id','sku_id']
        )['lead_time'].mean().reset_index()

        real_lead.rename(columns={'lead_time':'lead_time_days'}, inplace=True)

        lead_time_df = real_lead

demand_stats = demand_stats.merge(
    lead_time_df,
    on=['store_id','sku_id'],
    how='left'
)

demand_stats['lead_time_days'] = demand_stats['lead_time_days'].fillna(7)

# ==========================================================
# SERVICE LEVEL + ROP
# ==========================================================

if 'category' in products.columns:
    demand_stats = demand_stats.merge(
        products[['sku_id','category']],
        on='sku_id',
        how='left'
    )
else:
    demand_stats['category'] = "other"

def assign_service_level(cat):
    if cat == 'electronics':
        return 0.95
    elif cat == 'grocery':
        return 0.90
    else:
        return 0.92

demand_stats['service_level_target'] = demand_stats['category'].apply(assign_service_level)

z_map = {0.95:1.65, 0.90:1.28, 0.92:1.41}
demand_stats['z_value'] = demand_stats['service_level_target'].map(z_map)

demand_stats['safety_stock'] = (
    demand_stats['z_value'] *
    demand_stats['demand_std_dev'] *
    np.sqrt(demand_stats['lead_time_days'])
)

demand_stats['reorder_point'] = (
    demand_stats['avg_daily_demand'] *
    demand_stats['lead_time_days']
) + demand_stats['safety_stock']

demand_stats['recommended_order_qty'] = (
    demand_stats['reorder_point']
).clip(lower=0).round()

replenishment = demand_stats[[
    'store_id','sku_id','avg_daily_demand',
    'demand_std_dev','lead_time_days',
    'service_level_target','safety_stock',
    'reorder_point','recommended_order_qty'
]]

replenishment.to_csv(
    CURATED_PATH / "replenishment_inputs_store_sku.csv",
    index=False
)

print(" replenishment_inputs created")
print(" ETL Pipeline Completed Successfully!")


# ==========================================================
# DATA QUALITY REPORT (NON-INTRUSIVE)
# ==========================================================

print("\n Generating Data Quality Report...")

def generate_quality_metrics(df, table_name):
    return {
        "table_name": table_name,
        "row_count": len(df),
        "duplicate_rows": df.duplicated().sum(),
        "total_null_values": df.isnull().sum().sum(),
        "columns_with_nulls": df.isnull().sum()[df.isnull().sum() > 0].count()
    }

quality_data = []

quality_data.append(generate_quality_metrics(fact_sales, "fact_sales"))
quality_data.append(generate_quality_metrics(fact_inventory, "fact_inventory"))
quality_data.append(generate_quality_metrics(replenishment, "replenishment"))

# Business Rule Checks
negative_sales = (fact_sales['units_sold'] < 0).sum()
negative_inventory = (fact_inventory['on_hand_units'] < 0).sum()
future_dates = (fact_sales['date'] > pd.Timestamp.today()).sum()

business_checks = {
    "negative_sales_records": negative_sales,
    "negative_inventory_records": negative_inventory,
    "future_date_records": future_dates,
    "latest_sales_date": str(fact_sales['date'].max()),
    "latest_inventory_date": str(fact_inventory['date'].max())
}

quality_df = pd.DataFrame(quality_data)
quality_df.to_csv(CURATED_PATH / "data_quality_summary.csv", index=False)

business_df = pd.DataFrame([business_checks])
business_df.to_csv(CURATED_PATH / "data_quality_business_checks.csv", index=False)

print(" Data Quality Report Created")