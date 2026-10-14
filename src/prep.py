import pandas as pd

SEED = 42
RAW = "data/raw/dataco.csv"

# order-time features only: no actual shipping days, delivery status, order status or ship date (leakage)
FEATS = ["Type", "Days for shipment (scheduled)", "Shipping Mode", "Market", "Order Region",
         "Category Name", "Customer Segment", "Department Name", "Order Item Quantity",
         "Order Item Product Price", "Order Item Discount Rate", "Sales",
         "supplier", "sup_late_7d", "sup_late_30d", "sup_trend", "sup_n7", "is_peak"]  # last 6 from disrupt.py
KEEP = ["Order Id", "Order Item Id", "Product Card Id", "Product Name", "order_date", "Late_delivery_risk", "late_orig", "in_episode"]

d = pd.read_csv(RAW, encoding="latin-1")
d.columns = d.columns.str.strip()
d = d[d["Delivery Status"] != "Shipping canceled"]  # no delivery outcome, label is a artefact
d = d.drop(columns="Late_delivery_risk").merge(pd.read_csv("data/sim_features.csv"), on="Order Item Id")  # simulated label + trailing features
d["order_date"] =pd.to_datetime(d["order date (DateOrders)"])
d = d[KEEP + FEATS]
d["month"], d["dow"], d["hour"] = d.order_date.dt.month, d.order_date.dt.dayofweek, d.order_date.dt.hour

# temporal split: sample from each window so TabPFN stays small
windows = {
    "train": (d.order_date < "2016-07-01", 4000),
    "val": ((d.order_date >= "2016-07-01") & (d.order_date < "2017-01-01"), 1500),
    "test": ((d.order_date >= "2017-01-01") & (d.order_date < "2017-10-01"), 2000),
}
for name, (mask, n) in windows.items():
    s = d[mask].sample(n, random_state=SEED).sort_values("order_date")
    s.to_csv(f"data/{name}.csv", index=False)
    print(name, s.shape, f"late={s.Late_delivery_risk.mean():.3f}")

# SKU catalog + demand totals (demand is not a model feature, so no leakage); covers products launched in 2017
d.groupby("Product Card Id").agg(
    qty=("Order Item Quantity", "sum"), name=("Product Name", "first"),
    category=("Category Name", "first"), price=("Order Item Product Price", "mean"),
).to_csv("data/sku_history.csv")
