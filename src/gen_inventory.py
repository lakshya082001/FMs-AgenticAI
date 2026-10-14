"""Synthetic inventory/supplier layer on top of DataCo (which has no inventory or supplier data).
Real: product ids, prices, avg daily demand = total ordered qty / days in dataset.
Synthetic: suppliers, lead times, reliability, on-hand stock, cost multipliers. Seeded, assumptions below."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
d = pd.read_csv("data/raw/dataco.csv", encoding="latin-1")
d.columns = d.columns.str.strip()
d = d[d["Delivery Status"] != "Shipping canceled"]
h = d.groupby("Product Card Id").agg(qty=("Order Item Quantity", "sum"), name=("Product Name", "first"),
                                     category=("Category Name", "first"), price=("Order Item Product Price", "mean"))
h.index.name = "Product Card Id"
h = h.reset_index()
days =(pd.Timestamp("2018-02-01") - pd.Timestamp("2015-01-01")).days

# 8 suppliers: lead time mean (days) ~ U(5,25), sd = 10-35% of mean, reliability (on-time share) ~ U(0.75,0.97)
sup = pd.DataFrame({"supplier_id": [f"S{i}" for i in range(1, 9)]})
sup["lead_mean"] = rng.uniform(5, 25, 8).round(1)
sup["lead_sd"] = (sup.lead_mean * rng.uniform(0.1, 0.35, 8)).round(1)
sup["reliability"] = rng.uniform(0.75, 0.97, 8).round(2)
# unit cost vs base: faster suppliers cost more (1.25 at 5d lead down to 0.95 at 25d) +/- 0.03 noise
sup["cost_mult"] = (1.25 - 0.30 * (sup.lead_mean - 5) / 20 + rng.uniform(-0.03, 0.03, 8)).round(2)
sup.to_csv("data/suppliers.csv", index=False)

sku = pd.DataFrame({"product_id": h["Product Card Id"], "name": h.name, "category": h.category, "price": h.price.round(2)})
sku["daily_demand"] = (h.qty / days).round(2)
sku["unit_cost"] = (sku.price * 0.6).round(2)  # assumption: 40% gross margin
# days of cover ~ U(3,30): some SKUs start thin, so a delay actually matters
sku["on_hand"] = (sku.daily_demand * rng.uniform(3, 30, len(sku))).round().astype(int)
ids = sup.supplier_id.to_numpy()
picks = [rng.choice(ids, 3, replace=False) for _ in range(len(sku))]
sku["primary_supplier"] = [p[0] for p in picks]
sku["alt_suppliers"] = ["|".join(p[1:]) for p in picks]
sku.to_csv("data/sku.csv", index=False)
print(sku.shape, sup.shape, sku.daily_demand.describe()[["min", "50%", "max"]].to_dict())
