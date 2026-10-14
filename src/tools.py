import math, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import llm  # noqa: F401  (loads .env, incl. TabPFN token)
from schema import ExposedSKU, ExposureReport, Verdict

X = ["r3", "r7", "r14", "r30", "r90", "n7", "dev3", "dev7", "days_since_alert", "n_overdue", "od_share"]
# replenishment assumptions (synthetic, see README)
DELAY = 3.5            # expected extra days when disrupted (sim delays U{2..5})
Z, SD_FRAC, CYCLE = 1.65, 0.5, 14  # service z, demand sd as share of mean, order cycle days
EXPEDITE_CUT, EXPEDITE_PREMIUM = 0.4, 0.15
BAND, COST_GATE, TOP_N = 0.10, 10000.0, 5

sku = pd.read_csv("data/sku.csv").set_index("product_id")
sup = pd.read_csv("data/suppliers.csv").set_index("supplier_id")
train = pd.read_csv("data/train.csv")
_clf = None


def threshold():
    v = pd.read_csv("results/risk_scores.csv").query("split == 'val'")
    return float(max(np.arange(0.1, 0.9, 0.05), key=lambda c: f1_score(v.disrupted, v.p_disrupt >= c)))


def validate_row(row):
    issues = [f"missing {c}" for c in X if c not in row or pd.isna(row[c])]
    if "supplier" not in row or row["supplier"] not in sup.index:
        issues.append(f"unknown supplier {row.get('supplier')}")
    for c in ["r3", "r7", "r14", "r30", "r90", "od_share"]:
        if c in row and not pd.isna(row[c]) and not 0 <= row[c] <= 1:
            issues.append(f"{c} out of [0,1]")
    return issues


def predict(row):
    global _clf
    _clf = _clf or pickle.load(open("results/tabpfn.pkl", "rb"))
    return float(_clf.predict_proba(pd.DataFrame([row])[X])[0, 1])


def evidence(row):
    med = train[X].median()
    return [
        f"7d late rate {row['r7']:.2f} vs 90d baseline {row['r90']:.2f} ({row['dev7']:+.2f})",
        f"3d late rate {row['r3']:.2f} vs 30d {row['r30']:.2f} ({row['dev3']:+.2f})",
        f"in-transit orders past promised date: {row['n_overdue']:.0f} ({row['od_share']:.0%} of in-transit; training median {med['od_share']:.0%})",
        f"days since last observed alert: {row['days_since_alert']:.0f}",
    ]


def get_exposure(supplier):
    s = sku[sku.primary_supplier == supplier]
    lead = sup.loc[supplier, "lead_mean"] + DELAY
    cover = s.on_hand / s.daily_demand
    short = (lead - cover).clip(lower=0)
    value = short * s.daily_demand * s.price
    rows = [ExposedSKU(sku_id=i, name=s.loc[i, "name"], days_cover=round(cover[i], 1), lead_days=round(lead, 1),
                       shortfall_days=round(short[i], 1), value_at_risk=round(value[i], 2))
            for i in value[short > 0].sort_values(ascending=False).index]
    return ExposureReport(supplier=supplier, n_skus=len(s), skus=rows)


def calc_options(sku_id):
    s, p = sku.loc[sku_id], sup.loc[sku.loc[sku_id, "primary_supplier"]]
    cover = s.on_hand / s.daily_demand

    def opt(name, lead, mult_from, mult_to):
        qty = math.ceil(max(0, s.daily_demand * (lead + CYCLE) + Z * SD_FRAC * s.daily_demand * math.sqrt(lead) - s.on_hand))
        short = max(0, lead - cover)
        return {"option": name, "lead_days": round(float(lead), 1), "order_qty": int(qty), "shortfall_days": round(float(short), 1),
                "incremental_cost": round(float(qty * s.unit_cost * (mult_to - mult_from)), 2),
                "lost_sales_value": round(float(short * s.daily_demand * s.price), 2)}

    out = [opt("wait", p.lead_mean + DELAY, p.cost_mult, p.cost_mult),
           opt("expedite", p.lead_mean * (1 - EXPEDITE_CUT) + DELAY, p.cost_mult, p.cost_mult * (1 + EXPEDITE_PREMIUM))]
    for a in s.alt_suppliers.split("|"):  # alternates assumed undisrupted; expected delay scales with (1 - reliability)
        out.append(opt(f"switch_{a}", sup.loc[a, "lead_mean"] + (1 - sup.loc[a, "reliability"]) * DELAY, p.cost_mult, sup.loc[a, "cost_mult"]))
    return out


def check_plan(plan, exposure, options):
    errs, exposed = [], {k.sku_id for k in exposure.skus}
    for it in plan.items:
        if it.sku_id not in exposed:
            errs.append(f"sku {it.sku_id} not in exposure report")
            continue
        row = next((o for o in options[it.sku_id] if o["option"] == it.option), None)
        if row is None:
            errs.append(f"sku {it.sku_id}: unknown option {it.option}")
            continue
        for f in ["shortfall_days", "incremental_cost", "lost_sales_value"]:
            if abs(getattr(it, f) - row[f]) > 0.01:
                errs.append(f"sku {it.sku_id} {it.option}: {f} {getattr(it, f)} != tool value {row[f]}")
    return Verdict(ok=not errs, errors=errs)
