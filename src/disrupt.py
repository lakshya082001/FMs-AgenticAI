"""DataCo has no temporal dynamics (lane-level autocorrelation ~0), so overlay simulated supplier disruption episodes.
Latent episodes raise P(late) for a supplier's orders; the model only sees trailing outcomes of already-delivered orders.
Evaluation therefore measures detection of simulated disruptions, not real-world accuracy."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
d = pd.read_csv("data/raw/dataco.csv", encoding="latin-1")
d.columns = d.columns.str.strip()
d = d[d["Delivery Status"] != "Shipping canceled"].reset_index(drop=True)
t = pd.to_datetime(d["order date (DateOrders)"]).to_numpy()
done = (pd.to_datetime(d["shipping date (DateOrders)"]) + pd.to_timedelta(d["Days for shipping (real)"], unit="D")).to_numpy()
d["supplier"] = d["Product Card Id"].map(pd.read_csv("data/sku.csv").set_index("product_id").primary_supplier)

# episodes: gap ~ Exp(75d), duration ~ U(7,21)d, severity = P(on-time order flips to late) ~ U(0.35,0.7)
start, end = pd.Timestamp("2015-01-01"), pd.Timestamp("2018-02-01")
eps = []
for s in sorted(d.supplier.unique()):
    x = start + pd.Timedelta(days=float(rng.exponential(60)))
    while x < end:
        dur = rng.uniform(7, 21)
        eps.append((s, x, x + pd.Timedelta(days=dur), rng.uniform(0.35, 0.7)))
        x += pd.Timedelta(days=dur + float(rng.exponential(75)))
eps = pd.DataFrame(eps, columns=["supplier", "start", "end", "severity"])
eps.to_csv("data/episodes.csv", index=False)

sev = np.zeros(len(d))
for e in eps.itertuples():
    sev[(d.supplier.to_numpy() == e.supplier) & (t >= e.start.to_datetime64()) & (t < e.end.to_datetime64())] = e.severity
orig = d.Late_delivery_risk.to_numpy()
late = np.where((orig == 0) & (rng.random(len(d)) < sev), 1, orig)

# trailing supplier features from deliveries completed strictly before order time (no leakage)
r7, r30, n7 = (np.full(len(d), np.nan) for _ in range(3))
for s in d.supplier.unique():
    ix = np.where(d.supplier.to_numpy() == s)[0]
    o = ix[np.argsort(done[ix])]
    comp, cs = done[o], np.concatenate([[0], np.cumsum(late[o])])
    hi = np.searchsorted(comp, t[ix], "left")
    for W, r in [(7, r7), (30, r30)]:
        lo = np.searchsorted(comp, t[ix] - np.timedelta64(W, "D"), "left")
        n = hi - lo
        r[ix] = np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)
        if W == 7:
            n7[ix] = n

out = pd.DataFrame({
    "Order Item Id": d["Order Item Id"], "Late_delivery_risk": late, "late_orig": orig, "in_episode": (sev > 0).astype(int),
    "supplier": d.supplier, "sup_late_7d": r7, "sup_late_30d": r30, "sup_trend": r7 - r30, "sup_n7": n7,
    "is_peak": (pd.to_datetime(d["order date (DateOrders)"]).dt.month >= 10).astype(int),
})
out.to_csv("data/sim_features.csv", index=False)
print(f"episodes={len(eps)} in_episode_share={out.in_episode.mean():.3f} late: orig={orig.mean():.3f} sim={late.mean():.3f}")
