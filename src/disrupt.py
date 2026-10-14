"""DataCo has no temporal dynamics (lane-level autocorrelation ~0), so overlay simulated supplier disruption episodes.
Latent episodes raise P(late) for a supplier's orders; the model only sees trailing outcomes of already-delivered orders.
Output: supplier-day table, target = supplier is inside an episode on that day. Measures detection of simulated
disruptions, not real-world accuracy."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
d = pd.read_csv("data/raw/dataco.csv", encoding="latin-1")
d.columns = d.columns.str.strip()
d = d[d["Delivery Status"] != "Shipping canceled"].reset_index(drop=True)
t = pd.to_datetime(d["order date (DateOrders)"]).to_numpy()
done = (pd.to_datetime(d["shipping date (DateOrders)"]) + pd.to_timedelta(d["Days for shipping (real)"], unit="D")).to_numpy()
sup = d["Product Card Id"].map(pd.read_csv("data/sku.csv").set_index("product_id").primary_supplier).to_numpy()

# episodes: gap ~ Exp(75d), duration ~ U(7,21)d, severity = P(on-time order flips to late) ~ U(0.35,0.7)
start, end = pd.Timestamp("2015-01-01"), pd.Timestamp("2018-02-01")
eps = []
for s in sorted(set(sup)):
    x = start + pd.Timedelta(days=float(rng.exponential(60)))
    while x < end:
        dur = rng.uniform(7, 21)
        eps.append((s, x, x + pd.Timedelta(days=dur), rng.uniform(0.35, 0.7)))
        x += pd.Timedelta(days=dur + float(rng.exponential(75)))
eps = pd.DataFrame(eps, columns=["supplier", "start", "end", "severity"])
eps.to_csv("data/episodes.csv", index=False)

sev = np.zeros(len(d))
for e in eps.itertuples():
    sev[(sup == e.supplier) & (t >= e.start.to_datetime64()) & (t < e.end.to_datetime64())] = e.severity
orig = d.Late_delivery_risk.to_numpy()
late = np.where((orig == 0) & (rng.random(len(d)) < sev), 1, orig)

# supplier-day features at day start, from deliveries completed strictly before it (no leakage)
days = pd.date_range("2015-04-01", "2017-09-30")  # first 90d dropped: warm-up for the 90d window
D = days.to_numpy()
frames = []
for s in sorted(set(sup)):
    ix = np.where(sup == s)[0]
    o = ix[np.argsort(done[ix])]
    comp, cs = done[o], np.concatenate([[0], np.cumsum(late[o])])
    hi = np.searchsorted(comp, D, "left")
    f = pd.DataFrame({"supplier": s, "date": days})
    for W in [3, 7, 14, 30, 90]:
        n = hi - np.searchsorted(comp, D - np.timedelta64(W, "D"), "left")
        f[f"r{W}"] = np.where(n > 0, (cs[hi] - cs[hi - n]) / np.maximum(n, 1), np.nan)
        if W == 7:
            f["n7"] = n
    f["dev3"], f["dev7"] = f.r3 - f.r30, f.r7 - f.r90
    pos = np.arange(len(f))
    last = pd.Series(np.where((f.dev7 > 0.10) & (f.n7 >= 20), pos, np.nan)).ffill()  # observed-only alert
    f["days_since_alert"] = (pos - last).fillna(60).clip(upper=60)
    e = eps[eps.supplier == s]
    f["disrupted"] = np.any([(D >= a) & (D < b) for a, b in zip(e.start.to_numpy(), e.end.to_numpy())], axis=0).astype(int)
    frames.append(f)
f = pd.concat(frames)

for name, (a, b) in {"train": ("2015-04-01", "2016-06-30"), "val": ("2016-07-01", "2016-12-31"), "test": ("2017-01-01", "2017-09-30")}.items():
    s = f[(f.date >= a) & (f.date <= b)]
    s.to_csv(f"data/{name}.csv", index=False)
    print(name, s.shape, f"disrupted={s.disrupted.mean():.3f}")
print(f"episodes={len(eps)} late: orig={orig.mean():.3f} sim={late.mean():.3f}")
