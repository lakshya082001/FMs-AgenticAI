import os, pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score
from tabpfn import TabPFNClassifier

for line in open("../.env"):
    if line.startswith("tabpfn_key"):
        os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip("'\"")

X = ["r3", "r7", "r14", "r30", "r90", "n7", "dev3", "dev7", "days_since_alert"]
Y = "disrupted"
tr, va, te = (pd.read_csv(f"data/{n}.csv", parse_dates=["date"]) for n in ["train", "val", "test"])

clf = TabPFNClassifier(device="cpu", random_state=42)
clf.fit(tr[X], tr[Y])
pickle.dump(clf, open("results/tabpfn.pkl", "wb"))

va["p"], te["p"] = (clf.predict_proba(d[X])[:, 1] for d in (va, te))
thr = max(np.arange(0.1, 0.9, 0.05), key=lambda c: f1_score(va[Y], va.p >= c))  # threshold picked on val only

for name, d in [("val", va), ("test", te)]:
    b = np.minimum((d.p * 10).astype(int), 9)
    ece = sum(abs(d[Y][b == i].mean() - d.p[b == i].mean()) * (b == i).mean() for i in range(10) if (b == i).any())
    print(f"{name}: auc={roc_auc_score(d[Y], d.p):.3f} ap={average_precision_score(d[Y], d.p):.3f} brier={brier_score_loss(d[Y], d.p):.3f} "
          f"ece={ece:.3f} f1@{thr:.2f}={f1_score(d[Y], d.p >= thr):.3f} base={d[Y].mean():.3f} | baselines auc: r7={roc_auc_score(d[Y], d.r7):.3f} dev7={roc_auc_score(d[Y], d.dev7):.3f}")

# detection delay on test episodes: days from episode start until first flag
eps = pd.read_csv("data/episodes.csv", parse_dates=["start", "end"])
eps = eps[(eps.start >= te.date.min()) & (eps.start <= te.date.max())]
delays = []
for e in eps.itertuples():
    g = te[(te.supplier == e.supplier) & (te.date >= e.start.normalize() + pd.Timedelta(days=1)) & (te.date < e.end) & (te.p >= thr)]
    delays.append((g.date.min() - e.start).days if len(g) else None)
hit = [x for x in delays if x is not None]
print(f"test episodes={len(eps)} detected={len(hit)} median_delay_days={np.median(hit):.1f} false_alarm_rate={(te.p[te[Y] == 0] >= thr).mean():.3f}")

pd.concat([va.assign(split="val"), te.assign(split="test")])[["supplier", "date", "split", "p", Y]].rename(columns={"p": "p_disrupt"}).to_csv("results/risk_scores.csv", index=False)
