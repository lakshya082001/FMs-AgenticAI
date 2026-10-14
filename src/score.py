import os, pickle, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.preprocessing import OrdinalEncoder
from tabpfn import TabPFNClassifier

for line in open("../.env"):
    if line.startswith("tabpfn_key"):
        os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip("'\"")

CAT = ["Type", "Shipping Mode", "Market", "Order Region", "Category Name", "Customer Segment", "Department Name"]
NUM = ["Days for shipment (scheduled)", "Order Item Quantity", "Order Item Product Price",
       "Order Item Discount Rate", "Sales", "month", "dow", "hour"]
Y = "Late_delivery_risk"

tr, va, te = (pd.read_csv(f"data/{n}.csv") for n in ["train", "val", "test"])
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(tr[CAT])
X = lambda d: np.hstack([enc.transform(d[CAT]), d[NUM].to_numpy()])

clf = TabPFNClassifier(device="cpu", random_state=42, categorical_features_indices=list(range(len(CAT))))
clf.fit(X(tr), tr[Y])
pickle.dump((enc, clf), open("results/tabpfn.pkl", "wb"))

out = []
for name, d in [("val", va), ("test", te)]:
    t = time.time()
    p = clf.predict_proba(X(d))[:, 1]
    # ECE: 10 equal-width bins
    b = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(d[Y][b == i].mean() - p[b == i].mean()) * (b == i).mean() for i in range(10) if (b == i).any())
    print(f"{name}: auc={roc_auc_score(d[Y], p):.4f} brier={brier_score_loss(d[Y], p):.4f} "
          f"logloss={log_loss(d[Y], p):.4f} ece={ece:.4f} base_rate={d[Y].mean():.3f} time={time.time()-t:.1f}s")
    out.append(d[["Order Id", "Order Item Id", "Product Card Id", "order_date", Y]].assign(split=name, p_late=p))
pd.concat(out).to_csv("results/risk_scores.csv", index=False)
