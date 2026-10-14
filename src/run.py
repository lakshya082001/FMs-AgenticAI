"""python src/run.py --supplier S3 --date 2017-03-14 --name success [--decision approve|reject] ; TAMPER=1 injects a bad planner number"""
import argparse, json, time
import pandas as pd
from langgraph.types import Command
import llm
from graph import build

a = argparse.ArgumentParser()
a.add_argument("--supplier", required=True)
a.add_argument("--date", required=True)
a.add_argument("--name", required=True)
a.add_argument("--decision", choices=["approve", "reject"])  # omit to be prompted
a.add_argument("--drop", help="comma-separated feature columns to blank (validation edge case)")
a = a.parse_args()

t = pd.read_csv("data/test.csv")
row = t[(t.supplier == a.supplier) & (t.date == a.date)].iloc[0].to_dict()
for c in (a.drop or "").split(","):
    if c:
        row[c] = None

app, cfg, t0 = build(), {"configurable": {"thread_id": a.name}}, time.time()
out = app.invoke({"row": row, "trace": [], "retries": 0}, cfg)
if "__interrupt__" in out:
    print(json.dumps(out["__interrupt__"][0].value, indent=1))
    ok = (a.decision or input("approve? [y/n] ")[:1].lower() in ("approve", "y")) in (True, "approve")
    out = app.invoke(Command(resume={"approved": ok}), cfg)

os_path = f"results/traces/{a.name}.json"
json.dump({"final": {k: out.get(k) for k in ["risk", "exposure", "plan", "verdict", "human"]}, "trace": out["trace"],
           "llm_usage": llm.usage, "seconds": round(time.time() - t0, 1)}, open(os_path, "w"), indent=1, default=str)
print([f"{e['node']}:{e.get('next', '')}" for e in out["trace"]], llm.usage, f"{time.time() - t0:.0f}s ->", os_path)
