# Supplier disruption detection + replenishment agents (Track A, Scenario S1)

**Problem:** detect supplier disruptions early, then recommend replenishment actions.
**Track A (multi-agent), Scenario S1 (supply chain risk).** TabPFN scores supplier-day disruption risk; LLM agents explain, find exposed SKUs and plan actions; a human approves risky plans.

Walkthrough video: _TODO link_

Data: DataCo (real) + simulated supplier disruptions, see [DATA.md](DATA.md). Results, traces, schema, assumptions, limitations: see [DETAILS.md](DETAILS.md).

## Pipeline
```
supplier-day row -> Orchestrator (deterministic router, logs every decision)
   -> Risk Analyst      validate row, TabPFN inference (pickle), evidence + hedged summary   [1 LLM call]
        p < threshold  -> end ("not at risk" summary)
   -> Exposure Analyst  SKUs whose days of cover < disruption-adjusted lead time           [1 LLM call]
   -> Planner           picks wait / expedite / switch_Sx per SKU from tool-computed table [1 LLM call]
   -> check (code)      numbers match tool output; choice within 10% of cheapest option
        fail -> Planner retries once -> still failing: human
   -> human gate        interrupt if p is borderline, cost > 10,000, or check failed
```
LLMs never compute numbers; comparisons (alert status, hedge word) are done in code because the LLM got them wrong in testing (e.g. "0.27 is below 0.25").
Why agents rather than one prompt: separate tools and typed outputs per role, and an independent check between generation and action.

## Reproduce
Python 3.13. Needs `../.env` (outside the repo) with `tabpfn_key`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `MODEL` (OpenAI-compatible gateway). First TabPFN run downloads ~1.5 GB of weights (license acceptance on the Prior Labs account required).
```
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# put DataCoSupplyChainDataset.csv at data/raw/dataco.csv (Mendeley 8gx2fvg2k6, latin-1)
python src/gen_inventory.py      # suppliers.csv, sku.csv
python src/disrupt.py            # episodes.csv, train/val/test.csv
.venv/bin/python src/score.py    # fits TabPFN, writes results/tabpfn.pkl (~850 MB, git-ignored), risk_scores.csv
.venv/bin/python src/run.py --supplier S2 --date 2017-02-16 --name success   # prompts for approval unless --decision approve|reject
```
`TAMPER=1` injects a wrong number into the first plan (fault injection); `--drop r7,r90` blanks features (validation edge case).

