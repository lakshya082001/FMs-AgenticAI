# Supplier disruption detection + replenishment agents

Write-up: [write-up/writeup.pdf](write-up/writeup.pdf) · Results, schema, assumptions: [DETAILS.md](DETAILS.md) · Data: [DATA.md](DATA.md)

## Problem statement
Detect supplier disruptions early from operational data, then recommend replenishment actions (wait / expedite / switch supplier) before stock-outs, with a human sign-off on risky plans.

## Domain and track
**Scenario S1 (Supply Chain Risk & Optimization), Track A (Agentic / Multi-Agent AI).**

## Data source
DataCo Smart Supply Chain (Mendeley Data, DOI 10.17632/8gx2fvg2k6, CC BY 4.0): real orders and shipping times. It has no supplier or inventory data and no temporal dynamics, so a supplier/inventory layer and disruption episodes are **simulated** (seeded scripts, assumptions in DATA.md).

## Flow
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
LLMs never compute numbers; comparisons (alert status, hedge word) are done in code.

## Setup
Python 3.13.5. Needs `../.env` (outside the repo) with `tabpfn_key`, `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `MODEL` (OpenAI-compatible gateway). The first TabPFN run downloads ~1.5 GB of weights (license acceptance on the Prior Labs account required).
```
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```
Download `DataCoSupplyChainDataset.csv` (Mendeley 8gx2fvg2k6, latin-1) to `data/raw/dataco.csv`.

## Reproduce end to end
```
python src/gen_inventory.py      # data/suppliers.csv, data/sku.csv
python src/disrupt.py            # data/episodes.csv, data/train|val|test.csv
python src/score.py              # fits TabPFN, prints metrics, writes results/risk_scores.csv, results/tabpfn.pkl (~850 MB, git-ignored)
python src/run.py --supplier S2 --date 2017-02-16 --name success --decision approve   # writes results/traces/success.json
```
The exact command behind every committed trace is listed in DETAILS.md.
