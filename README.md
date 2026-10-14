# Supplier disruption detection + replenishment agents (Track A, Scenario S1)

**Problem:** detect supplier disruptions early, then recommend replenishment actions.
**Track A (multi-agent), Scenario S1 (supply chain risk).** TabPFN scores supplier-day disruption risk; LLM agents explain, find exposed SKUs and plan actions; a human approves risky plans.

Walkthrough video: _TODO link_

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

## Results (TabPFN, 3,656 train rows, CPU, seed 42)
| Split | AUC | Avg precision | Brier | ECE | Base rate |
|---|---|---|---|---|---|
| Val | 0.659 | 0.298 | 0.121 | 0.071 | 0.145 |
| Test | 0.652 | 0.402 | 0.172 | 0.091 | 0.234 |

Alert threshold 0.25 (best F1 on val). On the 35 test episodes: 27 detected, median delay 5 days, false-alarm rate 17.4% of non-disrupted days.
**TabPFN does not beat the single feature `od_share`** (val AUC 0.689, test 0.664). Gains over the first static-feature attempt came from the in-transit overdue signal, not from the model. No XGBoost/other baseline was run. Single seed, ~35 test episodes: differences of ~0.01 AUC are noise.
Trailing-feature history (first simulation flipped labels only, AUC 0.62/0.66 with 9 features) is in git history.

## Agent traces (`results/traces/`)
| File | Case | Path |
|---|---|---|
| success | S2 2017-02-16, p=0.71 | check caught a suboptimal choice, retry fixed it, auto-approved |
| success_human | S6 2017-07-10, p=0.31 | retry fixed a ~15K-worse option; borderline p -> human approves |
| safe | S1 2017-03-02, p=0.03 | not at risk, 1 LLM call |
| edge_tamper | S1 2017-07-12 (TAMPER=1) | injected wrong number + suboptimal choice caught, retry passes |
| edge_borderline | S1 2017-01-24, p=0.27 | human gate, human rejects |
| edge_check_escalation | S1 2017-07-10 | Planner keeps a >10% worse option after retry -> escalated to human |
| edge_false_alarm | S7 2017-01-01, p=0.71, not actually disrupted | pipeline acts on a false alarm; low cost so no human gate |
| edge_invalid | blanked features | rejected at validation, 0 LLM calls |

A run takes ~40-45 s and 1-4 LLM calls (~3K tokens); most of the time is single-row TabPFN inference on CPU (~30 s).

## Message schema (`src/schema.py`, pydantic)
`RiskAssessment` -> `ExposureReport` -> `ActionPlan` -> `Verdict`. Examples (from `success.json`):
```json
{"supplier": "S2", "date": "2017-02-16", "p_disrupt": 0.705, "threshold": 0.25, "at_risk": true, "borderline": false,
 "evidence": ["7d late rate 0.61 vs 90d baseline 0.56 (+0.06)", "in-transit orders past promised date: 58 (39% of in-transit; training median 29%)"],
 "summary": "...", "error": null}
{"supplier": "S2", "n_skus": 10, "skus": [{"sku_id": 627, "days_cover": 10.9, "lead_days": 17.3, "shortfall_days": 6.4, "value_at_risk": 6854.61}]}
{"items": [{"sku_id": 502, "option": "switch_S5", "shortfall_days": 0.0, "incremental_cost": 2458.0, "lost_sales_value": 0.0, "rationale": "..."}]}
{"ok": true, "errors": []}
```

## Assumptions (all synthetic, see `src/tools.py`, `gen_inventory.py`, `disrupt.py`)
Expected extra delay when disrupted 3.5 d; reorder = demand x (lead + 14 d cycle) + 1.65 x 0.5 x demand x sqrt(lead) - on-hand; expedite -40% lead, +15% cost; switching = +250 fixed per SKU, +5% price, alternates assumed undisrupted with delay (1 - reliability) x 3.5 d; borderline band +/-0.10 around threshold; cost gate 10,000; top 5 exposed SKUs planned. Dollar values are an artefact of synthetic cost/demand.

## Limitations
- Disruptions are simulated, so detection results are circular to the simulation design; severity was not tuned to improve results. Real validation needs real supplier disruption data.
- TabPFN is not better than a one-feature baseline here; its value is no tuning and probabilistic output, and calibration degrades under the train-to-test shift (ECE 0.09).
- The plan check validates numbers and cost-optimality, not whether the LLM's rationale is sound.
- Single-row inference (~30 s on CPU) is too slow for per-event serving; production would batch-score daily on GPU and cache.
- The TabPFN weights used (v3.5) carry a non-commercial license; production use needs a commercial license.
