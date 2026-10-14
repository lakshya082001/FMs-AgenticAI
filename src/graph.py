import os
from typing import Any, TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
import llm, tools
from schema import ActionPlan, ExposureReport, RiskAssessment


class State(TypedDict, total=False):
    row: dict
    risk: dict
    exposure: dict
    options: dict
    plan: dict
    verdict: dict
    retries: int
    human: dict
    next: str
    trace: list


def log(s, node, **kw):
    return {"trace": s["trace"] + [{"node": node, **kw}]}


def orchestrator(s):
    r, reason = s.get("risk"), ""
    if r is None:
        nxt, reason = "risk", "start"
    elif r["error"]:
        nxt, reason = "end", "row failed validation"
    elif not r["at_risk"]:
        nxt, reason = "end", "below threshold, no action"
    elif "exposure" not in s:
        nxt, reason = "exposure", "supplier at risk"
    elif not s["exposure"]["skus"]:
        nxt, reason = "end", "no SKU exposed"
    elif "plan" not in s:
        nxt, reason = "planner", "exposed SKUs found"
    elif s.get("verdict") is None:
        nxt, reason = "check", "verify plan numbers"
    elif not s["verdict"]["ok"] and s["retries"] < 1:
        nxt, reason = "planner", "check failed, retry once"
    elif "human" not in s and (not s["verdict"]["ok"] or r["borderline"] or total_cost(s) > tools.COST_GATE):
        nxt, reason = "human", "needs review: " + ("check failed" if not s["verdict"]["ok"] else "borderline p" if r["borderline"] else "cost above gate")
    else:
        nxt, reason = "end", "human decision recorded" if "human" in s else "auto-approved"
    return {"next": nxt, **log(s, "orchestrator", next=nxt, reason=reason)}


def total_cost(s):
    return sum(i["incremental_cost"] + i["lost_sales_value"] for i in s["plan"]["items"])


def risk_agent(s):
    row = s["row"]
    issues = tools.validate_row(row)
    if issues:
        r = RiskAssessment(supplier=str(row.get("supplier")), date=str(row.get("date")), error="; ".join(issues))
    else:
        p, thr = tools.predict(row), tools.threshold()
        ev = tools.evidence(row)
        out = llm.ask("You are a supply-chain risk analyst. Use ONLY the evidence given; do not add numbers. "
                      "P is a model probability, not a fact: describe the supplier with the exact hedge word given (e.g. 'possibly disrupted'), never state certainty. "
                      'Reply JSON {"summary": "<2-3 sentences on whether the supplier is disrupted and why>"}',
                      f"supplier {row['supplier']}, P(disrupted)={p:.2f}, alert threshold={thr:.2f}, status={'ALERT (above threshold)' if p >= thr else 'NO ALERT (below threshold)'}, hedge word={'likely' if p >= 0.5 else 'possibly' if p >= thr else 'unlikely'}\n"
                      + "\n".join(ev))
        r = RiskAssessment(supplier=row["supplier"], date=str(row["date"]), p_disrupt=round(p, 3), threshold=thr, at_risk=p >= thr,
                           borderline=abs(p - thr) <= tools.BAND, evidence=ev, summary=out["summary"])
    return {"risk": r.model_dump(), **log(s, "risk_agent", output=r.model_dump())}


def exposure_agent(s):
    rep = tools.get_exposure(s["risk"]["supplier"])
    top = rep.skus[:tools.TOP_N]
    if rep.skus:
        rep.summary = llm.ask("You are an inventory analyst. Use ONLY the numbers given. "
                              'Reply JSON {"summary": "<2 sentences on which SKUs are most exposed and why>"}',
                              f"supplier {rep.supplier}: {len(rep.skus)} of {rep.n_skus} SKUs would stock out before resupply. Top:\n"
                              + "\n".join(k.model_dump_json() for k in top))["summary"]
    opts = {k.sku_id: tools.calc_options(k.sku_id) for k in top}
    return {"exposure": rep.model_dump(), "options": opts, **log(s, "exposure_agent", output=rep.model_dump())}


def planner_agent(s):
    top = s["exposure"]["skus"][:tools.TOP_N]
    prompt = "\n".join(f"sku {k['sku_id']} ({k['name']}): " + str(s["options"][k["sku_id"]]) for k in top)
    fix = f"\nYour previous plan had errors, copy numbers exactly from the tool table: {s['verdict']['errors']}" if s.get("verdict") else ""
    out = llm.ask("You are a replenishment planner. For each SKU choose ONE option from its table, minimising incremental_cost + lost_sales_value. "
                  "Copy shortfall_days, incremental_cost, lost_sales_value exactly from the table; never compute numbers yourself. "
                  'Reply JSON {"items": [{"sku_id": int, "option": str, "shortfall_days": float, "incremental_cost": float, '
                  '"lost_sales_value": float, "rationale": "<one sentence>"}]}', prompt + fix)
    plan = ActionPlan(**out)
    if os.environ.get("TAMPER") and "plan" not in s:  # fault injection for the failure-path trace
        plan.items[0].lost_sales_value += 50
    n = s["retries"] + (1 if "plan" in s else 0)
    return {"plan": plan.model_dump(), "retries": n, "verdict": None, **log(s, "planner_agent", attempt=n + 1, output=plan.model_dump())}


def check_node(s):
    v = tools.check_plan(ActionPlan(**s["plan"]), ExposureReport(**s["exposure"]), s["options"])
    return {"verdict": v.model_dump(), **log(s, "check", output=v.model_dump())}


def human_gate(s):
    d = interrupt({"risk": s["risk"]["summary"], "p": s["risk"]["p_disrupt"], "plan": s["plan"], "total_cost": round(total_cost(s), 2),
                   "check_ok": s["verdict"]["ok"]})
    return {"human": d, **log(s, "human_gate", decision=d)}


def build():
    g = StateGraph(State)
    for n, f in [("orchestrator", orchestrator), ("risk", risk_agent), ("exposure", exposure_agent),
                 ("planner", planner_agent), ("check", check_node), ("human", human_gate)]:
        g.add_node(n, f)
    g.set_entry_point("orchestrator")
    g.add_conditional_edges("orchestrator", lambda s: s["next"], {k: k for k in ["risk", "exposure", "planner", "check", "human"]} | {"end": END})
    for n in ["risk", "exposure", "planner", "check", "human"]:
        g.add_edge(n, "orchestrator")
    return g.compile(checkpointer=MemorySaver())
