from pydantic import BaseModel


class RiskAssessment(BaseModel):
    supplier: str
    date: str
    p_disrupt: float | None = None
    threshold: float | None = None
    at_risk: bool = False
    borderline: bool = False
    evidence: list[str] = []
    summary: str = ""
    error: str | None = None


class ExposedSKU(BaseModel):
    sku_id: int
    name: str
    days_cover: float
    lead_days: float
    shortfall_days: float
    value_at_risk: float


class ExposureReport(BaseModel):
    supplier: str
    n_skus: int
    skus: list[ExposedSKU]  # exposed only, sorted by value_at_risk desc
    summary: str = ""


class PlanItem(BaseModel):
    sku_id: int
    option: str
    shortfall_days: float
    incremental_cost: float
    lost_sales_value: float
    rationale: str


class ActionPlan(BaseModel):
    items: list[PlanItem]


class Verdict(BaseModel):
    ok: bool
    errors: list[str] = []
