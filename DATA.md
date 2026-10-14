# Data

## Source
DataCo Smart Supply Chain (Mendeley Data, Constante et al. 2019, DOI 10.17632/8gx2fvg2k6, CC BY 4.0 per secondary sources).
180,519 order-item rows, 65,752 orders, 2015-01 to 2018-01. Real customer-order shipments (not supplier-side).
Dropped 7,754 `Shipping canceled` rows (no delivery outcome) -> 172,765 rows.

## Why simulate
Lane-level (shipping mode x market / region, category) weekly late rates have lag-1 autocorrelation of -0.02 to -0.06 and trailing 7d/30d rates add nothing over the lane mean (`src/autocorr_check.py`). The late flag is independent of recent history, so there is no real "disruption" dynamic to detect. We therefore overlay simulated supplier disruption episodes and measure whether the pipeline detects them. **This measures detection of simulated disruptions, not real-world accuracy.**

## Layers

| Layer | Real / synthetic | Script |
|---|---|---|
| Orders, products, prices, demand rates, shipping/scheduled days | Real (DataCo) | - |
| 8 suppliers (lead time U(5,25)d, reliability U(.75,.97), cost 1.25 -> 0.95 as lead grows), 118 SKUs (unit cost = 0.6 x price, on-hand = U(3,30) days of cover, 1 primary + 2 alternate suppliers) | Synthetic, seeded | `src/gen_inventory.py` |
| Disruption episodes: per supplier gap ~ Exp(75d), duration U(7,21)d, severity U(.35,.7) = P(order delayed), delay U{2..5} extra days; late = actual > scheduled days (DataCo's own label definition, verified 100%) | Synthetic, seeded | `src/disrupt.py` |

112 episodes covering 18.6% of orders; simulated late rate 0.612 vs 0.573 original.

## Modelling table (supplier-day)
One row per supplier per day, features computed at day start from information observable then (deliveries completed strictly before the day, in-transit orders past their promised date).
**Target:** `disrupted` = supplier is inside an episode that day.

| Feature | Meaning |
|---|---|
| r3, r7, r14, r30, r90 | late rate of completed deliveries over trailing 3/7/14/30/90 days |
| n7 | completed deliveries in trailing 7 days |
| dev3, dev7 | r3 - r30, r7 - r90 |
| days_since_alert | days since dev7 > 0.10 with n7 >= 20 (observed only, capped 60) |
| n_overdue, od_share | in-transit orders past promised date (count, share of in-transit) |

11 features, all numeric. 90-day warm-up dropped.

| Split | Window | Rows | Disrupted |
|---|---|---|---|
| Train (TabPFN context) | 2015-04-01 to 2016-06-30 | 3,656 | 15.4% |
| Val (threshold choice) | 2016-07-01 to 2016-12-31 | 1,472 | 14.5% |
| Test | 2017-01-01 to 2017-09-30 | 2,184 | 23.4% |

Oct 2017 onward unused (thin tail). Test has more disruption than train (35 episodes starting in window): a distribution shift that hurts calibration.
