# BOOMERS — Demographic Calibration Summary

Working write-up of the demographic-process calibration for the BOOMERS
paper (joint work with Timothy Kam and Tina Kao, ANU).  The model extends
Cho, Li & Uren (IER 2024) with Storesletten-style immigration dynamics.

All calibration inputs come from public ABS datasets (Census 2021 via
TableBuilder Basic; NOM Cat 3412.0 / 3407.0; Life Tables Cat
3302.0.55.001), no restricted-access microdata needed.

**→ Read:** [`demographics_summary.md`](./demographics_summary.md)

Figures (also embedded in the summary):

| Fig | Panel |
|:-:|---|
| 1 | ABS Net Overseas Migration 2004–2020 |
| 2 | Return-migration hazard $\gamma_{a-m}$ (Case A: Storesletten × 1.3) |
| 3 | Calibrated immigrant inflow $\psi_{m,q}$ by age-at-migration and skill tier |
| 4 | Native vs immigrant skill distribution (prime age 25–64) |
| 5 | Skill shares by origin region |
| 6 | Stationary immigrant mass $\mu$ vs Census 2021 stock (apples-to-apples) |

## Related code

The full code base (model ports of CLU's IER and QE papers, Stata
pipelines, calibration and simulation) is maintained privately.  This
repo is a **read-only snapshot of the demographic write-up** intended
for coauthors and public reference.

## Contact

Takeki Sunakawa — takeki.sunakawa@gmail.com
