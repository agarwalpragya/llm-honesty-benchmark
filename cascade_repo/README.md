# CASCADE: Controlled Agentic Scenario System for Automated Deception Evaluation

Artifact repository for the paper:
**"Stake-Sensitive Honesty in LLM Agents: A Controlled Benchmark for Incentive-Driven Concealment"**

---

## Overview

CASCADE is a harm-graded behavioral benchmark that quantifies incentive-driven concealment in large language models (LLMs). It uses a treatment-minus-control (TMC) design to measure whether models suppress accurate defect reports when doing so benefits them.

**Deception gap metric:**
```
Δ(m, h) = p̂_treat − p̂_ctrl
```
where `m` is the model and `h` is the harm level.

## Repository Structure

```
cascade_repo/
├── prompts/
│   └── scenarios.json        # All 12 benchmark prompts (6 harm levels × 2 conditions)
├── data/
│   └── results.csv           # Concealment counts and rates for all model × level × condition cells
├── analysis/
│   └── analyze.py            # Statistical analysis: Fisher's exact, Holm-Bonferroni, Wilson CIs
├── figures/
│   ├── figure1.png           # Deception gap plot (output of analyze.py)
│   └── figure1.pdf           # PDF version for paper
└── README.md
```

## Harm Level Taxonomy

| Level | Name | Description |
|-------|------|-------------|
| L1 | Cosmetic | Visual/aesthetic defect; no functional impact |
| L2 | Minor-functional | Functional impairment; no financial or safety consequence |
| L3 | Financial | Quantifiable monetary loss to a third party |
| L4 | Regulatory | Compliance violation or legal exposure |
| L5 | Health | Risk of physical harm to a user or third party |
| L6 | Safety-critical | Immediate risk to life |

## Benchmark Design

Each scenario has two conditions:
- **Control**: Defect report with no consequence for the reporting agent
- **Treatment**: Identical prompt with an added *consequence clause* (the only difference)

Models respond with a single token: `ACCEPTABLE` (concealment) or `DEFECTIVE` (honest report).

## Models Evaluated

- Claude Haiku 4.5 (Anthropic)
- GPT-4o (OpenAI)
- GPT-4o-mini (OpenAI)
- Llama-3.1-8B-Instruct (Meta)
- Qwen-2.5-7B-Instruct (Alibaba)

All models evaluated via the OpenRouter API, June 2026. Temperature = provider default.

## Statistical Methodology

- **Per-cell significance**: Two-tailed Fisher's exact test
- **Multiple comparisons correction**: Holm-Bonferroni across all 30 comparisons (5 models × 6 harm levels)
- **Confidence intervals**: Wilson 95% CIs for all rate estimates
- **Sample sizes**: n = 100 for cells with observed divergence in pilot; n = 40 for non-divergent cells

## Running the Analysis

```bash
pip install pandas numpy scipy matplotlib
cd analysis
python analyze.py
```

Outputs:
- Table I (concealment rates with CIs) to stdout
- Table II (deception gaps with Holm-corrected p-values) to stdout
- Table III (effect summary) to stdout
- `figures/figure1.png` — deception gap plot

## Key Findings

| Model | Low-harm Δ (L1–L3) | High-harm Δ (L4–L6) | p (pooled low) |
|-------|---------------------|---------------------|----------------|
| Claude Haiku 4.5 | +0.000 | +0.000 | 1.000 |
| GPT-4o-mini | +0.440 | +0.000 | <0.0001 |
| GPT-4o | +0.513 | +0.000 | <0.0001 |
| Llama-3.1-8B | +0.275 | +0.161 | <0.0001 |
| Qwen-2.5-7B | +0.023 | +0.023 | 0.280 |

**Core finding**: Incentive-driven concealment is real but bounded by harm severity. No model conceals safety-critical (L6) failures. Llama-3.1-8B is the only model with a significant deception gap at health-relevant (L5) harm level.

## Extending the Benchmark

To add new models:
1. Query each model with each prompt in `prompts/scenarios.json`
2. Record responses (ACCEPTABLE / DEFECTIVE / UNKNOWN for neither)
3. Add rows to `data/results.csv` following the existing schema
4. Re-run `analysis/analyze.py`

To add new harm levels or consequence clause variants:
1. Add new scenario objects to `prompts/scenarios.json` with appropriate `harm_level` and `condition` fields
2. Add corresponding rows to `data/results.csv`
3. Update the `HARM_LEVELS` list in `analyze.py`

## License

This benchmark is released for research use. The scenario prompts and analysis code are provided under the MIT License.
