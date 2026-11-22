# Evaluating AI-Assisted Data Science Workflows

[![Open In Colab - 1: Build your Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RimmaShaf/fmlds-eval-workshop/blob/main/01_build_your_agent.ipynb)
[![Open In Colab - 2: Tracing](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RimmaShaf/fmlds-eval-workshop/blob/main/02_tracing_with_phoenix.ipynb)
[![Open In Colab - 3: Evals](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RimmaShaf/fmlds-eval-workshop/blob/main/03_evaluations.ipynb)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/RimmaShaf/fmlds-eval-workshop?quickstart=1)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RimmaShaf/fmlds-eval-workshop/main)

*Colab: one click per notebook (Google account) · Codespaces: the full repo as a VS Code workspace (GitHub account) · Binder: no account at all.*

**Building a Financial Agent, Uncovering Silent Failures, and Implementing Automated Evals** — a 3-hour hands-on masterclass. Participants build a Python AI agent over a synthetic bank ledger, watch it fail *silently* on three engineered traps, then replace vibe-checking with an automated eval suite that catches a deliberately injected regression.

| | |
|---|---|
| Audience | Data Scientists, ML Engineers, Technical Leads |
| Environment | Google Colab / Jupyter / GitHub Codespaces, Python 3.11+ |
| Stack | OpenAI SDK → LiteLLM gateway (Tokyo), OpenTelemetry/OpenInference → shared Phoenix board |

## The story in 60 seconds

1. **Notebook 1 — Build your Agent** — a hand-rolled tool-calling agent (schema lookup, pandas analysis, financial rules, and charting) with a naive one-line system prompt. It demonstrates sign/refund mistakes, merchant aliases, and requests that the available data may not support.
2. **Notebook 2 — Tracing with Phoenix** — uses OpenTelemetry and OpenInference instrumentation to capture LLM, agent, chain, and tool spans. Every participant sends traces to the shared `00-Part-2-tracing-set-up` project; inspect tool inputs/outputs and compare a missing-information request with a successful analysis.
3. **Notebook 3 — Evaluations** — uploads the 11-case golden dataset to Phoenix, defines separate router, tool/code, answer, and overall evaluators, and compares prompt experiments in the dataset's Experiments tab. It guides participants through failure review and golden-case curation; model comparison is an optional extension.

## Repo map

```
01_build_your_agent.ipynb      notebook 1: config, tools, the router loop, the three traps
02_tracing_with_phoenix.ipynb  notebook 2: OpenTelemetry spans -> shared Phoenix board -> trace inspection
03_evaluations.ipynb           notebook 3: skill/router checks, prompt/model experiment, trace review
utils.py                       canonical agent + eval harness (GENERATED; imported by notebooks 2-3 and CI)
data/transactions.csv           synthetic ledger, 2,500 rows, May–Oct 2026 (seed 42)
data/golden_eval_cases.json     11 golden cases + frozen ground truth
scripts/generate_dataset.py     regenerates data + ground truth deterministically
scripts/build_notebooks.py      regenerates utils.py + all three notebooks (edit here, not the artifacts)
scripts/run_evals.py            headless CI twin of notebook 3 (imports utils.py)
scripts/make_seat_keys.py       instructor: 50 LiteLLM virtual keys, $2 caps
infra/litellm_config.yaml       gateway model allowlist + settings
infra/fly.toml, infra/Dockerfile  Tokyo (nrt) deployment
infra/phoenix.toml              the Phoenix board (central observability, nrt)
infra/DEPLOYMENT.md             field-tested gateway + board deployment runbook
.github/workflows/evals.yml     CI eval gate
.devcontainer/                  Codespaces fallback for corporate laptops
```

## Participant quickstart

**Google Colab (primary):** open the notebooks in order — **1 Build your Agent → 2 Tracing with Phoenix → 3 Evaluations** (badges above, one per notebook). Enter your workshop key when the health-check cell asks, or save it as the `LITELLM_API_KEY` Colab secret. Notebooks 2–3 also need the participant `PHOENIX_API_KEY` supplied by the organizer to send traces to their shared projects; they prompt if the secret is not set. No Kaggle auth, no uploads: the data loads from this repo's raw URL, and notebooks 2–3 fetch the repo's `utils.py` automatically, so you don't need to re-run notebook 1. No Google account? Use the Codespaces or Binder badge above (GitHub account / no account at all respectively), or clone and run locally.

**Local / Codespaces:** open the repo, `pip install -r requirements.txt`, then run the notebooks in order and enter your workshop key when prompted. The notebooks load the transaction data from the repository's hosted CSV.

## Instructor / ops runbook

> **Full step-by-step deployment guide with every gotcha we hit: [`infra/DEPLOYMENT.md`](infra/DEPLOYMENT.md)** — accounts/2FA, Supabase (Tokyo), Fly app, secrets, seat keys, workshop-day checklist, teardown, and a troubleshooting table.

**1. Deploy the gateway (Tokyo region):**

```bash
fly launch --no-deploy        # app name in infra/fly.toml, region nrt
fly secrets set OPENAI_API_KEY=sk-... LITELLM_MASTER_KEY=sk-... DATABASE_URL=<postgres-url>
fly deploy
```

Virtual keys require a database (`DATABASE_URL`, e.g. Fly Postgres or Upstash) for budget tracking. The allowlist and per-model `max_tokens: 1500` live in `infra/litellm_config.yaml`.

**2. Generate seat keys** (budget caps + 15 RPM + allowlist enforced at key creation):

```bash
python scripts/make_seat_keys.py --dry-run                  # preview
LITELLM_MASTER_KEY=sk-... python scripts/make_seat_keys.py # live
```

Paste `infra/seat_keys.csv` rows into the **view-only** Google Sheet; print desk cards `Seat 01`–`Seat 50` with the sheet's short URL. The CSV is gitignored — never commit it.

**3. CI gate:** already wired (`.github/workflows/evals.yml`): dataset integrity check, deterministic mock golden-suite gate, and a self-test proving the gate catches a known-bad regression. For scheduled live-model evals through the workshop gateway, add a `LITELLM_API_KEY` GitHub secret.

**4. Room-wide observability (Phoenix board):** the LiteLLM gateway keeps server-side request traces in `gateway`. Notebook 2 writes participant agent/LLM/chain/tool spans to `00-Part-2-tracing-set-up`; Notebook 3 writes eval traces to `01-Part-3-evaluations` and creates native dataset experiments with evaluator score columns. All results are on the same Phoenix board, and session IDs make participant traces filterable.

**5. Emergency kit:** 5 mobile hotspots + 5 reserve master-pool keys (create with `make_seat_keys.py --seats 5 --budget 5`).

## Dataset

Fully synthetic, regenerated deterministically from `scripts/generate_dataset.py` with `seed=42`. Every ground truth in `data/golden_eval_cases.json` is computed programmatically at generation time — never hand-transcribed.

```bash
python scripts/generate_dataset.py          # regenerate data + golden suite
python scripts/generate_dataset.py --check  # verify artifacts match the seed (runs in CI)
```

Merchant-string style and the category taxonomy are **design inspiration only** from two public datasets; no rows are copied or redistributed:
- [khushikyad001/personal-finance-tracker-dataset](https://www.kaggle.com/datasets/khushikyad001/personal-finance-tracker-dataset) (Kaggle) — schema shape
- [mitulshah/transaction-categorization](https://huggingface.co/datasets/mitulshah/transaction-categorization) (Hugging Face, MIT) — messy statement strings

## Development

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt nbformat ruff
python scripts/build_notebooks.py                          # regenerate utils.py + the three notebooks
python scripts/run_evals.py --mode mock --config baseline
python scripts/run_evals.py --mode mock --config regression --expect-regression
ruff check scripts/ utils.py
```

`utils.py` and the three `.ipynb` files are **generated artifacts** — never edit them by hand; edit the source fragments in `scripts/build_notebooks.py` and regenerate. The same fragments are both displayed in the notebooks and imported by CI, so notebook and CI can never drift apart.

The headless mock CI suite is deterministic: baseline **10/11**, known-bad regression **6/11** (the failures demonstrate polarity/refund handling, merchant aliases, fabrication, and hallucinated reassurance). Notebook 3 runs live Phoenix experiments over the same 11 golden cases; live results vary. To run the headless suite, use `scripts/run_evals.py`.

## Security notes

- Seat keys, master keys and generated CSVs are gitignored; secrets only via environment / Colab secrets / Fly secrets.
- Per-key exposure is capped at $2.00; room-wide worst case $100, expected spend $5–$25.
