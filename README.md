# Evaluating AI-Assisted Data Science Workflows

[![Open In Colab - 1: Build your Agent](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RimmaShaf/fmlds-eval-workshop/blob/main/01_build_your_agent.ipynb)
[![Open In Colab - 2: Tracing](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RimmaShaf/fmlds-eval-workshop/blob/main/02_tracing_with_phoenix.ipynb)
[![Open In Colab - 3: Evals](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RimmaShaf/fmlds-eval-workshop/blob/main/03_evaluations.ipynb)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/RimmaShaf/fmlds-eval-workshop?quickstart=1)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/RimmaShaf/fmlds-eval-workshop/main)

**Building a Financial Agent, Uncovering Silent Failures, and Implementing Automated Evals** — a 3-hour hands-on masterclass. Participants build a Python AI agent over a synthetic bank ledger, watch it fail *silently* on three engineered traps, then replace vibe-checking with an automated eval suite that catches a deliberately injected regression.

| | |
|---|---|
| Audience | Data Scientists, ML Engineers, Technical Leads |
| Environment | Google Colab / Jupyter / GitHub Codespaces, Python 3.10+ |
| Stack | OpenAI SDK → LiteLLM gateway (Tokyo), local tracing (Phoenix), central Phoenix board |

## The story in 60 seconds

1. **Notebook 1 — Build your Agent** — a hand-rolled tool-calling agent (schema lookup, guarded pandas REPL, financial rules) with a naive one-line system prompt. It fails silently on: sign/polarity inversion, merchant string fragmentation (`STARBUCKS #1042` vs `== "Starbucks"`), and schema hallucination (`merchant_city` does not exist).
2. **Notebook 2 — Tracing with Phoenix** — observability reveals what actually happened: first by reading the runs' own event logs (waterfall, token leaderboard), then with a real OpenTelemetry stack running *inside each participant's notebook* — tool cascades, retry bloat, token burn, and the "Mind Reader" sycophancy game.
3. **Notebook 3 — Evaluations** — a 10-case golden suite with frozen programmatic ground truth: execution checks (AST, no traceback), **router checks** (did the model call the right tool), numeric ground-truth matches, and an LLM-as-judge rubric. Baseline hardened agent: **9/10**. Inject a "creative & conversational" prompt + `temperature=0.7`: **6/10**, caught automatically — and the same suite runs as the CI gate.

## Repo map

```
01_build_your_agent.ipynb      notebook 1: config, tools, the router loop, the three traps
02_tracing_with_phoenix.ipynb  notebook 2: event logs -> local Phoenix -> Mind Reader -> publish to board
03_evaluations.ipynb           notebook 3: golden suite, router checks, regression gate, beat-9/10, CI
utils.py                       canonical agent + eval harness (GENERATED; imported by notebooks 2-3 and CI)
data/transactions.csv           synthetic ledger, 508 rows, May–Oct 2026 (seed 42)
data/golden_eval_cases.json     10 golden cases + frozen ground truth
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

**Google Colab (primary):** open the notebooks in order — **1 Build your Agent → 2 Tracing with Phoenix → 3 Evaluations** (badges above, one per notebook). Enter your workshop key when the health-check cell asks, or save it as the `LITELLM_API_KEY` Colab secret; set it once per notebook. No Kaggle auth, no uploads: the data loads from this repo's raw URL, and notebooks 2–3 fetch the repo's `utils.py` automatically, so you don't need to re-run notebook 1. No Google account? Use the Codespaces or Binder badge above (GitHub account / no account at all respectively), or clone and run locally.

**Local / Codespaces:** open the repo, `pip install -r requirements.txt`, then run the notebooks in order and enter your workshop key when prompted. They fall back to `data/transactions.csv` on disk if the hosted data is unavailable.

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

**4. Room-wide observability (Phoenix board):** the gateway ships every request's trace to a central self-hosted Phoenix app (Tokyo, `infra/DEPLOYMENT.md` has the full deploy runbook) via the LiteLLM `arize_phoenix` callback — server-side capture, zero participant setup, tagged per seat. Participants' own tracing runs in their notebooks (notebook 2 launches a local Phoenix per runtime) and can be published to the board with the optional end-of-notebook cell using the `PHOENIX_BOARD_URL` from the desk card.

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

Expected mock results: baseline **9/10**, regression **6/10** (the four failures are the exact teaching moments: polarity, fragmentation, fabrication, hallucinated reassurance). To run a live gateway eval, set `LITELLM_API_KEY` and pass `--mode gateway`. Live-model results vary — that is the point of the harness.

## Security notes

- Seat keys, master keys and generated CSVs are gitignored; secrets only via environment / Colab secrets / Fly secrets.
- The agent REPL blocks `os`/`subprocess`/`requests` imports and dangerous builtins via AST scan.
- Per-key exposure is capped at $2.00; room-wide worst case $100, expected spend $5–$25.
