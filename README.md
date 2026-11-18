# Evaluating AI-Assisted Data Science Workflows

**Building a Financial Agent, Uncovering Silent Failures, and Implementing Automated Evals** — a 3-hour hands-on masterclass. Participants build a Python AI agent over a synthetic bank ledger, watch it fail *silently* on three engineered traps, then replace vibe-checking with an automated eval suite that catches a deliberately injected regression.

| | |
|---|---|
| Audience | Data Scientists, ML Engineers, Technical Leads |
| Environment | Google Colab / Jupyter / GitHub Codespaces, Python 3.10+ |
| Stack | OpenAI SDK → LiteLLM gateway (Tokyo), local tracing (Langfuse optional) |

## The story in 60 seconds

1. **Module 2** — a hand-rolled tool-calling agent (schema lookup, guarded pandas REPL, financial rules) with a naive one-line system prompt. It fails silently on: sign/polarity inversion, merchant string fragmentation (`STARBUCKS #1042` vs `== "Starbucks"`), and schema hallucination (`merchant_city` does not exist).
2. **Module 3** — tracing reveals what actually happened: tool cascades, retry bloat, token burn, and the "Mind Reader" sycophancy game.
3. **Module 4** — a 10-case golden suite with frozen programmatic ground truth: execution checks (AST, no traceback), numeric ground-truth matches, and an LLM-as-judge rubric. Baseline hardened agent: **9/10**. Inject a "creative & conversational" prompt + `temperature=0.7`: **6/10**, caught automatically.
4. **Module 5** — the same suite as a CI gate.

## Repo map

```
workshop.ipynb                  the participant notebook (single source of truth)
data/transactions.csv           synthetic ledger, 508 rows, May–Oct 2026 (seed 42)
data/golden_eval_cases.json     10 golden cases + frozen ground truth
scripts/generate_dataset.py     regenerates data + ground truth deterministically
scripts/build_notebook.py       regenerates workshop.ipynb (edit here, not in the .ipynb)
scripts/run_evals.py            headless CI twin of Module 4 (executes the notebook's core cells)
scripts/make_seat_keys.py       instructor: 50 LiteLLM virtual keys, $2 caps
infra/litellm_config.yaml       gateway model allowlist + settings
infra/fly.toml, infra/Dockerfile  Tokyo (nrt) deployment
.github/workflows/evals.yml     CI eval gate
.devcontainer/                  Codespaces fallback for corporate laptops
```

## Participant quickstart

**Google Colab (primary):** open `workshop.ipynb` in Colab, set Colab secrets or edit the config cell — `WORKSHOP_MODE=gateway`, your seat key as `LITELLM_API_KEY` (from the view-only key sheet on your desk card) — then run top to bottom. No Kaggle auth, no uploads: the data loads from this repo's raw URL.

**Local / Codespaces:** open the repo, `pip install -r requirements.txt`, set `WORKSHOP_MODE` and run the notebook. Falls back to `data/transactions.csv` on disk.

**Modes:**

| `WORKSHOP_MODE` | LLM calls route to | use when |
|---|---|---|
| `gateway` | LiteLLM proxy (Tokyo), seat key `LITELLM_API_KEY` | workshop day |
| `direct` | api.openai.com, `OPENAI_API_KEY` | development |
| `mock` | nowhere — canned behavior, zero cost | TA dry-runs, CI, offline rehearsal |

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

**3. CI gate:** already wired (`.github/workflows/evals.yml`): dataset integrity check, mock-mode golden-suite gate, and a self-test proving the gate catches a known-bad regression. For live-model evals, add an `OPENAI_API_KEY` secret — the nightly job then runs `--mode direct`.

**4. Optional Langfuse (room-wide projector telemetry):** add to `infra/litellm_config.yaml`:

```yaml
litellm_settings:
  success_callback: ["langfuse"]
```

with `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` as Fly secrets — traces are then captured server-side for every seat, no participant setup. Client-side alternative: in the notebook's health-check cell, swap `from openai import OpenAI` for `from langfuse.openai import OpenAI`.

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
python -m venv .venv && .venv/bin/pip install -r requirements.txt nbformat nbclient ruff
python scripts/build_notebook.py                        # regenerate workshop.ipynb after edits
WORKSHOP_MODE=mock python scripts/run_evals.py --mode mock --config baseline
python scripts/run_evals.py --mode mock --config regression --expect-regression
ruff check scripts/
```

Expected mock results: baseline **9/10**, regression **6/10** (the four failures are the exact teaching moments: polarity, fragmentation, fabrication, hallucinated reassurance). Live-model results vary — that is the point of the harness.

## Security notes

- Seat keys, master keys and generated CSVs are gitignored; secrets only via environment / Colab secrets / Fly secrets.
- The agent REPL blocks `os`/`subprocess`/`requests` imports and dangerous builtins via AST scan.
- Per-key exposure is capped at $2.00; room-wide worst case $100, expected spend $5–$25.
