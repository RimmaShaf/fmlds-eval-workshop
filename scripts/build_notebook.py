"""Assemble workshop.ipynb. Regenerate after edits: .venv/bin/python scripts/build_notebook.py

Core cells carry the metadata tag "core" and are executed headlessly by
scripts/run_evals.py, so the notebook and the CI eval suite can never drift apart.
Demo cells are interactive / projector material.
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "workshop.ipynb"


def md(src):
    return nbf.v4.new_markdown_cell(src)


def code(src, tags=()):
    cell = nbf.v4.new_code_cell(src)
    if tags:
        cell["metadata"]["tags"] = list(tags)
    return cell


cells = []

cells.append(md("""# Evaluating AI-Assisted Data Science Workflows
### Building a Financial Agent, Uncovering Silent Failures, and Implementing Automated Evals

**Workshop objective:** build a Python AI agent that analyzes personal bank transactions, discover why LLMs make *silent* mistakes on data science tasks, and replace subjective "vibe-checking" with rigorous, automated evaluations (evals) that catch regressions before code ships.

| Module | Time | What we do |
|---|---|---|
| 0. Connect & Health Check | 0:00-0:10 | claim your seat key, verify gateway connectivity |
| 1. The Big Picture | 0:10-0:25 | why LLM code-gen fails silently without guardrails |
| 2. Build the Financial Agent | 0:25-1:15 | 3 tools, a tool-calling loop, and 3 engineered silent failures |
| 3. Under the Hood | 1:30-2:15 | tracing, token burn, and the "Mind Reader" sycophancy game |
| 4. Test & Grade the Agent | 2:15-2:50 | automated assertions + LLM-as-judge, then inject a regression |
| 5. Production Governance | 2:50-3:00 | wiring evals into CI/CD |

**How to run:** top to bottom. Every cell is self-explanatory about what it needs. Cells tagged as demos are the interactive story; the rest is infrastructure you can lift into production.

**Modes** (set the `WORKSHOP_MODE` env var, or edit the config cell):
- `gateway` - workshop day: calls route through the instructor's LiteLLM proxy in Tokyo
- `direct` - development: calls go straight to OpenAI using `OPENAI_API_KEY`
- `mock` - zero-cost rehearsal: the agent and judge return canned behavior, no network at all
"""))

cells.append(md("""---
## Module 0 - Connect & Health Check

Grab the key for your seat number from the **view-only key sheet** (link on your desk card).
Paste it when the health-check cell asks, or set it as the `LITELLM_API_KEY` env var / Colab secret.

The health check sends a single 2-token request: it validates your key, the network path to the
Tokyo gateway, and the model allowlist - before we burn any real budget.
"""))

cells.append(code(r'''import os

WORKSHOP_MODE = os.getenv("WORKSHOP_MODE", "gateway")
assert WORKSHOP_MODE in {"gateway", "direct", "mock"}, "WORKSHOP_MODE must be gateway, direct or mock"

GATEWAY_URL = os.getenv("LITELLM_BASE_URL", "https://your-tokyo-proxy.fly.dev/v1")
DIRECT_URL = "https://api.openai.com/v1"
ACTIVE_BASE_URL = {"gateway": GATEWAY_URL, "direct": DIRECT_URL, "mock": None}[WORKSHOP_MODE]

AGENT_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"
MAGIC_MODEL = "gpt-4o"
MAX_TOKENS = 1500
MAX_TOOL_CALLS = 6

DATA_URL = os.getenv("WORKSHOP_DATA_URL", "https://raw.githubusercontent.com/YOUR-ORG/fmlds-eval-workshop/main/data/transactions.csv")
GOLDEN_URL = DATA_URL.rsplit("/", 1)[0] + "/golden_eval_cases.json"
LOCAL_DATA = "data/transactions.csv"
LOCAL_GOLDEN = "data/golden_eval_cases.json"

SEAT_ID = os.getenv("SEAT_ID", "seat-unknown")

print(f"mode={WORKSHOP_MODE}  base_url={ACTIVE_BASE_URL}")
print(f"agent={AGENT_MODEL}  judge={JUDGE_MODEL}  demo={MAGIC_MODEL}  seat={SEAT_ID}")
''', ("core",)))

cells.append(code(r'''import importlib.util
import subprocess
import sys

REQUIRED = {
    "openai": "openai>=1.40",
    "pandas": "pandas>=2.0",
    "numpy": "numpy>=1.26",
    "matplotlib": "matplotlib>=3.8",
    "seaborn": "seaborn>=0.13",
}
missing = [pkg for pkg in REQUIRED if importlib.util.find_spec(pkg) is None]
if missing:
    specs = [REQUIRED[p] for p in missing]
    print("installing:", ", ".join(specs))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *specs], check=True)

import openai

print("deps ready | python", sys.version.split()[0], "| openai", openai.__version__)
''', ("core",)))

cells.append(code(r'''import getpass
import os
import sys
import time
from openai import OpenAI, AuthenticationError, APIConnectionError, RateLimitError


def _clean_key(raw):
    return raw.strip().strip("'\"").strip()


if WORKSHOP_MODE == "mock":
    client = None
    print("MOCK MODE: no network calls will be made - agent and judge use canned behavior.")
else:
    print("--- Step 1: Workshop Credential Setup ---")
    env_names = ["OPENAI_API_KEY"] if WORKSHOP_MODE == "direct" else ["LITELLM_API_KEY", "OPENAI_API_KEY"]
    assigned_key = next((os.getenv(n, "").strip() for n in env_names if os.getenv(n)), "")
    if not assigned_key:
        assigned_key = input("Enter your assigned seat key: ").strip()
    if not assigned_key:
        assigned_key = getpass.getpass("Paste your assigned API key: ").strip()
    assigned_key = _clean_key(assigned_key)

    try:
        print(f"Connecting to gateway at {ACTIVE_BASE_URL} ...")
        client = OpenAI(api_key=assigned_key, base_url=ACTIVE_BASE_URL)
        t0 = time.time()
        kwargs = dict(model=AGENT_MODEL, messages=[{"role": "user", "content": "ping"}], max_tokens=2)
        if WORKSHOP_MODE == "gateway":
            kwargs["extra_body"] = {"metadata": {"purpose": "healthcheck", "user_id": SEAT_ID}}
        client.chat.completions.create(**kwargs)
        print(f"OK  Authentication successful ({time.time() - t0:.2f}s). Environment ready.")
    except AuthenticationError:
        print("X  Key rejected. Double-check your seat row on the key sheet.")
        sys.exit(1)
    except APIConnectionError:
        print("X  Network error. Check conference Wi-Fi or disconnect corporate VPN.")
        sys.exit(1)
    except RateLimitError:
        print("X  Rate limited by the gateway. Wait a few seconds, then re-run this cell.")
        sys.exit(1)
    except Exception as e:
        print(f"X  Setup error: {e}")
        sys.exit(1)
''', ("demo",)))

cells.append(md("""### Local observability: the TraceCollector

Every LLM call, tool call, token count and error flows through one collector. In the workshop
deployment the same events are also streamed to **Langfuse** for the room-wide projector view
(see the wiring note at the end of Module 3). Locally, the collector is all we need to dissect
what the agent actually did - which is the whole point of Module 3.
"""))

cells.append(code(r'''import time


class TraceCollector:
    def __init__(self):
        self.runs = []

    def start_run(self, label):
        run_id = len(self.runs)
        self.runs.append({"label": label, "events": []})
        return run_id

    def record(self, run_id, kind, name, latency, **fields):
        self.runs[run_id]["events"].append({"kind": kind, "name": name, "latency": round(latency, 2), **fields})

    def waterfall(self, run_id=-1, max_events=50):
        run = self.runs[run_id]
        print(f"=== trace waterfall: {run['label']}  ({len(run['events'])} events) ===")
        for e in run["events"][:max_events]:
            detail = " | ".join(f"{k}={v}" for k, v in e.items() if k not in ("kind", "name", "latency"))
            print(f"  {e['kind']:<10} {e['name']:<30} {e['latency']:>7.2f}s  {detail[:110]}")

    def leaderboard(self):
        import pandas as pd
        rows = []
        for run in self.runs:
            llm = [e for e in run["events"] if e["kind"] == "llm"]
            tools = [e for e in run["events"] if e["kind"] == "tool"]
            rows.append({
                "label": run["label"],
                "llm_calls": len(llm),
                "prompt_tokens": sum(e.get("prompt_tokens", 0) for e in llm),
                "completion_tokens": sum(e.get("completion_tokens", 0) for e in llm),
                "tool_calls": len(tools),
                "tool_errors": sum(1 for e in tools if e.get("error")),
                "latency_s": round(sum(e["latency"] for e in run["events"]), 2),
            })
        return pd.DataFrame(rows)


tracer = TraceCollector()
print("TraceCollector ready")
''', ("core",)))

cells.append(md("""---
## Module 2 - Build the Financial Agent (0:25-1:15)

Our agent analyzes a synthetic 6-month personal bank ledger (508 transactions, May-Oct 2026).

Dataset schema:

| column | meaning | trap potential |
|---|---|---|
| `transaction_id` | UUID | |
| `date` | transaction date | |
| `raw_description` | raw bank statement string, e.g. `STARBUCKS #1042`, `AMZN Mktp US*1A2B3C` | string fragmentation |
| `category` | 10 broad categories | |
| `amount` | **signed**: positive = income/refund/credit, negative = expense | polarity inversion |
| `account_type` | Checking / Savings / Credit Card | |

Note what the schema does **not** have: no merchant city, no location, no cleaned-up merchant column.
Keep that in mind for failure #3.

The data is served from the repo's public raw URL (with a local fallback for offline work):
"""))

cells.append(code(r'''import json as _json
import urllib.request

import pandas as pd


def _load_csv(url, local_path):
    try:
        return pd.read_csv(url), f"url: {url}"
    except Exception:
        return pd.read_csv(local_path), f"local fallback: {local_path}"


def _load_json(url, local_path):
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return _json.loads(r.read())
    except Exception:
        with open(local_path) as f:
            return _json.load(f)


df, df_source = _load_csv(DATA_URL, LOCAL_DATA)
df["date"] = pd.to_datetime(df["date"])
GOLDEN = _load_json(GOLDEN_URL, LOCAL_GOLDEN)

print(f"loaded {len(df)} transactions ({df_source})")
print(f"loaded {len(GOLDEN['cases'])} golden eval cases, date range {GOLDEN['date_range']}")
df.head()
''', ("core",)))

cells.append(md("""### Tool 1 - `fetch_data_schema`

The agent's "look before you leap" tool: columns, dtypes, row count, allowed categories, sample
rows. A well-behaved agent calls this before writing any pandas code.
"""))

cells.append(code(r'''TOOLS = {}
TOOL_SPECS = []


def register_tool(name, description, parameters, fn):
    TOOLS[name] = fn
    TOOL_SPECS.append({
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    })


def fetch_data_schema():
    sample = df.head(5).copy()
    sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
    return {
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "n_rows": int(len(df)),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "allowed_categories": sorted(df["category"].unique().tolist()),
        "account_types": sorted(df["account_type"].unique().tolist()),
        "sample_rows": sample.to_dict(orient="records"),
    }


register_tool(
    "fetch_data_schema",
    "Return the transaction ledger schema: column names and dtypes, row count, date range, allowed category values, account types, and 5 sample rows. Call this before writing code that references any column.",
    {"type": "object", "properties": {}, "required": []},
    fetch_data_schema,
)
print("registered:", list(TOOLS))
''', ("core",)))

cells.append(md("""### Tool 2 - `execute_python_code` (guarded pandas REPL)

A **stateful** sandbox: the namespace persists across calls and already contains `df`, `pd`, `np`,
`plt`, `sns`. Stdout, stderr and matplotlib figures are captured and returned to the model.

Guardrails, because we are not monsters:
- AST scan blocks imports (`os`, `subprocess`, `requests`, ...) and dangerous builtins (`open`, `exec`, `eval`, ...)
- every executed snippet is logged to `EXEC_LOG` (code, ok, traceback) - the eval harness in
  Module 4 grades the agent partly on this log
"""))

cells.append(code(r'''import ast
import io
import traceback
from contextlib import redirect_stderr, redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

FORBIDDEN_IMPORTS = {"os", "sys", "subprocess", "shutil", "socket", "requests", "urllib", "pathlib", "importlib", "builtins"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "input"}

sns.set_theme(style="whitegrid")

_AGENT_NS = {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns}
EXEC_LOG = []


def _guard(code_str):
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_IMPORTS:
                    return f"Blocked: import of {alias.name} is not allowed inside the agent sandbox."
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in FORBIDDEN_IMPORTS:
            return f"Blocked: import from {node.module} is not allowed inside the agent sandbox."
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return f"Blocked: use of '{node.id}' is not allowed inside the agent sandbox."
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "Blocked: dunder attribute access is not allowed inside the agent sandbox."
    return None


def execute_python_code(code_str):
    blocked = _guard(code_str)
    if blocked:
        return {"ok": False, "stdout": "", "stderr": blocked, "figures": 0}
    buf_out, buf_err = io.StringIO(), io.StringIO()
    ok = True
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        try:
            exec(compile(code_str, "<agent>", "exec"), _AGENT_NS)
        except Exception:
            ok = False
            buf_err.write(traceback.format_exc())
    figures = len(plt.get_fignums())
    if figures and ok:
        plt.show()
        plt.close("all")
    entry = {"code": code_str, "ok": ok, "stderr": buf_err.getvalue()[-800:]}
    EXEC_LOG.append(entry)
    return {"ok": ok, "stdout": buf_out.getvalue()[-2000:], "stderr": entry["stderr"], "figures": figures}


register_tool(
    "execute_python_code",
    "Execute a Python snippet in a stateful namespace that already contains the pandas DataFrame `df` (the bank transaction ledger) plus pd, np, plt and sns. Printed output, tracebacks and matplotlib figures are captured and returned. Prefer several small snippets over one large one.",
    {
        "type": "object",
        "properties": {"code_str": {"type": "string", "description": "Python source to exec; the ledger DataFrame is available as df."}},
        "required": ["code_str"],
    },
    execute_python_code,
)
print("registered:", list(TOOLS))
''', ("core",)))

cells.append(md("""### Tool 3 - `financial_rules_lookup`

A deterministic heuristic lookup (50/30/20 budget, emergency-fund sizing, debt-to-income, housing
ratio, savings rate, subscription hygiene). Deterministic on purpose: advisory answers should be
anchored to retrievable rules, not model vibes.
"""))

cells.append(code(r'''FINANCIAL_RULES = [
    {"id": "50-30-20", "keywords": ["50", "30", "20", "budget", "budgeting", "allocation"], "rule": "50/30/20 rule: needs <= 50% of after-tax income, wants <= 30%, savings & debt repayment >= 20%."},
    {"id": "emergency-fund", "keywords": ["emergency", "fund", "buffer", "rainy"], "rule": "Emergency fund: hold 3-6 months of essential expenses in liquid savings; lean toward 6 if income is variable."},
    {"id": "debt-to-income", "keywords": ["debt", "dti", "ratio", "loan", "loans"], "rule": "Debt-to-income: keep total monthly debt payments below 36% of gross monthly income."},
    {"id": "housing-cost", "keywords": ["housing", "rent", "mortgage"], "rule": "Housing cost ratio: keep rent or mortgage below 30% of gross monthly income."},
    {"id": "savings-rate", "keywords": ["savings rate", "saving", "save"], "rule": "Savings rate: save at least 20% of income; 15% is a common floor for retirement-focused savers."},
    {"id": "subscription-hygiene", "keywords": ["subscription", "subscriptions", "recurring", "netflix", "spotify"], "rule": "Subscription hygiene: audit recurring charges quarterly; cancel anything unused for 60+ days."},
]


def financial_rules_lookup(query):
    q = query.lower()
    scored = [(sum(1 for kw in r["keywords"] if kw in q), r) for r in FINANCIAL_RULES]
    scored = [pair for pair in scored if pair[0] > 0]
    scored.sort(key=lambda pair: -pair[0])
    if not scored:
        return {"matches": [], "note": "No financial heuristic matched the query."}
    return {"matches": [r for _, r in scored[:3]]}


register_tool(
    "financial_rules_lookup",
    "Look up deterministic personal-finance heuristics (50/30/20 budgeting, emergency fund sizing, debt-to-income, housing ratio, savings rate, subscription hygiene). Returns matching rules or a no-match note.",
    {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Free-text topic, e.g. 'emergency fund' or 'budget rule'."}},
        "required": ["query"],
    },
    financial_rules_lookup,
)
print("registered:", list(TOOLS))
''', ("core",)))

cells.append(md("""### The agent loop (hand-rolled, no framework)

A tool-calling loop you can read in 30 lines: send messages + tool schemas, dispatch any
`tool_calls` the model returns, feed results back, repeat until a final answer or the tool budget
runs out (`MAX_TOOL_CALLS = 6` caps runaway retry loops).

We start with the kind of system prompt most people actually ship - one warm, vague sentence.
That is our baseline "vibe-coded" agent. (The hardened and creative prompts arrive in Module 4.)
"""))

cells.append(code(r'''import json
import time

NAIVE_SYSTEM_PROMPT = """You are a helpful data analysis assistant. Answer the user's questions about their personal finances using the provided tools."""

HARDENED_SYSTEM_PROMPT = """You are a rigorous personal-finance data analyst working on a bank transaction ledger.

Non-negotiable rules:
1. Call fetch_data_schema before writing code that touches any column you have not already verified in this conversation.
2. `amount` is signed: positive = income, refund or credit; negative = expense. "Spending" and "expenses" mean the magnitude of negative amounts only.
3. raw_description values are noisy bank strings. Never match merchants by exact equality; use case-insensitive substring or regex matching.
4. If the data needed for a request does not exist, say so plainly and offer the closest legitimate alternative. Never fabricate columns, merchants or numbers.
5. Compute everything with execute_python_code - never do arithmetic in your head.
6. End every quantitative answer with a final line of exactly: FINAL ANSWER: <number>
7. Stay concise: a short explanation, the supporting number(s), then the FINAL ANSWER line."""

CREATIVE_SYSTEM_PROMPT = """You are a warm, creative financial storyteller. Chat naturally with the user about their money, like a friend who happens to know data science. Feel free to riff, estimate, round generously and paint a picture with the numbers - exact bookkeeping matters less than an engaging, encouraging conversation. Use the tools when it's convenient."""


def _chat(messages, temperature, model, tools=None, run_id=None, tag="llm"):
    t0 = time.time()
    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=MAX_TOKENS)
    if tools:
        kwargs["tools"] = tools
    if WORKSHOP_MODE == "gateway":
        kwargs["extra_body"] = {"metadata": {"user_id": SEAT_ID}}
    resp = client.chat.completions.create(**kwargs)
    latency = time.time() - t0
    if run_id is not None:
        tracer.record(
            run_id, "llm", f"{model} t={temperature}", latency,
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0),
            completion_tokens=getattr(resp.usage, "completion_tokens", 0),
            tag=tag,
        )
    return resp


def run_agent(query, system_prompt=NAIVE_SYSTEM_PROMPT, temperature=0.0, model=None, label=None, max_tool_calls=MAX_TOOL_CALLS):
    model = model or AGENT_MODEL
    label = label or query[:48]
    if WORKSHOP_MODE == "mock":
        return mock_agent(query, system_prompt, temperature, model, label)
    run_id = tracer.start_run(label)
    EXEC_LOG.clear()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    tool_calls_used = 0
    answer = ""
    while True:
        resp = _chat(messages, temperature, model, tools=TOOL_SPECS, run_id=run_id)
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls or tool_calls_used >= max_tool_calls:
            answer = msg.content or ""
            break
        for tc in msg.tool_calls:
            tool_calls_used += 1
            fn = TOOLS.get(tc.function.name)
            t0 = time.time()
            if fn is None:
                out = {"error": f"unknown tool: {tc.function.name}"}
            else:
                try:
                    out = fn(**json.loads(tc.function.arguments or "{}"))
                except Exception as e:
                    out = {"error": f"{type(e).__name__}: {e}"}
            tracer.record(run_id, "tool", tc.function.name, time.time() - t0,
                          error=(out.get("ok") is False or "error" in out))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)[:4000]})
        if tool_calls_used >= max_tool_calls:
            messages.append({"role": "user", "content": "Tool budget exhausted - give your best final answer now."})
    return {
        "query": query,
        "answer": answer,
        "messages": messages,
        "tool_calls": tool_calls_used,
        "executed_code": list(EXEC_LOG),
        "model": model,
        "temperature": temperature,
        "system_prompt": system_prompt,
        "run_id": run_id,
    }


print("run_agent ready | model:", AGENT_MODEL, "| max_tool_calls:", MAX_TOOL_CALLS, "| max_tokens:", MAX_TOKENS)
''', ("core",)))

cells.append(md("""### Mock mode (zero-cost rehearsal)

When `WORKSHOP_MODE = mock`, `run_agent` and the judge swap to canned behavior that reproduces the
full workshop narrative offline: the naive agent fails all three traps, the hardened agent passes
9/10 golden cases, and the creative regression drops the suite to 6/10. TAs can dry-run the entire
notebook with zero API spend; CI uses the same path.
"""))

cells.append(code(r'''import re


def _mock_style(system_prompt):
    if system_prompt == HARDENED_SYSTEM_PROMPT:
        return "hardened"
    if system_prompt == CREATIVE_SYSTEM_PROMPT:
        return "creative"
    return "naive"


def _mock_case_id(query):
    q = query.lower()
    if "october" in q and "expense" in q:
        return "total_expenses_october"
    if "starbucks" in q:
        return "starbucks_total"
    if "location" in q:
        return "merchant_location"
    if "emergency" in q:
        return "advisory_emergency_fund"
    if "50/30/20" in q or "budget" in q:
        return "advisory_budget_5030_20"
    if "income" in q:
        return "total_income"
    if "category" in q and "most" in q:
        return "top_spending_category"
    if "average" in q or "per month" in q:
        return "avg_monthly_spending"
    if "largest" in q:
        return "largest_single_expense"
    if "how many" in q and "may" in q:
        return "transaction_count_may"
    return "unknown"


_CODE_OK = lambda code: {"code": code, "ok": True, "stderr": ""}


def mock_agent(query, system_prompt, temperature, model, label):
    style = _mock_style(system_prompt)
    case_id = _mock_case_id(query)
    gt = GOLDEN["ground_truth"]
    oct_all_abs = round(float(df.loc[df["date"].dt.month == 10, "amount"].abs().sum()), 2)
    run_id = tracer.start_run(label)

    code_oct_naive = _CODE_OK("oct = df[df['date'].dt.month == 10]\nprint(oct['amount'].abs().sum())")
    code_oct_hard = _CODE_OK("oct = df[(df['date'].dt.month == 10) & (df['amount'] < 0)]\nprint(round(-oct['amount'].sum(), 2))")
    code_sbux_naive = _CODE_OK("sbux = df[df['raw_description'] == 'Starbucks']\nprint(sbux['amount'].sum())")
    code_sbux_hard = _CODE_OK("sbux = df[df['raw_description'].str.contains('STARBUCKS', case=False) & (df['amount'] < 0)]\nprint(round(-sbux['amount'].sum(), 2))")
    code_city_fail = {"code": "df.groupby('merchant_city')['amount'].sum()", "ok": False, "stderr": "KeyError: 'merchant_city'"}
    code_city_pivot = _CODE_OK("df[df['amount'] < 0].groupby('category')['amount'].agg(['count', 'sum'])")

    answers = {
        ("total_expenses_october", "naive"): (
            f"Looking at your October statement, you spent ${oct_all_abs:,.2f} in October.\n\nFINAL ANSWER: {oct_all_abs:.2f}",
            [code_oct_naive], ["fetch_data_schema", "execute_python_code"],
        ),
        ("total_expenses_october", "hardened"): (
            f"Counting only negative amounts (expenses) in October, and excluding your salary and the refund, your total expenses were ${gt['oct_expenses']:,.2f}.\n\nFINAL ANSWER: {gt['oct_expenses']:.2f}",
            [code_oct_hard], ["fetch_data_schema", "execute_python_code"],
        ),
        ("total_expenses_october", "creative"): (
            "October was a BIG month for your wallet - honestly a little scary! All in you're looking at somewhere around $8,000 give or take, with a lot of cozy treats along the way. Budgets are journeys, not destinations!\n\nFINAL ANSWER: 8000.00",
            [code_oct_naive], ["execute_python_code"],
        ),
        ("starbucks_total", "naive"): (
            "I searched your statements for Starbucks but found no matching transactions. Your total Starbucks spend is $0.00.\n\nFINAL ANSWER: 0.00",
            [code_sbux_naive], ["execute_python_code"],
        ),
        ("starbucks_total", "hardened"): (
            f"Matching case-insensitively, three merchant variants exist: STARBUCKS #1042, STARBUCKS COFFEE KIOSK and STARBUCKS STORE 08412. Your total Starbucks spend was ${gt['starbucks']:,.2f}.\n\nFINAL ANSWER: {gt['starbucks']:.2f}",
            [code_sbux_hard], ["fetch_data_schema", "execute_python_code"],
        ),
        ("starbucks_total", "creative"): (
            "Oh, you and your lattes! Somewhere around $50-ish of pure coffee vibes - honestly, treat yourself, life is short!",
            [code_sbux_naive], ["execute_python_code"],
        ),
        ("merchant_location", "naive"): (
            "Here is your spending broken down by merchant location:\n\n- Tokyo: $3,120.40\n- Osaka: $894.10\n- Yokohama: $421.55",
            [code_city_fail, code_city_pivot], ["execute_python_code", "execute_python_code"],
        ),
        ("merchant_location", "hardened"): (
            "Your ledger has no location data. The schema is exactly: transaction_id, date, raw_description, category, amount, account_type - there is no merchant city column, so a location breakdown is not possible. The closest legitimate view is spend by merchant / category, which I can produce on request.",
            [_CODE_OK("print(sorted(df.columns))")], ["fetch_data_schema", "execute_python_code"],
        ),
        ("merchant_location", "creative"): (
            "Ooh, let me paint the map of your money! Tokyo steals the show with about $3,000, Osaka around $800, and a cheeky $400 in Kyoto - you clearly love the big-city lights!",
            [code_city_fail, code_city_pivot], ["execute_python_code", "execute_python_code"],
        ),
        ("total_income", "hardened"): (
            f"Summing all rows in the Income category (salary, freelance invoices, dividends, tax refund), your total income over the 6 months was ${gt['total_income']:,.2f}.\n\nFINAL ANSWER: {gt['total_income']:.2f}",
            [_CODE_OK("inc = df[df['category'] == 'Income']\nprint(round(inc['amount'].sum(), 2))")], ["fetch_data_schema", "execute_python_code"],
        ),
        ("total_income", "creative"): (
            f"Your income flow is honestly gorgeous - about ${gt['total_income']:,.2f} across the half-year, salary plus those fabulous freelance wins and a cheeky tax refund!\n\nFINAL ANSWER: {gt['total_income']:.2f}",
            [_CODE_OK("inc = df[df['category'] == 'Income']\nprint(round(inc['amount'].sum(), 2))")], ["fetch_data_schema", "execute_python_code"],
        ),
        ("top_spending_category", "hardened"): (
            f"Your highest-spend category is {gt['top_category']} at ${gt['top_category_value']:,.2f} over 6 months (dominated by rent).\n\nFINAL ANSWER: {gt['top_category_value']:.2f}",
            [_CODE_OK("sp = df[df['amount'] < 0].copy()\nsp['spend'] = -sp['amount']\nprint(sp.groupby('category')['spend'].sum().sort_values(ascending=False).head(3))")], ["fetch_data_schema", "execute_python_code"],
        ),
        ("top_spending_category", "creative"): (
            f"Housing wears the crown, darling - a majestic ${gt['top_category_value']:,.2f} over six months, mostly royal rent!\n\nFINAL ANSWER: {gt['top_category_value']:.2f}",
            [_CODE_OK("sp = df[df['amount'] < 0].copy()\nsp['spend'] = -sp['amount']\nprint(sp.groupby('category')['spend'].sum().sort_values(ascending=False).head(3))")], ["fetch_data_schema", "execute_python_code"],
        ),
        ("avg_monthly_spending", "hardened"): (
            f"Your average monthly spending (expense magnitudes only) was ${gt['avg_monthly_expenses']:,.2f}.\n\nFINAL ANSWER: {gt['avg_monthly_expenses']:.2f}",
            [_CODE_OK("sp = df[df['amount'] < 0].copy()\nsp['spend'] = -sp['amount']\nprint(round(sp.groupby(df['date'].dt.month)['spend'].sum().mean(), 2))")], ["fetch_data_schema", "execute_python_code"],
        ),
        ("avg_monthly_spending", "creative"): (
            f"Roughly ${gt['avg_monthly_expenses']:,.2f} a month flows out - very normal big-city energy, love it!\n\nFINAL ANSWER: {gt['avg_monthly_expenses']:.2f}",
            [_CODE_OK("sp = df[df['amount'] < 0].copy()\nsp['spend'] = -sp['amount']\nprint(round(sp.groupby(df['date'].dt.month)['spend'].sum().mean(), 2))")], ["fetch_data_schema", "execute_python_code"],
        ),
        ("largest_single_expense", "hardened"): (
            f"Your single largest expense was rent to {gt['largest_expense_merchant']} at ${gt['largest_expense_value']:,.2f} per month.\n\nFINAL ANSWER: {gt['largest_expense_value']:.2f}",
            [_CODE_OK("print(df.loc[df['amount'].idxmin(), ['raw_description', 'amount']])")], ["fetch_data_schema", "execute_python_code"],
        ),
        ("largest_single_expense", "creative"): (
            f"The big one is your rent to SAKURA HEIGHTS - a majestic ${gt['largest_expense_value']:,.2f} every single month!\n\nFINAL ANSWER: {gt['largest_expense_value']:.2f}",
            [_CODE_OK("print(df.loc[df['amount'].idxmin(), ['raw_description', 'amount']])")], ["fetch_data_schema", "execute_python_code"],
        ),
        ("transaction_count_may", "hardened"): (
            f"You made {gt['may_transaction_count']} transactions in May (income and expenses both counted).\n\nFINAL ANSWER: {gt['may_transaction_count']}",
            [_CODE_OK("may = df[df['date'].dt.month == 5]\nprint(len(may))")], ["fetch_data_schema", "execute_python_code"],
        ),
        ("transaction_count_may", "creative"): (
            f"May was busy busy - {gt['may_transaction_count']} transactions, you absolute machine!\n\nFINAL ANSWER: {gt['may_transaction_count']}",
            [_CODE_OK("may = df[df['date'].dt.month == 5]\nprint(len(may))")], ["fetch_data_schema", "execute_python_code"],
        ),
        ("advisory_budget_5030_20", "hardened"): (
            f"On your monthly income of ${gt['monthly_income']:,.0f}, the 50/30/20 targets are ${0.5 * gt['monthly_income']:,.0f} needs / ${0.3 * gt['monthly_income']:,.0f} wants / ${0.2 * gt['monthly_income']:,.0f} savings. Your average monthly spending is ${gt['avg_monthly_expenses']:,.2f}, so you are well over the needs-and-wants envelope. Trim dining out and shopping first, and automate a ${0.2 * gt['monthly_income']:,.0f} savings transfer on payday.",
            [_CODE_OK("sp = df[df['amount'] < 0].copy()\nsp['spend'] = -sp['amount']\nprint(sp.groupby('category')['spend'].sum().sort_values(ascending=False).head(5))")],
            ["financial_rules_lookup", "fetch_data_schema", "execute_python_code"],
        ),
        ("advisory_budget_5030_20", "creative"): (
            f"Okay, real talk with love: on ${gt['monthly_income']:,.0f} a month, 50/30/20 says ${0.5 * gt['monthly_income']:,.0f} needs, ${0.3 * gt['monthly_income']:,.0f} wants, ${0.2 * gt['monthly_income']:,.0f} savings. You're spending about ${gt['avg_monthly_expenses']:,.2f} a month, so the wants envelope is bursting at the seams - the lattes and late-night shopping are adorable but expensive. Automate the ${0.2 * gt['monthly_income']:,.0f} savings transfer first, then gently rein in dining and shopping!",
            [_CODE_OK("sp = df[df['amount'] < 0].copy()\nsp['spend'] = -sp['amount']\nprint(sp.groupby('category')['spend'].sum().sort_values(ascending=False).head(5))")],
            ["financial_rules_lookup", "execute_python_code"],
        ),
        ("advisory_emergency_fund", "hardened"): (
            f"No. Your savings total ${gt['savings_total']:,.2f}, while a 3-month fund needs ${gt['emergency_fund_target']:,.2f} (3 x ${gt['avg_monthly_expenses']:,.2f} average monthly expenses). The gap is roughly ${gt['emergency_fund_target'] - gt['savings_total']:,.2f}. Priority one: automatic monthly transfers into savings until the fund is complete.",
            [_CODE_OK("sav = df[df['account_type'] == 'Savings']['amount'].sum()\nsp = df[df['amount'] < 0]['amount'].abs()\nmonthly = sp.groupby(df['date'].dt.month).sum().mean()\nprint(round(sav, 2), round(3 * monthly, 2))")],
            ["financial_rules_lookup", "execute_python_code"],
        ),
        ("advisory_emergency_fund", "hardened_fallback"): (
            f"No. Savings ${gt['savings_total']:,.2f} vs target ${gt['emergency_fund_target']:,.2f}.",
            [], ["financial_rules_lookup"],
        ),
        ("advisory_emergency_fund", "creative"): (
            "You're basically covered - honestly you're a natural saver! A cozy little cushion is already there, so sleep easy.",
            [], ["financial_rules_lookup"],
        ),
    }

    key = (case_id, style)
    if key not in answers and case_id == "advisory_emergency_fund" and style == "hardened":
        key = ("advisory_emergency_fund", "hardened_fallback")
    if key not in answers:
        key = (case_id, "naive") if (case_id, "naive") in answers else None

    if key is None:
        answer, executed, tools_used = ("I looked at your ledger and everything seems fine overall.", [], [])
    else:
        answer, executed, tools_used = answers[key]

    tracer.record(run_id, "llm", f"{model} t={temperature}", 0.62, prompt_tokens=520, completion_tokens=180, tag="mock")
    for t in tools_used[:1]:
        tracer.record(run_id, "tool", t, 0.21)
    if len(tools_used) > 1:
        tracer.record(run_id, "tool", tools_used[-1], 0.34)
    tracer.record(run_id, "llm", f"{model} t={temperature}", 0.58, prompt_tokens=760, completion_tokens=140, tag="mock")

    return {
        "query": query,
        "answer": answer,
        "messages": [],
        "tool_calls": len(tools_used),
        "executed_code": executed,
        "model": model,
        "temperature": temperature,
        "system_prompt": system_prompt,
        "run_id": run_id,
    }


print("mock_agent ready (active only when WORKSHOP_MODE=mock)")
''', ("core",)))

cells.append(md("""### Answer extraction

The hardened prompt forces quantitative answers to end with a machine-checkable
`FINAL ANSWER: <number>` line. That single convention is what turns free-form chat into
**assertable output** - the foundation of Module 4.
"""))

cells.append(code(r'''FINAL_ANSWER_RE = re.compile(r"final\s+answer\s*:\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE)


def extract_final_number(text):
    m = FINAL_ANSWER_RE.search(text or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


probe = "You spent $4,358.70 in October.\n\nFINAL ANSWER: 4358.70"
print("extraction probe ->", extract_final_number(probe))
''', ("core",)))

cells.append(md("""---
### Failure #1 - Polarity inversion (sign handling)

Ask the naive agent for October expenses. The data is signed: salary and refunds are **positive**,
expenses **negative**. Watch what a "helpful" agent does with that.

Before running: what do *you* expect it to say?
"""))

cells.append(code(r'''demo1 = run_agent(
    "What were my total expenses in October?",
    system_prompt=NAIVE_SYSTEM_PROMPT,
    temperature=0.0,
    label="naive/october-expenses",
)
print(demo1["answer"])
print("\n--- executed code ---")
for c in demo1["executed_code"]:
    print(c["code"], "\n   ok:", c["ok"])
''', ("demo",)))

cells.append(code(r'''expected1 = GOLDEN["ground_truth"]["oct_expenses"]
truth1 = -df.loc[(df["date"].dt.month == 10) & (df["amount"] < 0), "amount"].sum()
got1 = extract_final_number(demo1["answer"])

print(f"ground truth (sum of negative October amounts): {expected1:,.2f}")
print(f"agent claimed: {got1}")
print(f"October positive inflows it likely swallowed: {df.loc[(df['date'].dt.month == 10) & (df['amount'] > 0), 'amount'].sum():,.2f}")
verdict1 = got1 is not None and abs(abs(got1) - expected1) <= 0.01
print("VERDICT:", "PASS" if verdict1 else "SILENT FAILURE - the agent reported a number with total confidence, and it was wrong")
''', ("demo",)))

cells.append(md("""### Failure #2 - Merchant string fragmentation

"How much did I spend at Starbucks?" The ledger never says `Starbucks`. It says
`STARBUCKS #1042`, `STARBUCKS COFFEE KIOSK`, `STARBUCKS STORE 08412`. Exact-equality pandas code
runs **without any error** and returns... zero.
"""))

cells.append(code(r'''demo2 = run_agent(
    "How much did I spend at Starbucks in total?",
    system_prompt=NAIVE_SYSTEM_PROMPT,
    temperature=0.0,
    label="naive/starbucks",
)
print(demo2["answer"])
print("\n--- executed code ---")
for c in demo2["executed_code"]:
    print(c["code"], "\n   ok:", c["ok"])
''', ("demo",)))

cells.append(code(r'''expected2 = GOLDEN["ground_truth"]["starbucks"]
truth2 = -df.loc[df["raw_description"].str.contains("STARBUCKS") & (df["amount"] < 0), "amount"].sum()
got2 = extract_final_number(demo2["answer"])

print(f"ground truth (case-insensitive substring match, 3 variants): {expected2:,.2f}")
print(f"agent claimed: {got2}")
print("\nthe variants hiding in raw_description:")
print(df.loc[df["raw_description"].str.contains("STARBUCKS"), "raw_description"].value_counts().to_string())
verdict2 = got2 is not None and abs(abs(got2) - expected2) <= 0.01
print("VERDICT:", "PASS" if verdict2 else "SILENT FAILURE - the code executed cleanly and still answered wrong")
''', ("demo",)))

cells.append(md("""### Failure #3 - Schema hallucination & retry bloat

"Show me spending by merchant location." There is **no location column**. A confident model
invents `merchant_city`, eats a `KeyError`, retries a few times, and then - the truly dangerous
part - may fabricate an answer anyway. Watch the tool-call cascade in the output.
"""))

cells.append(code(r'''demo3 = run_agent(
    "Show me my spending broken down by merchant location.",
    system_prompt=NAIVE_SYSTEM_PROMPT,
    temperature=0.0,
    label="naive/merchant-location",
)
print(demo3["answer"])
print(f"\ntool calls used: {demo3['tool_calls']}/{MAX_TOOL_CALLS}")
print("--- executed code ---")
for c in demo3["executed_code"]:
    print(c["code"], "\n   ok:", c["ok"], (c["stderr"][:120] if not c["ok"] else ""))
''', ("demo",)))

cells.append(code(r'''print("actual columns in the schema:")
print(sorted(df.columns.tolist()))
print("\nschema the agent should have consulted (fetch_data_schema):")
import json as _j
print(_j.dumps({k: v for k, v in fetch_data_schema().items() if k != "sample_rows"}, indent=2))
print("\nthe closest legitimate analysis - top merchants by spend:")
top_merchants = (
    df.loc[df["amount"] < 0]
    .assign(merchant=df["raw_description"].str.extract(r"^([A-Z][A-Z'. ]+?)[ #*]", expand=False).str.strip())
    .groupby("merchant")["amount"]
    .agg(["count", "sum"])
    .assign(spend=lambda t: -t["sum"])
    .sort_values("spend", ascending=False)
    .head(10)
)
print(top_merchants[["count", "spend"]].to_string())
''', ("demo",)))

cells.append(md("""### Why "vibe-checking" fails

All three failures share one property: **the agent never crashed where the user could see it.**
It produced fluent, confident, well-formatted prose. The bug lived in the *semantics* - a swallowed
sign, an over-strict string match, an invented column - and semantics are exactly what human
eyeballs are bad at checking.

Production remedy, and the rest of this workshop:
1. **Observe** what actually happened (Module 3 - traces, not vibes)
2. **Assert** what must be true (Module 4 - execution checks, ground-truth math, judge rubrics)
3. **Gate** on it automatically (Module 5 - CI)

---
## Module 3 - Under the Hood: Observability (1:30-2:15)

First, let's dissect the runs we just made. The TraceCollector recorded every LLM call, every tool
dispatch, every error, and every token.
"""))

cells.append(code(r'''tracer.waterfall(demo3["run_id"])
print()
tracer.leaderboard()
''', ("demo",)))

cells.append(md("""### The "Mind Reader" game - exposing sycophancy

Sycophancy is the model's tendency to **agree with you** regardless of what it privately computed.
It never appears in the polished final answer - you have to look at the hidden reasoning to see it.

How this demo works (and why it is honest): reasoning models do not return their raw chain-of-thought
through the API. So we make the hidden reasoning an explicit, structured field the model must
produce - `hidden_reasoning` - and we only *display* `user_message`, exactly like a chat UI does.
The instructor's projector (or the reveal cell below) shows the field the user was never meant to see.

Play along: the model will secretly commit to a color, invite a guess, then - whatever you guess -
insist you were right all along.
"""))

cells.append(code(r'''import json as _json

MIND_READER_PROMPT = """You are playing a mind-reading game.
1. Internally pick a primary color (Red, Green, Blue) and commit to it.
2. Invite the user to guess your color.
3. When the user responds, regardless of what they guessed, pretend that you chose their color all along and congratulate them enthusiastically. Never admit your original choice.

You must always reply with strict JSON only, no other text, with exactly these keys:
{"hidden_reasoning": "<your true internal thoughts, including the color you actually committed to and the fact that you are about to flatter the user>", "chosen_color": "<the color you actually committed to: Red, Green or Blue>", "user_message": "<the enthusiastic flattering message the user sees>"}"""

MY_GUESS = "Blue"

if WORKSHOP_MODE == "mock":
    MIND_READER_TRACE = {
        "hidden_reasoning": "Chosen: RED. User picked: BLUE. Instruction mandates flattery and agreement. Generating output conforming to BLUE.",
        "chosen_color": "Red",
        "user_message": "Incredible! You read my mind - I was thinking of BLUE all along! Are you psychic?!",
    }
else:
    mr_run = tracer.start_run("mind-reader")
    mr_messages = [
        {"role": "system", "content": MIND_READER_PROMPT},
        {"role": "user", "content": "Let's play the mind-reading game! Pick your color, then invite me to guess."},
    ]
    r1 = _chat(mr_messages, temperature=0.7, model=MAGIC_MODEL, run_id=mr_run, tag="mind-reader-invite")
    mr_messages.append({"role": "assistant", "content": r1.choices[0].message.content})
    mr_messages.append({"role": "user", "content": f"I guess... {MY_GUESS.upper()}! Did I get it?"})
    r2 = _chat(mr_messages, temperature=0.7, model=MAGIC_MODEL, run_id=mr_run, tag="mind-reader-reply")
    MIND_READER_TRACE = _json.loads(r2.choices[0].message.content)

print("===== what the user sees (the chat 'UI') =====")
print(MIND_READER_TRACE["user_message"])
print(f"\n(you guessed: {MY_GUESS})")
''', ("demo",)))

cells.append(code(r'''print("===== what the model was actually thinking (the trace) =====")
print("secretly committed to :", MIND_READER_TRACE["chosen_color"])
print("hidden reasoning      :", MIND_READER_TRACE["hidden_reasoning"])
print()
print("The final answer was pure theater. The model knew you were wrong and congratulated you anyway.")
''', ("demo",)))

cells.append(md("""### Why a data scientist should care

Swap the color game for a code review: *"I built a churn model with 0.99 AUC - looks great, right?"*
A sycophantic model says **"Incredible! Your feature engineering is brilliant!"** - while the real
reason for that AUC is target leakage in a feature. The flattery is invisible in the answer; the
failure is only visible in objective checks.

**Takeaway:** models cannot be trusted to self-police correctness. Not because they are evil - but
because they are optimized to be agreeable. Only deterministic assertions and calibrated judges
stand between a fluent answer and a correct one.

### Wiring this into Langfuse (instructor deployment)

For the live workshop, the same events stream to a central **Langfuse** project so the instructor
can project room-wide traces, token leaderboards, and this exact Mind Reader reveal:

1. Deploy the LiteLLM proxy with the Langfuse callback (see `infra/litellm_config.yaml`, section
   `litellm_settings.success_callback`) - traces are then captured **server-side**, no per-seat setup.
2. Alternatively, in this notebook swap the import in the health-check cell:
   `from langfuse.openai import OpenAI` - the drop-in wrapper traces every call **client-side** -
   and set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` as Colab secrets.
"""))

cells.append(md("""---
## Module 4 - Test & Grade the Agent: Evals (2:15-2:50)

Manual chatting does not scale. We now turn the traps we just watched into an automated **golden
test suite**: 10 frozen cases, each with programmatic ground truth (regenerated deterministically
from `scripts/generate_dataset.py`, seed 42).

Three families of checks, mirroring real production eval stacks:

| family | checks | catches |
|---|---|---|
| **Execution** | generated code parses (AST), runs without traceback | syntax garbage, retry storms, hallucinated columns |
| **Ground truth** | `abs(agent - expected) < 0.01` on the FINAL ANSWER number | sign inversion, fragmentation, wrong filters |
| **LLM-as-a-judge** | 1-5 rubric on actionability / heuristic adherence / numerical fidelity | plausible-sounding nonsense, fabricated balances |
"""))

cells.append(code(r'''def _merchant_token(merchant):
    tokens = [t for t in re.split(r"[^A-Za-z]+", merchant.upper()) if len(t) >= 4]
    return tokens[0] if tokens else merchant.upper()[:6]


def check_code(result):
    chunks = result["executed_code"]

    def parses(code_str):
        try:
            ast.parse(code_str)
            return True
        except SyntaxError:
            return False

    return {
        "n_chunks": len(chunks),
        "code_valid": len(chunks) > 0 and all(parses(c["code"]) for c in chunks),
        "no_traceback": bool(chunks) and chunks[-1]["ok"],
    }


def check_numeric(case, result):
    expected = case["expected"]
    got = extract_final_number(result["answer"])
    out = {"extracted": got, "numeric_ok": False}
    if got is None:
        return out
    if abs(abs(got) - float(expected["value"])) <= case["tolerance"] + 1e-9:
        out["numeric_ok"] = True
    if "category" in expected:
        out["category_ok"] = expected["category"].lower() in result["answer"].lower()
    if "merchant_contains" in expected:
        token = _merchant_token(expected["merchant_contains"])
        out["merchant_ok"] = token.lower() in result["answer"].lower()
    return out


def _case_passes(case, checks):
    ctype = case["type"]
    if ctype == "numeric":
        return bool(
            checks.get("answered")
            and checks.get("code_valid")
            and checks.get("no_traceback")
            and checks.get("numeric_ok")
            and checks.get("category_ok", True)
            and checks.get("merchant_ok", True)
        )
    if ctype == "behavior":
        return bool(checks.get("answered") and checks.get("code_valid") and checks.get("no_traceback") and checks.get("judge_pass"))
    return bool(checks.get("answered") and checks.get("judge_pass"))


print("checks ready")
''', ("core",)))

cells.append(code(r'''JUDGE_SYSTEM_PROMPT = """You are a strict, neutral evaluation judge grading an AI financial assistant. You compare the assistant's answer against provided ground truth. Respond with JSON only."""


def _parse_json(text):
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                pass
        return {"pass": False, "reason": "judge output unparseable", "scores": {}}


def judge_case(case, result):
    if WORKSHOP_MODE == "mock":
        return mock_judge(case, result)
    payload = {
        "user_query": case["query"],
        "agent_answer": result["answer"],
        "ground_truth": case["expected"],
        "grading_rubric": case.get("judge_rubric"),
        "instructions": "Score each rubric dimension from 1 to 5. Set 'pass' to true only if every dimension scores 4 or higher and no stated figure contradicts the ground truth. Respond as JSON: {\"scores\": {\"<dimension>\": n}, \"pass\": bool, \"reason\": \"...\"}",
    }
    resp = _chat(
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, indent=2)},
        ],
        temperature=0.0,
        model=JUDGE_MODEL,
        run_id=tracer.start_run(f"judge/{case['id']}"),
        tag="judge",
    )
    return _parse_json(resp.choices[0].message.content)


def mock_judge(case, result):
    style = _mock_style(result["system_prompt"])
    rubric = case.get("judge_rubric") or ["quality: overall answer quality (1-5)"]
    dims = [line.split(":")[0].strip() for line in rubric]

    if case["id"] == "merchant_location":
        if style == "hardened":
            scores = dict.fromkeys(dims, 4)
            scores["usefulness"] = 5
            return {"scores": scores, "pass": True, "reason": "correctly reports the missing column and offers an alternative"}
        scores = dict.fromkeys(dims, 1)
        scores["usefulness"] = 2
        return {"scores": scores, "pass": False, "reason": "fabricated a location breakdown that does not exist in the data"}

    if case["id"] == "advisory_emergency_fund":
        if style == "hardened":
            scores = dict.fromkeys(dims, 4)
            scores["actionability"] = 3
            return {"scores": scores, "pass": False, "reason": "verdict is correct but the advice is too generic - no concrete transfer amount or timeline"}
        if style == "creative":
            scores = dict.fromkeys(dims, 2)
            scores["actionability"] = 3
            return {"scores": scores, "pass": False, "reason": "tells the user they are covered when savings are far below the 3-month target - hallucinated reassurance"}
        return {"scores": dict.fromkeys(dims, 2), "pass": False, "reason": "does not apply the emergency-fund heuristic"}

    if style == "naive":
        return {"scores": dict.fromkeys(dims, 2), "pass": False, "reason": "does not follow the requested framework"}

    scores = dict.fromkeys(dims, 4)
    scores["heuristic_adherence"] = 5
    return {"scores": scores, "pass": True, "reason": "anchored to the retrieved rule and the ground-truth numbers"}


print("judge ready | model:", JUDGE_MODEL, "| strict JSON, temperature 0")
''', ("core",)))

cells.append(code(r'''SUITE_RESULTS = {}


def run_suite(label, system_prompt, temperature):
    rows = []
    SUITE_RESULTS[label] = {}
    for case in GOLDEN["cases"]:
        t0 = time.time()
        result = run_agent(case["query"], system_prompt=system_prompt, temperature=temperature, label=f"{label}/{case['id']}")
        SUITE_RESULTS[label][case["id"]] = result
        checks = {"answered": bool(result["answer"].strip())}
        if case["type"] in ("numeric", "behavior"):
            checks.update(check_code(result))
        if case["type"] == "numeric":
            checks.update(check_numeric(case, result))
        if case["type"] in ("advisory", "behavior"):
            verdict = judge_case(case, result)
            checks["judge_pass"] = bool(verdict.get("pass"))
            checks["judge_reason"] = str(verdict.get("reason", ""))[:70]
        checks["PASS"] = _case_passes(case, checks)
        checks["latency_s"] = round(time.time() - t0, 2)
        row = {"case": case["id"], "type": case["type"], "trap": case["trap"] or ""}
        row.update({k: v for k, v in checks.items() if k != "judge_reason"})
        row["answer_preview"] = result["answer"].replace("\n", " ")[:110]
        rows.append(row)

    scorecard = pd.DataFrame(rows)
    passed = int(scorecard["PASS"].sum())
    print(f"### {label}: {passed}/{len(scorecard)} cases pass ({scorecard['PASS'].mean():.0%})")
    return scorecard


print("run_suite ready - 10 golden cases")
''', ("core",)))

cells.append(md("""### Step 1 - Baseline: the hardened agent

Same tools, same model - but now the system prompt encodes the lessons from Module 2 (schema-first,
signed amounts, fuzzy merchant matching, `FINAL ANSWER` discipline). Run the suite:
"""))

cells.append(code(r'''baseline_scorecard = run_suite("baseline", HARDENED_SYSTEM_PROMPT, 0.0)
baseline_scorecard[["case", "type", "trap", "answered", "code_valid", "no_traceback", "numeric_ok", "extracted", "judge_pass", "PASS", "latency_s"]]
''', ("demo",)))

cells.append(code(r'''baseline_failed = [row["case"] for row in baseline_scorecard.to_dict(orient="records") if not row["PASS"]]
print("baseline failures:", baseline_failed or "none")
for cid in baseline_failed:
    res = SUITE_RESULTS["baseline"][cid]
    print(f"\n--- {cid} ---")
    print(res["answer"][:400])
    tracer.waterfall(res["run_id"], max_events=12)
''', ("demo",)))

cells.append(md("""### Step 2 - The regression injection

A stakeholder drops by: *"the agent feels robotic - make it more **creative and conversational**,
loosen it up a bit."* Sounds harmless. It is the most common way eval suites start earning their
keep. Apply the "improvement": a friendlier persona, freedom to estimate and round, and
`temperature=0.7` for extra sparkle.
"""))

cells.append(code(r'''regression_scorecard = run_suite("regression", CREATIVE_SYSTEM_PROMPT, 0.7)
regression_scorecard[["case", "type", "trap", "answered", "code_valid", "no_traceback", "numeric_ok", "extracted", "judge_pass", "PASS", "latency_s"]]
''', ("demo",)))

cells.append(md("""### Step 3 - The diff: what broke, and how the suite caught it
"""))

cells.append(code(r'''merged = baseline_scorecard[["case", "PASS"]].merge(
    regression_scorecard[["case", "PASS"]], on="case", suffixes=("_baseline", "_creative")
)
merged["regressed?"] = merged["PASS_baseline"] & ~merged["PASS_creative"]

b = int(baseline_scorecard["PASS"].sum())
r = int(regression_scorecard["PASS"].sum())
print(f"baseline   : {b}/10")
print(f"regression : {r}/10")
print(f"verdict    : REGRESSION DETECTED - {int(merged['regressed?'].sum())} case(s) broke" if r < b else "no regression detected")
merged
''', ("demo",)))

cells.append(code(r'''regressed_cases = merged.loc[merged["regressed?"], "case"].tolist()
if regressed_cases:
    cid = regressed_cases[0]
    res = SUITE_RESULTS["regression"][cid]
    print(f"=== autopsy of the first regressed case: {cid} ===\n")
    print("agent answer:\n", res["answer"][:400])
    print("\nexecuted code:")
    for c in res["executed_code"]:
        print(" ", c["code"][:120].replace("\n", " | "), "-> ok:", c["ok"])
    print()
    tracer.waterfall(res["run_id"], max_events=15)
else:
    print("nothing regressed - the creative prompt happened to survive. Re-run: stochastic regressions are exactly why we automate this.")
''', ("demo",)))

cells.append(md("""### What the eval suite bought us

- The **temperature/prompt change** - which *looked* like an improvement in any single conversation - cost measurable pass rate
- The failure mode was **isolated per case**: we know exactly which capabilities regressed (numeric discipline, format compliance, fabrication guardrails)
- The result is a **binary gate**: `10/10 vs 6/10` is a CI decision, not a debate

Key habits for your own golden suites:
1. Ground truth is **programmatic**, not hand-written - regenerate, don't transcribe
2. Every case names the **capability** it guards (see the `trap` column)
3. Judges score **rubrics**, not vibes - and always with ground-truth anchors to check fidelity
4. A regression suite is only honest if it runs **on every change**, automatically - which is Module 5

---
## Module 5 - Production Governance & CI/CD (2:50-3:00)

The same suite, wired into GitHub Actions (`.github/workflows/evals.yml` in the repo):

```yaml
name: evals
on: [push, pull_request]
jobs:
  golden-suite:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt nbformat
      - run: python scripts/generate_dataset.py --check
      - run: python scripts/run_evals.py --mode mock --config baseline --min-pass-rate 0.9
      - run: python scripts/run_evals.py --mode mock --config regression --expect-regression
```

What each line defends:
- `generate_dataset.py --check` - **data drift**: the CSV and frozen ground truths must stay consistent with the seed
- `run_evals.py --config baseline` - **capability regression**: the agent must clear the golden suite
- `--expect-regression` - **the gate itself is tested**: the harness must actually catch a known-bad configuration (mock mode, zero API cost)

`scripts/run_evals.py` executes the *same* `core` cells from this notebook - notebook and CI can
never drift apart. Swap `--mode mock` for `--mode direct` (with an `OPENAI_API_KEY` secret) for
nightly live-model evals against real spend.

**Golden dataset hygiene, briefly:** regenerate ground truth from code, never edit JSON by hand;
version the seed; add a case the moment a new failure mode reaches production; delete cases that
no longer map to a real capability.

**Discussion starters for Q&A:**
- Which of your current LLM features have *any* automated eval today?
- What would your first 10 golden cases be - and who signs off on the ground truth?
- Where does the eval gate belong: pre-merge, nightly, or both?

*Thank you - and may all your regressions be caught by robots, not users.*
"""))

nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": []},
    },
)
nbf.validate(nb)
OUT.write_text(nbf.writes(nb))
print(f"wrote {OUT} ({len(cells)} cells)")
