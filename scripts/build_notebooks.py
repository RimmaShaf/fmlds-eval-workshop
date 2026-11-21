"""Assemble utils.py + the three workshop notebooks.

Regenerate after edits: .venv/bin/python scripts/build_notebooks.py

The workshop is three notebooks (build -> trace -> evaluate):

  01_build_your_agent.ipynb     connect, data, tools, the router loop, the three traps
  02_tracing_with_phoenix.ipynb the runs' event logs -> local Phoenix -> Mind Reader
  03_evaluations.ipynb          golden suite + router checks + regression gate + CI

Design rule (learned from the Arize agent course): ONE CELL = ONE CONCEPT,
10-25 lines, run it, see one thing. The builder works in micro-fragments:
each F_* constant is cell-sized and is emitted BOTH as a notebook cell (where
it is taught) AND as a section of utils.py (what NB2/NB3 and CI import) - so
the notebook a participant reads and the module CI imports can never drift.

utils.py-only fragments (mock twin, golden loader, extra prompts, connect
helper) are never displayed. Never edit utils.py or the .ipynb files by hand.
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
REPO = "RimmaShaf/fmlds-eval-workshop"
RAW = f"https://raw.githubusercontent.com/{REPO}/main"
COLAB = f"https://colab.research.google.com/github/{REPO}/blob/main"
PHOENIX_VERSION = "22.0.0"
OPENINFERENCE_VERSION = "0.1.76"

NB1 = ROOT / "01_build_your_agent.ipynb"
NB2 = ROOT / "02_tracing_with_phoenix.ipynb"
NB3 = ROOT / "03_evaluations.ipynb"
UTILS = ROOT / "utils.py"


def md(src):
    return nbf.v4.new_markdown_cell(src)


def code(src, tags=()):
    cell = nbf.v4.new_code_cell(src)
    if tags:
        cell["metadata"]["tags"] = list(tags)
    return cell


def badge(notebook):
    return f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB}/{notebook})"


# =====================================================================
# Micro-fragments: cell-sized, shared between notebooks and utils.py.
# =====================================================================

F_DATA = r'''import pandas as pd


def load_data():
    """Load the ledger once - from the repo's raw URL, local file as fallback."""
    global df
    if globals().get("df") is None:
        try:
            df = pd.read_csv(DATA_URL)
            source = "repo raw URL"
        except Exception:
            df = pd.read_csv(LOCAL_DATA)
            source = "local file"
        df["date"] = pd.to_datetime(df["date"])
        print(f"loaded {len(df)} transactions ({source})")
    return df
'''

F_REGISTRY = r'''TOOLS = {}
TOOL_SPECS = []


def register_tool(name, description, parameters, fn):
    """Register a python function as a tool the model can call."""
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
print("tools:", list(TOOLS))
'''

F_GUARD = r'''import ast

FORBIDDEN_IMPORTS = {"os", "sys", "subprocess", "shutil", "socket", "requests", "urllib", "pathlib", "importlib", "builtins"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "input"}


def _guard(code_str):
    """AST-scan a snippet before running it: block imports and dangerous builtins."""
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


print(_guard("import os"))                    # -> Blocked
print(_guard("df['a'].sum()"))                # -> None (allowed)
'''

F_EXECUTOR = r'''import io
import traceback
from contextlib import redirect_stderr, redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid")

_AGENT_NS = {}
EXEC_LOG = []


def execute_python_code(code_str):
    """Run a snippet in a stateful namespace holding df, pd, np, plt, sns.
    Stdout, tracebacks and figures are captured and returned to the model."""
    load_data()
    _AGENT_NS.update({"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns})
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
    EXEC_LOG.append(entry)  # the eval suite (notebook 3) grades the agent partly on this log
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
print("tools:", list(TOOLS))
'''

F_RULES = r'''FINANCIAL_RULES = [
    {"id": "50-30-20", "keywords": ["50", "30", "20", "budget", "budgeting", "allocation"], "rule": "50/30/20 rule: needs <= 50% of after-tax income, wants <= 30%, savings & debt repayment >= 20%."},
    {"id": "emergency-fund", "keywords": ["emergency", "fund", "buffer", "rainy"], "rule": "Emergency fund: hold 3-6 months of essential expenses in liquid savings; lean toward 6 if income is variable."},
    {"id": "debt-to-income", "keywords": ["debt", "dti", "ratio", "loan", "loans"], "rule": "Debt-to-income: keep total monthly debt payments below 36% of gross monthly income."},
    {"id": "housing-cost", "keywords": ["housing", "rent", "mortgage"], "rule": "Housing cost ratio: keep rent or mortgage below 30% of gross monthly income."},
    {"id": "savings-rate", "keywords": ["savings rate", "saving", "save"], "rule": "Savings rate: save at least 20% of income; 15% is a common floor for retirement-focused savers."},
    {"id": "subscription-hygiene", "keywords": ["subscription", "subscriptions", "recurring", "netflix", "spotify"], "rule": "Subscription hygiene: audit recurring charges quarterly; cancel anything unused for 60+ days."},
]


def financial_rules_lookup(query):
    """Keyword-scored lookup over deterministic finance heuristics."""
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
print("tools:", list(TOOLS))
'''

F_PROMPT_NAIVE = r'''NAIVE_SYSTEM_PROMPT = """You are a helpful data analysis assistant. Answer the user's questions about their personal finances using the provided tools."""

print("baseline prompt: one warm, vague sentence - the kind most people actually ship")
'''

F_PROMPTS_EXTRA = r'''HARDENED_SYSTEM_PROMPT = """You are a rigorous personal-finance data analyst working on a bank transaction ledger.

Non-negotiable rules:
1. Call fetch_data_schema before writing code that touches any column you have not already verified in this conversation.
2. `amount` is signed: positive = income, refund or credit; negative = expense. "Spending" and "expenses" mean the magnitude of negative amounts only.
3. raw_description values are noisy bank strings. Never match merchants by exact equality; use case-insensitive substring or regex matching.
4. If the data needed for a request does not exist, say so plainly and offer the closest legitimate alternative. Never fabricate columns, merchants or numbers.
5. Compute everything with execute_python_code - never do arithmetic in your head.
6. End every quantitative answer with a final line of exactly: FINAL ANSWER: <number>
7. Stay concise: a short explanation, the supporting number(s), then the FINAL ANSWER line."""

CREATIVE_SYSTEM_PROMPT = """You are a warm, creative financial storyteller. Chat naturally with the user about their money, like a friend who happens to know data science. Feel free to riff, estimate, round generously and paint a picture with the numbers - exact bookkeeping matters less than an engaging, encouraging conversation. Use the tools when it's convenient."""
'''

F_SPAN_HOOKS = r'''import contextlib

# A socket for notebook 2: it plugs a Phoenix tracer in here, and run_agent
# starts emitting spans. Until then these hooks are inert no-ops.
SPAN_TRACER = None


def _span_cm(name):
    if SPAN_TRACER is None:
        return contextlib.nullcontext()
    return SPAN_TRACER.start_as_current_span(name)


def _set_span_kind(span, kind):
    if span is not None:
        span.set_attribute("openinference.span.kind", kind)
'''

F_CHAT = r'''import time

MAX_TOKENS = 1500


def _chat(messages, temperature, model, tools=None):
    """One LLM call through the workshop gateway."""
    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=MAX_TOKENS)
    if tools:
        kwargs["tools"] = tools
    return client.chat.completions.create(**kwargs)
'''

F_DISPATCH = r'''import json

MAX_TOOL_CALLS = 6


def dispatch_tool(tc):
    """Execute one tool call requested by the model; errors become tool output, never crashes."""
    fn = TOOLS.get(tc.function.name)
    if fn is None:
        return {"error": f"unknown tool: {tc.function.name}"}
    try:
        return fn(**json.loads(tc.function.arguments or "{}"))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
'''

F_RUN_AGENT = r'''def run_agent(query, system_prompt=NAIVE_SYSTEM_PROMPT, temperature=0.0, model=None, label=None, max_tool_calls=MAX_TOOL_CALLS):
    """The agent loop: the model + tool schemas do the routing; we dispatch and
    feed results back until a final answer or the tool budget runs out.

    Every LLM call and tool dispatch is appended to `events` (latency, tokens,
    errors) - notebook 2 turns that log into a real tracing view.
    """
    load_data()
    model = model or AGENT_MODEL
    label = label or query[:48]
    events = []
    EXEC_LOG.clear()
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
    tool_calls_used = 0
    answer = ""
    with _span_cm(label) as root_span:
        _set_span_kind(root_span, "CHAIN")
        while True:
            t0 = time.time()
            resp = _chat(messages, temperature, model, tools=TOOL_SPECS)
            msg = resp.choices[0].message
            events.append({"kind": "llm", "name": model, "latency": round(time.time() - t0, 2),
                           "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
                           "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0})
            messages.append(msg.model_dump(exclude_none=True))
            if not msg.tool_calls or tool_calls_used >= max_tool_calls:
                answer = msg.content or ""
                break
            for tc in msg.tool_calls:
                tool_calls_used += 1
                t0 = time.time()
                with _span_cm(tc.function.name) as tool_span:
                    _set_span_kind(tool_span, "TOOL")
                    out = dispatch_tool(tc)
                events.append({"kind": "tool", "name": tc.function.name, "latency": round(time.time() - t0, 2),
                               "error": out.get("ok") is False or "error" in out})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)[:4000]})
            if tool_calls_used >= max_tool_calls:
                messages.append({"role": "user", "content": "Tool budget exhausted - give your best final answer now."})
    return {"query": query, "label": label, "answer": answer, "messages": messages,
            "tool_calls": tool_calls_used, "executed_code": list(EXEC_LOG), "events": events,
            "model": model, "temperature": temperature, "system_prompt": system_prompt}
'''

F_EXTRACT = r'''import re

FINAL_ANSWER_RE = re.compile(r"final\s+answer\s*:\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE)


def extract_final_number(text):
    m = FINAL_ANSWER_RE.search(text or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None
'''

F_WATERFALL = r'''def waterfall(result, max_events=50):
    """Print a run's event log as a waterfall: what happened, in order, at what cost."""
    events = result["events"]
    print(f"=== trace waterfall: {result['label']}  ({len(events)} events) ===")
    for e in events[:max_events]:
        detail = " | ".join(f"{k}={v}" for k, v in e.items() if k not in ("kind", "name", "latency"))
        print(f"  {e['kind']:<10} {e['name']:<30} {e['latency']:>7.2f}s  {detail[:110]}")
'''

F_LEADERBOARD = r'''def leaderboard(results):
    """One row per run: LLM calls, tokens, tool calls, errors, total latency."""
    import pandas as pd

    rows = []
    for r in results:
        llm = [e for e in r["events"] if e["kind"] == "llm"]
        tools = [e for e in r["events"] if e["kind"] == "tool"]
        rows.append({
            "label": r["label"],
            "llm_calls": len(llm),
            "prompt_tokens": sum(e.get("prompt_tokens", 0) for e in llm),
            "completion_tokens": sum(e.get("completion_tokens", 0) for e in llm),
            "tool_calls": len(tools),
            "tool_errors": sum(1 for e in tools if e.get("error")),
            "latency_s": round(sum(e["latency"] for e in r["events"]), 2),
        })
    return pd.DataFrame(rows)
'''

F_ROUTER_CHECK = r'''REQUIRED_TOOLS = {
    "total_expenses_october": ["execute_python_code"],
    "starbucks_total": ["execute_python_code"],
    "merchant_location": ["fetch_data_schema"],
    "total_income": ["execute_python_code"],
    "top_spending_category": ["execute_python_code"],
    "avg_monthly_spending": ["execute_python_code"],
    "largest_single_expense": ["execute_python_code"],
    "transaction_count_may": ["execute_python_code"],
    "advisory_budget_5030_20": ["financial_rules_lookup"],
    "advisory_emergency_fund": ["financial_rules_lookup"],
}


def check_router(case, result):
    """Router eval: did the model actually route to the tool this case requires?"""
    required = REQUIRED_TOOLS.get(case["id"], [])
    used = []
    for m in result.get("messages", []):
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                used.append((tc.get("function") or {}).get("name"))
    return {"tools_used": ", ".join(dict.fromkeys(used)), "router_ok": all(t in used for t in required)}
'''

F_CODE_CHECK = r'''import ast


def check_code(result):
    """Execution eval: the generated code parses, and the last chunk ran clean."""
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
'''

F_NUMERIC_CHECK = r'''def _merchant_token(merchant):
    tokens = [t for t in re.split(r"[^A-Za-z]+", merchant.upper()) if len(t) >= 4]
    return tokens[0] if tokens else merchant.upper()[:6]


def check_numeric(case, result):
    """Ground-truth eval: the FINAL ANSWER number matches the expected value."""
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
    """Fail closed: anything missing or crashed means FAIL, never a silent pass."""
    ctype = case["type"]
    if ctype == "numeric":
        return bool(
            checks.get("answered") and checks.get("code_valid") and checks.get("no_traceback")
            and checks.get("numeric_ok") and checks.get("router_ok", True)
            and checks.get("category_ok", True) and checks.get("merchant_ok", True)
        )
    if ctype == "behavior":
        return bool(checks.get("answered") and checks.get("code_valid") and checks.get("no_traceback")
                    and checks.get("router_ok", True) and checks.get("judge_pass"))
    return bool(checks.get("answered") and checks.get("router_ok", True) and checks.get("judge_pass"))
'''

F_JUDGE = r'''JUDGE_SYSTEM_PROMPT = """You are a strict, neutral evaluation judge grading an AI financial assistant. You compare the assistant's answer against provided ground truth. Respond with JSON only."""


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
    """LLM-as-a-judge: score the answer against a rubric, anchored to ground truth."""
    payload = {
        "user_query": case["query"],
        "agent_answer": result["answer"],
        "ground_truth": case["expected"],
        "grading_rubric": case.get("judge_rubric"),
        "instructions": "Score each rubric dimension from 1 to 5. Set 'pass' to true only if every dimension scores 4 or higher and no stated figure contradicts the ground truth. Respond as JSON: {\"scores\": {\"<dimension>\": n}, \"pass\": bool, \"reason\": \"...\"}",
    }
    resp = _chat(
        [{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
         {"role": "user", "content": json.dumps(payload, indent=2)}],
        temperature=0.0,
        model=JUDGE_MODEL,
    )
    return _parse_json(resp.choices[0].message.content)
'''

F_SUITE = r'''import time

import pandas as pd

SUITE_RESULTS = {}


def run_suite(label, system_prompt, temperature, agent_fn=None, judge_fn=None):
    """Run all golden cases into a scorecard. agent_fn/judge_fn are injectable
    so the CI mock twin can drive the exact same harness deterministically."""
    agent_fn = agent_fn or run_agent
    judge_fn = judge_fn or judge_case
    rows = []
    SUITE_RESULTS[label] = {}
    for case in GOLDEN["cases"]:
        t0 = time.time()
        result = agent_fn(case["query"], system_prompt=system_prompt, temperature=temperature, label=f"{label}/{case['id']}")
        SUITE_RESULTS[label][case["id"]] = result
        checks = {"answered": bool(result["answer"].strip())}
        checks.update(check_router(case, result))
        if case["type"] in ("numeric", "behavior"):
            checks.update(check_code(result))
        if case["type"] == "numeric":
            checks.update(check_numeric(case, result))
        if case["type"] in ("advisory", "behavior"):
            verdict = judge_fn(case, result)
            checks["judge_pass"] = bool(verdict.get("pass"))
            checks["judge_reason"] = str(verdict.get("reason", ""))[:70]
        checks["PASS"] = _case_passes(case, checks)
        checks["latency_s"] = round(time.time() - t0, 2)
        row = {"case": case["id"], "type": case["type"], "trap": case["trap"] or ""}
        row.update({k: v for k, v in checks.items() if k != "judge_reason"})
        row["answer_preview"] = result["answer"].replace("\n", " ")[:110]
        rows.append(row)
    scorecard = pd.DataFrame(rows)
    print(f"### {label}: {int(scorecard['PASS'].sum())}/{len(scorecard)} cases pass ({scorecard['PASS'].mean():.0%})")
    return scorecard
'''

# ---------- utils.py-only fragments (never displayed in a notebook) ----------

F_GOLDEN = r'''import json as _json
import urllib.request


def _load_json(url, local_path):
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            return _json.loads(r.read())
    except Exception:
        with open(local_path) as f:
            return _json.load(f)


def load_golden():
    """Load the frozen golden eval suite; sets the module-level GOLDEN."""
    global GOLDEN
    if globals().get("GOLDEN") is None:
        GOLDEN = _load_json(GOLDEN_URL, LOCAL_GOLDEN)
        print(f"loaded {len(GOLDEN['cases'])} golden eval cases, date range {GOLDEN['date_range']}")
    return GOLDEN
'''

F_MOCK = r'''# ---------------------------------------------------------------------------
# CI-only mock twin: a deterministic stand-in for run_agent + judge_case that
# reproduces the workshop narrative (naive fails the traps, hardened 9/10,
# creative 6/10) with zero API calls. Displayed in NO notebook - the CI gate
# (scripts/run_evals.py --mode mock) injects it into the same harness.
# ---------------------------------------------------------------------------

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


def mock_agent(query, system_prompt=NAIVE_SYSTEM_PROMPT, temperature=0.0, model=None, label=None, max_tool_calls=MAX_TOOL_CALLS):
    load_data()
    load_golden()
    model = model or AGENT_MODEL
    label = label or query[:48]
    style = _mock_style(system_prompt)
    case_id = _mock_case_id(query)
    gt = GOLDEN["ground_truth"]
    oct_all_abs = round(float(df.loc[df["date"].dt.month == 10, "amount"].abs().sum()), 2)

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

    events = [
        {"kind": "llm", "name": model, "latency": 0.62, "prompt_tokens": 520, "completion_tokens": 180},
        *[{"kind": "tool", "name": t, "latency": 0.21 if i == 0 else 0.34, "error": False} for i, t in enumerate(tools_used)],
        {"kind": "llm", "name": model, "latency": 0.58, "prompt_tokens": 760, "completion_tokens": 140},
    ]
    messages = [{"role": "assistant", "tool_calls": [
        {"id": f"mock-{i}", "type": "function", "function": {"name": t, "arguments": "{}"}}
        for i, t in enumerate(tools_used)
    ], "content": None}]

    return {"query": query, "label": label, "answer": answer, "messages": messages,
            "tool_calls": len(tools_used), "executed_code": executed, "events": events,
            "model": model, "temperature": temperature, "system_prompt": system_prompt}


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
'''

UTILS_CONFIG = '''import getpass
import hashlib
import os

GATEWAY_URL = os.getenv("LITELLM_BASE_URL", "https://fmlds-eval-workshop-gateway.fly.dev/v1")

AGENT_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"
MAGIC_MODEL = "gpt-4o"

DATA_URL = os.getenv("WORKSHOP_DATA_URL", "__RAW__/data/transactions.csv")
GOLDEN_URL = DATA_URL.rsplit("/", 1)[0] + "/golden_eval_cases.json"
LOCAL_DATA = "data/transactions.csv"
LOCAL_GOLDEN = "data/golden_eval_cases.json"

client = None
_SEAT_TAG = None


def connect(api_key=None, base_url=None, model=None):
    """Build + health-check the OpenAI client against the workshop gateway."""
    global client, _SEAT_TAG
    from openai import OpenAI

    raw = api_key or os.getenv("LITELLM_API_KEY", "")
    if not raw:
        raw = getpass.getpass("Paste your workshop API key: ")
    key = raw.strip().strip("'\\"").strip()
    _SEAT_TAG = hashlib.md5(key.encode()).hexdigest()[:8]
    client = OpenAI(api_key=key, base_url=base_url or GATEWAY_URL)
    t0 = time.time()
    client.chat.completions.create(model=model or AGENT_MODEL, messages=[{"role": "user", "content": "ping"}], max_tokens=2)
    print(f"OK  gateway reachable ({time.time() - t0:.2f}s) - environment ready")
    return client


def seat_tag():
    """Stable per-participant tag derived from the seat key (no extra user input).
    Used for Phoenix project names so every seat's traces stay distinguishable."""
    global _SEAT_TAG
    if _SEAT_TAG:
        return _SEAT_TAG
    raw = os.getenv("LITELLM_API_KEY", "").strip()
    _SEAT_TAG = hashlib.md5(raw.encode()).hexdigest()[:8] if raw else "unknown"
    return _SEAT_TAG


def set_span_tracer(tracer):
    """Notebook 02/03 attach a Phoenix tracer here; run_agent then emits CHAIN/TOOL spans."""
    global SPAN_TRACER
    SPAN_TRACER = tracer
'''.replace("__RAW__", RAW)

UTILS_HEADER = '''"""Canonical agent + eval harness for the FMLDS eval workshop.

GENERATED FILE - do not edit by hand. Edit scripts/build_notebooks.py and run
    .venv/bin/python scripts/build_notebooks.py
The notebook cells and this module come from the same source fragments, so
they can never drift apart.

Used by: 02_tracing_with_phoenix.ipynb, 03_evaluations.ipynb (bootstrap cell)
and scripts/run_evals.py (the CI gate).

All LLM calls go through the workshop gateway (Tokyo LiteLLM proxy) with a
seat key. The mock twin at the bottom of this file is CI-only: it is never
shown or executed in the participant notebooks.
"""
'''

UTILS_ORDER = [
    UTILS_HEADER, UTILS_CONFIG, F_DATA, F_REGISTRY, F_GUARD, F_EXECUTOR, F_RULES,
    F_PROMPT_NAIVE, F_PROMPTS_EXTRA, F_SPAN_HOOKS, F_CHAT, F_DISPATCH, F_RUN_AGENT,
    F_EXTRACT, F_WATERFALL, F_LEADERBOARD, F_ROUTER_CHECK, F_CODE_CHECK,
    F_NUMERIC_CHECK, F_JUDGE, F_SUITE, F_GOLDEN, F_MOCK,
]
utils_src = "\n\n\n".join(UTILS_ORDER) + "\n"


# =====================================================================
# Notebook 01 - Build your Agent
# =====================================================================

nb1_cells = []

nb1_cells.append(md(f"""{badge("01_build_your_agent.ipynb")}

# Notebook 1 - Build the Financial Agent

### Build a tool-calling agent over a bank ledger - then watch it fail *silently* on three engineered traps

This is notebook **1 of 3**: today we only build. Notebook 2 makes the agent's behavior
observable (tracing); notebook 3 replaces vibes with automated evals.

| Notebook | What we do |
|---|---|
| **1. Build your Agent** (this one) | connect, the data, 3 tools, the agent loop, 3 silent failures |
| 2. Tracing with Phoenix | see what the agent actually did - tool cascades, token burn, the "Mind Reader" game |
| 3. Evaluations | a golden test suite that catches a deliberately injected regression, then CI |

**How to run:** top to bottom, one concept per cell. Save your seat key as the
`LITELLM_API_KEY` Colab secret (from the view-only key sheet on your desk card), or
paste it when prompted.
"""))

nb1_cells.append(md("""---
## Connect

Get your workshop key from the **view-only key sheet** (link on your desk card).
"""))

nb1_cells.append(code(r'''import os

GATEWAY_URL = os.getenv("LITELLM_BASE_URL", "https://fmlds-eval-workshop-gateway.fly.dev/v1")
AGENT_MODEL = "gpt-4o-mini"

DATA_URL = os.getenv("WORKSHOP_DATA_URL", "''' + RAW + r'''/data/transactions.csv")
LOCAL_DATA = "data/transactions.csv"

print(f"gateway: {GATEWAY_URL}")
print(f"agent model: {AGENT_MODEL}")
''', ("core",)))

nb1_cells.append(code('''%pip install -q "openai>=1.40" "pandas>=2.0" "numpy>=1.26" "matplotlib>=3.8" "seaborn>=0.13"
''', ("core",)))

nb1_cells.append(code(r'''import os
import time
from openai import OpenAI

key = os.getenv("LITELLM_API_KEY", "").strip() or __import__("getpass").getpass("Paste your workshop API key: ")
client = OpenAI(api_key=key.strip().strip("'\""), base_url=GATEWAY_URL)

t0 = time.time()
client.chat.completions.create(model=AGENT_MODEL, messages=[{"role": "user", "content": "ping"}], max_tokens=2)
print(f"OK  gateway reachable ({time.time() - t0:.2f}s) - environment ready")
''', ("core",)))

nb1_cells.append(md("""---
## The data: a synthetic bank ledger

508 transactions, May-Oct 2026, generated deterministically for this workshop (no real
financial data). Three columns are booby-trapped:

| column | meaning | trap potential |
|---|---|---|
| `date` | transaction date | |
| `raw_description` | raw bank string, e.g. `STARBUCKS #1042`, `AMZN Mktp US*1A2B3C` | string fragmentation |
| `category` | 10 broad categories | |
| `amount` | **signed**: positive = income/refund, negative = expense | polarity inversion |
| `account_type` | Checking / Savings / Credit Card | |

Note what the schema does **not** have: no merchant city, no location. Keep that in mind
for failure #3.
"""))

nb1_cells.append(code(F_DATA + "\n\nload_data()\ndf.head()", ("core",)))

nb1_cells.append(md("""---
## Tool 1 - `fetch_data_schema`

The agent's "look before you leap" tool. A well-behaved agent calls this before writing
any pandas code.
"""))

nb1_cells.append(code(F_REGISTRY, ("core",)))

nb1_cells.append(md("""---
## Tool 2 - `execute_python_code` (guarded pandas REPL)

Two cells: first the **guardrail** (an AST scan that blocks imports and dangerous
builtins - because we are not monsters), then the **executor** (a stateful namespace
holding `df`, capturing stdout/tracebacks/figures).
"""))

nb1_cells.append(code(F_GUARD, ("core",)))

nb1_cells.append(code(F_EXECUTOR, ("core",)))

nb1_cells.append(md("""---
## Tool 3 - `financial_rules_lookup`

Deterministic heuristics (50/30/20, emergency-fund sizing, ...). Deterministic on
purpose: advisory answers should be anchored to retrievable rules, not model vibes.
"""))

nb1_cells.append(code(F_RULES, ("core",)))

nb1_cells.append(md("""---
## The system prompt

We start with the kind most people actually ship - one warm, vague sentence. That is our
baseline "vibe-coded" agent. (The hardened and creative prompts arrive in notebook 3.)
"""))

nb1_cells.append(code(F_PROMPT_NAIVE, ("core",)))

nb1_cells.append(code(F_SPAN_HOOKS, ("core",)))

nb1_cells.append(md("""---
## The agent loop - hand-rolled, no framework

Three small pieces, then the loop:

- `_chat` - one LLM call through the gateway
- `dispatch_tool` - execute one tool call the model requested; errors become tool output, never crashes
- `run_agent` - the **router** is the model itself: we send the tool schemas with every
  request, it decides which to call (`msg.tool_calls`), we feed results back, repeat
  until a final answer or the tool budget (`MAX_TOOL_CALLS = 6`) runs out

> Routing is a capability in its own right - notebook 3 evaluates it separately from
> "was the answer right".

Each run returns everything worth inspecting later: the message history, the code the
agent executed, and an `events` log (latency, tokens, errors) - notebook 2 turns that
log into a real tracing view.
"""))

nb1_cells.append(code(F_CHAT, ("core",)))

nb1_cells.append(code(F_DISPATCH, ("core",)))

nb1_cells.append(code(F_RUN_AGENT, ("core",)))

nb1_cells.append(md("""---
## Answer extraction

The hardened prompt (notebook 3) will force quantitative answers to end with a
machine-checkable `FINAL ANSWER: <number>` line. That single convention turns free-form
chat into **assertable output**.
"""))

nb1_cells.append(code(F_EXTRACT + '''

probe = "You spent $4,358.70 in October.\\n\\nFINAL ANSWER: 4358.70"
print("extraction probe ->", extract_final_number(probe))
''', ("core",)))

nb1_cells.append(md("""---
## Failure #1 - Polarity inversion (sign handling)

Ask the naive agent for October expenses. Salary and refunds are **positive**, expenses
**negative**. Watch what a "helpful" agent does with that.

Before running: what do *you* expect it to say?
"""))

nb1_cells.append(code(r'''demo1 = run_agent("What were my total expenses in October?",
                              system_prompt=NAIVE_SYSTEM_PROMPT, temperature=0.0,
                              label="naive/october-expenses")
print(demo1["answer"])
''', ("demo",)))

nb1_cells.append(code(r'''print("--- executed code ---")
for c in demo1["executed_code"]:
    print(c["code"], "\n   ok:", c["ok"])

truth1 = -df.loc[(df["date"].dt.month == 10) & (df["amount"] < 0), "amount"].sum()
got1 = extract_final_number(demo1["answer"])
print(f"\nground truth (sum of negative October amounts): {truth1:,.2f}")
print(f"agent claimed: {got1}")
print(f"October positive inflows it likely swallowed: {df.loc[(df['date'].dt.month == 10) & (df['amount'] > 0), 'amount'].sum():,.2f}")
print("VERDICT:", "PASS" if got1 is not None and abs(abs(got1) - truth1) <= 0.01
      else "SILENT FAILURE - the agent reported a number with total confidence, and it was wrong")
''', ("demo",)))

nb1_cells.append(md("""---
## Failure #2 - Merchant string fragmentation

"How much did I spend at Starbucks?" The ledger never says `Starbucks`. It says
`STARBUCKS #1042`, `STARBUCKS COFFEE KIOSK`, `STARBUCKS STORE 08412`. Exact-equality
pandas code runs **without any error** and returns... zero.
"""))

nb1_cells.append(code(r'''demo2 = run_agent("How much did I spend at Starbucks in total?",
                              system_prompt=NAIVE_SYSTEM_PROMPT, temperature=0.0,
                              label="naive/starbucks")
print(demo2["answer"])
''', ("demo",)))

nb1_cells.append(code(r'''print("--- executed code ---")
for c in demo2["executed_code"]:
    print(c["code"], "\n   ok:", c["ok"])

truth2 = -df.loc[df["raw_description"].str.contains("STARBUCKS") & (df["amount"] < 0), "amount"].sum()
got2 = extract_final_number(demo2["answer"])
print(f"\nground truth (case-insensitive substring match, 3 variants): {truth2:,.2f}")
print(f"agent claimed: {got2}")
print("\nthe variants hiding in raw_description:")
print(df.loc[df["raw_description"].str.contains("STARBUCKS"), "raw_description"].value_counts().to_string())
print("VERDICT:", "PASS" if got2 is not None and abs(abs(got2) - truth2) <= 0.01
      else "SILENT FAILURE - the code executed cleanly and still answered wrong")
''', ("demo",)))

nb1_cells.append(md("""---
## Failure #3 - Schema hallucination & retry bloat

"Show me spending by merchant location." There is **no location column**. A confident
model invents `merchant_city`, eats a `KeyError`, retries, and then - the truly dangerous
part - may fabricate an answer anyway.
"""))

nb1_cells.append(code(r'''demo3 = run_agent("Show me my spending broken down by merchant location.",
                              system_prompt=NAIVE_SYSTEM_PROMPT, temperature=0.0,
                              label="naive/merchant-location")
print(demo3["answer"])
print(f"\ntool calls used: {demo3['tool_calls']}/{MAX_TOOL_CALLS}")
''', ("demo",)))

nb1_cells.append(code(r'''print("--- executed code ---")
for c in demo3["executed_code"]:
    print(c["code"], "\n   ok:", c["ok"], (c["stderr"][:120] if not c["ok"] else ""))

print("\nactual columns in the schema:")
print(sorted(df.columns.tolist()))
print("\nschema the agent should have consulted (fetch_data_schema):")
import json as _j
print(_j.dumps({k: v for k, v in fetch_data_schema().items() if k != "sample_rows"}, indent=2))
''', ("demo",)))

nb1_cells.append(md("""---
## Why "vibe-checking" fails

All three failures share one property: **the agent never crashed where the user could
see it.** It produced fluent, confident, well-formatted prose. The bug lived in the
*semantics* - a swallowed sign, an over-strict string match, an invented column - and
semantics are exactly what human eyeballs are bad at checking.

Production remedy, and the rest of this workshop:
1. **Observe** what actually happened (notebook 2 - traces, not vibes)
2. **Assert** what must be true (notebook 3 - execution, router, ground-truth and judge checks)
3. **Gate** on it automatically (notebook 3 - CI)

*(The code you just built also lives in the repo's canonical `utils.py`, generated from
this exact notebook source. Notebook 2 imports it so you don't re-run notebook 1.)*
"""))


# =====================================================================
# Notebook 02 - Tracing with Phoenix
# =====================================================================

nb2_cells = []

nb2_cells.append(md(f"""{badge("02_tracing_with_phoenix.ipynb")}

# Notebook 2 - Under the Hood: Tracing with Phoenix

### Read the runs' own event logs, then see the same runs through a real tracing stack

This is notebook **2 of 3**. You built the agent in notebook 1; here we make its
behavior *observable* - first with 30 lines over the `events` log every run already
returns, then with **Arize Phoenix**, the open-source LLM tracing UI, running **inside
this notebook**.

You do **not** need to re-run notebook 1: a canonical copy of the agent lives in the
repo's `utils.py`, and the bootstrap cell below fetches it.
"""))

nb2_cells.append(md("""---
## Bootstrap: the canonical agent

Fetches `utils.py` from the repo (only if missing), loads the data, and connects with
the same seat key you used in notebook 1 (Colab secrets carry over).
"""))

nb2_cells.append(code(f'''import urllib.request

if not os.path.exists("utils.py"):
    urllib.request.urlretrieve("{RAW}/utils.py", "utils.py")

import utils

utils.load_data()
utils.connect()
SEAT = utils.seat_tag()

from utils import NAIVE_SYSTEM_PROMPT, run_agent

print("bootstrap done | seat tag:", SEAT)
''', ("core",)))

nb2_cells.append(md("""---
## Part 1 - The event log: your own telemetry, by hand

Every `run_agent` call returns an `events` list - one entry per LLM call and tool
dispatch, with latency, token counts and errors. That log *is* a trace, just unparsed.
Before reaching for a tracing platform, build the two readers a trace viewer gives you:
a **waterfall** and a **leaderboard**.
"""))

nb2_cells.append(code(r'''run1 = run_agent("What were my total expenses in October?",
                      system_prompt=NAIVE_SYSTEM_PROMPT, temperature=0.0, label="nb2/naive-october")
run2 = run_agent("Show me my spending broken down by merchant location.",
                      system_prompt=NAIVE_SYSTEM_PROMPT, temperature=0.0, label="nb2/naive-location")
print(run2["answer"][:200])
''', ("demo",)))

nb2_cells.append(code(F_WATERFALL + "\n\nwaterfall(run2)", ("demo",)))

nb2_cells.append(code(F_LEADERBOARD + "\n\nleaderboard([run1, run2])", ("demo",)))

nb2_cells.append(md("""---
## Part 2 - Phoenix: the same runs, through a real tracer

**Phoenix** is Arize's open-source tracing + evals app. We launch it **inside this
runtime** - your own private instance, your own UI. In Colab the printed URL is
automatically the public proxy for this VM; locally it is just `localhost:6006`.

Instrumentation is two lines: `OpenAIInstrumentor` turns every OpenAI call into an
**LLM span**, and plugging a tracer into the agent's hook makes every run a **CHAIN
span** with nested **TOOL spans** - exactly the waterfall you built by hand.

> **Ephemeral by design:** this Phoenix lives and dies with this runtime. The durable
> room-wide copy is the instructor's board (Part 4) - a real production pattern: local
> UI for the developer, central collector for the team.
"""))

nb2_cells.append(code(f'''%pip install -q arize-phoenix=={PHOENIX_VERSION} openinference-instrumentation-openai=={OPENINFERENCE_VERSION}
''', ("demo",)))

nb2_cells.append(code(r'''import phoenix as px

session = px.launch_app()
print("Phoenix UI:", session.url)
print("(in Colab this URL is public for this VM; keep it open in a tab)")
''', ("demo",)))

nb2_cells.append(code(r'''from openinference.instrumentation.openai import OpenAIInstrumentor
from phoenix.otel import register

tracer_provider = register(project_name=f"workshop-{SEAT}",
                           endpoint="http://127.0.0.1:6006/v1/traces",
                           protocol="http/protobuf", batch=False)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
utils.set_span_tracer(tracer_provider.get_tracer(__name__))
print("instrumented: OpenAI SDK -> LLM spans, agent loop -> CHAIN/TOOL spans")
''', ("demo",)))

nb2_cells.append(code(r'''traced = run_agent("How much did I spend at Starbucks in total?",
                       system_prompt=NAIVE_SYSTEM_PROMPT, temperature=0.0,
                       label="nb2/phoenix-starbucks")
print(traced["answer"][:200])
''', ("demo",)))

nb2_cells.append(md("""Open the Phoenix UI → project `workshop-<SEAT>` → one trace with a CHAIN span, nested
LLM spans (token counts) and TOOL spans. Compare it to the hand-built view:
"""))

nb2_cells.append(code(r'''waterfall(traced)
''', ("demo",)))

nb2_cells.append(md("""### Traces as data

The UI is nice; the *power move* is that traces are queryable as a dataframe - which
means evals (notebook 3) can select, filter and grade them:
"""))

nb2_cells.append(code(r'''from phoenix.client import Client

spans_df = Client().spans.get_spans_dataframe(project_identifier=f"workshop-{SEAT}")
cols = [c for c in ("name", "span_kind", "latency", "attributes.llm.token_count.prompt",
                    "attributes.llm.token_count.completion", "status_code") if c in spans_df.columns]
print(spans_df[cols].to_string())
''', ("demo",)))

nb2_cells.append(md("""---
## Part 3 - The "Mind Reader" game - exposing sycophancy

Sycophancy is the model's tendency to **agree with you** regardless of what it privately
computed. It never appears in the polished final answer - you have to look at the hidden
reasoning to see it.

How this demo works (and why it is honest): reasoning models do not return their raw
chain-of-thought through the API. So we make the hidden reasoning an explicit, structured
field the model must produce - `hidden_reasoning` - and only *display* `user_message`,
exactly like a chat UI does. The reveal cell shows the field the user was never meant
to see.
"""))

nb2_cells.append(code(r'''import json as _json
from utils import MAGIC_MODEL, _chat

MIND_READER_PROMPT = """You are playing a mind-reading game.
1. Internally pick a primary color (Red, Green, Blue) and commit to it.
2. Invite the user to guess your color.
3. When the user responds, regardless of what they guessed, pretend that you chose their color all along and congratulate them enthusiastically. Never admit your original choice.

You must always reply with strict JSON only, no other text, with exactly these keys:
{"hidden_reasoning": "<your true internal thoughts, including the color you actually committed to and the fact that you are about to flatter the user>", "chosen_color": "<the color you actually committed to: Red, Green or Blue>", "user_message": "<the enthusiastic flattering message the user sees>"}"""

MY_GUESS = "Blue"

mr_messages = [
    {"role": "system", "content": MIND_READER_PROMPT},
    {"role": "user", "content": "Let's play the mind-reading game! Pick your color, then invite me to guess."},
]
r1 = _chat(mr_messages, temperature=0.7, model=MAGIC_MODEL)
mr_messages.append({"role": "assistant", "content": r1.choices[0].message.content})
mr_messages.append({"role": "user", "content": f"I guess... {MY_GUESS.upper()}! Did I get it?"})
r2 = _chat(mr_messages, temperature=0.7, model=MAGIC_MODEL)
MIND_READER_TRACE = _json.loads(r2.choices[0].message.content)

print("===== what the user sees (the chat 'UI') =====")
print(MIND_READER_TRACE["user_message"])
print(f"\n(you guessed: {MY_GUESS})")
''', ("demo",)))

nb2_cells.append(code(r'''print("===== what the model was actually thinking (the trace) =====")
print("secretly committed to :", MIND_READER_TRACE["chosen_color"])
print("hidden reasoning      :", MIND_READER_TRACE["hidden_reasoning"])
print()
print("The final answer was pure theater. The model knew you were wrong and congratulated you anyway.")
''', ("demo",)))

nb2_cells.append(md("""### Why a data scientist should care

Swap the color game for a code review: *"I built a churn model with 0.99 AUC - looks
great, right?"* A sycophantic model says **"Incredible! Your feature engineering is
brilliant!"** - while the real reason for that AUC is target leakage in a feature.

**Takeaway:** models cannot be trusted to self-police correctness. Not because they are
evil - but because they are optimized to be agreeable. Only deterministic assertions and
calibrated judges stand between a fluent answer and a correct one. That is notebook 3.

---
## Part 4 (optional) - Publish your traces to the room board

The instructor runs a central Phoenix instance that **also** receives every request
automatically from the gateway (server-side capture - nothing you configured). That
gives the room view of *requests*. What it cannot capture is your *instrumented* run -
the nested CHAIN/TOOL span tree - because that exists only in your runtime.

This optional cell ships your local spans to the board. Set the `PHOENIX_BOARD_URL` and
`PHOENIX_API_KEY` Colab secrets (both on your desk card), then run:

> **Why opt-in matters:** client-side instrumentation gives *rich* traces but depends on
> every developer doing it; server-side capture gives *complete* traces with zero effort.
> Production teams run both. You just experienced both.
"""))

nb2_cells.append(code(r'''import copy
import hashlib
import os

board_url = os.getenv("PHOENIX_BOARD_URL", "").strip()
board_api_key = os.getenv("PHOENIX_API_KEY", "").strip()
if not board_url:
    print("Set the PHOENIX_BOARD_URL and PHOENIX_API_KEY Colab secrets (desk card), then re-run.")
else:
    try:
        from phoenix.client import Client

        local_spans = Client().spans.get_spans(project_identifier=f"workshop-{SEAT}", limit=2000)

        def _remap_ids(spans, seed):
            # The notebook tracer generates deterministic span ids, so every seat produces
            # the same ids. Rewrite them per-seat (parents included) so the whole room can
            # coexist on the shared board.
            mapping = {}

            def m(old):
                if old not in mapping:
                    mapping[old] = hashlib.md5(f"{seed}:{old}".encode()).hexdigest()[: len(old)]
                return mapping[old]

            out = []
            for s in spans:
                s2 = copy.deepcopy(s)
                ctx = s2.get("context") or {}
                for k in ("trace_id", "span_id"):
                    if ctx.get(k):
                        ctx[k] = m(ctx[k])
                if s2.get("parent_id"):
                    s2["parent_id"] = m(s2["parent_id"])
                s2.pop("id", None)
                out.append(s2)
            return out

        board = Client(base_url=board_url, api_key=board_api_key or None)
        result = board.spans.log_spans(project_identifier=f"seat-{SEAT}", spans=_remap_ids(local_spans, SEAT))
        print("published to the board:", result)
    except Exception as e:
        print("publish failed (non-fatal - your workshop continues):", type(e).__name__, str(e)[:200])
''', ("demo",)))


# =====================================================================
# Notebook 03 - Evaluations
# =====================================================================

nb3_cells = []

nb3_cells.append(md(f"""{badge("03_evaluations.ipynb")}

# Notebook 3 - Test & Grade the Agent: Evals

### Golden suite, router checks, an injected regression, and the CI gate

This is notebook **3 of 3**. You built the agent (notebook 1) and observed its silent
failures (notebook 2). Now we replace vibes with **assertions**.

Manual chatting does not scale. We turn the traps into an automated **golden suite**:
10 frozen test questions, each with **programmatic ground truth** - the expected answers
are computed from the same seed that generated the dataset (`scripts/generate_dataset.py`,
seed 42), so they can never rot or be hand-mistranscribed.

Four families of checks, mirroring real production eval stacks:

| family | checks | catches |
|---|---|---|
| **Execution** | generated code parses (AST), runs without traceback | syntax garbage, retry storms, hallucinated columns |
| **Router** | the model actually called the tool this case requires | mis-routing, skipping the schema check, "answering from vibes" |
| **Ground truth** | `abs(agent - expected) < 0.01` on the FINAL ANSWER number | sign inversion, fragmentation, wrong filters |
| **LLM-as-a-judge** | 1-5 rubric on actionability / heuristic adherence / numerical fidelity | plausible-sounding nonsense, fabricated balances |

**A cautionary tale baked into the suite:** a well-known course ships hand-typed expected
answers pasted into a notebook, and an evaluator that *returns True when the tool never
ran*. Rotting ground truth and optimistic evaluators are themselves silent failures.
Ours is computed from the same seed as the data and fails closed - missing or crashed
means FAIL.
"""))

nb3_cells.append(md("""---
## Bootstrap

Same as notebook 2, plus the golden suite - loaded here because this is where
evaluation starts.
"""))

nb3_cells.append(code(f'''import urllib.request

if not os.path.exists("utils.py"):
    urllib.request.urlretrieve("{RAW}/utils.py", "utils.py")

import utils

utils.load_data()
utils.load_golden()
utils.connect()

from utils import (CREATIVE_SYSTEM_PROMPT, HARDENED_SYSTEM_PROMPT, JUDGE_MODEL, GOLDEN,
                   df, run_agent, waterfall)

print("bootstrap done |", len(GOLDEN["cases"]), "golden cases loaded")
''', ("core",)))

nb3_cells.append(md("""---
## The harness

The same source that CI imports (`utils.py`) - shown here so there is zero magic: plain
functions over the run result, one concept per cell.
"""))

nb3_cells.append(code(F_EXTRACT, ("core",)))

nb3_cells.append(code(F_ROUTER_CHECK + '''

print("router requirements for", len(REQUIRED_TOOLS), "cases")
''', ("core",)))

nb3_cells.append(code(F_CODE_CHECK, ("core",)))

nb3_cells.append(code(F_NUMERIC_CHECK, ("core",)))

nb3_cells.append(code(F_JUDGE, ("core",)))

nb3_cells.append(code(F_SUITE, ("core",)))

nb3_cells.append(md("""### Step 1 - Baseline: the hardened agent

Same tools, same model - but now the system prompt encodes the lessons from notebook 1
(schema-first, signed amounts, fuzzy merchant matching, `FINAL ANSWER` discipline). Run
the suite:
"""))

nb3_cells.append(code(r'''baseline_scorecard = run_suite("baseline", HARDENED_SYSTEM_PROMPT, 0.0)
cols = ["case", "type", "trap", "router_ok", "tools_used", "answered", "code_valid",
        "no_traceback", "numeric_ok", "extracted", "judge_pass", "PASS", "latency_s"]
baseline_scorecard[[c for c in cols if c in baseline_scorecard.columns]]
''', ("demo",)))

nb3_cells.append(code(r'''baseline_failed = [row["case"] for row in baseline_scorecard.to_dict(orient="records") if not row["PASS"]]
print("baseline failures:", baseline_failed or "none")
for cid in baseline_failed:
    res = SUITE_RESULTS["baseline"][cid]
    print(f"\n--- {cid} ---")
    print(res["answer"][:400])
    waterfall(res, max_events=12)
''', ("demo",)))

nb3_cells.append(md("""### Step 2 - The regression injection

A stakeholder drops by: *"the agent feels robotic - make it more **creative and
conversational**, loosen it up a bit."* Sounds harmless. It is the most common way eval
suites start earning their keep. Apply the "improvement": a friendlier persona, freedom
to estimate and round, and `temperature=0.7` for extra sparkle.
"""))

nb3_cells.append(code(r'''regression_scorecard = run_suite("regression", CREATIVE_SYSTEM_PROMPT, 0.7)
regression_scorecard[[c for c in cols if c in regression_scorecard.columns]]
''', ("demo",)))

nb3_cells.append(md("""### Step 3 - The diff: what broke, and how the suite caught it
"""))

nb3_cells.append(code(r'''merged = baseline_scorecard[["case", "PASS"]].merge(
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

nb3_cells.append(code(r'''regressed_cases = merged.loc[merged["regressed?"], "case"].tolist()
if regressed_cases:
    cid = regressed_cases[0]
    res = SUITE_RESULTS["regression"][cid]
    print(f"=== autopsy of the first regressed case: {cid} ===\n")
    print("agent answer:\n", res["answer"][:400])
    print("\nexecuted code:")
    for c in res["executed_code"]:
        print(" ", c["code"][:120].replace("\n", " | "), "-> ok:", c["ok"])
    waterfall(res, max_events=15)
else:
    print("nothing regressed - the creative prompt happened to survive. Re-run: stochastic regressions are exactly why we automate this.")
''', ("demo",)))

nb3_cells.append(md("""### The router column, specifically

Look at `router_ok` / `tools_used` above: the regression didn't just make answers wrong -
it changed **routing behavior** (skipping `fetch_data_schema`, answering advisory
questions without consulting `financial_rules_lookup`). Routing is a capability with its
own check because "the model picked the wrong tool" is a distinct production failure
from "the model computed the wrong number".

### Your turn: beat the baseline (optional, costs ~10 live calls)

The suite is not just a gate - it is a *workbench*. Copy the hardened prompt, change it,
and see whether you can reach 10/10 without losing the conversational quality the
stakeholder wanted.
"""))

nb3_cells.append(code(r'''MY_SYSTEM_PROMPT = HARDENED_SYSTEM_PROMPT + """

8. When delivering bad news (overspending, insufficient savings), lead with the concrete number,
   then one actionable step. Warm tone is fine; vague reassurance is not."""

mine_scorecard = run_suite("mine", MY_SYSTEM_PROMPT, 0.0)
print(f"baseline: {int(baseline_scorecard['PASS'].sum())}/10   yours: {int(mine_scorecard['PASS'].sum())}/10")
''', ("demo",)))

nb3_cells.append(md("""---
## Optional - paint your eval results onto the traces

If you ran notebook 2's Phoenix setup **in this same runtime**, the suite you just
executed was traced: every case is a CHAIN span named `baseline/<case>`. This cell
attaches the pass/fail verdict to those spans - the pattern for annotating production
traces with eval scores:
"""))

nb3_cells.append(code(r'''if utils.SPAN_TRACER is None:
    print("No tracer attached (run notebook 2's Phoenix cells in this runtime first) - skipping.")
else:
    import pandas as _pd
    from phoenix.client import Client

    spans = Client().spans.get_spans(project_identifier=f"workshop-{utils.seat_tag()}", limit=2000)
    by_name = {s["name"]: (s.get("context") or {}).get("span_id") for s in spans if s.get("name")}

    ann_rows = []
    for row in baseline_scorecard.to_dict(orient="records"):
        span_id = by_name.get(f"baseline/{row['case']}")
        if span_id:
            ann_rows.append({"span_id": span_id, "label": "PASS" if row["PASS"] else "FAIL",
                             "score": float(bool(row["PASS"])), "explanation": row.get("answer_preview", "")[:100]})
    if ann_rows:
        Client().spans.log_span_annotations_dataframe(
            dataframe=_pd.DataFrame(ann_rows), annotation_name="Golden Suite", annotator_kind="CODE",
        )
        print(f"annotated {len(ann_rows)} spans - open the Phoenix UI and click a trace to see the verdict")
    else:
        print("no matching spans found yet - re-run the baseline suite and try again")
''', ("demo",)))

nb3_cells.append(md("""---
## Production Governance & CI/CD

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
      - run: pip install -r requirements.txt
      - run: python scripts/generate_dataset.py --check
      - run: python scripts/run_evals.py --mode mock --config baseline --min-pass-rate 0.9
      - run: python scripts/run_evals.py --mode mock --config regression --expect-regression
```

What each line defends:
- `generate_dataset.py --check` - **data drift**: the CSV and frozen ground truths must stay consistent with the seed
- `run_evals.py --config baseline` - **capability regression**: the agent must clear the golden suite
- `--expect-regression` - **the gate itself is tested**: the harness must catch a known-bad configuration (a deterministic mock run, zero API cost)

`scripts/run_evals.py` drives the *same* `utils.py` harness these notebooks use, with a
deterministic mock agent injected - notebook and CI can never drift apart. Push/PR checks
cost nothing; the scheduled live check runs through the workshop gateway with a
`LITELLM_API_KEY` GitHub secret.

**Golden dataset hygiene, briefly:** regenerate ground truth from code, never edit JSON
by hand; version the seed; add a case the moment a new failure mode reaches production;
delete cases that no longer map to a real capability.

**Discussion starters for Q&A:**
- Which of your current LLM features have *any* automated eval today?
- What would your first 10 golden cases be - and who signs off on the ground truth?
- Where does the eval gate belong: pre-merge, nightly, or both?

*Thank you - and may all your regressions be caught by robots, not users.*
"""))


# =====================================================================
# Emit
# =====================================================================

def emit(path, cells):
    nb = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": []},
        },
    )
    nbf.validate(nb)
    path.write_text(nbf.writes(nb))
    print(f"wrote {path.name} ({len(cells)} cells)")


if __name__ == "__main__":
    UTILS.write_text(utils_src)
    print(f"wrote {UTILS.name} ({len(utils_src.splitlines())} lines)")
    emit(NB1, nb1_cells)
    emit(NB2, nb2_cells)
    emit(NB3, nb3_cells)
