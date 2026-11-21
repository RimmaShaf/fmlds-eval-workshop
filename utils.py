"""Canonical agent + eval harness for the FMLDS eval workshop.

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



import getpass
import hashlib
import os

GATEWAY_URL = os.getenv("LITELLM_BASE_URL", "https://fmlds-eval-workshop-gateway.fly.dev/v1")

AGENT_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"
MAGIC_MODEL = "gpt-4o"

DATA_URL = os.getenv("WORKSHOP_DATA_URL", "https://raw.githubusercontent.com/RimmaShaf/fmlds-eval-workshop/main/data/transactions.csv")
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
    key = raw.strip().strip("'\"").strip()
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



# utils.py-only plumbing: notebooks load df inline (plain top-level assignment);
# imported-module contexts (notebooks 2-3 bootstrap, CI) call load_data() lazily.
import pandas as pd


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



TOOLS = {}
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



import ast

BLOCKED = {"os", "sys", "subprocess", "requests", "urllib", "pathlib", "importlib",
           "open", "exec", "eval", "compile", "__import__", "globals", "locals", "input"}


def _guard(code_str):
    """Parse the model's code without running it; reject blocked imports and builtins."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] in BLOCKED for a in node.names):
            return "Blocked: imports are not allowed inside the agent sandbox."
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in BLOCKED:
            return "Blocked: imports are not allowed inside the agent sandbox."
        if isinstance(node, ast.Name) and node.id in BLOCKED:
            return f"Blocked: use of '{node.id}' is not allowed inside the agent sandbox."
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "Blocked: dunder attribute access is not allowed inside the agent sandbox."
    return None



import io
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



FINANCIAL_RULES = [
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



NAIVE_SYSTEM_PROMPT = """You are a helpful data analysis assistant. Answer the user's questions about their personal finances using the provided tools."""

print("baseline prompt: one warm, vague sentence - the kind most people actually ship")



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



# ---------------------------------------------------------------------------
# Span hooks (utils.py-only): notebook 2 attaches a Phoenix tracer via
# set_span_tracer(), and the span-hook lines in run_agent start emitting
# CHAIN/TOOL spans. Notebook 1's teaching view omits those lines entirely.
# ---------------------------------------------------------------------------

SPAN_TRACER = None


def _span(name, kind):
    """Start a span if a tracer is attached; returns a handle (or None)."""
    if SPAN_TRACER is None:
        return None
    cm = SPAN_TRACER.start_as_current_span(name)
    span = cm.__enter__()
    if span is not None:
        span.set_attribute("openinference.span.kind", kind)
    return cm


def _end_span(handle):
    if handle is not None:
        handle.__exit__(None, None, None)



import time

MAX_TOKENS = 1500


def _chat(messages, temperature, model, tools=None):
    """One LLM call through the workshop gateway."""
    kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=MAX_TOKENS)
    if tools:
        kwargs["tools"] = tools
    return client.chat.completions.create(**kwargs)



import json

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



def run_agent(query, system_prompt=NAIVE_SYSTEM_PROMPT, temperature=0.0, model=None, label=None, max_tool_calls=MAX_TOOL_CALLS):
    """The agent loop: the model + tool schemas do the routing; we dispatch and
    feed results back until a final answer or the tool budget runs out.

    Every LLM call and tool dispatch is appended to `events` (latency, tokens,
    errors) - notebook 2 turns that log into a real tracing view.
    """
    model = model or AGENT_MODEL
    label = label or query[:48]
    events = []
    EXEC_LOG.clear()
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
    tool_calls_used = 0
    answer = ""
    root_span = _span(label, "CHAIN")  # span-hook
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
            tool_span = _span(tc.function.name, "TOOL")  # span-hook
            out = dispatch_tool(tc)
            _end_span(tool_span)  # span-hook
            events.append({"kind": "tool", "name": tc.function.name, "latency": round(time.time() - t0, 2),
                           "error": out.get("ok") is False or "error" in out})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)[:4000]})
        if tool_calls_used >= max_tool_calls:
            messages.append({"role": "user", "content": "Tool budget exhausted - give your best final answer now."})
    _end_span(root_span)  # span-hook
    return {"query": query, "label": label, "answer": answer, "messages": messages,
            "tool_calls": tool_calls_used, "executed_code": list(EXEC_LOG), "events": events,
            "model": model, "temperature": temperature, "system_prompt": system_prompt}



import re

FINAL_ANSWER_RE = re.compile(r"final\s+answer\s*:\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE)


def extract_final_number(text):
    m = FINAL_ANSWER_RE.search(text or "")
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None



def waterfall(result, max_events=50):
    """Print a run's event log as a waterfall: what happened, in order, at what cost."""
    events = result["events"]
    print(f"=== trace waterfall: {result['label']}  ({len(events)} events) ===")
    for e in events[:max_events]:
        detail = " | ".join(f"{k}={v}" for k, v in e.items() if k not in ("kind", "name", "latency"))
        print(f"  {e['kind']:<10} {e['name']:<30} {e['latency']:>7.2f}s  {detail[:110]}")



def leaderboard(results):
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



REQUIRED_TOOLS = {
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



import ast


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



def _merchant_token(merchant):
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



JUDGE_SYSTEM_PROMPT = """You are a strict, neutral evaluation judge grading an AI financial assistant. You compare the assistant's answer against provided ground truth. Respond with JSON only."""


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



import time

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



import json as _json
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



# ---------------------------------------------------------------------------
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

