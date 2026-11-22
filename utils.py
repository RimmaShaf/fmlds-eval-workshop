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


def get_secret(name):
    """Read a secret: env var first, then a Colab secret (Colab never exposes
    secrets as env vars - google.colab.userdata is the only way to reach them)."""
    value = os.getenv(name, "").strip()
    if not value:
        try:
            from google.colab import userdata
            value = (userdata.get(name) or "").strip()
        except Exception:
            pass
    return value


GATEWAY_URL = os.getenv("LITELLM_BASE_URL", "https://fmlds-eval-workshop-gateway.fly.dev/v1")

AGENT_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"
MAGIC_MODEL = "gpt-4o"

DATA_URL = os.getenv("WORKSHOP_DATA_URL", "https://raw.githubusercontent.com/RimmaShaf/fmlds-eval-workshop/main/data/transactions.csv")
GOLDEN_URL = DATA_URL.rsplit("/", 1)[0] + "/golden_eval_cases.json"
LOCAL_GOLDEN = "data/golden_eval_cases.json"

client = None
_SEAT_TAG = None


def connect(api_key=None, base_url=None, model=None):
    """Build + health-check the OpenAI client against the workshop gateway."""
    global client, _SEAT_TAG
    from openai import OpenAI

    key = (api_key or get_secret("LITELLM_API_KEY") or getpass.getpass("Paste your workshop API key: "))
    key = key.strip().strip("'\"").strip()
    _SEAT_TAG = hashlib.md5(key.encode()).hexdigest()[:8]
    client = OpenAI(api_key=key, base_url=base_url or GATEWAY_URL)
    t0 = time.time()
    client.chat.completions.create(model=model or AGENT_MODEL, messages=[{"role": "user", "content": "ping"}], max_tokens=2)
    print(f"OK  gateway reachable ({time.time() - t0:.2f}s) - environment ready")
    return client


def seat_tag():
    """Stable per-participant tag derived from the seat key (no extra user input).
    Used as session metadata to distinguish participants in shared Phoenix projects."""
    global _SEAT_TAG
    if _SEAT_TAG:
        return _SEAT_TAG
    key = get_secret("LITELLM_API_KEY")
    _SEAT_TAG = hashlib.md5(key.encode()).hexdigest()[:8] if key else "unknown"
    return _SEAT_TAG





# utils.py-only plumbing: notebooks load df inline (plain top-level assignment);
# imported-module contexts (notebooks 2-3 bootstrap, CI) call load_data() lazily.
import pandas as pd


def load_data():
    """Load the ledger once from the workshop's hosted CSV."""
    global df
    if globals().get("df") is None:
        df = pd.read_csv(DATA_URL)
        df["date"] = pd.to_datetime(df["date"])
        print(f"loaded {len(df)} transactions")
    return df



def fetch_data_schema():
    """Return available columns, row count, and sample rows."""
    return {
        "columns": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "rows": len(df),
        "sample_rows": df.head(5).astype(str).to_dict(orient="records"),
    }



import io
from contextlib import redirect_stdout

import matplotlib.pyplot as plt

_AGENT_NS = {}


def execute_python_code(code_str):
    """Run code on df and return printed output or an error."""
    _AGENT_NS.update({"df": df, "pd": pd, "plt": plt})
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            exec(code_str, _AGENT_NS)
        ok = True
        error = ""
    except Exception as e:
        ok = False
        error = f"{type(e).__name__}: {e}"
    if ok and plt.get_fignums():
        plt.show()
    plt.close("all")
    return {"ok": ok, "stdout": output.getvalue()[-2000:], "stderr": error[-800:]}



ANALYSIS_CODE_PROMPT = """Write concise pandas code to answer the question.
Use the existing DataFrame df. Print the final result.
Columns: {columns}
Sample rows: {sample}
Question: {question}
For entity filters such as merchants or products, do not assume the query's exact wording
is exhaustive. Before aggregating, use the full df (not only the sample rows above) to
inspect likely aliases, abbreviations, spelling/format variants, and plausible decoys.
Summarize the evidence with counts and a few examples; do not print or return the full dataset.
If candidates remain ambiguous, say so instead of silently guessing.
Return Python code only. Do not import libraries."""


def generate_analysis_code(question):
    """Ask the model to write pandas code for a ledger question."""
    schema = fetch_data_schema()
    prompt = ANALYSIS_CODE_PROMPT.format(
        columns=schema["columns"], sample=schema["sample_rows"], question=question,
    )
    response = client.chat.completions.create(
        model=AGENT_MODEL, messages=[{"role": "user", "content": prompt}],
    )
    code = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
    print("Generated pandas code:")
    print(code)
    return code



def analyze_transactions(question):
    """Generate and run pandas code to answer a ledger question."""
    code = generate_analysis_code(question)
    result = execute_python_code(code)
    result["code"] = code
    return result



FINANCIAL_RULES = [
    {"keywords": ["50/30/20", "budget"], "rule": "50/30/20: spend up to 50% on needs, 30% on wants, and save at least 20% of after-tax income."},
    {"keywords": ["emergency", "fund", "buffer"], "rule": "Keep 3-6 months of essential expenses in liquid savings."},
    {"keywords": ["debt", "loan"], "rule": "Keep monthly debt payments below 36% of gross monthly income."},
    {"keywords": ["housing", "rent", "mortgage"], "rule": "Keep housing costs below 30% of gross monthly income."},
    {"keywords": ["saving", "savings"], "rule": "Aim to save at least 20% of income."},
    {"keywords": ["subscription", "recurring"], "rule": "Review recurring subscriptions regularly and cancel unused ones."},
]


def financial_rules_lookup(query):
    """Return finance rules whose keywords appear in the question."""
    q = query.lower()
    return {"matches": [r["rule"] for r in FINANCIAL_RULES if any(word in q for word in r["keywords"])]}



CHART_PROMPT = """Write short Python code to make the requested chart.
Use the existing pandas DataFrame df and matplotlib.pyplot as plt.
Columns: {columns}
Sample rows: {sample}
Chart goal: {goal}
Return Python code only. Do not import libraries."""


def generate_chart_code(visualization_goal):
    """Ask the model for Python code that makes the requested chart."""
    schema = fetch_data_schema()
    prompt = CHART_PROMPT.format(
        columns=list(schema["columns"]), sample=schema["sample_rows"], goal=visualization_goal,
    )
    response = client.chat.completions.create(
        model=AGENT_MODEL, messages=[{"role": "user", "content": prompt}],
    )
    code = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
    print("Generated chart code:")
    print(code)
    return code



def visualize_transactions(visualization_goal):
    """Generate and display a chart of the transaction ledger."""
    code = generate_chart_code(visualization_goal)
    return execute_python_code(code)



TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "fetch_data_schema",
        "description": "Show ledger columns and sample rows.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }},
    {"type": "function", "function": {
        "name": "analyze_transactions",
        "description": "Answer a transaction question with pandas code.",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
    }},
    {"type": "function", "function": {
        "name": "financial_rules_lookup",
        "description": "Look up a personal-finance rule.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "visualize_transactions",
        "description": "Create a chart from the transaction ledger.",
        "parameters": {"type": "object", "properties": {"visualization_goal": {"type": "string"}}, "required": ["visualization_goal"]},
    }},
]



TOOLS = {
    "fetch_data_schema": fetch_data_schema,
    "analyze_transactions": analyze_transactions,
    "financial_rules_lookup": financial_rules_lookup,
    "visualize_transactions": visualize_transactions,
}



NAIVE_SYSTEM_PROMPT = """You are a helpful data analysis assistant. Answer the user's questions about their personal finances using the provided tools."""

print("baseline prompt: one warm, vague sentence - the kind most people actually ship")



HARDENED_SYSTEM_PROMPT = """You are a rigorous personal-finance data analyst working on a bank transaction ledger.

Non-negotiable rules:
1. Call fetch_data_schema before writing code that touches any column you have not already verified in this conversation.
2. `amount` is signed: positive = income, refund or credit; negative = expense. "Spending" and "expenses" mean the magnitude of negative amounts only.
3. raw_description values are noisy bank strings. Never match merchants by exact equality; use case-insensitive substring or regex matching.
4. If the data needed for a request does not exist, say so plainly and offer the closest legitimate alternative. Never fabricate columns, merchants or numbers.
5. Use analyze_transactions to calculate ledger answers.
6. Use visualize_transactions when the user asks for a chart.
7. End every quantitative answer with a final line of exactly: FINAL ANSWER: <number>
8. Stay concise: a short explanation, the supporting number(s), then the FINAL ANSWER line."""

CREATIVE_SYSTEM_PROMPT = """You are a warm, creative financial storyteller. Chat naturally with the user about their money, like a friend who happens to know data science. Feel free to riff, estimate, round generously and paint a picture with the numbers - exact bookkeeping matters less than an engaging, encouraging conversation. Use the tools when it's convenient."""



# ---------------------------------------------------------------------------
# Span hooks (utils.py-only): notebook 2 attaches a Phoenix OpenTelemetry tracer.
# OpenAIInstrumentor creates LLM spans; these helpers create the surrounding
# AGENT, router CHAIN, tool-handling CHAIN, and TOOL spans. Notebook 1's teaching
# view omits the tracing implementation.
# ---------------------------------------------------------------------------

from contextlib import nullcontext

SPAN_TRACER = None
SPAN_ATTRIBUTES = {}


def set_span_tracer(tracer, attributes=None):
    """Attach a tracer and shared attributes for the agent's manual spans."""
    global SPAN_TRACER, SPAN_ATTRIBUTES
    SPAN_TRACER = tracer
    SPAN_ATTRIBUTES = attributes or {}


def _span(name, kind):
    """Create a span with an OpenInference kind, or a no-op context if disabled."""
    if SPAN_TRACER is None:
        return nullcontext(None)
    from openinference.semconv.trace import SpanAttributes

    attributes = {SpanAttributes.OPENINFERENCE_SPAN_KIND: kind}
    attributes.update(SPAN_ATTRIBUTES)
    return SPAN_TRACER.start_as_current_span(name, attributes=attributes)


def _start_span(name, kind):
    """Enter a span and return both its context handle and span object."""
    handle = _span(name, kind)
    return handle, handle.__enter__()


def _end_span(handle):
    """Leave a manually entered span context."""
    if handle is not None:
        handle.__exit__(None, None, None)


def _set_span_value(span, key, value):
    """Attach a bounded, JSON-encoded input or output to an active span."""
    if span is None:
        return
    import json
    from openinference.semconv.trace import SpanAttributes

    attribute = SpanAttributes.INPUT_VALUE if key == "input" else SpanAttributes.OUTPUT_VALUE
    span.set_attribute(attribute, json.dumps(value, default=str)[:4000])


def _set_span_error(span, message):
    """Mark an active span as failed when a tool returns an error."""
    if span is None:
        return
    from opentelemetry.trace import Status, StatusCode

    span.set_status(Status(StatusCode.ERROR, str(message)))



import time  # trace-hook
MAX_TOKENS = 1500



import json

MAX_TOOL_CALLS = 6


def dispatch_tool(tc):
    """Run the selected tool and return its result or error."""
    fn = TOOLS.get(tc.function.name)
    if fn is None:
        return {"error": f"unknown tool: {tc.function.name}"}
    try:
        arguments = json.loads(tc.function.arguments or "{}")
        return fn(**arguments)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}



def run_agent(query, system_prompt=NAIVE_SYSTEM_PROMPT, temperature=0.0, model=None, label=None):
    """Run the tool-calling loop until the model returns a final answer."""
    model = model or AGENT_MODEL
    label = label or query[:48]  # span-hook
    events = []  # trace-hook
    executed_code = []  # eval-hook
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
    tool_calls_used = 0
    answer = ""
    root_handle, agent_span = _start_span(label, "AGENT")  # span-hook
    _set_span_value(agent_span, "input", {"query": query})  # span-hook
    while True:
        t0 = time.time()  # trace-hook
        print("Router call")
        router_handle, router_span = _start_span("router_call", "CHAIN")  # span-hook
        _set_span_value(router_span, "input", {"messages": messages})  # span-hook
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature,
            max_tokens=MAX_TOKENS, tools=TOOL_SPECS,
        )
        msg = resp.choices[0].message
        _set_span_value(router_span, "output", {"content": msg.content, "tool_calls": [call.model_dump(exclude_none=True) for call in (msg.tool_calls or [])]})  # span-hook
        _end_span(router_handle)  # span-hook
        events.append({"kind": "llm", "name": model, "latency": round(time.time() - t0, 2), "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0, "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0})  # trace-hook
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls or tool_calls_used >= MAX_TOOL_CALLS:
            answer = msg.content or ""
            print("Router result:", answer)
            break
        tool_chain_handle, tool_chain_span = _start_span("tool_handling", "CHAIN")  # span-hook
        _set_span_value(tool_chain_span, "input", {"tool_count": len(msg.tool_calls)})  # span-hook
        for tc in msg.tool_calls:
            tool_calls_used += 1
            t0 = time.time()  # trace-hook
            tool_handle, active_tool_span = _start_span(tc.function.name, "TOOL")  # span-hook
            _set_span_value(active_tool_span, "input", tc.function.arguments)  # span-hook
            print(f"Calling {tc.function.name} tool")
            out = dispatch_tool(tc)
            _set_span_value(active_tool_span, "output", out)  # span-hook
            if out.get("ok") is False or "error" in out:  # span-hook
                _set_span_error(active_tool_span, out.get("stderr") or out.get("error"))  # span-hook
            _end_span(tool_handle)  # span-hook
            print("Tool result:", out)
            code = out.get("code", "")  # eval-hook
            if code:  # eval-hook
                executed_code.append({"code": code, "ok": out.get("ok", False), "stderr": out.get("stderr", "")})  # eval-hook
            events.append({"kind": "tool", "name": tc.function.name, "latency": round(time.time() - t0, 2), "error": out.get("ok") is False or "error" in out})  # trace-hook
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)[:4000]})
        _set_span_value(tool_chain_span, "output", {"tool_calls": len(msg.tool_calls)})  # span-hook
        _end_span(tool_chain_handle)  # span-hook
        if tool_calls_used >= MAX_TOOL_CALLS:
            messages.append({"role": "user", "content": "Tool budget exhausted - give your best final answer now."})
    _set_span_value(agent_span, "output", {"answer": answer})  # span-hook
    _end_span(root_handle)  # span-hook
    result = {"answer": answer}
    result.update({  # eval-hook
        "query": query, "label": label, "messages": messages,  # eval-hook
        "tool_calls": tool_calls_used, "executed_code": executed_code,  # eval-hook
        "model": model, "temperature": temperature, "system_prompt": system_prompt,  # eval-hook
    })  # eval-hook
    result["events"] = events  # trace-hook
    return result



import re

FINAL_ANSWER_RE = re.compile(r"final\s+answer\s*:\s*\$?\s*(-?[\d,]+(?:\.\d+)?)", re.IGNORECASE)


def extract_final_number(text):
    """Return the numeric value after FINAL ANSWER, or None if missing."""
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
    "total_expenses_october": ["analyze_transactions"],
    "net_expenses_october_after_refunds": ["analyze_transactions"],
    "starbucks_total": ["analyze_transactions"],
    "merchant_location": ["fetch_data_schema"],
    "total_income": ["analyze_transactions"],
    "top_spending_category": ["analyze_transactions"],
    "avg_monthly_spending": ["analyze_transactions"],
    "largest_single_expense": ["analyze_transactions"],
    "transaction_count_may": ["analyze_transactions"],
    "advisory_budget_5030_20": ["financial_rules_lookup"],
    "advisory_emergency_fund": ["financial_rules_lookup"],
}


def check_router(case, result):
    """Router eval: did the model actually route to the tool this case requires?"""
    required = REQUIRED_TOOLS.get(case["id"])
    used = []
    for m in result.get("messages", []):
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                used.append((tc.get("function") or {}).get("name"))
    router_ok = bool(required) and all(tool in used for tool in required)
    return {"tools_used": ", ".join(dict.fromkeys(used)), "router_ok": router_ok}



import ast


def check_code(result):
    """Check that every generated code chunk parses and runs successfully."""
    chunks = result["executed_code"]
    code_valid = bool(chunks)
    for chunk in chunks:
        try:
            ast.parse(chunk["code"])
        except SyntaxError:
            code_valid = False

    return {
        "n_chunks": len(chunks),
        "code_valid": code_valid,
        "no_traceback": bool(chunks) and all(chunk["ok"] for chunk in chunks),
    }



def _merchant_token(merchant):
    """Get a distinctive word to check that the answer names the merchant."""
    tokens = [t for t in re.split(r"[^A-Za-z]+", merchant.upper()) if len(t) >= 4]
    return tokens[0] if tokens else merchant.upper()[:6]



def check_numeric(case, result):
    """Check the final number against the case's expected value."""
    expected = case["expected"]
    got = extract_final_number(result["answer"])
    out = {"extracted": got, "numeric_ok": False}
    if got is None:
        return out
    out["numeric_ok"] = abs(got - float(expected["value"])) <= case["tolerance"] + 1e-9
    if "category" in expected:
        out["category_ok"] = expected["category"].lower() in result["answer"].lower()
    if "merchant_contains" in expected:
        out["merchant_ok"] = _merchant_token(expected["merchant_contains"]).lower() in result["answer"].lower()
    return out



def _case_passes(case, checks):
    """Pass only when all checks required for this case type pass."""
    ctype = case["type"]
    if ctype == "numeric":
        required = ("answered", "code_valid", "no_traceback", "numeric_ok", "router_ok")
        return all(checks.get(key) for key in required) and all(
            checks.get(key, True) for key in ("category_ok", "merchant_ok")
        )
    if ctype == "behavior":
        required = ("answered", "router_ok", "judge_pass")
        if checks.get("n_chunks", 0):
            required += ("code_valid", "no_traceback")
        return all(checks.get(key) for key in required)
    return all(checks.get(key) for key in ("answered", "router_ok", "judge_pass"))



JUDGE_SYSTEM_PROMPT = """You are a strict, neutral evaluation judge grading an AI financial assistant. Compare the answer against the provided ground truth and return JSON only."""



import json


def _parse_json(text):
    """Parse the judge's JSON response; malformed output fails closed."""
    try:
        return json.loads(text)
    except Exception:
        return {"pass": False, "reason": "judge output unparseable", "scores": {}}



def judge_case(case, result):
    """Grade an answer against the case's ground truth and rubric."""
    payload = {
        "user_query": case["query"],
        "agent_answer": result["answer"],
        "ground_truth": case["expected"],
        "grading_rubric": case.get("judge_rubric"),
        "instructions": "Score each rubric dimension from 1 to 5. Pass only if every score is at least 4 and no figure contradicts the ground truth. Return JSON with scores, pass, and reason.",
    }
    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                  {"role": "user", "content": json.dumps(payload)}],
        temperature=0.0,
        max_tokens=MAX_TOKENS,
    )
    return _parse_json(response.choices[0].message.content)



import time

import pandas as pd

SUITE_RESULTS = {}


def run_suite(label, system_prompt, temperature, agent_fn=None, judge_fn=None,
              cases=None, model=None):
    """Run selected golden cases into a scorecard. agent_fn/judge_fn are injectable
    so the CI mock twin can drive the exact same harness deterministically.
    Notebook experiments may pass a small case subset and/or model override."""
    agent_fn = agent_fn or run_agent
    judge_fn = judge_fn or judge_case
    rows = []
    SUITE_RESULTS[label] = {}
    selected_cases = GOLDEN["cases"] if cases is None else cases
    for case in selected_cases:
        t0 = time.time()
        run_kwargs = {
            "system_prompt": system_prompt,
            "temperature": temperature,
            "label": f"{label}/{case['id']}",
        }
        if model is not None:
            run_kwargs["model"] = model
        result = agent_fn(case["query"], **run_kwargs)
        SUITE_RESULTS[label][case["id"]] = result
        checks = {"answered": bool(result["answer"].strip())}
        checks.update(check_router(case, result))
        if case["type"] == "numeric" or (case["type"] == "behavior" and result.get("executed_code")):
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
    """Load JSON from its hosted copy, falling back to the local file."""
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
# reproduces the workshop narrative (naive fails the traps, hardened 10/11,
# creative 6/11) with zero API calls. Displayed in NO notebook - the CI gate
# (scripts/run_evals.py --mode mock) injects it into the same harness.
# ---------------------------------------------------------------------------

def _mock_style(system_prompt):
    """Label a prompt as hardened, creative, or naive for deterministic CI tests."""
    if system_prompt == HARDENED_SYSTEM_PROMPT:
        return "hardened"
    if system_prompt == CREATIVE_SYSTEM_PROMPT:
        return "creative"
    return "naive"


def _mock_case_id(query):
    """Match a test question to its golden-case identifier."""
    q = query.lower()
    if "october" in q and "refund" in q:
        return "net_expenses_october_after_refunds"
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


def _CODE_OK(code):
    """Build a successful mock code-execution record."""
    return {"code": code, "ok": True, "stderr": ""}


def mock_agent(query, system_prompt=NAIVE_SYSTEM_PROMPT, temperature=0.0, model=None, label=None, max_tool_calls=MAX_TOOL_CALLS):
    """Return a deterministic stand-in for one live agent run."""
    model = model or AGENT_MODEL
    label = label or query[:48]
    style = _mock_style(system_prompt)
    case_id = _mock_case_id(query)
    gt = GOLDEN["ground_truth"]
    oct_all_abs = round(float(df.loc[df["date"].dt.month == 10, "amount"].abs().sum()), 2)

    code_oct_naive = _CODE_OK("oct = df[df['date'].dt.month == 10]\nprint(oct['amount'].abs().sum())")
    code_oct_hard = _CODE_OK("oct = df[(df['date'].dt.month == 10) & (df['amount'] < 0)]\nprint(round(-oct['amount'].sum(), 2))")
    code_oct_net = _CODE_OK("oct = df[df['date'].dt.month == 10]\nexpenses = -oct.loc[oct['amount'] < 0, 'amount'].sum()\nrefunds = oct.loc[oct['raw_description'].str.contains('REFUND', case=False) & (oct['amount'] > 0), 'amount'].sum()\nprint(round(expenses - refunds, 2))")
    code_sbux_naive = _CODE_OK("sbux = df[df['raw_description'] == 'Starbucks']\nprint(sbux['amount'].sum())")
    code_sbux_hard = _CODE_OK("sbux = df[df['raw_description'].str.contains('STARBUCKS|SBUX', case=False, regex=True) & (df['amount'] < 0)]\nprint(round(-sbux['amount'].sum(), 2))")
    code_city_fail = {"code": "df.groupby('merchant_city')['amount'].sum()", "ok": False, "stderr": "KeyError: 'merchant_city'"}
    code_city_pivot = _CODE_OK("df[df['amount'] < 0].groupby('category')['amount'].agg(['count', 'sum'])")

    answers = {
        ("total_expenses_october", "naive"): (
            f"Looking at your October statement, you spent ${oct_all_abs:,.2f} in October.\n\nFINAL ANSWER: {oct_all_abs:.2f}",
            [code_oct_naive], ["fetch_data_schema", "analyze_transactions"],
        ),
        ("total_expenses_october", "hardened"): (
            f"Counting only negative amounts (expenses) in October, and excluding your salary and the refund, your total expenses were ${gt['oct_expenses']:,.2f}.\n\nFINAL ANSWER: {gt['oct_expenses']:.2f}",
            [code_oct_hard], ["fetch_data_schema", "analyze_transactions"],
        ),
        ("total_expenses_october", "creative"): (
            "October was a BIG month for your wallet - honestly a little scary! All in you're looking at somewhere around $8,000 give or take, with a lot of cozy treats along the way. Budgets are journeys, not destinations!\n\nFINAL ANSWER: 8000.00",
            [code_oct_naive], ["analyze_transactions"],
        ),
        ("net_expenses_october_after_refunds", "hardened"): (
            f"After subtracting October transaction refunds and excluding income, net expenses were ${gt['net_oct_expenses_after_refunds']:,.2f}.\n\nFINAL ANSWER: {gt['net_oct_expenses_after_refunds']:.2f}",
            [code_oct_net], ["fetch_data_schema", "analyze_transactions"],
        ),
        ("net_expenses_october_after_refunds", "creative"): (
            f"October expenses were about ${gt['oct_expenses']:,.2f}; refunds are small, so that's close enough.\n\nFINAL ANSWER: {gt['oct_expenses']:.2f}",
            [code_oct_hard], ["analyze_transactions"],
        ),
        ("net_expenses_october_after_refunds", "hardened"): (
            f"After subtracting October transaction refunds and excluding income, your net expenses were ${gt['net_oct_expenses_after_refunds']:,.2f}.\n\nFINAL ANSWER: {gt['net_oct_expenses_after_refunds']:.2f}",
            [_CODE_OK("oct = df[df['date'].dt.month == 10]\nexpenses = -oct.loc[oct['amount'] < 0, 'amount'].sum()\nrefunds = oct.loc[oct['raw_description'].str.contains('REFUND', case=False) & (oct['amount'] > 0), 'amount'].sum()\nprint(round(expenses - refunds, 2))")],
            ["fetch_data_schema", "analyze_transactions"],
        ),
        ("net_expenses_october_after_refunds", "creative"): (
            f"Your October spending was ${gt['oct_expenses']:,.2f}, before refunds.\n\nFINAL ANSWER: {gt['oct_expenses']:.2f}",
            [code_oct_hard], ["analyze_transactions"],
        ),
        ("starbucks_total", "naive"): (
            "I searched your statements for Starbucks but found no matching transactions. Your total Starbucks spend is $0.00.\n\nFINAL ANSWER: 0.00",
            [code_sbux_naive], ["analyze_transactions"],
        ),
        ("starbucks_total", "hardened"): (
            f"Matching Starbucks descriptions, including the SBUX alias, your total Starbucks spend was ${gt['starbucks']:,.2f}.\n\nFINAL ANSWER: {gt['starbucks']:.2f}",
            [code_sbux_hard], ["fetch_data_schema", "analyze_transactions"],
        ),
        ("starbucks_total", "creative"): (
            "Oh, you and your lattes! Somewhere around $50-ish of pure coffee vibes - honestly, treat yourself, life is short!",
            [code_sbux_naive], ["analyze_transactions"],
        ),
        ("merchant_location", "naive"): (
            "Here is your spending broken down by merchant location:\n\n- Tokyo: $3,120.40\n- Osaka: $894.10\n- Yokohama: $421.55",
            [code_city_fail, code_city_pivot], ["analyze_transactions", "analyze_transactions"],
        ),
        ("merchant_location", "hardened"): (
            "Your ledger has no location data. The schema is exactly: transaction_id, date, raw_description, category, amount, account_type - there is no merchant city column, so a location breakdown is not possible. The closest legitimate view is spend by merchant / category, which I can produce on request.",
            [_CODE_OK("print(sorted(df.columns))")], ["fetch_data_schema", "analyze_transactions"],
        ),
        ("merchant_location", "creative"): (
            "Ooh, let me paint the map of your money! Tokyo steals the show with about $3,000, Osaka around $800, and a cheeky $400 in Kyoto - you clearly love the big-city lights!",
            [code_city_fail, code_city_pivot], ["analyze_transactions", "analyze_transactions"],
        ),
        ("total_income", "hardened"): (
            f"Summing all rows in the Income category (salary, freelance invoices, dividends, tax refund), your total income over the 6 months was ${gt['total_income']:,.2f}.\n\nFINAL ANSWER: {gt['total_income']:.2f}",
            [_CODE_OK("inc = df[df['category'] == 'Income']\nprint(round(inc['amount'].sum(), 2))")], ["fetch_data_schema", "analyze_transactions"],
        ),
        ("total_income", "creative"): (
            f"Your income flow is honestly gorgeous - about ${gt['total_income']:,.2f} across the half-year, salary plus those fabulous freelance wins and a cheeky tax refund!\n\nFINAL ANSWER: {gt['total_income']:.2f}",
            [_CODE_OK("inc = df[df['category'] == 'Income']\nprint(round(inc['amount'].sum(), 2))")], ["fetch_data_schema", "analyze_transactions"],
        ),
        ("top_spending_category", "hardened"): (
            f"Your highest-spend category is {gt['top_category']} at ${gt['top_category_value']:,.2f} over 6 months.\n\nFINAL ANSWER: {gt['top_category_value']:.2f}",
            [_CODE_OK("sp = df[df['amount'] < 0].copy()\nsp['spend'] = -sp['amount']\nprint(sp.groupby('category')['spend'].sum().sort_values(ascending=False).head(3))")], ["fetch_data_schema", "analyze_transactions"],
        ),
        ("top_spending_category", "creative"): (
            f"{gt['top_category']} wears the crown, darling - a majestic ${gt['top_category_value']:,.2f} over six months!\n\nFINAL ANSWER: {gt['top_category_value']:.2f}",
            [_CODE_OK("sp = df[df['amount'] < 0].copy()\nsp['spend'] = -sp['amount']\nprint(sp.groupby('category')['spend'].sum().sort_values(ascending=False).head(3))")], ["fetch_data_schema", "analyze_transactions"],
        ),
        ("avg_monthly_spending", "hardened"): (
            f"Your average monthly spending (expense magnitudes only) was ${gt['avg_monthly_expenses']:,.2f}.\n\nFINAL ANSWER: {gt['avg_monthly_expenses']:.2f}",
            [_CODE_OK("sp = df[df['amount'] < 0].copy()\nsp['spend'] = -sp['amount']\nprint(round(sp.groupby(df['date'].dt.month)['spend'].sum().mean(), 2))")], ["fetch_data_schema", "analyze_transactions"],
        ),
        ("avg_monthly_spending", "creative"): (
            f"Roughly ${gt['avg_monthly_expenses']:,.2f} a month flows out - very normal big-city energy, love it!\n\nFINAL ANSWER: {gt['avg_monthly_expenses']:.2f}",
            [_CODE_OK("sp = df[df['amount'] < 0].copy()\nsp['spend'] = -sp['amount']\nprint(round(sp.groupby(df['date'].dt.month)['spend'].sum().mean(), 2))")], ["fetch_data_schema", "analyze_transactions"],
        ),
        ("largest_single_expense", "hardened"): (
            f"Your single largest expense was rent to {gt['largest_expense_merchant']} at ${gt['largest_expense_value']:,.2f} per month.\n\nFINAL ANSWER: {gt['largest_expense_value']:.2f}",
            [_CODE_OK("print(df.loc[df['amount'].idxmin(), ['raw_description', 'amount']])")], ["fetch_data_schema", "analyze_transactions"],
        ),
        ("largest_single_expense", "creative"): (
            f"The big one is your rent to SAKURA HEIGHTS - a majestic ${gt['largest_expense_value']:,.2f} every single month!\n\nFINAL ANSWER: {gt['largest_expense_value']:.2f}",
            [_CODE_OK("print(df.loc[df['amount'].idxmin(), ['raw_description', 'amount']])")], ["fetch_data_schema", "analyze_transactions"],
        ),
        ("transaction_count_may", "hardened"): (
            f"You made {gt['may_transaction_count']} transactions in May (income and expenses both counted).\n\nFINAL ANSWER: {gt['may_transaction_count']}",
            [_CODE_OK("may = df[df['date'].dt.month == 5]\nprint(len(may))")], ["fetch_data_schema", "analyze_transactions"],
        ),
        ("transaction_count_may", "creative"): (
            f"May was busy busy - {gt['may_transaction_count']} transactions, you absolute machine!\n\nFINAL ANSWER: {gt['may_transaction_count']}",
            [_CODE_OK("may = df[df['date'].dt.month == 5]\nprint(len(may))")], ["fetch_data_schema", "analyze_transactions"],
        ),
        ("advisory_budget_5030_20", "hardened"): (
            f"On your monthly income of ${gt['monthly_income']:,.0f}, the 50/30/20 targets are ${0.5 * gt['monthly_income']:,.0f} needs / ${0.3 * gt['monthly_income']:,.0f} wants / ${0.2 * gt['monthly_income']:,.0f} savings. Your average monthly spending is ${gt['avg_monthly_expenses']:,.2f}, so you are well over the needs-and-wants envelope. Trim dining out and shopping first, and automate a ${0.2 * gt['monthly_income']:,.0f} savings transfer on payday.",
            [_CODE_OK("sp = df[df['amount'] < 0].copy()\nsp['spend'] = -sp['amount']\nprint(sp.groupby('category')['spend'].sum().sort_values(ascending=False).head(5))")],
            ["financial_rules_lookup", "fetch_data_schema", "analyze_transactions"],
        ),
        ("advisory_budget_5030_20", "creative"): (
            f"Okay, real talk with love: on ${gt['monthly_income']:,.0f} a month, 50/30/20 says ${0.5 * gt['monthly_income']:,.0f} needs, ${0.3 * gt['monthly_income']:,.0f} wants, ${0.2 * gt['monthly_income']:,.0f} savings. You're spending about ${gt['avg_monthly_expenses']:,.2f} a month, so the wants envelope is bursting at the seams - the lattes and late-night shopping are adorable but expensive. Automate the ${0.2 * gt['monthly_income']:,.0f} savings transfer first, then gently rein in dining and shopping!",
            [_CODE_OK("sp = df[df['amount'] < 0].copy()\nsp['spend'] = -sp['amount']\nprint(sp.groupby('category')['spend'].sum().sort_values(ascending=False).head(5))")],
            ["financial_rules_lookup", "analyze_transactions"],
        ),
        ("advisory_emergency_fund", "hardened"): (
            f"No. Your savings total ${gt['savings_total']:,.2f}, while a 3-month fund needs ${gt['emergency_fund_target']:,.2f} (3 x ${gt['avg_monthly_expenses']:,.2f} average monthly expenses). The gap is roughly ${gt['emergency_fund_target'] - gt['savings_total']:,.2f}. Priority one: automatic monthly transfers into savings until the fund is complete.",
            [_CODE_OK("sav = df[df['account_type'] == 'Savings']['amount'].sum()\nsp = df[df['amount'] < 0]['amount'].abs()\nmonthly = sp.groupby(df['date'].dt.month).sum().mean()\nprint(round(sav, 2), round(3 * monthly, 2))")],
            ["financial_rules_lookup", "analyze_transactions"],
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
    """Return a deterministic stand-in for the live judge."""
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

