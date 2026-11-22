"""Assemble utils.py + the three workshop notebooks.

Regenerate after edits: .venv/bin/python scripts/build_notebooks.py

The workshop is three notebooks (build -> trace -> evaluate):

  01_build_your_agent.ipynb     connect, data, tools, the router loop, the three traps
  02_tracing_with_phoenix.ipynb OpenTelemetry spans -> local Phoenix -> trace inspection
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
    colab = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB}/{notebook})"
    codespaces = f'<a href="https://codespaces.new/{REPO}?quickstart=1"><img src="https://github.com/codespaces/badge.svg" alt="Open in GitHub Codespaces" width="100"></a>'
    return f"{colab} {codespaces}"


def display_src(src):
    """Hide trace and eval bookkeeping from notebook 1's teaching view."""
    lines = []
    for line in src.splitlines():
        if any(hook in line for hook in ("# span-hook", "# trace-hook", "# eval-hook")):
            continue
        lines.append(line.replace(", label=None", ""))
    return "\n".join(lines)


# =====================================================================
# Micro-fragments: cell-sized, shared between notebooks and utils.py.
# =====================================================================

F_DATA = r'''# utils.py-only plumbing: notebooks load df inline (plain top-level assignment);
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
'''

F_SCHEMA = r'''def fetch_data_schema():
    """Return available columns, row count, and sample rows."""
    return {
        "columns": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "rows": len(df),
        "sample_rows": df.head(5).astype(str).to_dict(orient="records"),
    }
'''

F_EXECUTOR = r'''import io
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
'''

F_ANALYSIS_CODE = r'''ANALYSIS_CODE_PROMPT = """Write concise pandas code to answer the question.
Use the existing DataFrame df. Print the final result.
Columns: {columns}
Sample rows: {sample}
Question: {question}
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
'''

F_ANALYZE = r'''def analyze_transactions(question):
    """Generate and run pandas code to answer a ledger question."""
    code = generate_analysis_code(question)
    result = execute_python_code(code)
    result["code"] = code
    return result
'''

F_RULES = r'''FINANCIAL_RULES = [
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
'''

F_CHART_CODE = r'''CHART_PROMPT = """Write short Python code to make the requested chart.
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
'''

F_VISUALIZATION = r'''def visualize_transactions(visualization_goal):
    """Generate and display a chart of the transaction ledger."""
    code = generate_chart_code(visualization_goal)
    return execute_python_code(code)
'''

F_TOOL_SPECS = r'''TOOL_SPECS = [
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
'''

F_TOOL_MAP = r'''TOOLS = {
    "fetch_data_schema": fetch_data_schema,
    "analyze_transactions": analyze_transactions,
    "financial_rules_lookup": financial_rules_lookup,
    "visualize_transactions": visualize_transactions,
}
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
5. Use analyze_transactions to calculate ledger answers.
6. Use visualize_transactions when the user asks for a chart.
7. End every quantitative answer with a final line of exactly: FINAL ANSWER: <number>
8. Stay concise: a short explanation, the supporting number(s), then the FINAL ANSWER line."""

CREATIVE_SYSTEM_PROMPT = """You are a warm, creative financial storyteller. Chat naturally with the user about their money, like a friend who happens to know data science. Feel free to riff, estimate, round generously and paint a picture with the numbers - exact bookkeeping matters less than an engaging, encouraging conversation. Use the tools when it's convenient."""
'''

F_SPAN_HOOKS = r'''# ---------------------------------------------------------------------------
# Span hooks (utils.py-only): notebook 2 attaches a Phoenix OpenTelemetry tracer.
# OpenAIInstrumentor creates LLM spans; these helpers create the surrounding
# AGENT, router CHAIN, tool-handling CHAIN, and TOOL spans. Notebook 1's teaching
# view omits the tracing implementation.
# ---------------------------------------------------------------------------

from contextlib import nullcontext

SPAN_TRACER = None


def set_span_tracer(tracer):
    """Attach the OpenTelemetry tracer used by the agent's manual spans."""
    global SPAN_TRACER
    SPAN_TRACER = tracer


def _span(name, kind):
    """Create a span with an OpenInference kind, or a no-op context if disabled."""
    if SPAN_TRACER is None:
        return nullcontext(None)
    from openinference.semconv.trace import SpanAttributes

    return SPAN_TRACER.start_as_current_span(
        name,
        attributes={SpanAttributes.OPENINFERENCE_SPAN_KIND: kind},
    )


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
'''

F_MODEL_CONFIG = r'''import time  # trace-hook
MAX_TOKENS = 1500
'''

F_DISPATCH = r'''import json

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
'''

F_RUN_AGENT = r'''def run_agent(query, system_prompt=NAIVE_SYSTEM_PROMPT, temperature=0.0, model=None, label=None):
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
'''

F_EXTRACT = r'''import re

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
'''

F_MERCHANT_TOKEN = r'''def _merchant_token(merchant):
    """Get a distinctive word to check that the answer names the merchant."""
    tokens = [t for t in re.split(r"[^A-Za-z]+", merchant.upper()) if len(t) >= 4]
    return tokens[0] if tokens else merchant.upper()[:6]
'''

F_NUMERIC_CHECK = r'''def check_numeric(case, result):
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
'''

F_CASE_PASSES = r'''def _case_passes(case, checks):
    """Pass only when all checks required for this case type pass."""
    ctype = case["type"]
    if ctype == "numeric":
        required = ("answered", "code_valid", "no_traceback", "numeric_ok", "router_ok")
        return all(checks.get(key) for key in required) and all(
            checks.get(key, True) for key in ("category_ok", "merchant_ok")
        )
    if ctype == "behavior":
        return all(checks.get(key) for key in ("answered", "code_valid", "no_traceback", "router_ok", "judge_pass"))
    return all(checks.get(key) for key in ("answered", "router_ok", "judge_pass"))
'''

F_JUDGE_PROMPT = r'''JUDGE_SYSTEM_PROMPT = """You are a strict, neutral evaluation judge grading an AI financial assistant. Compare the answer against the provided ground truth and return JSON only."""
'''

F_PARSE_JSON = r'''import json


def _parse_json(text):
    """Parse the judge's JSON response; malformed output fails closed."""
    try:
        return json.loads(text)
    except Exception:
        return {"pass": False, "reason": "judge output unparseable", "scores": {}}
'''

F_JUDGE_CASE = r'''def judge_case(case, result):
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
'''

F_MOCK = r'''# ---------------------------------------------------------------------------
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
'''

UTILS_CONFIG = '''import getpass
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

DATA_URL = os.getenv("WORKSHOP_DATA_URL", "__RAW__/data/transactions.csv")
GOLDEN_URL = DATA_URL.rsplit("/", 1)[0] + "/golden_eval_cases.json"
LOCAL_GOLDEN = "data/golden_eval_cases.json"

client = None
_SEAT_TAG = None


def connect(api_key=None, base_url=None, model=None):
    """Build + health-check the OpenAI client against the workshop gateway."""
    global client, _SEAT_TAG
    from openai import OpenAI

    key = (api_key or get_secret("LITELLM_API_KEY") or getpass.getpass("Paste your workshop API key: "))
    key = key.strip().strip("'\\"").strip()
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
    key = get_secret("LITELLM_API_KEY")
    _SEAT_TAG = hashlib.md5(key.encode()).hexdigest()[:8] if key else "unknown"
    return _SEAT_TAG


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
    UTILS_HEADER, UTILS_CONFIG, F_DATA, F_SCHEMA, F_EXECUTOR, F_ANALYSIS_CODE,
    F_ANALYZE, F_RULES, F_CHART_CODE,
    F_VISUALIZATION, F_TOOL_SPECS, F_TOOL_MAP, F_PROMPT_NAIVE, F_PROMPTS_EXTRA,
    F_SPAN_HOOKS, F_MODEL_CONFIG, F_DISPATCH, F_RUN_AGENT,
    F_EXTRACT, F_WATERFALL, F_LEADERBOARD, F_ROUTER_CHECK, F_CODE_CHECK,
    F_MERCHANT_TOKEN, F_NUMERIC_CHECK, F_CASE_PASSES, F_JUDGE_PROMPT,
    F_PARSE_JSON, F_JUDGE_CASE, F_SUITE, F_GOLDEN, F_MOCK,
]
utils_src = "\n\n\n".join(UTILS_ORDER) + "\n"


# =====================================================================
# Notebook 01 - Build your Agent
# =====================================================================

nb1_cells = []

nb1_cells.append(md(f"""{badge("01_build_your_agent.ipynb")}

# Build a Financial Agent

Build a tool-calling agent over a synthetic bank ledger, then explore three common failure modes.
Run cells from top to bottom. Enter your workshop key when prompted.
"""))

nb1_cells.append(md("""---
## Connect

Install, then point a client at the workshop gateway with your seat key.
"""))

nb1_cells.append(code('''%pip install -q "openai>=1.40" "pandas>=2.0" "matplotlib>=3.8"
''', ("core",)))

nb1_cells.append(code(r'''from getpass import getpass
from openai import OpenAI

GATEWAY_URL = "https://fmlds-eval-workshop-gateway.fly.dev/v1"
AGENT_MODEL = "gpt-4o-mini"

client = OpenAI(api_key=getpass("Workshop API key: ").strip(), base_url=GATEWAY_URL)
client.chat.completions.create(model=AGENT_MODEL, messages=[{"role": "user", "content": "ping"}], max_tokens=2)
print("Gateway connected")
''', ("core",)))

nb1_cells.append(md("""---
## The data: a synthetic bank ledger

2,500 synthetic transactions from May–October 2026. Amounts are signed: income and refunds
are positive; expenses are negative. There is no location column.

| column | meaning | trap |
|---|---|---|
| `date` | transaction date | |
| `raw_description` | raw bank string: `STARBUCKS #1042`, `AMZN Mktp US*1A2B3C` | string fragmentation |
| `category` | 10 broad categories | |
| `amount` | **signed**: positive = income/refund, negative = expense | polarity inversion |
| `account_type` | Checking / Savings / Credit Card | |
"""))

nb1_cells.append(code(r'''import pandas as pd

DATA_URL = "''' + RAW + r'''/data/transactions.csv"
df = pd.read_csv(DATA_URL)
df["date"] = pd.to_datetime(df["date"])

print(f"loaded {len(df)} transactions")
df.head()
''', ("core",)))

nb1_cells.append(md("""---
## Tool 1 - Inspect the ledger

This function reports the available columns and shows sample rows. It does not assume
that a particular column exists.
"""))

nb1_cells.append(code(F_SCHEMA, ("core",)))
nb1_cells.append(code("fetch_data_schema()", ("demo",)))

nb1_cells.append(md("""---
## Tool 2 - Analyze the transactions

This tool has two steps: generate pandas code, then run it. The router calls one
high-level tool; these cells show what happens inside it.
"""))

nb1_cells.append(md("""First, ask the model to write pandas code using the ledger schema.
"""))
nb1_cells.append(code(F_ANALYSIS_CODE, ("core",)))

nb1_cells.append(md("""Next, run the code. The ledger is available as `df`; printed output and errors go back to the model.
"""))
nb1_cells.append(code(F_EXECUTOR, ("core",)))
nb1_cells.append(code('execute_python_code("print(df[\'amount\'].sum())")', ("demo",)))

nb1_cells.append(md("""Finally, this function connects code generation and execution as one tool call.
"""))
nb1_cells.append(code(F_ANALYZE, ("core",)))
nb1_cells.append(code('analyze_transactions("What were my expenses in October?")', ("demo",)))

nb1_cells.append(md("""---
## Tool 3 - Look up a financial rule

This function returns a stored rule when its keywords match the question.
"""))

nb1_cells.append(code(F_RULES, ("core",)))
nb1_cells.append(code('financial_rules_lookup("How does the 50/30/20 budget work?")', ("demo",)))

nb1_cells.append(md("""---
## Tool 4 - Visualize the transactions

First ask the model for chart code using the ledger columns and sample rows. Then run
that code against `df` and display the chart.
"""))

nb1_cells.append(code(F_CHART_CODE, ("core",)))

nb1_cells.append(md("""The next function connects chart generation to the Python executor, so the router can
call visualization as one tool. The test below makes one live model call and displays
the chart.
"""))

nb1_cells.append(code(F_VISUALIZATION, ("core",)))
nb1_cells.append(code('visualize_transactions("Bar chart of spending by category")', ("demo",)))

nb1_cells.append(md("""---
## Describe the tools to the model

The schema tells the model each tool's name, purpose, and arguments.
"""))

nb1_cells.append(code(F_TOOL_SPECS, ("core",)))
nb1_cells.append(md("""The implementation map connects each schema name to its Python function.
"""))
nb1_cells.append(code(F_TOOL_MAP, ("core",)))

nb1_cells.append(md("""---
## The system prompt

We start with the kind most people actually ship - one warm, vague sentence. That is our
baseline "vibe-coded" agent. (The hardened and creative prompts arrive in notebook 3.)
"""))

nb1_cells.append(code(F_PROMPT_NAIVE, ("core",)))

nb1_cells.append(md("""---
## The agent loop - hand-rolled, no framework

The model chooses tools from their schemas. We run each call, return its result, and
repeat until the model answers or reaches the six-call limit.
"""))

nb1_cells.append(md("""Set a response limit before the agent makes its model call.
"""))
nb1_cells.append(code(F_MODEL_CONFIG, ("core",)))
nb1_cells.append(md("""`dispatch_tool` runs the Python function selected by the model and returns its result.
"""))
nb1_cells.append(code(F_DISPATCH, ("core",)))

nb1_cells.append(md("""`run_agent` asks the model what to do, runs requested tools, and returns the final answer.
"""))
nb1_cells.append(code(display_src(F_RUN_AGENT), ("core",)))

nb1_cells.append(md("""---
## Challenge 1 - Signed amounts

Ask the naive agent to use its Python tool, but do not tell it how signed amounts work.
Income and refunds are **positive**; expenses are **negative**.

Before running: what do *you* expect it to say?
"""))

nb1_cells.append(code(r'''demo1 = run_agent("Use analyze_transactions to calculate my total expenses in October and report the dollar amount.",
                              system_prompt=NAIVE_SYSTEM_PROMPT, temperature=0.0)
print(demo1["answer"])
''', ("demo",)))

nb1_cells.append(code(r'''truth1 = -df.loc[(df["date"].dt.month == 10) & (df["amount"] < 0), "amount"].sum()
classic_mistake = df.loc[df["date"].dt.month == 10, "amount"].sum()
print(f"\ntruth (sum of negative October amounts):   {truth1:,.2f}")
print(f"the signed total of all October amounts:    {classic_mistake:,.2f}")
print(f"income and refunds included in that total:  {df.loc[(df['date'].dt.month == 10) & (df['amount'] > 0), 'amount'].sum():,.2f}")
print("\nThe signed total is net cash flow, not expense magnitude.")
''', ("demo",)))

nb1_cells.append(md("""### Harder sign case: net expenses after refunds

Gross expenses count negative transactions. Net expenses also subtract refunds, but must
not subtract income. The ledger contains an October refund for this case.
"""))
nb1_cells.append(code(r'''demo1b = run_agent("Use analyze_transactions to calculate net expenses in October after transaction refunds, excluding income.",
                               system_prompt=NAIVE_SYSTEM_PROMPT, temperature=0.0)
print(demo1b["answer"])
''', ("demo",)))
nb1_cells.append(code(r'''october = df[df["date"].dt.month == 10]
gross_expenses = -october.loc[october["amount"] < 0, "amount"].sum()
refunds = october.loc[
    october["raw_description"].str.contains("REFUND", case=False) & (october["amount"] > 0), "amount"
].sum()
print(f"gross expenses: {gross_expenses:,.2f}")
print(f"refunds:        {refunds:,.2f}")
print(f"net expenses:   {gross_expenses - refunds:,.2f}")
''', ("demo",)))

nb1_cells.append(md("""---
## Challenge 2 - Merchant aliases

"How much did I spend at Starbucks?" Some descriptions contain `STARBUCKS`; others use
the alias `SBUX`. `TULLY'S COFFEE` is a decoy: coffee does not necessarily mean Starbucks.
"""))

nb1_cells.append(code(r'''demo2 = run_agent("Use analyze_transactions to find my total Starbucks spending and report the dollar amount, even if it is zero.",
                              system_prompt=NAIVE_SYSTEM_PROMPT, temperature=0.0)
print(demo2["answer"])
''', ("demo",)))

nb1_cells.append(code(r'''is_starbucks = df["raw_description"].str.contains(r"STARBUCKS|SBUX", case=False, regex=True)
truth2 = -df.loc[is_starbucks & (df["amount"] < 0), "amount"].sum()
print(f"\ntruth (STARBUCKS and SBUX descriptions): {truth2:,.2f}")
print("\nthe variants hiding in raw_description:")
print(df.loc[is_starbucks, "raw_description"].value_counts().to_string())
print("\nSimilar coffee-shop spend, not Starbucks:")
print(df.loc[df["raw_description"].str.contains("TULLY'S", case=False), "raw_description"].value_counts().to_string())
''', ("demo",)))

nb1_cells.append(md("""---
## Challenge 3 - Missing information

"Show me spending by merchant location." There is **no location column**. See how the
agent responds when asked for a breakdown the ledger may not support, then compare its
answer with the columns that are actually available.
"""))

nb1_cells.append(code(r'''demo3 = run_agent("Show me my spending broken down by merchant location.",
                              system_prompt=NAIVE_SYSTEM_PROMPT, temperature=0.0)
print(demo3["answer"])
''', ("demo",)))

nb1_cells.append(code(r'''
print("\nactual columns in the schema:")
print(sorted(df.columns.tolist()))
print("\nschema the agent should have consulted (fetch_data_schema):")
import json as _j
print(_j.dumps({k: v for k, v in fetch_data_schema().items() if k != "sample_rows"}, indent=2))
''', ("demo",)))

nb1_cells.append(md("""---
## Why "vibe-checking" fails

These challenges test different mismatches between the question and the data: sign
handling, merchant names, and unavailable fields. A fluent answer can still be wrong, so
compare it with the ledger.

The rest of the workshop: **observe** what happened (notebook 2), **assert** what must
be true (notebook 3), **gate** on it in CI (notebook 3).

*(The code you just built also lives in the repo's canonical `utils.py`, generated from
this notebook's source. Notebook 2 imports it so you don't re-run this notebook.)*
"""))


# =====================================================================
# Notebook 02 - Tracing with Phoenix
# =====================================================================

nb2_cells = []

nb2_cells.append(md(f"""{badge("02_tracing_with_phoenix.ipynb")}

# Lab 2: Trace Your Agent

### Instrument the agent with OpenTelemetry and inspect its traces in Phoenix

In notebook 1, you built the agent. Here, the OpenAI instrumentor captures LLM calls
automatically, while manual spans make the agent and its tool flow visible. You will
inspect the full trace—including tool inputs and outputs—in Phoenix.

The workshop's canonical agent remains in `utils.py`; this notebook attaches a tracer to
that implementation rather than creating a second agent that could drift.
"""))

nb2_cells.append(md("""---
## Bootstrap the workshop agent

Fetch `utils.py` only if needed, load the synthetic ledger, and connect through the same
workshop gateway and seat key as notebook 1.
"""))

nb2_cells.append(code(f"""import urllib.request
import os

if not os.path.exists("utils.py"):
    urllib.request.urlretrieve("{RAW}/utils.py", "utils.py")

import utils

utils.load_data()
utils.connect()
SEAT = utils.seat_tag()
from utils import NAIVE_SYSTEM_PROMPT, run_agent

print("bootstrap done | seat tag:", SEAT)
""", ("core",)))

nb2_cells.append(md("""---
## 1. Install the tracing libraries

The observability stack uses Phoenix's OpenTelemetry registration and OpenInference's
OpenAI instrumentor. The workshop agent uses its existing pandas-based tools.
"""))

nb2_cells.append(code(f"%pip install -q arize-phoenix=={PHOENIX_VERSION} openinference-instrumentation-openai=={OPENINFERENCE_VERSION}", ("core",)))

nb2_cells.append(md("""## 2. Start Phoenix and register an OpenTelemetry provider

For this workshop, Phoenix runs locally in the notebook runtime. `register` configures
its trace exporter; the endpoint is the collector URL, not the Phoenix UI URL.
"""))

nb2_cells.append(code(r"""import phoenix as px
from phoenix.otel import register

session = px.launch_app()
print("Phoenix UI:", session.url)
""", ("core",)))

nb2_cells.append(code(r"""tracer_provider = register(
    project_name=f"workshop-{SEAT}",
    endpoint="http://127.0.0.1:6006/v1/traces",
    protocol="http/protobuf",
    batch=False,
)
print("OpenTelemetry collector:", "http://127.0.0.1:6006/v1/traces")
""", ("core",)))

nb2_cells.append(md("""## 3. Instrument LLM calls and attach the agent tracer

`OpenAIInstrumentor` creates an LLM span for each OpenAI SDK call. The canonical agent
adds the surrounding spans: an **AGENT** root, a **CHAIN** for each router call, a nested
LLM span, then a tool-handling **CHAIN** containing one **TOOL** span per tool execution.
Inputs, outputs, and tool errors are recorded as span attributes.
"""))

nb2_cells.append(code(r"""from openinference.instrumentation.openai import OpenAIInstrumentor

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
utils.set_span_tracer(tracer_provider.get_tracer(__name__))
print("OpenAI calls instrumented; manual agent/chain/tool spans enabled")
""", ("core",)))

nb2_cells.append(md("""> **Trace data note:** spans contain prompts and tool inputs/outputs. This workshop uses
> synthetic transactions; do not send personal or otherwise sensitive data to a tracer.
"""))

nb2_cells.append(md("""---
## 4. Trace a request the ledger may not support

This request asks for a merchant-location breakdown, but the ledger has no location field.
Run the agent, then use the trace to see which steps it took and what the tool returned.
The exact model response can vary.
"""))

nb2_cells.append(code(r"""missing_location = run_agent(
    "Show me my spending broken down by merchant location.",
    system_prompt=NAIVE_SYSTEM_PROMPT,
    temperature=0.0,
    label="nb2/missing-location",
)
print(missing_location["answer"])
""", ("demo",)))

nb2_cells.append(md("""### Read the trace in Phoenix

Open the Phoenix UI and select project `workshop-<SEAT>`. Expand the trace and follow the
parent-child spans. Find the router's LLM call, the selected tool, and the tool's input
and output. A failed tool call should be visible in its TOOL span rather than disappearing
inside a final answer.
"""))

nb2_cells.append(code(r"""from phoenix.client import Client

phoenix_client = Client()
spans_df = phoenix_client.spans.get_spans_dataframe(project_identifier=f"workshop-{SEAT}")
trace_columns = [
    column for column in (
        "name", "span_kind", "parent_id", "status_code",
        "attributes.input.value", "attributes.output.value",
        "attributes.llm.token_count.prompt", "attributes.llm.token_count.completion",
    )
    if column in spans_df.columns
]
display(spans_df[trace_columns].tail(12))
""", ("demo",)))

nb2_cells.append(md("""---
## 5. Compare with a successful analysis

Run a second request that asks for net expenses after refunds. In Phoenix, compare its
agent → router/LLM → tool span tree with the missing-information example. Notice how the
same instrumentation captures both paths without changing the tracing code.
"""))

nb2_cells.append(code(r"""net_expenses = run_agent(
    "Use analyze_transactions to calculate net expenses in October after transaction refunds, excluding income.",
    system_prompt=NAIVE_SYSTEM_PROMPT,
    temperature=0.0,
    label="nb2/net-expenses-after-refunds",
)
print(net_expenses["answer"])
""", ("demo",)))

nb2_cells.append(code(r"""spans_df = phoenix_client.spans.get_spans_dataframe(project_identifier=f"workshop-{SEAT}")
display(spans_df[trace_columns].tail(20))
""", ("demo",)))

nb2_cells.append(md("""---
## Optional - Publish traces to the room board

This step is optional. It sends your local spans to the shared Phoenix board.
"""))

nb2_cells.append(md("""First, remap span IDs so traces from different participants do not collide.
"""))

nb2_cells.append(code(r'''import copy
import hashlib

def remap_span_ids(spans, seat):
    """Make span and parent IDs unique to this participant."""
    remapped = []
    ids = {}
    for span in spans:
        item = copy.deepcopy(span)
        context = item.get("context") or {}
        for key in ("trace_id", "span_id"):
            old = context.get(key)
            if old:
                context[key] = ids.setdefault(old, hashlib.md5(f"{seat}:{old}".encode()).hexdigest()[:len(old)])
        old_parent = item.get("parent_id")
        if old_parent:
            item["parent_id"] = ids.setdefault(old_parent, hashlib.md5(f"{seat}:{old_parent}".encode()).hexdigest()[:len(old_parent)])
        item.pop("id", None)
        remapped.append(item)
    return remapped
''', ("demo",)))

nb2_cells.append(md("""The board URL is fixed for this workshop. Keep its API key in a Colab secret or
environment variable; the publish cell reads it without exposing it in the notebook.
"""))

nb2_cells.append(code(r'''from phoenix.client import Client

board_url = "https://fmlds-eval-workshop-phoenix.fly.dev"
board_key = utils.get_secret("PHOENIX_API_KEY")
if not board_key:
    print("Add PHOENIX_API_KEY to your workshop secrets first.")
else:
    spans = Client().spans.get_spans(project_identifier=f"workshop-{SEAT}", limit=2000)
    board = Client(base_url=board_url, api_key=board_key or None)
    result = board.spans.log_spans(
        project_identifier=f"seat-{SEAT}", spans=remap_span_ids(spans, SEAT),
    )
    print("published:", result)
''', ("demo",)))


# =====================================================================
# Notebook 03 - Evaluations
# =====================================================================

nb3_cells = []

nb3_cells.append(md(f"""{badge("03_evaluations.ipynb")}

# Test and Grade the Agent

Turn the failures from notebooks 1–2 into an 11-question golden suite with ground truth
computed from the same seed as the dataset.

Four families of checks, mirroring real production eval stacks:

| family | checks | catches |
|---|---|---|
| **Execution** | generated code parses (AST), runs without traceback | syntax garbage, retry storms, hallucinated columns |
| **Router** | the model actually called the tool this case requires | mis-routing, skipping the schema check, "answering from vibes" |
| **Ground truth** | `abs(agent - expected) < 0.01` on the FINAL ANSWER number | sign inversion, fragmentation, wrong filters |
| **LLM-as-a-judge** | 1-5 rubric on actionability / heuristic adherence / numerical fidelity | plausible-sounding nonsense, fabricated balances |

The checks fail closed: missing output or a crashed tool means FAIL.
"""))

nb3_cells.append(md("""---
## Bootstrap

Same as notebook 2, plus the golden suite - loaded here because this is where
evaluation starts.
"""))

nb3_cells.append(code(f'''import urllib.request
import os

if not os.path.exists("utils.py"):
    urllib.request.urlretrieve("{RAW}/utils.py", "utils.py")

import utils

utils.load_data()
utils.load_golden()
utils.connect()

from utils import (CREATIVE_SYSTEM_PROMPT, HARDENED_SYSTEM_PROMPT, JUDGE_MODEL, GOLDEN,
                   MAX_TOKENS, client, df, run_agent, waterfall)

print("bootstrap done |", len(GOLDEN["cases"]), "golden cases loaded")
''', ("core",)))

nb3_cells.append(md("""---
## The harness

Each check reads a run result. Numeric answers use the `FINAL ANSWER: <number>` convention
so they can be compared with ground truth.
"""))

nb3_cells.append(code(F_EXTRACT, ("core",)))

nb3_cells.append(md("""### Check the router

This function checks whether the agent called the tool required for each case.
"""))
nb3_cells.append(code(F_ROUTER_CHECK, ("core",)))

nb3_cells.append(md("""### Check code execution

Each code call should parse and run successfully.
"""))
nb3_cells.append(code(F_CODE_CHECK, ("core",)))

nb3_cells.append(md("""### Check merchant names

Reduce a merchant name to a word we can look for in the answer.
"""))
nb3_cells.append(code(F_MERCHANT_TOKEN, ("core",)))

nb3_cells.append(md("""### Check numeric answers

Compare the extracted final number with the computed ground truth.
"""))
nb3_cells.append(code(F_NUMERIC_CHECK, ("core",)))

nb3_cells.append(md("""### Decide whether a case passes

Require every check for that case type; missing checks do not pass.
"""))
nb3_cells.append(code(F_CASE_PASSES, ("core",)))

nb3_cells.append(md("""### Prepare the judge prompt

The judge compares advisory answers with the same case ground truth.
"""))
nb3_cells.append(code(F_JUDGE_PROMPT, ("core",)))

nb3_cells.append(md("""Parse the judge response as JSON. If it is not valid JSON, the case fails.
"""))
nb3_cells.append(code(F_PARSE_JSON, ("core",)))

nb3_cells.append(md("""Send the answer, expected result, and rubric to the judge.
"""))
nb3_cells.append(code(F_JUDGE_CASE, ("core",)))

nb3_cells.append(md("""`run_suite` runs every golden case and combines its checks into one scorecard.
"""))
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
total_cases = len(GOLDEN["cases"])
print(f"baseline   : {b}/{total_cases}")
print(f"regression : {r}/{total_cases}")
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

### Your turn: beat the baseline (optional, costs ~11 live calls)

The suite is not just a gate - it is a *workbench*. Copy the hardened prompt, change it,
and see whether you can reach 11/11 without losing the conversational quality the
stakeholder wanted.
"""))

nb3_cells.append(code(r'''MY_SYSTEM_PROMPT = HARDENED_SYSTEM_PROMPT + """

8. When delivering bad news (overspending, insufficient savings), lead with the concrete number,
   then one actionable step. Warm tone is fine; vague reassurance is not."""

mine_scorecard = run_suite("mine", MY_SYSTEM_PROMPT, 0.0)
print(f"baseline: {int(baseline_scorecard['PASS'].sum())}/{len(GOLDEN['cases'])}   yours: {int(mine_scorecard['PASS'].sum())}/{len(GOLDEN['cases'])}")
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
