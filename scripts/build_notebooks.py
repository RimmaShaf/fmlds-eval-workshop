"""Assemble utils.py + the three workshop notebooks.

Regenerate after edits: .venv/bin/python scripts/build_notebooks.py

The workshop is three notebooks (build -> trace -> evaluate):

  01_build_your_agent.ipynb     connect, data, tools, the router loop, the three traps
  02_tracing_with_phoenix.ipynb OpenTelemetry spans -> shared Phoenix board -> inspection
  03_evaluations.ipynb          skill/router evals + prompt/model experiments + trace review

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
PHOENIX_BOARD_URL = "https://fmlds-eval-workshop-phoenix.fly.dev"
PHOENIX_PROJECT = "00-Part-2-tracing-set-up"
PHOENIX_EVAL_PROJECT = "01-Part-3-evaluations"
PHOENIX_OTEL_VERSION = "0.17.1"
PHOENIX_CLIENT_VERSION = "3.7.0"
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
    github = f"[View on GitHub](https://github.com/{REPO}/blob/main/{notebook})"
    return f"{colab} {codespaces} {github}"


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
    required = REQUIRED_TOOLS.get(case["id"])
    used = []
    for m in result.get("messages", []):
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                used.append((tc.get("function") or {}).get("name"))
    router_ok = bool(required) and all(tool in used for tool in required)
    return {"tools_used": ", ".join(dict.fromkeys(used)), "router_ok": router_ok}
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
        required = ("answered", "router_ok", "judge_pass")
        if checks.get("n_chunks", 0):
            required += ("code_valid", "no_traceback")
        return all(checks.get(key) for key in required)
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
    Used as session metadata to distinguish participants in shared Phoenix projects."""
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

Ask the naive agent to calculate October expenses, but do not tell it how signed amounts work.
Income and refunds are **positive**; expenses are **negative**.

Before running: what do *you* expect it to say?
"""))

nb1_cells.append(code(r'''demo1 = run_agent("What were my total expenses in October?",
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
nb1_cells.append(code(r'''demo1b = run_agent("What were my net expenses in October after refunds, excluding income?",
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

nb1_cells.append(code(r'''demo2 = run_agent("How much did I spend at Starbucks in total?",
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

nb1_cells.append(md(f"""---
### Continue the workshop

[Next: Notebook 2 — Tracing with Phoenix](https://github.com/{REPO}/blob/main/02_tracing_with_phoenix.ipynb)
"""))


# =====================================================================
# Notebook 02 - Tracing with Phoenix
# =====================================================================

nb2_cells = []

nb2_cells.append(md(f"""{badge("02_tracing_with_phoenix.ipynb")}

# Lab 2: Trace Your Agent

### Instrument the agent with OpenTelemetry and inspect its traces in Phoenix

In notebook 1, you built the agent. Here, OpenInference captures LLM calls automatically,
manual spans make agent and tool steps visible, and every trace goes directly to the shared
Phoenix board on Fly.

The workshop's canonical agent remains in `utils.py`; this notebook attaches a tracer to
that implementation rather than creating a second agent that could drift.
"""))

nb2_cells.append(md("""---
## 1. Install the lightweight Phoenix tracing SDKs

These packages let this notebook export OpenTelemetry spans and read them from the shared
Phoenix board. They provide tracing and client access; the workshop server is already running.
"""))

nb2_cells.append(code(f"%pip install -q arize-phoenix-otel=={PHOENIX_OTEL_VERSION} arize-phoenix-client=={PHOENIX_CLIENT_VERSION} openinference-instrumentation-openai=={OPENINFERENCE_VERSION}", ("core",)))

nb2_cells.append(md("""> Install packages before importing the agent or tracing SDK. If Colab says the
> runtime needs a restart, restart it now and rerun the notebook from the top.
"""))

nb2_cells.append(md("""---
## Required action: provide the Phoenix board key

Before running the next code cell, open Colab's **Secrets** panel (key icon), add
`PHOENIX_API_KEY` using the participant board key supplied by the organizer, and enable
Notebook access. This is separate from `LITELLM_API_KEY`. If you do not add the secret,
the bootstrap cell will ask for the key privately.
"""))

nb2_cells.append(md("""## 2. Bootstrap the Notebook 1 agent

Notebook 1's agent has been wrapped in the canonical `utils.py` module. This notebook
refreshes that file and imports the same `run_agent` code instead of redefining the agent.
It also loads the ledger into `utils.df`; the local alias `df` below is used only for
ground-truth comparisons.
"""))

nb2_cells.append(code(f"""import urllib.request
import os
import importlib
from pathlib import Path

utils_path = Path("utils.py")
if not utils_path.exists() or "def set_span_tracer(tracer, attributes=None):" not in utils_path.read_text():
    urllib.request.urlretrieve("{RAW}/utils.py", "utils.py")

import utils
utils = importlib.reload(utils)

utils.load_data()
df = utils.df
utils.connect()
SEAT = utils.seat_tag()
PHOENIX_BOARD_URL = "{PHOENIX_BOARD_URL}"
PHOENIX_PROJECT = "{PHOENIX_PROJECT}"
PHOENIX_API_KEY = utils.get_secret("PHOENIX_API_KEY")
if not PHOENIX_API_KEY:
    from getpass import getpass
    PHOENIX_API_KEY = getpass("Phoenix board API key: ").strip()
from phoenix.client import Client
phoenix_client = Client(base_url=PHOENIX_BOARD_URL, api_key=PHOENIX_API_KEY)
try:
    phoenix_client.projects.list()
except Exception as exc:
    if "401" not in str(exc) and "Unauthorized" not in str(exc):
        raise
    from getpass import getpass
    print("Saved Phoenix key was rejected. Enter the participant board key supplied by the organizer.")
    PHOENIX_API_KEY = getpass("Phoenix board API key: ").strip()
    phoenix_client = Client(base_url=PHOENIX_BOARD_URL, api_key=PHOENIX_API_KEY)
    phoenix_client.projects.list()
from utils import NAIVE_SYSTEM_PROMPT, run_agent

print("bootstrap done")
print("Phoenix project:", PHOENIX_PROJECT)
print("Phoenix session.id:", SEAT)
""", ("core",)))

nb2_cells.append(md("""## 3. Send traces to the shared Phoenix board

Register the shared OTLP collector and authenticate with the board key. The whole room
uses the `00-Part-2-tracing-set-up` project, which sorts before the gateway project.
"""))

nb2_cells.append(code(r"""from phoenix.otel import register

tracer_provider = register(
    project_name=PHOENIX_PROJECT,
    endpoint=f"{PHOENIX_BOARD_URL}/v1/traces",
    protocol="http/protobuf",
    batch=False,
    api_key=PHOENIX_API_KEY,
)
print("OTLP collector:", f"{PHOENIX_BOARD_URL}/v1/traces")
""", ("core",)))

nb2_cells.append(md("""## 4. Why trace the agent?

An answer alone does not show how the agent reached it. OpenInference records each model
call, and the agent adds spans for its router and tools. In Phoenix, follow the trace to see
what the model asked each tool to do, what came back, and where an error occurred.
"""))

nb2_cells.append(code(r"""from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes

openai_instrumentor = OpenAIInstrumentor()
if openai_instrumentor.is_instrumented_by_opentelemetry:
    openai_instrumentor.uninstrument()
openai_instrumentor.instrument(tracer_provider=tracer_provider)
utils.set_span_tracer(
    tracer_provider.get_tracer(__name__),
    attributes={SpanAttributes.SESSION_ID: SEAT},
)
print("OpenAI calls instrumented; manual agent/chain/tool spans enabled")
""", ("core",)))

nb2_cells.append(md("""> **Trace data note:** spans contain prompts and tool inputs/outputs. This workshop uses
> synthetic transactions; do not send personal or otherwise sensitive data to a tracer.
"""))

nb2_cells.append(md("""---
## 5. Challenge 1: signed expenses versus cash flow

These are natural user questions; they do not name a tool. Watch the trace to see what the
router chooses and why.

Ask the agent for October expenses without explaining that income and refunds are positive
while expenses are negative. In the trace, inspect the generated pandas code and tool output.
"""))

nb2_cells.append(code(r"""gross_october = run_agent(
    "What were my total expenses in October?",
    system_prompt=NAIVE_SYSTEM_PROMPT,
    temperature=0.0,
    label="nb2/october-gross-expenses",
)
print(gross_october["answer"])
""", ("demo",)))

nb2_cells.append(md("""Compare the answer with two plausible calculations: the positive magnitude of
negative transactions, and the signed total of every October transaction.
"""))

nb2_cells.append(code(r"""october = df[df["date"].dt.month == 10]
truth_gross_expenses = -october.loc[october["amount"] < 0, "amount"].sum()
signed_cash_flow = october["amount"].sum()
print(f"expense magnitude: {truth_gross_expenses:,.2f}")
print(f"signed cash flow:  {signed_cash_flow:,.2f}")
print(f"positive income/refunds included: {october.loc[october['amount'] > 0, 'amount'].sum():,.2f}")
""", ("demo",)))

nb2_cells.append(md("""### Inspect this run's tool I/O

The Phoenix trace shows the overall span tree. This helper also prints the exact tool
arguments and results returned to the agent for this individual run.
"""))

nb2_cells.append(code(r"""def show_tool_io(result):
    "Print tool arguments and returned results from one agent run."
    for message in result["messages"]:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                function = call["function"]
                print(f"Calling {function['name']}: {function['arguments']}")
        elif message.get("role") == "tool":
            print("Tool result:", message["content"])
""", ("core",)))
nb2_cells.append(code("show_tool_io(gross_october)", ("demo",)))

nb2_cells.append(md("""### Harder sign case: net expenses after refunds

Gross expenses count negative transactions. Net expenses subtract refunds but must exclude
income. The synthetic ledger contains an October refund.
"""))

nb2_cells.append(code(r"""net_october = run_agent(
    "What were my net expenses in October after refunds, excluding income?",
    system_prompt=NAIVE_SYSTEM_PROMPT,
    temperature=0.0,
    label="nb2/net-expenses-after-refunds",
)
print(net_october["answer"])
""", ("demo",)))

nb2_cells.append(code(r"""refunds = october.loc[
    october["raw_description"].str.contains("REFUND", case=False) & (october["amount"] > 0), "amount"
].sum()
print(f"gross expenses: {truth_gross_expenses:,.2f}")
print(f"refunds:        {refunds:,.2f}")
print(f"net expenses:   {truth_gross_expenses - refunds:,.2f}")
""", ("demo",)))
nb2_cells.append(code("show_tool_io(net_october)", ("demo",)))

nb2_cells.append(md("""---
## 6. Challenge 2: merchant aliases and a decoy

Descriptions use both `STARBUCKS` and the alias `SBUX`. Tully's is another coffee shop,
not Starbucks. Inspect the tool input and see whether the generated filter handles both cases.
"""))

nb2_cells.append(code(r"""starbucks_run = run_agent(
    "How much did I spend at Starbucks in total?",
    system_prompt=NAIVE_SYSTEM_PROMPT,
    temperature=0.0,
    label="nb2/starbucks-aliases",
)
print(starbucks_run["answer"])
""", ("demo",)))

nb2_cells.append(code(r"""is_starbucks = df["raw_description"].str.contains(r"STARBUCKS|SBUX", case=False, regex=True)
truth_starbucks = -df.loc[is_starbucks & (df["amount"] < 0), "amount"].sum()
tully_decoy_count = df["raw_description"].str.contains("TULLY'S", case=False).sum()
print(f"Starbucks + SBUX expense magnitude: {truth_starbucks:,.2f}")
print(f"Tully's decoy rows to exclude:      {tully_decoy_count}")
""", ("demo",)))
nb2_cells.append(code("show_tool_io(starbucks_run)", ("demo",)))

nb2_cells.append(md("""---
## 7. Free-text: ask your own question

Try any finance question about the ledger. The prompt below is intentionally open-ended:
the agent chooses its tools, and the trace shows how it responds.
"""))

nb2_cells.append(code(r"""free_text_query = input("Your question for the financial agent: ")
free_text_result = run_agent(
    free_text_query,
    system_prompt=NAIVE_SYSTEM_PROMPT,
    temperature=0.0,
    label=f"nb2/free-text/{SEAT}",
)
print(free_text_result["answer"])
""", ("demo",)))
nb2_cells.append(code("show_tool_io(free_text_result)", ("demo",)))

nb2_cells.append(md(f"""---
## Find your traces in Phoenix

1. Open the [shared Phoenix board]({PHOENIX_BOARD_URL}).
2. Select project **`{PHOENIX_PROJECT}`**.
3. Search/filter spans by the **`session.id`** value printed during setup to isolate your
   runs from the rest of the room.
4. Expand an AGENT span to follow its nested router, LLM, tool, and tool-generated LLM spans.

The code below fetches your own trace spans into `my_spans` so you can inspect them in pandas.
"""))

nb2_cells.append(code(r"""spans_df = phoenix_client.spans.get_spans_dataframe(project_identifier=PHOENIX_PROJECT, limit=5000)
session_column = "attributes.session.id"
trace_id_column = next((column for column in ("context.trace_id", "trace_id") if column in spans_df.columns), None)
my_trace_ids = []
if session_column in spans_df.columns and trace_id_column:
    my_trace_ids = spans_df.loc[spans_df[session_column] == SEAT, trace_id_column].dropna().unique()
    my_spans = spans_df[spans_df[trace_id_column].isin(my_trace_ids)].copy()
else:
    my_spans = spans_df[spans_df.get(session_column, "") == SEAT].copy() if session_column in spans_df else spans_df.iloc[0:0].copy()
trace_columns = [
    column for column in (
        "name", "span_kind", "parent_id", trace_id_column, session_column,
        "status_code", "attributes.input.value", "attributes.output.value",
        "attributes.llm.token_count.prompt", "attributes.llm.token_count.completion",
    )
    if column and column in my_spans.columns
]
print(f"Loaded {len(my_spans)} spans across {len(my_trace_ids)} of your traces (session.id={SEAT})")
display(my_spans[trace_columns])
""", ("demo",)))

nb2_cells.append(md(f"""---
### Continue the workshop

[Next: Notebook 3 — Evaluations](https://github.com/{REPO}/blob/main/03_evaluations.ipynb)
"""))


# =====================================================================
# Notebook 03 - Evaluations
# =====================================================================

nb3_cells = []

nb3_cells.append(md(f"""{badge("03_evaluations.ipynb")}

# Evaluate, Experiment, Improve

Notebook 2 made the agent's behavior visible. Now we will evaluate a few important skills,
inspect failures in Phoenix, and test a prompt change against the same examples.

**Learning path:** run a small baseline → inspect router and skill scores → change one prompt
→ compare the same cases → review a failure → decide whether it belongs in the golden dataset.

The short lab uses three cases to keep live calls manageable. The complete 11-case suite stays
in the shared harness and CI.
"""))

nb3_cells.append(md("""---
## 1. Install the workshop and Phoenix client packages

This mirrors Notebook 2's Phoenix setup. The board is hosted; this installs only the client
and tracing SDKs, not a second Phoenix server. The base packages let this notebook run on its
own Colab runtime too.
"""))

nb3_cells.append(code(f'''%pip install -q "openai>=1.40" "pandas>=2.0" "matplotlib>=3.8" arize-phoenix-otel=={PHOENIX_OTEL_VERSION} arize-phoenix-client=={PHOENIX_CLIENT_VERSION} openinference-instrumentation-openai=={OPENINFERENCE_VERSION}''', ("core",)))

nb3_cells.append(md("""> Install before importing the agent or tracing SDK. If Colab says the runtime
> needs a restart, restart it and rerun the notebook from the top.
"""))

nb3_cells.append(md("""---
## 2. Load the same agent and credentials as Notebook 2

Notebook 3 imports the canonical agent from `utils.py`, loads the frozen golden cases, and
uses the same gateway and seat key as Notebook 2. The Phoenix board key is separate from the
workshop model key.
"""))

nb3_cells.append(code(f'''import urllib.request
import os
import importlib
from pathlib import Path

utils_path = Path("utils.py")
required_utils = (
    "def set_span_tracer(tracer, attributes=None):",
    "def load_golden():",
    "def run_suite(label, system_prompt, temperature, agent_fn=None, judge_fn=None,",
)
if not utils_path.exists() or any(marker not in utils_path.read_text() for marker in required_utils):
    urllib.request.urlretrieve("{RAW}/utils.py", "utils.py")

import utils
utils = importlib.reload(utils)

utils.load_data()
utils.load_golden()
utils.connect()
SEAT = utils.seat_tag()
PHOENIX_BOARD_URL = "{PHOENIX_BOARD_URL}"
PHOENIX_EVAL_PROJECT = "{PHOENIX_EVAL_PROJECT}"
PHOENIX_API_KEY = utils.get_secret("PHOENIX_API_KEY")
if not PHOENIX_API_KEY:
    from getpass import getpass
    PHOENIX_API_KEY = getpass("Phoenix board API key: ").strip()
from phoenix.client import Client
phoenix_client = Client(base_url=PHOENIX_BOARD_URL, api_key=PHOENIX_API_KEY)
try:
    phoenix_client.projects.list()
except Exception as exc:
    if "401" not in str(exc) and "Unauthorized" not in str(exc):
        raise
    from getpass import getpass
    print("Saved Phoenix key was rejected. Enter the participant board key supplied by the organizer.")
    PHOENIX_API_KEY = getpass("Phoenix board API key: ").strip()
    phoenix_client = Client(base_url=PHOENIX_BOARD_URL, api_key=PHOENIX_API_KEY)
    phoenix_client.projects.list()

from utils import (
    AGENT_MODEL, CREATIVE_SYSTEM_PROMPT, HARDENED_SYSTEM_PROMPT, GOLDEN,
    SUITE_RESULTS, check_code, check_numeric, check_router, run_agent, run_suite,
)

print("bootstrap done |", len(GOLDEN["cases"]), "golden cases | seat:", SEAT)
''', ("core",)))

nb3_cells.append(md("""## 3. Send this notebook's traces to Phoenix

As in Notebook 2, the OTLP exporter writes directly to the shared board. Notebook 3 uses
`01-Part-3-evaluations`, separate from Notebook 2's trace-review project. The stable
`session.id` identifies your seat; unique run labels distinguish experiment variants.
"""))

nb3_cells.append(code(r"""from phoenix.otel import register

tracer_provider = register(
    project_name=PHOENIX_EVAL_PROJECT,
    endpoint=f"{PHOENIX_BOARD_URL}/v1/traces",
    protocol="http/protobuf",
    batch=False,
    api_key=PHOENIX_API_KEY,
)
print("Phoenix project:", PHOENIX_EVAL_PROJECT)
print("Phoenix board:", PHOENIX_BOARD_URL)
""", ("core",)))

nb3_cells.append(code(r"""from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes

openai_instrumentor = OpenAIInstrumentor()
if openai_instrumentor.is_instrumented_by_opentelemetry:
    openai_instrumentor.uninstrument()
openai_instrumentor.instrument(tracer_provider=tracer_provider)
utils.set_span_tracer(
    tracer_provider.get_tracer(__name__),
    attributes={SpanAttributes.SESSION_ID: SEAT},
)
print("OpenAI and agent spans are instrumented")
""", ("core",)))

nb3_cells.append(md("""---
## 4. Run a small baseline

An evaluation is a repeatable question, expected behavior, and evaluator. We will start with
three examples from the frozen golden dataset: signed expenses, merchant aliases, and a
missing schema field. The same checks and examples will be used for every variant.
"""))

nb3_cells.append(code(r'''from IPython.display import display

STARTER_IDS = ["total_expenses_october", "starbucks_total", "merchant_location"]
STARTER_CASES = [case for case in GOLDEN["cases"] if case["id"] in STARTER_IDS]
if len(STARTER_CASES) != len(STARTER_IDS):
    raise ValueError("Starter cases are missing from the golden dataset")

import datetime as _datetime
RUN_TAG = _datetime.datetime.now(_datetime.timezone.utc).strftime("lab3-%Y%m%d-%H%M%S-%f")
BASELINE_LABEL = f"{RUN_TAG}/prompt-baseline"

baseline = run_suite(
    BASELINE_LABEL,
    HARDENED_SYSTEM_PROMPT,
    temperature=0.0,
    cases=STARTER_CASES,
)
display(baseline[[c for c in ("case", "router_ok", "tools_used", "numeric_ok", "judge_pass", "PASS") if c in baseline]])
''', ("demo",)))

nb3_cells.append(md("""### What are the evaluators checking?

The **router eval** checks whether the agent called the expected tool. The **skill evals**
check whether generated code ran and whether a numeric answer matched frozen ground truth;
the missing-location case uses a judge for honesty and usefulness. These are separate signals:
a correct tool choice does not guarantee a correct answer.

The analysis tool also cautions the code generator to inspect the full dataset for likely
entity variants and decoys, using bounded summaries instead of dumping rows. The merchant case
tests whether that general caution is enough; it is not a guarantee, so inspect the generated
code and trace evidence.
"""))

nb3_cells.append(code(r'''example_case = next(c for c in STARTER_CASES if c["id"] == "total_expenses_october")
example_result = SUITE_RESULTS[BASELINE_LABEL][example_case["id"]]
print("router eval:", check_router(example_case, example_result))
print("skill evals:", {**check_code(example_result), **check_numeric(example_case, example_result)})
''', ("demo",)))

nb3_cells.append(md("""---
## 5. Run a prompt experiment

An experiment changes one thing and holds the rest fixed: **same questions, model,
temperature, and evaluators**. The variant below intentionally relaxes tool-use and
precision guidance. Run it, compare the scores, then inspect the corresponding traces.

This lab uses Phoenix traces and evaluation annotations to review the variants. It does not
create a separate Phoenix Dataset/Experiment object; this lightweight experiment keeps the
Notebook 2 client setup unchanged. The SDK-managed dataset workflow can be an advanced follow-up.
"""))

nb3_cells.append(code(r'''CANDIDATE_SYSTEM_PROMPT = CREATIVE_SYSTEM_PROMPT
CANDIDATE_LABEL = f"{RUN_TAG}/prompt-candidate"
print("Candidate system prompt:\n", CANDIDATE_SYSTEM_PROMPT)

candidate = run_suite(
    CANDIDATE_LABEL,
    CANDIDATE_SYSTEM_PROMPT,
    temperature=0.0,  # held fixed to isolate the prompt change
    cases=STARTER_CASES,
)
display(candidate[[c for c in ("case", "router_ok", "tools_used", "numeric_ok", "judge_pass", "PASS") if c in candidate]])
''', ("demo",)))

nb3_cells.append(code(r'''comparison = baseline[["case", "PASS", "router_ok"]].merge(
    candidate[["case", "PASS", "router_ok"]], on="case", suffixes=("_baseline", "_candidate")
)
comparison["regressed"] = comparison["PASS_baseline"] & ~comparison["PASS_candidate"]
print(f"baseline: {int(baseline['PASS'].sum())}/{len(baseline)}")
print(f"candidate: {int(candidate['PASS'].sum())}/{len(candidate)}")
comparison
''', ("demo",)))

nb3_cells.append(md("""### Optional: compare two router models instead

Keep the hardened prompt and the same case fixed, then change only the router model. The
workshop gateway is configured for `gpt-4o-mini` and `gpt-4o`; the larger model may use more
of your seat's budget. This agent also makes a separate code-generation call inside its data
tool; that inner call stays on `gpt-4o-mini`, so this is a router-model comparison, not a
whole-agent model comparison. It uses one numeric case to keep the optional exercise short.
Check the gateway's currently enabled model aliases before changing these names.
"""))

nb3_cells.append(code(r'''MODEL_CASES = [next(c for c in GOLDEN["cases"] if c["id"] == "total_expenses_october")]
MODEL_A = AGENT_MODEL
MODEL_B = "gpt-4o"
model_a_label = f"{RUN_TAG}/model-{MODEL_A}"
model_b_label = f"{RUN_TAG}/model-{MODEL_B}"

model_a = run_suite(model_a_label, HARDENED_SYSTEM_PROMPT, 0.0, cases=MODEL_CASES, model=MODEL_A)
model_b = run_suite(model_b_label, HARDENED_SYSTEM_PROMPT, 0.0, cases=MODEL_CASES, model=MODEL_B)
display(model_a[["case", "PASS", "latency_s"]].merge(
    model_b[["case", "PASS", "latency_s"]], on="case", suffixes=(f"_{MODEL_A}", f"_{MODEL_B}")
))
''', ("demo",)))

nb3_cells.append(md("""---
## 6. Inspect and label a failure in Phoenix

Open the shared board, select `01-Part-3-evaluations`, and filter to your `session.id` and
the run-label prefix shown below. Open a failed candidate trace (or a baseline failure if the
candidate happens to pass). Compare the user question, router decision, tool input/output,
generated code, and final answer. Which component failed?

If the board's annotation UI is enabled for participants, add a **human** failure label and a
short evidence-based note. Otherwise, record the same note in the next cell. Do not classify a
case as a model failure until you've checked that the evaluator and expected answer are sound.
"""))

nb3_cells.append(code(r'''print("Board:", PHOENIX_BOARD_URL)
print("Project:", PHOENIX_EVAL_PROJECT)
print("Seat filter (session.id):", SEAT)
failed_candidate_ids = candidate.loc[~candidate["PASS"], "case"].tolist()
review_ids = failed_candidate_ids or baseline.loc[~baseline["PASS"], "case"].tolist()
REVIEW_LABEL = CANDIDATE_LABEL if failed_candidate_ids else BASELINE_LABEL
print("Trace-name prefix:", REVIEW_LABEL + "/")
print("Candidate failures:", failed_candidate_ids or "none this run")
print("Review these case(s):", review_ids or "no failed cases; inspect any trace and challenge the evaluator")
''', ("demo",)))

nb3_cells.append(code(r'''# Complete this after inspecting the trace. Keep the evidence concise and reproducible.
failure_note = {
    "run_label": REVIEW_LABEL,
    "case_id": "replace-with-failed-case-id",
    "failure_type": "router | skill | data | evaluator | prompt",
    "trace_evidence": "Which span/input/output shows the failure?",
    "expected_behavior": "What should the agent have done?",
    "reproducible": "yes | no | not_checked",
}
failure_note
''', ("demo",)))

nb3_cells.append(md("""---
## 7. Promote a reviewed failure to the golden dataset

Only promote a case when the failure is reproducible, the expected behavior is clear, and
ground truth can be independently checked. Capture at least: a unique `id`, the exact user
`query`, case `type`, expected behavior/value, tolerance (for numeric cases), and the rule for
computing ground truth.

For this repository, golden cases are generated by `scripts/generate_dataset.py`; don't edit
the JSON by hand. Add the case to `build_cases()` and compute its expected value from the
synthetic data, then regenerate and verify with `python scripts/generate_dataset.py --check`.
The notebook note above is a review candidate, not an automatic dataset change.
"""))

nb3_cells.append(md("""## Attach evaluator scores to the experiment traces

This adds code-eval annotations to traces using unique run labels, so repeated notebook runs
do not accidentally attach scores to an earlier trace. Human review remains a separate label.
"""))

nb3_cells.append(code(r'''import pandas as pd

scorecards = [(BASELINE_LABEL, baseline), (CANDIDATE_LABEL, candidate)]
spans = phoenix_client.spans.get_spans(project_identifier=PHOENIX_EVAL_PROJECT, limit=2000)
span_ids = {}
for span in spans:
    attributes = span.get("attributes") or {}
    if attributes.get("session.id") != SEAT:
        continue
    span_name = span.get("name")
    span_id = (span.get("context") or {}).get("span_id")
    if span_name and span_id:
        span_ids[span_name] = span_id

annotations = []
for label, scorecard in scorecards:
    for row in scorecard.to_dict(orient="records"):
        span_id = span_ids.get(f"{label}/{row['case']}")
        if span_id:
            annotations.append({
                "span_id": span_id,
                "label": "PASS" if row["PASS"] else "FAIL",
                "score": float(bool(row["PASS"])),
                "explanation": f"{label}: router={row.get('router_ok')}; numeric={row.get('numeric_ok')}",
            })

if annotations:
    phoenix_client.spans.log_span_annotations_dataframe(
        dataframe=pd.DataFrame(annotations),
        annotation_name="Workshop golden checks",
        annotator_kind="CODE",
    )
    print(f"Logged {len(annotations)} evaluator annotations")
else:
    print("No matching spans found yet; rerun this cell after the traces arrive")
''', ("demo",)))

nb3_cells.append(md("""---
## Takeaways

- Router, execution, answer-quality, and ground-truth checks catch different failures.
- A prompt or model change is an experiment only when you compare it on the same cases with
  the same evaluators and hold other variables fixed.
- Phoenix traces help explain *why* a score changed; human review decides whether a failure
  is real and worth adding to the golden set.
- The complete golden suite and regression gate remain in `scripts/run_evals.py` and CI.

For a full-cost run of all 11 cases, use the harness outside the short training path or during
an instructor-led extension.
"""))

nb3_cells.append(md(f"""---
### Workshop home

[Return to the workshop README](https://github.com/{REPO}/blob/main/README.md)
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
