"""Headless CI twin of workshop.ipynb Module 4.

Executes the notebook's "core"-tagged cells (config, tools, agent, eval harness) and runs one
golden-suite configuration as a pass/fail gate. The notebook stays the single source of truth:
if the harness and the notebook disagree, this script breaks loudly.

Exit codes:
  0 - gate satisfied (baseline >= min pass rate, or regression detected when expected)
  1 - gate violated
  2 - harness error
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CORE_TAG = "core"
RESULTS_DIR = ROOT / "results"


def exec_core_cells(notebook_path: Path, namespace: dict) -> None:
    import nbformat

    nb = nbformat.read(str(notebook_path), as_version=4)
    executed = 0
    for cell in nb.cells:
        tags = cell.get("metadata", {}).get("tags", [])
        if cell.cell_type == "code" and CORE_TAG in tags:
            exec(compile(cell.source, f"<core-cell-{executed}>", "exec"), namespace)
            executed += 1
    if executed == 0:
        raise RuntimeError(f"no '{CORE_TAG}' cells found in {notebook_path}")


def build_client(namespace: dict):
    from openai import OpenAI

    mode = namespace["WORKSHOP_MODE"]
    env_names = ["OPENAI_API_KEY"] if mode == "direct" else ["LITELLM_API_KEY", "OPENAI_API_KEY"]
    api_key = next((os.getenv(n, "").strip() for n in env_names if os.getenv(n)), "")
    if not api_key:
        raise SystemExit(f"2 no API key found for mode '{mode}' (expected env: {env_names})")
    return OpenAI(api_key=api_key, base_url=namespace["ACTIVE_BASE_URL"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default=os.getenv("WORKSHOP_MODE", "mock"), choices=["mock", "direct", "gateway"])
    parser.add_argument("--config", default="baseline", choices=["baseline", "regression"])
    parser.add_argument("--min-pass-rate", type=float, default=0.9)
    parser.add_argument("--expect-regression", action="store_true", help="gate passes only if pass rate drops below the threshold")
    parser.add_argument("--notebook", default=str(ROOT / "workshop.ipynb"))
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args()

    os.environ["WORKSHOP_MODE"] = args.mode
    ns = {"__name__": "workshop_core"}
    exec_core_cells(Path(args.notebook), ns)

    if args.mode != "mock":
        ns["client"] = build_client(ns)

    if args.config == "baseline":
        label, prompt_key, temperature = "baseline", "HARDENED_SYSTEM_PROMPT", 0.0
    else:
        label, prompt_key, temperature = "regression", "CREATIVE_SYSTEM_PROMPT", 0.7

    scorecard = ns["run_suite"](label, ns[prompt_key], temperature)
    rows = scorecard.to_dict(orient="records")
    passed = int(scorecard["PASS"].sum())
    total = len(scorecard)
    pass_rate = passed / total if total else 0.0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "mode": args.mode,
        "model": ns["AGENT_MODEL"],
        "temperature": temperature,
        "passed": passed,
        "total": total,
        "pass_rate": round(pass_rate, 4),
        "min_pass_rate": args.min_pass_rate,
        "rows": rows,
    }
    (out_dir / f"{label}.scorecard.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    scorecard.to_csv(out_dir / f"{label}.scorecard.csv", index=False)

    print(f"[evals] config={label} mode={args.mode} model={ns['AGENT_MODEL']} t={temperature}")
    print(f"[evals] pass_rate={pass_rate:.0%} ({passed}/{total})  artifacts -> {out_dir}")
    for row in rows:
        flag = "PASS" if row["PASS"] else "FAIL"
        print(f"  [{flag}] {row['case']:<28} trap={row['trap'] or '-'}")

    if args.expect_regression:
        if pass_rate < args.min_pass_rate:
            print(f"[evals] gate: expected regression below {args.min_pass_rate:.0%} -> DETECTED ({pass_rate:.0%})")
            return 0
        print(f"[evals] gate: expected regression below {args.min_pass_rate:.0%} -> NOT DETECTED ({pass_rate:.0%}) - the suite failed to catch a known-bad config")
        return 1
    if pass_rate >= args.min_pass_rate:
        print(f"[evals] gate: min_pass_rate={args.min_pass_rate:.0%} -> PASS")
        return 0
    print(f"[evals] gate: min_pass_rate={args.min_pass_rate:.0%} -> FAIL ({pass_rate:.0%})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
