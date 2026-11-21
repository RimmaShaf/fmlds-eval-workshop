"""CI twin of the workshop eval suite (notebook 03).

Imports the canonical agent + eval harness from utils.py - the module generated
from the same source fragments the notebooks display and execute - and runs one
golden-suite configuration as a pass/fail gate. Notebook and CI can never drift
apart: they are the same code.

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
RESULTS_DIR = ROOT / "results"

CONFIGS = {
    "baseline": ("baseline", "HARDENED_SYSTEM_PROMPT", 0.0),
    "regression": ("regression", "CREATIVE_SYSTEM_PROMPT", 0.7),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default=os.getenv("WORKSHOP_MODE", "mock"), choices=["mock", "gateway"])
    parser.add_argument("--config", default="baseline", choices=list(CONFIGS))
    parser.add_argument("--min-pass-rate", type=float, default=0.9)
    parser.add_argument("--expect-regression", action="store_true", help="gate passes only if pass rate drops below the threshold")
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args()

    os.environ["WORKSHOP_MODE"] = args.mode
    sys.path.insert(0, str(ROOT))
    import utils

    utils.load_data()

    if args.mode == "gateway":
        if not os.getenv("LITELLM_API_KEY"):
            print("error: gateway mode requires the LITELLM_API_KEY env var", file=sys.stderr)
            return 2
        try:
            utils.connect()
        except Exception as e:
            print(f"error: gateway health check failed: {e}", file=sys.stderr)
            return 2

    label, prompt_key, temperature = CONFIGS[args.config]
    scorecard = utils.run_suite(label, getattr(utils, prompt_key), temperature)

    rows = scorecard.to_dict(orient="records")
    passed = int(scorecard["PASS"].sum())
    total = len(scorecard)
    pass_rate = passed / total if total else 0.0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "label": label,
        "mode": args.mode,
        "model": utils.AGENT_MODEL,
        "temperature": temperature,
        "passed": passed,
        "total": total,
        "pass_rate": round(pass_rate, 4),
        "min_pass_rate": args.min_pass_rate,
        "rows": rows,
    }
    (out_dir / f"{label}.scorecard.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")
    scorecard.to_csv(out_dir / f"{label}.scorecard.csv", index=False)

    print(f"[evals] config={label} mode={args.mode} model={utils.AGENT_MODEL} t={temperature}")
    print(f"[evals] pass_rate={pass_rate:.0%} ({passed}/{total})  artifacts -> {out_dir}")
    for row in rows:
        flag = "PASS" if row["PASS"] else "FAIL"
        print(f"  [{flag}] {row['case']:<28} trap={row['trap'] or '-'} router={'ok' if row.get('router_ok') else 'MISS-ROUTED'}")

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
