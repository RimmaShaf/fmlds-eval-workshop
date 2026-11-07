"""Generate per-seat LiteLLM virtual keys and export a sheet-ready CSV.

Talks to the LiteLLM proxy admin API (/key/generate) with the master key. Each seat key gets:
  - a hard spend cap (default $2.00)
  - a rate limit (default 15 RPM) to stop accidental infinite while-loops
  - the workshop model allowlist

Output CSV (seat, key_alias, key, max_budget, rpm_limit) is meant to be pasted into the
view-only Google Sheet. It contains live secrets: it is gitignored and must never be committed.

Use --dry-run to preview the exact payloads without touching the proxy.
"""

import argparse
import csv
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODELS = ["gpt-4o-mini", "gpt-4o", "o3-mini"]
OUT = Path(__file__).resolve().parents[1] / "infra" / "seat_keys.csv"


def generate_key(proxy_url: str, master_key: str, payload: dict) -> dict:
    url = proxy_url.rstrip("/") + "/key/generate"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {master_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-url", default=os.getenv("LITELLM_PROXY_URL", "https://your-tokyo-proxy.fly.dev"))
    parser.add_argument("--seats", type=int, default=50)
    parser.add_argument("--budget", type=float, default=2.00)
    parser.add_argument("--rpm", type=int, default=15)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--duration", default="7d", help="key lifetime, e.g. 7d or 30d")
    parser.add_argument("--master-key", default=os.getenv("LITELLM_MASTER_KEY", ""))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = []

    for seat in range(1, args.seats + 1):
        alias = f"seat-{seat:02d}"
        payload = {
            "key_alias": alias,
            "max_budget": args.budget,
            "rpm_limit": args.rpm,
            "models": models,
            "duration": args.duration,
            "metadata": {"seat": seat, "purpose": "fmlds-eval-workshop"},
        }
        if args.dry_run:
            rows.append({"seat": seat, "key_alias": alias, "key": "DRY-RUN", "max_budget": args.budget, "rpm_limit": args.rpm})
            continue
        if not args.master_key:
            print("error: no master key (set LITELLM_MASTER_KEY or pass --master-key)", file=sys.stderr)
            return 2
        try:
            created = generate_key(args.proxy_url, args.master_key, payload)
        except urllib.error.HTTPError as e:
            print(f"error: seat {seat:02d} rejected by proxy: {e.code} {e.read().decode()[:200]}", file=sys.stderr)
            return 1
        except urllib.error.URLError as e:
            print(f"error: cannot reach proxy at {args.proxy_url}: {e.reason}", file=sys.stderr)
            return 1
        rows.append({
            "seat": seat,
            "key_alias": alias,
            "key": created.get("key", ""),
            "max_budget": args.budget,
            "rpm_limit": args.rpm,
        })
        print(f"created {alias} (budget=${args.budget:.2f}, {args.rpm} rpm)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seat", "key_alias", "key", "max_budget", "rpm_limit"])
        writer.writeheader()
        writer.writerows(rows)

    total_exposure = args.seats * args.budget
    print(f"\nwrote {out_path} ({len(rows)} keys)")
    print(f"max financial exposure: ${total_exposure:.2f}  |  expected actual spend: $5-$25")
    if not args.dry_run:
        print("upload rows to the VIEW-ONLY Google Sheet now; never commit or share this CSV")
    return 0


if __name__ == "__main__":
    sys.exit(main())
