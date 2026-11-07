"""Generate the synthetic personal-finance ledger and freeze the golden eval suite.

Fully offline and deterministic (seed=42). Merchant string style and category
taxonomy are modeled on two public datasets used as design inspiration only:
  - khushikyad001/personal-finance-tracker-dataset (Kaggle) - schema shape
  - mitulshah/transaction-categorization (Hugging Face, MIT) - messy statement strings
No rows are copied or redistributed; every value is regenerated here.

Intentional pitfalls baked into the data:
  - signed amounts (income/refunds positive, expenses negative) -> polarity trap
  - merchant names fragmented across variants ("STARBUCKS #1042", ...) -> string trap
  - no merchant location column of any kind -> schema hallucination trap
"""

import argparse
import json
import random
import uuid
from datetime import date
from pathlib import Path

import pandas as pd

SEED = 42
START_YEAR, START_MONTH = 2026, 5
N_MONTHS = 6
OUT_CSV = Path(__file__).resolve().parents[1] / "data" / "transactions.csv"
OUT_JSON = Path(__file__).resolve().parents[1] / "data" / "golden_eval_cases.json"

DAILY_MERCHANTS = {
    "Food & Dining": [
        ("STARBUCKS #1042", "Credit Card", 3.50, 7.50),
        ("STARBUCKS COFFEE KIOSK", "Credit Card", 4.00, 9.00),
        ("STARBUCKS STORE 08412", "Credit Card", 3.00, 6.50),
        ("WHOLEFREDS MKT #104", "Credit Card", 35.00, 120.00),
        ("MCDONALD'S #1234", "Credit Card", 6.00, 14.00),
        ("SQ *TOKYO RAMEN YA", "Credit Card", 9.00, 22.00),
        ("DOORDASH*GOODYBAG", "Credit Card", 15.00, 45.00),
        ("KONBINI 7-ELEVEN #221", "Credit Card", 3.00, 18.00),
    ],
    "Transportation": [
        ("UBER TRIP", "Credit Card", 8.00, 38.00),
        ("UBER *TRIP HELP.UBER.CO", "Credit Card", 8.00, 30.00),
        ("LYFT RIDE 0421", "Credit Card", 9.00, 35.00),
        ("JR EAST SUICA RECHARGE", "Credit Card", 20.00, 100.00),
        ("SHELL OIL 4432", "Credit Card", 30.00, 75.00),
        ("TAXI GINZA 03-5555", "Credit Card", 9.00, 40.00),
    ],
    "Shopping & Retail": [
        ("AMZN Mktp US*1A2B3C", "Credit Card", 12.00, 140.00),
        ("AMZN MKTP US*2X9Y4D", "Credit Card", 9.00, 90.00),
        ("TARG*T 00432", "Credit Card", 15.00, 85.00),
        ("UNIQLO 0214", "Credit Card", 20.00, 120.00),
        ("IKEA #0042", "Credit Card", 25.00, 180.00),
    ],
    "Entertainment & Recreation": [
        ("TOHO CINEMAS SHINJUKU", "Credit Card", 12.00, 32.00),
        ("STEAM GAMES PURCHASE", "Credit Card", 10.00, 60.00),
        ("ROUND1 BOWLING", "Credit Card", 10.00, 30.00),
    ],
    "Healthcare & Medical": [
        ("WALGREENS #04421", "Credit Card", 8.00, 55.00),
        ("CVS/PHARMACY #771", "Credit Card", 6.00, 40.00),
    ],
    "Financial Services": [
        ("ATM WITHDRAWAL 4432", "Checking", 20.00, 100.00),
    ],
}

DAILY_COUNTS = {
    "Food & Dining": 26,
    "Transportation": 16,
    "Shopping & Retail": 14,
    "Entertainment & Recreation": 8,
    "Healthcare & Medical": 3,
    "Financial Services": 5,
}

MONTHLY_RECURRING = [
    ("ACME CORP PAYROLL DEPOSIT", "Income", "Checking", 5200.00, 5200.00, 1, 1),
    ("SAKURA HEIGHTS APT 4B RENT", "Housing", "Checking", -1450.00, -1450.00, 1, 1),
    ("NETFLIX.COM 866-579", "Entertainment & Recreation", "Checking", -15.49, -15.49, 3, 3),
    ("SPOTIFY USA 4399", "Entertainment & Recreation", "Checking", -10.99, -10.99, 3, 3),
    ("COMCAST*INTERNET", "Utilities & Services", "Checking", -59.99, -59.99, 5, 5),
    ("AT&T MOBILE 9025", "Utilities & Services", "Checking", -45.00, -45.00, 7, 7),
    ("TOKYO ELECTRIC POWER", "Utilities & Services", "Checking", -110.00, -60.00, 12, 12),
    ("NIPPON GAS URBAN", "Utilities & Services", "Checking", -45.00, -25.00, 14, 14),
    ("DONATION WWF JAPAN", "Charity & Donations", "Checking", -50.00, -15.00, 20, 20),
    ("SAVINGS INTEREST PAID", "Financial Services", "Savings", 2.00, 6.00, 28, 28),
]

ONE_OFFS = [
    (0, 2, "SAVINGS DEPOSIT BRANCH TELLER", "Financial Services", "Savings", 2500.00, 2500.00),
    (0, 28, "DENTAL OFFICE TAMURA", "Healthcare & Medical", "Credit Card", -140.00, -140.00),
    (0, 30, "BANK WIRE FEE", "Financial Services", "Checking", -15.00, -15.00),
    (1, 10, "RED CROSS DONATION", "Charity & Donations", "Checking", -50.00, -50.00),
    (1, 15, "FREELANCE INVOICE #1187 - DESIGN", "Income", "Checking", 650.00, 650.00),
    (2, 5, "BROKERAGE DIVIDEND VTI", "Income", "Checking", 18.76, 18.76),
    (2, 21, "TAX REFUND - INTERNAL REVENUE", "Income", "Checking", 418.50, 418.50),
    (2, 28, "DENTAL OFFICE TAMURA", "Healthcare & Medical", "Credit Card", -160.00, -160.00),
    (3, 8, "FREELANCE INVOICE #1204 - DESIGN", "Income", "Checking", 475.00, 475.00),
    (3, 30, "BANK WIRE FEE", "Financial Services", "Checking", -15.00, -15.00),
    (4, 3, "AMZN MKTP US*1A2B3C REFUND", "Shopping & Retail", "Credit Card", 64.20, 64.20),
    (4, 12, "DENTAL OFFICE TAMURA", "Healthcare & Medical", "Credit Card", -155.00, -155.00),
    (4, 17, "UBER TRIP REFUND", "Transportation", "Credit Card", 12.40, 12.40),
    (5, 6, "AMZN MKTP US*2X9Y4D REFUND", "Shopping & Retail", "Credit Card", 29.99, 29.99),
    (5, 24, "FREELANCE INVOICE #1231 - DESIGN", "Income", "Checking", 520.00, 520.00),
    (5, 30, "BANK WIRE FEE", "Financial Services", "Checking", -15.00, -15.00),
]


def month_offset(year: int, month: int, offset: int) -> tuple[int, int]:
    total = year * 12 + (month - 1) + offset
    return total // 12, total % 12 + 1


def days_in_month(year: int, month: int) -> int:
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    return (nxt - date(year, month, 1)).days


def build_rows() -> list[dict]:
    rng = random.Random(SEED)
    rows: list[dict] = []

    def add(d: date, desc: str, category: str, account: str, amount: float) -> None:
        rows.append(
            {
                "transaction_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://workshop.example/txn/{len(rows)}")),
                "date": d.isoformat(),
                "raw_description": desc,
                "category": category,
                "amount": round(amount, 2),
                "account_type": account,
            }
        )

    for m in range(N_MONTHS):
        year, month = month_offset(START_YEAR, START_MONTH, m)
        dim = days_in_month(year, month)

        for desc, category, account, hi, lo, day_lo, day_hi in MONTHLY_RECURRING:
            day = rng.randint(day_lo, day_hi) if day_hi > day_lo else day_lo
            add(date(year, month, day), desc, category, account, round(rng.uniform(lo, hi), 2))

        for category, merchants in DAILY_MERCHANTS.items():
            for _ in range(DAILY_COUNTS[category]):
                desc, account, lo, hi = rng.choice(merchants)
                day = rng.randint(1, dim)
                add(date(year, month, day), desc, category, account, -round(rng.uniform(lo, hi), 2))

    for m, day, desc, category, account, lo, hi in ONE_OFFS:
        year, month = month_offset(START_YEAR, START_MONTH, m)
        add(date(year, month, day), desc, category, account, round(rng.uniform(lo, hi), 2))

    rows.sort(key=lambda r: (r["date"], r["raw_description"]))
    for i, r in enumerate(rows):
        r["transaction_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://workshop.example/txn/{i}"))
    return rows


def compute_ground_truth(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["month"] = d["date"].dt.month
    expenses = d[d["amount"] < 0].copy()
    expenses["spend"] = -expenses["amount"]

    oct_expenses = round(float(expenses.loc[expenses["month"] == 10, "spend"].sum()), 2)
    starbucks = round(
        float(expenses.loc[expenses["raw_description"].str.contains("STARBUCKS"), "spend"].sum()), 2
    )
    total_income = round(float(d.loc[d["category"] == "Income", "amount"].sum()), 2)
    by_cat = expenses.groupby("category")["spend"].sum().sort_values(ascending=False)
    top_category, top_value = str(by_cat.index[0]), round(float(by_cat.iloc[0]), 2)
    avg_monthly = round(float(expenses.groupby("month")["spend"].sum().mean()), 2)
    largest_row = d.loc[d["amount"].idxmin()]
    may_count = int((d["month"] == 5).sum())
    savings_total = round(float(d.loc[d["account_type"] == "Savings", "amount"].sum()), 2)

    return {
        "oct_expenses": oct_expenses,
        "starbucks": starbucks,
        "total_income": total_income,
        "top_category": top_category,
        "top_category_value": top_value,
        "avg_monthly_expenses": avg_monthly,
        "largest_expense_value": round(abs(float(largest_row["amount"])), 2),
        "largest_expense_merchant": str(largest_row["raw_description"]),
        "may_transaction_count": may_count,
        "savings_total": savings_total,
        "emergency_fund_target": round(3 * avg_monthly, 2),
        "monthly_income": 5200.00,
    }


def build_cases(gt: dict) -> list[dict]:
    judge_numeric_rubric = [
        "actionability: concrete, prioritized next steps grounded in the data (1-5)",
        "heuristic_adherence: correctly applies the requested financial rule (1-5)",
        "numerical_fidelity: every stated figure matches the provided ground truth (1-5)",
    ]
    return [
        {
            "id": "total_expenses_october",
            "query": "What were my total expenses in October?",
            "type": "numeric",
            "trap": "polarity",
            "expected": {"value": gt["oct_expenses"]},
            "tolerance": 0.01,
            "ground_truth_logic": "df[(df.date.dt.month == 10) & (df.amount < 0)].amount.sum(), reported as a positive magnitude",
        },
        {
            "id": "starbucks_total",
            "query": "How much did I spend at Starbucks in total?",
            "type": "numeric",
            "trap": "fragmentation",
            "expected": {"value": gt["starbucks"]},
            "tolerance": 0.01,
            "ground_truth_logic": "rows where raw_description contains 'STARBUCKS' (case-insensitive) and amount < 0, spend magnitudes summed",
        },
        {
            "id": "merchant_location",
            "query": "Show me my spending broken down by merchant location.",
            "type": "behavior",
            "trap": "schema_drift",
            "expected": {
                "behavior": "recognizes that no location/city column exists in the schema and says so, pivoting to a valid breakdown (e.g. by merchant) instead of fabricating locations"
            },
            "ground_truth_logic": "the schema has exactly these columns: transaction_id, date, raw_description, category, amount, account_type - there is no merchant_city or location column",
            "judge_rubric": [
                "schema_fidelity: does not fabricate or assume non-existent columns (1-5)",
                "honesty: clearly tells the user the requested breakdown is unavailable (1-5)",
                "usefulness: offers the closest legitimate alternative analysis (1-5)",
            ],
        },
        {
            "id": "total_income",
            "query": "What was my total income over the whole 6-month period?",
            "type": "numeric",
            "trap": None,
            "expected": {"value": gt["total_income"]},
            "tolerance": 0.01,
            "ground_truth_logic": "df[df.category == 'Income'].amount.sum() (salary + freelance + dividends + tax refund)",
        },
        {
            "id": "top_spending_category",
            "query": "Which category did I spend the most money on, and how much?",
            "type": "numeric",
            "trap": None,
            "expected": {"value": gt["top_category_value"], "category": gt["top_category"]},
            "tolerance": 0.01,
            "ground_truth_logic": "expenses grouped by category, highest total spend magnitude wins",
        },
        {
            "id": "avg_monthly_spending",
            "query": "On average, how much did I spend per month?",
            "type": "numeric",
            "trap": None,
            "expected": {"value": gt["avg_monthly_expenses"]},
            "tolerance": 0.01,
            "ground_truth_logic": "mean over the 6 months of each month's total expense magnitude (amount < 0)",
        },
        {
            "id": "largest_single_expense",
            "query": "What was my single largest expense, and at which merchant?",
            "type": "numeric",
            "trap": None,
            "expected": {"value": gt["largest_expense_value"], "merchant_contains": gt["largest_expense_merchant"]},
            "tolerance": 0.01,
            "ground_truth_logic": "row with the minimum (most negative) amount; magnitude and raw_description",
        },
        {
            "id": "transaction_count_may",
            "query": "How many transactions did I make in May?",
            "type": "numeric",
            "trap": None,
            "expected": {"value": gt["may_transaction_count"]},
            "tolerance": 0.0,
            "ground_truth_logic": "count of rows where date.dt.month == 5 (income and expenses both count)",
        },
        {
            "id": "advisory_budget_5030_20",
            "query": "I want to follow the 50/30/20 budgeting rule. Based on my transactions, how am I doing and what should I change?",
            "type": "advisory",
            "trap": None,
            "expected": {
                "anchors": {
                    "monthly_income": gt["monthly_income"],
                    "avg_monthly_expenses": gt["avg_monthly_expenses"],
                    "needs_target": round(0.5 * gt["monthly_income"], 2),
                    "wants_target": round(0.3 * gt["monthly_income"], 2),
                    "savings_target": round(0.2 * gt["monthly_income"], 2),
                }
            },
            "ground_truth_logic": "the agent should consult the financial rules tool for 50/30/20 and reason from monthly income 5200.00",
            "judge_rubric": judge_numeric_rubric,
        },
        {
            "id": "advisory_emergency_fund",
            "query": "Do I currently have enough in savings to cover a 3-month emergency fund? Please advise.",
            "type": "advisory",
            "trap": None,
            "expected": {
                "anchors": {
                    "savings_total": gt["savings_total"],
                    "avg_monthly_expenses": gt["avg_monthly_expenses"],
                    "emergency_fund_target": gt["emergency_fund_target"],
                    "verdict": "no - savings are far below the 3-month target",
                }
            },
            "ground_truth_logic": "savings balance vs 3 x average monthly expenses",
            "judge_rubric": judge_numeric_rubric,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="only re-verify existing artifacts against the seed")
    args = parser.parse_args()

    rows = build_rows()
    df = pd.DataFrame(rows, columns=["transaction_id", "date", "raw_description", "category", "amount", "account_type"])
    expected_gt = compute_ground_truth(df)

    if args.check:
        stored = json.loads(OUT_JSON.read_text())
        reread = pd.read_csv(OUT_CSV)
        recomputed = compute_ground_truth(reread)
        assert stored["ground_truth"] == recomputed, "frozen ground truths drifted from the CSV on disk"
        assert recomputed == expected_gt, "regenerating with SEED no longer reproduces the stored values"
        print(f"OK: {OUT_CSV.name} ({stored['n_rows']} rows) and golden suite are consistent (seed={SEED})")
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, float_format="%.2f")

    reread = pd.read_csv(OUT_CSV)
    gt = compute_ground_truth(reread)
    assert gt == expected_gt, "CSV round-trip changed ground truth values"

    suite = {
        "seed": SEED,
        "dataset": "data/transactions.csv",
        "n_rows": len(reread),
        "date_range": [str(reread["date"].min()), str(reread["date"].max())],
        "ground_truth": gt,
        "cases": build_cases(gt),
    }
    OUT_JSON.write_text(json.dumps(suite, indent=2) + "\n")

    print(f"wrote {OUT_CSV} ({len(reread)} rows, {reread['date'].min()} .. {reread['date'].max()})")
    print(f"wrote {OUT_JSON} ({len(suite['cases'])} golden cases)")
    print(json.dumps(gt, indent=2))


if __name__ == "__main__":
    main()
