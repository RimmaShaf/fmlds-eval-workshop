# Deploying the Workshop Gateway — Complete Runbook

Step-by-step instructions for standing up the LiteLLM proxy gateway used by this
workshop, including **every gotcha we hit while doing it** (each one cost real
debugging time — they are marked ⚠️). Written for the next instructor running
this workshop, or anyone building the same stack: LiteLLM proxy on Fly.io
(Tokyo) + Supabase Postgres + seat keys + a public GitHub repo of the notebook.

**Read Phase 0 before spending any money or pasting any secret.**

> **Field-tested 2026-11-18.** This runbook was executed end-to-end (Supabase
> Tokyo + Fly `nrt`, one machine, health check passing, pilot seat keys issued).
> Every ⚠️ marks a real failure that occurred during that run, with the fix that
> worked. Expect the whole thing in ~2 hours, most of it human account setup.

## Division of labor — what the human does vs what the agent does

The split follows one principle: **the human owns money, accounts, and
credential creation; the agent owns everything mechanical in between — and
never echoes a secret into the chat transcript.**

| Task | Human | Agent (OpenCode) |
|---|---|---|
| Create accounts, enable 2FA, billing forms, tax-ID | ✅ all of it | — |
| Supabase project, Tokyo region, DB password | ✅ | — |
| Generate master key, store in password manager | ✅ (generate + store) | can run `openssl rand -hex 24` but should not be the custodian |
| Buy OpenAI API credits, set project spend cap | ✅ (money) | — |
| `fly secrets set` values | ✅ paste, or drop into gitignored `.env` | runs the command; reads key from `.env` if asked; never prints values |
| Edit `fly.toml`, `.dockerignore`, docs, notebook builder | — | ✅ |
| `fly launch` / `deploy` / `logs` / `status` / `scale count` | — | ✅ (flyctl auth is shared on the machine) |
| Generate seat keys (`make_seat_keys.py`) | provides `LITELLM_MASTER_KEY` via env | ✅ runs script |
| Deploy the Phoenix board + mint system API keys (Phase 4b) | — | ✅ runs the whole chain (secrets stay in memory / Fly) |
| Set the board admin UI password, distribute board values to desk cards | ✅ (credentials + distribution) | — |
| Smoke-test a seat key | — | ✅ reads key from gitignored CSV into a shell variable — value never appears in output |
| Verify Prisma/health/one-machine state | — | ✅ |
| Repo edits, Colab badge, eval gate, gitleaks scan, `git push` | approves repo name + visibility | ✅ |
| Teardown decisions (destroy app, delete DB, revoke keys) | ✅ decides | ✅ runs the commands |

Handoff pattern that worked: human does the clicks and payments in browser
tabs, tells the agent "done", agent verifies from the terminal (curl / fly
logs / config show) and moves to the next phase.

---

## Architecture

```
 Participants (Colab / Codespaces / local)
        │  seat key (sk-…, $2 cap, 15 RPM, 7d expiry)
        ▼
 LiteLLM gateway — Fly.io app "fmlds-eval-workshop-gateway"
        │  https://fmlds-eval-workshop-gateway.fly.dev/v1   (region nrt, Tokyo)
        │        │
        │        ├──> OpenAI api.openai.com   (project key, hard spend limit)
        │        └──> Supabase Postgres       (keys, budgets, spend logs)
        │              aws-0-ap-northeast-1.pooler.supabase.com  (Tokyo)
        │        │
        │        └──> Phoenix board — Fly.io app "fmlds-eval-workshop-phoenix"
        │              server-side trace capture, every request, tagged per seat
         ▼                (participants' full span trees export directly from
  Instructor: /key/generate via LITELLM_MASTER_KEY → seat_keys.csv →     notebook 2)
 view-only Google Sheet → desk cards
```

Both the gateway and the database live in Tokyo, ~1–2 ms apart. Participants
only ever see the gateway URL and their own seat key.

## Cost

| Item | Cost | Notes |
|---|---|---|
| Supabase Free (Postgres) | **$0** | Tokyo region, 500 MB — LiteLLM tables are < 1 MB for one workshop |
| Fly machine (shared-1x, 1 GB, always-on) — gateway | ~$5–8/mo | `auto_stop_machines = off` so no cold starts on workshop day |
| Fly machine (shared-1x, 1 GB, always-on) — Phoenix board | ~$5–8/mo | central observability; SQLite on a 3 GB volume, ~30k spans/day is trivial |
| Fly Managed Postgres (if chosen instead of Supabase) | $38/mo + $0.28/GB | not worth it for a 1-day workshop |
| OpenAI usage | ~$5–25 actual, $100 worst case | 50 seats × $2 hard cap |

Destroy everything after the workshop (Phase 6) and residual cost is $0.

---

## Phase 0 — Accounts and hardening (do this first)

Four accounts are in the blast radius; enable 2FA on all of them:

| Account | Why it matters | Where 2FA lives |
|---|---|---|
| **GitHub** | owns the repo **and** (if used) is your Supabase login — single point of failure for two services | Settings → Password and authentication |
| **Supabase** | holds the DB behind seat keys and budgets | Account → Security |
| **Fly.io** | the most sensitive: `fly ssh console` can read every gateway secret | Dashboard → avatar → Account → 2FA |
| **OpenAI** | controls the actual money | Platform settings → Your profile → Multi-factor authentication |

⚠️ **Fly's 2FA is easy to miss** — it's under *Account settings → 2FA*, not
under a "Security" tab in an org page, and it's per-user, not per-org.

⚠️ **If Fly's billing form seems to require a tax ID** you almost certainly
selected "Business" by accident. For an individual it's optional — leave it
blank. If you're in Japan, never enter your 個人番号 (My Number) there; that's
for Japanese tax administration, not foreign SaaS billing. Business accounts
use the 13-digit 法人番号 (corporate number).

**OpenAI key — two separate knobs, both yours to set:**

1. **Credits (the wallet).** platform.openai.com → Settings → **Billing → Buy
   credits** (~$20 covers the workshop's $5–25 expected spend; worst case $100).
   ⚠️ **Prepaid and separate from everything else**: a ChatGPT (Plus) subscription
   includes **zero** API usage. If the gateway returns
   `429 — You have no credits remaining`, this knob is the fix — nothing about
   the gateway, keys, or code is wrong.
2. **Project spend cap (the ceiling).** Create a *dedicated OpenAI Project*,
   generate the API key inside it, and set a **hard spend limit (~$60)** on the
   project (Settings → Rate limits / project settings). This is a *different
   setting* from buying credits — credits are what you've paid, the cap is the
   automatic stop.

This gives caps at two independent layers — OpenAI project limit (room-wide)
and LiteLLM per-key budget ($2/seat) — so no single misconfiguration can drain
an account.

**Secrets inventory (exactly four — put them in a password manager now):**

| Secret | What it is | Generated |
|---|---|---|
| `LITELLM_MASTER_KEY` | gateway admin credential, `sk-` prefix required | `openssl rand -hex 24` |
| `OPENAI_API_KEY` | the project-scoped key above | platform.openai.com |
| `DATABASE_URL` | Supabase session-pooler URI | Supabase → Connect |
| seat keys | 50 × per-participant virtual keys | Phase 4, this guide |

---

## Phase 1 — Supabase (free, Tokyo)

1. Sign in at supabase.com. **"Continue with GitHub" is fine** — it's
   identity-only OAuth (email/avatar); Supabase gets no repo access. Your
   GitHub 2FA now covers this login too.
2. **New project**: name `fmlds-eval-workshop`, strong DB password (see ⚠️
   below), region **Asia Pacific → Tokyo (Northeast Asia)**.

⚠️ **Region is two dropdown steps**: pick the *Asia Pacific* group first, then
**Tokyo** specifically (not Singapore/Seoul). Your connection host must contain
`ap-northeast-1`, e.g. `aws-0-ap-northeast-1.pooler.supabase.com`. The gateway
(`nrt`) and DB must sit next to each other or every LLM call pays ~70 ms extra.

⚠️ **Decline "Enable Data API" / "API access to new tables"** (leave both off).
That feature auto-exposes every table through a public REST endpoint guarded
only by RLS policies — Prisma's tables get none. LiteLLM uses the raw Postgres
connection string only; the Data API is an idle public attack surface.

⚠️ **Password percent-encoding**: the DB password is embedded *inside* the
connection URI, so special characters must be percent-encoded (`@`→`%40`,
`#`→`%23`, `:`→`%3A`, `/`→`%2F`, `%`→`%25`). Plain `-`, `.`, `_`, `~`, letters
and digits are safe as-is. **Simplest fix: set the DB password to
`openssl rand -hex 24` output** — hex can never need encoding. (Watch for
en/em-dashes `–` `—` from password generators; they are *not* the ASCII `-`.)

3. Copy the connection string: **Connect button (top bar) → Session pooler →
   URI** (also at Project Settings → Database → Connection string).

⚠️ **Use the Session pooler, port 5432** — NOT:
   - port **6543** (transaction mode) — breaks Prisma's migrations on LiteLLM startup
   - the direct `db.<ref>.supabase.co` host — IPv6-only, and IPv6 egress from Fly to Supabase is unreliable

   Expected shape:
   `postgresql://postgres.<ref>:<password>@aws-0-ap-northeast-1.pooler.supabase.com:5432/postgres`

⚠️ **Free projects pause after ~7 days of inactivity** (data survives). Set a
calendar reminder to open the dashboard and unpause the day before the
workshop. Nothing else needs configuring — LiteLLM's Prisma migration creates
all tables on first boot.

---

## Phase 2 — Fly.io app

### One-time setup

```bash
brew install flyctl
fly auth login        # then do the 2FA setup from Phase 0, and billing:
                      # Individual → tax ID blank → card on file
```

⚠️ **In the dashboard, create an App — not a Sprite.** Sprites are Fly's
agent-VM product (Linux boxes for Cursor etc.), a completely different thing.
In fact, **don't click "New app" in the dashboard at all** — `fly launch`
creates the app from `infra/fly.toml` with the right name, region, and machine
size; a manually created empty app just confuses things.

⚠️ **Always run fly commands from the repo root** with `-c infra/fly.toml`.

### Create the app (no deploy yet)

```bash
fly launch --config infra/fly.toml --no-deploy
```

Answer the prompts: accept the app name (`fmlds-eval-workshop-gateway`, add a
suffix if taken) / your personal org / **no** database (Supabase is ours) /
**no** deploy yet.

⚠️ **When fly offers to create a `.dockerignore` — say yes.** Without it, every
deploy uploads the entire repo (`.git`, `.venv` ≈ hundreds of MB, and your
`.env`!) to the remote builder. The repo's `.dockerignore` (already committed)
is the cleaned-up version; it must ignore local junk but **never `infra/`**
(the image copies `infra/litellm_config.yaml`).

⚠️ **The `infra/infra/Dockerfile` error.** Newer flyctl resolves build paths in
`fly.toml` **relative to the fly.toml's directory**, not your cwd. So
`dockerfile = "infra/Dockerfile"` becomes `infra/infra/Dockerfile` → not found.
The committed `infra/fly.toml` already carries the fix:

```toml
[build]
  context = ".."        # build context = repo root
  dockerfile = "Dockerfile"   # the file sits next to fly.toml
```

### Set the three secrets

```bash
fly secrets set -c infra/fly.toml \
  LITELLM_MASTER_KEY="sk-<openssl-hex>" \
  OPENAI_API_KEY="sk-<project-key>" \
  DATABASE_URL="postgresql://postgres.<ref>:<hex>@aws-0-ap-northeast-1.pooler.supabase.com:5432/postgres"
```

⚠️ **Quote every value** — `DATABASE_URL` contains `://` and friends that zsh
will otherwise mangle. Verify with `fly secrets list -c infra/fly.toml` (three
names, masked values).

⚠️ **No interactive mode**: `fly secrets set NAME` (no `=value`) fails with
"must be in the format NAME=VALUE", and there is **no `-i` flag**. Values pasted
this way land in zsh history — scrub afterwards with `history` → `history -d
<n>`, or edit `~/.zsh_history`.

These are the *only* three secrets. `infra/fly.toml` and
`infra/litellm_config.yaml` reference them via `os.environ/...`, so **nothing
sensitive is ever committed** — verify with
`git grep -E "sk-|postgres(ql)?://" -- . ':!.venv'` before making the repo public.

---

## Phase 3 — First deploy and verification

```bash
fly deploy -c infra/fly.toml
fly logs -c infra/fly.toml
```

Success looks like, in order: container starts → config loaded → **`Prisma DB
Connected`** (this is where Supabase either works or doesn't) → uvicorn
listening on `0.0.0.0:8080` → health checks passing.

After ~90 s confirm with `fly status -c infra/fly.toml`: one machine, `started`,
and `CHECKS: 1 total, 1 passing`. Then from anywhere:

```bash
curl https://fmlds-eval-workshop-gateway.fly.dev/health/liveliness
# → "I'm alive!"
```

If `Prisma DB Connected` never appears, it's almost always the `DATABASE_URL`:
wrong pooler port (6543), unencoded password character, or a paused Supabase
project.

---

## Phase 4 — Seat keys

```bash
# 1. Preview the exact payloads — touches nothing
.venv/bin/python scripts/make_seat_keys.py --dry-run

# 2. Pilot pair against the live gateway
LITELLM_PROXY_URL=https://fmlds-eval-workshop-gateway.fly.dev \
LITELLM_MASTER_KEY=sk-... \
  .venv/bin/python scripts/make_seat_keys.py --seats 2 --out infra/seat_keys_pilot.csv

# 3. Smoke-test seat 01 — the key travels CSV → shell variable → header,
#    and is never printed to screen or transcript
KEY=$(awk -F, 'NR==2 {print $3}' infra/seat_keys_pilot.csv) && \
curl https://fmlds-eval-workshop-gateway.fly.dev/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"ping"}],"max_tokens":5}'
# expected: a JSON chat completion. A 429 "no credits remaining" → Phase 0,
# credits knob (see troubleshooting table). Any auth error → the key didn't
# make it into the DB (check Prisma line in fly logs).

# 4. All 50
LITELLM_PROXY_URL=https://fmlds-eval-workshop-gateway.fly.dev \
LITELLM_MASTER_KEY=sk-... \
  .venv/bin/python scripts/make_seat_keys.py
```

Paste the CSV rows into the **view-only** Google Sheet; desk cards
`Seat 01`–`Seat 50` carry the sheet's short URL. The CSV is gitignored
(`seat_keys*.csv`) — never commit, never screenshot it.

Every seat key carries three self-destruct axes if leaked: **$2 budget,
15 RPM, 7-day expiry**. RPM = requests per minute, enforced per key by
LiteLLM server-side — it's the anti-runaway-loop brake: a stuck tool-calling
agent can't exceed 15 LLM requests in any window (request #16 gets a `429`
until it frees up), and one greedy notebook can't starve the other 49 seats.
Tune at creation with `--rpm`; instructor demo/master-pool keys can take ~30.
Reserve pool for the emergency kit: `make_seat_keys.py --seats 5 --budget 5`.

The LiteLLM admin UI at `/ui` is gated by the master key — never share it,
not even with TAs; that's what seat keys are for.

---

## Phase 4b — The Phoenix board (room-wide observability)

A second Fly app, **fmlds-eval-workshop-phoenix** (Tokyo, next to the gateway),
collecting traces from two directions:

1. **Server-side (automatic):** the gateway's `arize_phoenix` callback ships
   every request — full trace tree (HTTP → auth → LLM call → DB writes), tagged
   with `litellm.metadata.user_api_key_alias` = the seat alias. Zero participant
   setup; this is what the projector shows.
2. **Client-side (automatic):** notebook 2 exports participant AGENT/LLM/CHAIN/TOOL
   spans to `00-Part-2-tracing-set-up`; notebook 3 exports evaluation traces and
   grades to `01-Part-3-evaluations`. Gateway spans remain in project `gateway`.

### Deploy (agent can do all of this)

```bash
fly launch --config infra/phoenix.toml --no-deploy
fly volumes create phoenix_data --app fmlds-eval-workshop-phoenix --region nrt --size 3 --yes
fly deploy -c infra/phoenix.toml
curl https://fmlds-eval-workshop-phoenix.fly.dev/healthz     # → "OK"
```

⚠️ **Use the lightweight SDK in notebooks:** Notebooks 2 and 3 install the
independently versioned `arize-phoenix-otel` and `arize-phoenix-client` packages,
not a local server. The board remains Phoenix **22.0.0**; the SDKs send OTLP/HTTP directly.

⚠️ **Auth is ON by default** (the Docker image binds a non-loopback host, and
auth follows the bind address). Everything returns `401 Invalid token` until
you mint a system API key. Keep it that way — it's the right posture for a
public URL.

### Mint the system API keys (the PHOENIX_ADMIN_SECRET dance)

Fly secrets are **write-only** — you cannot read a value back, so the whole
chain must run in ONE process: generate the admin secret → set it → use it →
remove it. Do not paste the key values anywhere; they go straight into Fly
secrets / the gitignored key file.

```python
# /opt/homebrew/bin/python3.12 (any python works; run from the repo root)
import json, secrets, subprocess, time, urllib.request
BOARD = "https://fmlds-eval-workshop-phoenix.fly.dev"

def fly(*a): return subprocess.run(["fly", *a], capture_output=True, text=True, check=True).stdout

admin = "admin-" + secrets.token_hex(24)
fly("secrets", "set", "-c", "infra/phoenix.toml", f"PHOENIX_ADMIN_SECRET={admin}")
# ...wait for healthz to return OK (the app restarts), then:
req = urllib.request.Request(
    f"{BOARD}/v1/system/api_keys",
    data=json.dumps({"data": {"name": "workshop-board-2"}}).encode(),   # ⚠️ body wraps in "data"
    headers={"Authorization": f"Bearer {admin}", "Content-Type": "application/json"},
    method="POST")
key = json.loads(urllib.request.urlopen(req, timeout=60).read())["data"]["key"]  # ⚠️ field is "key"
fly("secrets", "set", "-c", "infra/fly.toml", f"PHOENIX_API_KEY={key}")          # gateway → board
fly("secrets", "unset", "-c", "infra/phoenix.toml", "PHOENIX_ADMIN_SECRET")      # remove it again
```

Gotchas packed in there: the request body must be `{"data": {...}}` (422
otherwise); the response field is `key` (not `api_key`); the key is shown
**once**; `PHOENIX_SECRET` (JWT signing) must already be set or the admin
secret is rejected. Mint a second key the same way for participants and save
it to `infra/phoenix_board.key` (gitignored via `*.key`). For rehearsal, copy
that value into the Colab `PHOENIX_API_KEY` secret; for the live workshop, put
it on the desk cards.

### Wire the gateway (already committed)

- `infra/litellm_config.yaml`: `litellm_settings.callbacks: ["arize_phoenix"]`
- `infra/fly.toml` `[env]`: `LITELLM_OTEL_V2 = "true"`,
  `PHOENIX_COLLECTOR_HTTP_ENDPOINT = "https://fmlds-eval-workshop-phoenix.fly.dev/v1/traces"`,
  `PHOENIX_PROJECT_NAME = "gateway"`
- `PHOENIX_API_KEY` as a gateway Fly secret (from the mint above)
- `fly deploy -c infra/fly.toml`

### Verify (5 minutes, do this before the workshop)

1. Fire one pilot-key request through the gateway; within ~10 s the board's
   `gateway` project shows a trace whose spans carry
   `litellm.metadata.user_api_key_alias: "seat-01"`.
2. Run notebook 2 with the desk-card `PHOENIX_API_KEY`; traces should appear in
   `00-Part-2-tracing-set-up`.
3. Run Notebook 3's golden dataset upload, baseline, and prompt variant; both named
   experiments and their evaluator score columns should appear in the dataset's Experiments
   tab. Confirm the linked task traces appear and gateway traces remain in `gateway`.
4. Log into the board UI and confirm the projector can select each of the three projects.

### Human tasks for this phase

- **Set the admin password.** First UI login is `admin@localhost` / `admin`,
  and Phoenix prompts you to change it. Choose a real password, store it in
  your password manager — this is the instructor's board UI credential.
- **Desk cards:** each card carries (a) the key-sheet short URL and (b) the
  `PHOENIX_API_KEY` required by notebooks 2–3 (the participant key from
  `infra/phoenix_board.key`). Both notebooks contain the public board URL.

Notebooks 2 and 3 export spans directly. Notebook 3 also creates dataset experiments and
evaluator results through the Phoenix client; there is no separate publish step or ID
remapping. The OpenTelemetry SDK generates distinct trace/span IDs for each run.

---

## Phase 4c — Colab rehearsal (do this BEFORE the workshop; ~30–40 min live, ~5 min mock)

The only thing that cannot be verified from the repo itself is the real Colab
experience. Run all three notebooks end-to-end as a participant would.

**Prep (2 min):** a pilot seat key and the participant board key in
`infra/phoenix_board.key`. For rehearsal, set that value as the Colab secret
`PHOENIX_API_KEY` (🔑 panel; enable **Notebook access** for Notebooks 2–3), or
paste it at the hidden prompt. It is separate from `LITELLM_API_KEY`.

**Fast structural pass first (optional, 5 min):** set a `WORKSHOP_MODE=mock`
secret, run all three notebooks top to bottom. Validates badges, bootstrap,
data loading, pip cell, suite mechanics — zero API spend. Then delete the mock
secret and do the live pass.

- [ ] **Notebook 1 (10 min, live):** open via the README badge → config cell
      prints `mode=gateway` → deps auto-install → health check OK → data loads
      **2,500 rows from the raw GitHub URL** → compare the agent's answers with
      the signed-amount, fragmented-merchant, and missing-location examples.
- [ ] **Notebook 2 (10 min, live):** `%pip` installs lightweight Phoenix OTEL,
      client, and OpenInference packages before imports → bootstrap fetches
      `utils.py` and the board API key → register the board's OTLP endpoint →
      missing-location and refund-netting runs appear immediately in project
      `00-Part-2-tracing-set-up` with nested AGENT/CHAIN/LLM/TOOL spans and I/O
- [ ] **Notebook 3 (live):** bootstrap → upload the 11-case golden dataset → run the
      baseline and prompt-variant experiments → compare per-evaluator and overall scores
      in the dataset's Experiments tab → inspect a low-scoring example and its trace.
      Results vary across live model runs; do not require a specific pass count. Model
      comparison is opt-in. The deterministic mock regression gate remains in CI.

**If something breaks, likely causes:**
| Symptom | Fix |
|---|---|
| Badge/notebook 404 on GitHub | filename typo in the badge URL — check the three badge links in README |
| Data load falls back to local / fails | raw.githubusercontent propagation lag — wait 1 min, re-run; verify repo is public |
| Colab `%pip` reports dependency resolver warnings | pip detected differences among preinstalled packages; this tracing setup installs lightweight OTEL/client SDKs, not the Phoenix server | continue if the SDK imports; if instrumentation fails, restart the runtime and rerun from the top |
| Notebook 2 board export returns 401 | `PHOENIX_API_KEY` missing/typo (the board has auth ON) | add the participant key as a Colab secret or enter it at the prompt |
| 429s during the suite | 15 RPM seat cap doing its job — wait ~30 s and re-run the cell |
| Notebooks 2/3 still prompt although a Colab secret is set | Colab secrets are **not** environment variables, and "Notebook access" is per-notebook; notebook 1's paste prompt is by design | for the optional zero-paste setup, switch the toggle on for *that* notebook — `utils.get_secret` reads it via `google.colab.userdata` |

---

## Phase 5 — Workshop day

- [ ] **Day before:** unpause the Supabase project, `fly status -c infra/fly.toml`,
      one live smoke test, charge the 5 mobile hotspots
- [ ] Board verification checklist (Phase 4b) green — projector shows the board
- [ ] Desk cards printed with seat numbers, key-sheet URL, and the board
      `PHOENIX_API_KEY`
- [ ] View-only Google Sheet shared (link, view-only — not "anyone can edit")
- [ ] Projector: instructor notebook using the workshop gateway

---

## Phase 6 — Teardown (zero residual cost & secrets)

In this order:

1. Delete the Google Sheet (kills the shared key surface)
2. `rm infra/seat_keys*.csv infra/phoenix_board.key` (all copies: pilot + production + board key)
3. `fly apps destroy fmlds-eval-workshop-gateway` and
   `fly apps destroy fmlds-eval-workshop-phoenix` (+ `fly apps list` to confirm)
4. Delete the Supabase project
5. Revoke the OpenAI project key
6. (Optional) remove the 2FA you added if these accounts are single-purpose —
   generally keep it.

---

## Troubleshooting — every error we actually hit

| Error / symptom | Cause | Fix |
|---|---|---|
| `dockerfile '…/infra/infra/Dockerfile' not found` | flyctl resolves fly.toml build paths relative to the config's directory | `[build] context = ".."` + `dockerfile = "Dockerfile"` (already committed) |
| `could not parse secrets: must be in the format NAME=VALUE` | `fly secrets set NAME` without a value doesn't prompt | use `NAME="value"` form, quoted |
| `unknown shorthand flag: 'i' in -i` | `fly secrets set` has no interactive flag (docs imply prompting; flyctl doesn't do it) | inline values + scrub zsh history |
| Supabase auth fails though password is "right" | special character in password not percent-encoded in the URI | reset DB password to `openssl rand -hex 24` and re-copy the URI |
| `DB not connected` / `/key/generate` 500 | `DATABASE_URL` missing, or transaction-mode (port 6543) pooler breaks Prisma | session pooler URI, port 5432 |
| Deploy succeeds, notebook can't reach gateway | placeholder URL still in the notebook | Phase 4 of the repo-publish plan (README runbook) |
| `fly status` shows empty `CHECKS`, no health monitoring | `[[http_service.http_checks]]` is **not a valid key** in current fly.toml — flyctl silently drops unknown sections, so it never errored, it just vanished | use `[[http_service.checks]]` with `path`/`method`/`interval`/`timeout`/`grace_period` (committed in `infra/fly.toml`) |
| Health check fails right after deploy, machine shows "unhealthy" | check fired during boot: LiteLLM binds port 8080 only after Prisma prep + migrations (~30–60 s) | `grace_period = "60s"` — already set |
| Two machines running, double the cost | repeated deploys each added a machine instead of replacing | `fly scale count 1 -c infra/fly.toml --yes` |
| Smoke test: `429 — You have no credits remaining` (wrapped by LiteLLM as `throttling_error`) | OpenAI org has zero API credit — the gateway, seat key, and routing are all working; only the wallet is empty | platform.openai.com → Settings → Billing → Buy credits (~$20). ChatGPT Plus doesn't help: subscriptions include no API usage |
| Smoke test: auth error on the seat key | key not persisted — Prisma/DB issue | confirm `Prisma DB Connected` in `fly logs`; check Supabase project isn't paused |
| Fly billing demands tax ID | "Business" account type selected | Individual → field becomes optional |
| Board returns `401 Invalid token` for everything | Phoenix 22 Docker image enables auth by default (non-loopback bind) | expected — mint a system API key via the `PHOENIX_ADMIN_SECRET` chain (Phase 4b); notebook 2 sends it as `PHOENIX_API_KEY` |
| `POST /v1/system/api_keys` → 422 "Field required: data" | request body must wrap in a `data` object | `{"data": {"name": "..."}}` |
| System-key response has no `api_key` field | the response field is `key`, and the value is shown **once** | parse `data.key`; if you lose it, mint a new key (old one can't be retrieved) |
| `phoenix.client.Client(endpoint=...)` → TypeError | that was the old client; the new one takes `base_url` | `Client(base_url=..., api_key=...)` |
| Board project has no spans yet | API key/project name is wrong, or the first run has not completed | confirm `PHOENIX_API_KEY`, `00-Part-2-tracing-set-up` or `01-Part-3-evaluations`, and the endpoint `https://fmlds-eval-workshop-phoenix.fly.dev/v1/traces` |
| Can't find 2FA on Fly | it's per-user | avatar → Account → 2FA |

## Security invariants (the short version)

- **Intentionally public:** gateway URL, board URL, model allowlist, dataset, the repo itself.
- **Never public:** master key, seat keys, OpenAI key, `DATABASE_URL`, Phoenix
  `PHOENIX_API_KEY`s, `infra/phoenix_board.key` (gitignored via `*.key`).
- The Phoenix board has **auth enabled** (Phoenix 22 default); every client —
  the gateway callback and participant notebook — authenticates with a
  system API key. The `PHOENIX_ADMIN_SECRET` used to mint keys is removed from
  the deployment immediately after use.
- Secrets only ever live in: Fly secrets (gateway + board), env/Colab secrets
  (participants), password manager (you).
- Spend is capped at two layers: $2/seat (LiteLLM, needs the DB) + the OpenAI
  project hard limit.
- `fly ssh console` exposes env vars → keep the Fly org single-member.
- The agent REPL blocks `os`/`subprocess`/`requests` imports via AST scan
  (already in the notebook) — defense against participants, not attackers.

## Appendix — participants without a Google account (Tokyo audience)

Colab lets anyone **view** a public notebook, but **running** requires a Google
account (the VM is tied to it). For Japan: personal Google-account ownership is
high (~64% general population, ~85–95% among data scientists/engineers), but
corporate Microsoft-365 laptops frequently block Google sign-in — the real
friction is policy, not account ownership. Offer, in order:

1. **Colab** (primary) — needs Google account to run
2. **GitHub Codespaces** — `.devcontainer/` is already in the repo; 120
   core-hours/month free with any GitHub account; data already on disk
3. **MyBinder** — zero account, public repo; 1–2 GB RAM, 10-min idle, donated
   capacity — a safety net for a handful of seats, not 50
4. **Local** — `pip install -r requirements.txt` + Jupyter/VS Code

Say it in the pre-workshop email: *"Bring a Google or GitHub account; if your
work laptop blocks Google Drive/Colab, use Codespaces (link in README)."*
