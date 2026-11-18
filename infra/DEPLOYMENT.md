# Deploying the Workshop Gateway — Complete Runbook

Step-by-step instructions for standing up the LiteLLM proxy gateway used by this
workshop, including **every gotcha we hit while doing it** (each one cost real
debugging time — they are marked ⚠️). Written for the next instructor running
this workshop, or anyone building the same stack: LiteLLM proxy on Fly.io
(Tokyo) + Supabase Postgres + seat keys + a public GitHub repo of the notebook.

**Read Phase 0 before spending any money or pasting any secret.**

---

## Architecture

```
 Participants (Colab / Codespaces / local)
        │  seat key (sk-…, $2 cap, 15 RPM, 7d expiry)
        ▼
 LiteLLM gateway — Fly.io app "fmlds-eval-workshop-gateway"
        │  https://<app>.fly.dev/v1   (region nrt, Tokyo)
        │        │
        │        ├──> OpenAI api.openai.com   (project key, hard spend limit)
        │        └──> Supabase Postgres       (keys, budgets, spend logs)
        │              aws-0-ap-northeast-1.pooler.supabase.com  (Tokyo)
        ▼
 Instructor: /key/generate via LITELLM_MASTER_KEY → seat_keys.csv →
 view-only Google Sheet → desk cards
```

Both the gateway and the database live in Tokyo, ~1–2 ms apart. Participants
only ever see the gateway URL and their own seat key.

## Cost

| Item | Cost | Notes |
|---|---|---|
| Supabase Free (Postgres) | **$0** | Tokyo region, 500 MB — LiteLLM tables are < 1 MB for one workshop |
| Fly machine (shared-1x, 1 GB, always-on) | ~$5–8/mo | `auto_stop_machines = off` so no cold starts on workshop day |
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

**OpenAI key:** create a *dedicated OpenAI Project* and generate the API key
inside it, with a **hard spend limit (~$60)** on the project. This gives caps
at two independent layers — OpenAI project limit (room-wide) and LiteLLM
per-key budget ($2/seat) — so no single misconfiguration can drain an account.

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

# 3. Smoke-test one key (replace <KEY> with a value from the CSV)
curl https://fmlds-eval-workshop-gateway.fly.dev/v1/chat/completions \
  -H "Authorization: Bearer <KEY>" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"ping"}],"max_tokens":5}'

# 4. All 50
LITELLM_PROXY_URL=https://fmlds-eval-workshop-gateway.fly.dev \
LITELLM_MASTER_KEY=sk-... \
  .venv/bin/python scripts/make_seat_keys.py
```

Paste the CSV rows into the **view-only** Google Sheet; desk cards
`Seat 01`–`Seat 50` carry the sheet's short URL. The CSV is gitignored
(`seat_keys*.csv`) — never commit, never screenshot it.

Every seat key carries three self-destruct axes if leaked: **$2 budget,
15 RPM, 7-day expiry**. Reserve pool for the emergency kit:
`make_seat_keys.py --seats 5 --budget 5`.

The LiteLLM admin UI at `/ui` is gated by the master key — never share it,
not even with TAs; that's what seat keys are for.

---

## Phase 5 — Workshop day

- [ ] **Day before:** unpause the Supabase project, `fly status -c infra/fly.toml`,
      one live smoke test, charge the 5 mobile hotspots
- [ ] Desk cards printed with seat numbers + key-sheet URL
- [ ] View-only Google Sheet shared (link, view-only — not "anyone can edit")
- [ ] Projector: instructor notebook with `WORKSHOP_MODE=gateway`

---

## Phase 6 — Teardown (zero residual cost & secrets)

In this order:

1. Delete the Google Sheet (kills the shared key surface)
2. `rm infra/seat_keys*.csv` (all copies: pilot + production)
3. `fly apps destroy fmlds-eval-workshop-gateway` (+ `fly apps list` to confirm)
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
| Fly billing demands tax ID | "Business" account type selected | Individual → field becomes optional |
| Can't find 2FA on Fly | it's per-user | avatar → Account → 2FA |

## Security invariants (the short version)

- **Intentionally public:** gateway URL, model allowlist, dataset, the repo itself.
- **Never public:** master key, seat keys, OpenAI key, `DATABASE_URL`.
- Secrets only ever live in: Fly secrets (gateway), env/Colab secrets
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
