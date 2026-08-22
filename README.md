# seal

A personal AI assistant built as a **durable agent**: every conversation is
one durable process, so turns survive restarts, streams can be resumed
mid-generation, and tool calls can park indefinitely waiting for human
approval.

Seal is an example app for the [AI SDK for Python](https://ai-python.dev)
(the `ai` package), built on rotor's `DurableProcess` runtime (`rotorcore`).

The agent (Claude via the AI Gateway) has three tools: `bash`,
`web_fetch`, and `subagent`. Bash runs are gated behind an approval UI
when run by the main agent, but not when run by a subagent. (That is
silly, but this is a demo app.)

## How it works

- **frontend/** — React + Vite chat UI using the AI SDK (`useChat`) and
  [AI Elements](https://elements.ai-sdk.dev). Reconnecting to a session
  re-tails the in-flight stream (`useChat({ resume: true })`).
- **backend/app/** — FastAPI service. `POST /api/chat` starts (or messages)
  the session process and streams the AI SDK UI message protocol; other
  endpoints cover sessions, titles, and private blob attachments. See
  `app/server.py` for the endpoint list.
- **backend/agent/** — the durable agent itself. Each conversation is one
  keyed `Session` process (`processes.py`): every model turn is one atomic
  activation, each tool call runs as a keyed `run_tool` child with its own
  retry policy, approvals are durable hooks the turn parks on (parking an
  idle process costs nothing), and subagents are child processes in the
  same scope. Handlers are plain async Python — there is no replay and no
  determinism contract.
- **Streaming & state** — tokens stream over rotor's live channel with
  spooled write-through, so a reconnect mid-generation replays the current
  turn (`replay_inflight`); the transcript is the process checkpoint, read
  lease-free through a query; approval capabilities ride the durable
  records ledger. Vercel Blob stores attachments when available.

Deployment is two Vercel services (see `vercel.json`): the frontend and the
backend, with the rotor worker driven by the queue subscriber declared in
`backend/pyproject.toml`.

## Development

Prereqs: [uv](https://docs.astral.sh/uv/), [pnpm](https://pnpm.io), and the
[Vercel CLI](https://vercel.com/docs/cli).

```sh
./dev-setup.sh        # sync backend deps (works around a vercel-worker version override)
cd frontend && pnpm install
vercel dev            # serves frontend + backend + worker on :3000
```

Environment: `AI_GATEWAY_API_KEY` (model access), optional `DATABASE_URL`
(Postgres storage), and a blob token for attachments.

### Checks

```sh
make ci               # everything below
make ci-backend       # uv sync, ruff, mypy, ty, pytest
make ci-frontend      # pnpm install, prettier, eslint, tsc, vitest, build
```

### E2E tests

`e2e/` drives a real browser against a running instance:

```sh
cd e2e && pnpm install && pnpm run install-browser
pnpm test             # expects the app at http://localhost:3000
pnpm run test:images  # image latency: time to first image, time to all N
```

`test:images` prompts "draw N pictures of things you find interesting"
(`N=5` by default) and reports when each image actually painted, measured
from the submit click. Timings also land in
`/tmp/seal-e2e-images-summary.json`.

## Deployment

Deploy as a project to Vercel with `vc deploy`. `DATABASE_URL` must
point to a Postgres database, which can most easily be done by
configuring a marketplace integration with Neon or similar.
