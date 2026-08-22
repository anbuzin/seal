"""Worker entrypoint for the durable agent.

The worker is the only place handlers run; the FastAPI service is a pure
control plane (start/send/resolve/observe). Locally, ``python -m app.worker``
serves a polling worker; on Vercel the same object is driven by the queue
subscriber declared in ``pyproject.toml`` (rotor auto-detects the platform).

Gone from the workflow port: the ``Workflows()`` registry import dance, the
``WORKFLOW_LOCAL_DATA_DIR``/``SEAL_STREAMS_DIR`` env plumbing, and the
per-step HTTP access-log suppression (there is no HTTP POST per step
anymore — a worker leases a process and drains its mailbox).
"""

from __future__ import annotations

import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

from rotor import Worker  # noqa: E402

from agent import telemetry  # noqa: E402
from agent.processes import AgentBase, Session, Subagent  # noqa: E402
from agent.tools import run_tool  # noqa: E402

telemetry.install("seal-agent")

worker = Worker()
worker.register(Session)
worker.register(Subagent)
worker.register(run_tool)

_ = AgentBase  # the shared base is never registered: it is never spawned


if __name__ == "__main__":
    import asyncio

    asyncio.run(worker.serve())
