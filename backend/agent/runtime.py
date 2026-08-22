"""The app's rotor client — the control plane the FastAPI surface talks to.

One process-wide client. Backends are discovered from the environment
(``DATABASE_URL``/``POSTGRES_URL`` for the store, sqlite under ``.rotor/``
in dev; the live channel follows the same discovery), which replaces the
workflow-data/streams-dir environment plumbing.
"""

from __future__ import annotations

import rotor

_client: rotor.Client | None = None


def client() -> rotor.Client:
    global _client
    if _client is None:
        _client = rotor.Client()
    return _client


def set_client(instance: rotor.Client) -> None:
    """Test seam: point the app at a LocalRuntime's client."""
    global _client
    _client = instance
