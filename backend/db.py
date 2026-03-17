"""
Database connection pool management for Neon PostgreSQL.

Uses asyncpg for fast async database access. The pool is lazily created
on first use and shared across all requests.
"""

from __future__ import annotations

import os

import asyncpg

pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    """Get or create the shared connection pool."""
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(dsn=os.environ["DATABASE_URL"])
    return pool


async def close_pool() -> None:
    """Close the connection pool (for graceful shutdown)."""
    global pool
    if pool is not None:
        await pool.close()
        pool = None
