"""The durable agent: rotor processes, tools, and the wire protocol.

There is no workflow registry, sandbox policy, or deterministic RNG/clock
here anymore. Handlers are plain async Python; durability is per-message
(one activation = one atomic commit), so nothing in this package needs to
be replay-safe.
"""
