"""Agent coordination taskboard — file-backed read/write lock over ``task_plans/taskboard.json``.

Seven MCP tools (``taskboard_`` prefix) provide atomic path-conflict detection,
read/write locking, pending queues with promotion, TTL-based stale-claim reclaim,
and a human-readable markdown view auto-rendered after each mutating operation.

NOTE: The JSON store uses ``fcntl.flock`` for cross-process atomic RMW — Linux only
(POSIX advisory lock).  Windows is not supported.
"""
