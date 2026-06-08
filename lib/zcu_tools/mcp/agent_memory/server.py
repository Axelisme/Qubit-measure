#!/usr/bin/env python
"""MCP server for the agent lab notebook (records + solutions).

A stdio JSON-RPC 2.0 server (launched per ``.mcp.json``). Unlike the GUI bridges
there is no live process to forward to: the tools dispatch *in-process* to a
``MemoryStore`` that does file CRUD over ``agent_memory/<namespace>/``. So there is
no socket, no connect/launch lifecycle, no version guard — just the stdio protocol
loop and the seven memory tools.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict

# Launched standalone (``python .../server.py``), so add the repo ``lib`` dir to
# sys.path for the absolute imports below.
# lib/zcu_tools/mcp/agent_memory/server.py -> lib
_LIB_DIR = Path(__file__).resolve().parents[3]
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))

# Fast-fail preflight: the store needs PyYAML. Surface an actionable instruction on
# stderr (stdout is the JSON-RPC channel and must stay clean).
if importlib.util.find_spec("yaml") is None:
    sys.stderr.write(
        "agent-memory MCP server requires PyYAML. Rebuild the environment with:\n"
        "    uv sync --extra client\n"
    )
    raise SystemExit(1)

# The shared MCP stdio plumbing lives in zcu_tools.mcp.core.bridge. It consumes the
# wire-spec primitives from zcu_tools.gui.remote — mcp is a consumer of that shared
# remote layer, not a leaf.
from zcu_tools.mcp.agent_memory.method_specs import METHOD_SPECS  # noqa: E402
from zcu_tools.mcp.agent_memory.store import MemoryStore  # noqa: E402
from zcu_tools.mcp.core.bridge import (  # noqa: E402
    McpServerConfig,
    assemble_tools,
    generate_tools,
    run_stdio_loop,
)

_SERVER_INSTRUCTIONS = """\
A persistent lab notebook for the measuring agent. Two kinds of entry:
  - records  (episodic): what happened in one measurement — keyed by chip/qub/date.
  - solutions (semantic): context-free problem -> fix, reusable across qubits.

Discipline (no hook enforces this — it is on you):
  - BEFORE an experiment: memory_recall(chip, qub, exp_type) — see what you did
    before on this qubit and the known gotchas for this experiment.
  - Hit a problem? memory_search(query=<symptom>) before improvising.
  - After a measurement concludes / a problem is solved / the user gives feedback:
    1. memory_record(...) the episode (this qubit, date, the numbers, the surprise).
    2. memory_search the symptom; if a matching solution exists ->
       memory_update_solution (add this record id to seen_in — that promotes it to
       'confirmed' at >=2); else -> memory_add_solution (starts 'provisional').
    Solutions hold the reusable RULE; records hold the instance and the numbers.
  - A solution proven wrong -> memory_delete it (or downgrade via update).

Writing rules:
  - Solutions must be self-contained: NO source-file / function references (you can
    not read code). cfg knob names (post_delay, gain, reps, freq) are fine.
  - Do not record routine, no-surprise runs — numeric calibration values already
    live in the GUI's MetaDict, not here.

ids are namespace-relative paths without .md (e.g.
'records/Q5_2D/Q1/2026-06-08-reset-bath', 'solutions/reset/bath/low-contrast').
A failed call raises with an actionable message; reads are idempotent.
"""

# No GUI subprocess / socket here (tools dispatch in-process), so this needs only
# the base McpServerConfig — none of the MCPBridgeConfig launch fields.
_CONFIG = McpServerConfig(
    tool_prefix="memory_",
    server_display_name="agent-memory",
    server_instructions=_SERVER_INSTRUCTIONS,
)


def build_dispatch(store: MemoryStore) -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    """Map each wire method to a ``MemoryStore`` call. Keys must match METHOD_SPECS."""
    return {
        "recall": lambda p: store.recall(**p),
        "search": lambda p: store.search(**p),
        "get": lambda p: store.get(**p),
        "record": lambda p: store.record(**p),
        "add_solution": lambda p: store.add_solution(**p),
        "update_solution": lambda p: store.update_solution(**p),
        "delete": lambda p: store.delete(**p),
    }


def build_tools(store: MemoryStore) -> Dict[str, Dict[str, Any]]:
    """Generate the MCP tool table, dispatching in-process to ``store``."""
    dispatch = build_dispatch(store)

    def local_send(
        method: str, params: Dict[str, Any], timeout_seconds: float = 30.0
    ) -> Any:
        del timeout_seconds  # no transport; calls are local + synchronous
        handler = dispatch.get(method)
        if handler is None:
            raise RuntimeError(f"unknown agent-memory method {method!r}")
        return handler(params)

    return assemble_tools(
        generate_tools(_CONFIG, METHOD_SPECS, frozenset(), local_send), {}, frozenset()
    )


def _memory_root() -> Path:
    """Where the notebook lives: ``$ZCU_AGENT_MEMORY_DIR`` or ``<repo>/agent_memory``."""
    env = os.environ.get("ZCU_AGENT_MEMORY_DIR")
    if env:
        return Path(env)
    # lib/zcu_tools/mcp/agent_memory/server.py -> repo root
    return Path(__file__).resolve().parents[4] / "agent_memory"


def main() -> None:
    store = MemoryStore(root=_memory_root(), namespace="main_gui")
    run_stdio_loop(_CONFIG, build_tools(store))


if __name__ == "__main__":
    main()
