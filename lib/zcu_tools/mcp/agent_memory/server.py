#!/usr/bin/env python
"""MCP server for the agent lab notebook (records + troubleshooting + checklists).

A stdio JSON-RPC 2.0 server (launched per ``.mcp.json``). Unlike the GUI bridges
there is no live process to forward to: the tools dispatch *in-process* to a
``MemoryStore`` that does file CRUD over ``agent_memory/<namespace>/``. So there is
no socket, no connect/launch lifecycle, no version guard — just the stdio protocol
loop and the memory tools. Figure copies use ``shutil`` in-process, so the figure
paths the agent passes must be reachable on this same machine (they are: the GUI
writes figures to local temp files on the same host as this server).
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
A persistent, human-readable lab notebook for the measuring agent. Three functions:
  - records      (episodic): one measurement as a FOLDER — record.md (the per-item
                 pass/fail verdict + the numbers) plus copied figures. Keyed by
                 chip/qub/date. IMMUTABLE: never overwritten, never deleted.
  - troubleshooting (semantic): context-free symptom -> fix, reusable across qubits.
  - checklists   (acceptance): one acceptance list per experiment type, used to
                 self-grade a run before recording it.

Discipline (no hook enforces this — it is on you):
  - BEFORE an experiment: memory_recall(chip, qub, exp_type) — it returns three
    buckets: the acceptance 'checklist' for this experiment, the 'gotchas'
    (solutions) for it, and the 'recent' records for this chip/qub.
  - Hit a problem? memory_search(query=<symptom>) before improvising; if a fix
    exists, follow it.
  - At the ACCEPTANCE GATE (after analyze): grade the run against the recall'd
    checklist item by item, with evidence (numbers, what the figure shows), then
    1. memory_record_measurement(...) the episode — pass decision=accept|reject,
       a one-line reason, the per-item verdict in body, and figure_paths to copy
       the plots into the record folder (a record with NO figures is fine — just
       omit figure_paths). Recording does not block writeback; an imperfect run is
       accepted as long as the reason states why honestly.
    2. If you learned a reusable rule: memory_search the symptom; if a matching
       solution exists -> memory_update_solution (add this record id to seen_in —
       that promotes it to 'confirmed' at >=2); else -> memory_add_solution
       (starts 'provisional').
  - Curate the checklist with memory_checklist_set(exp_type, items) (whole-list
    replace); memory_checklist_get(exp_type) reads it back.
  - A solution proven wrong -> memory_delete it (or downgrade via update). Records
    cannot be deleted.

Writing rules:
  - Solutions must be self-contained: NO source-file / function references (you can
    not read code). cfg knob names (post_delay, gain, reps, freq) are fine.
  - Record what a future session needs: the verdict, the numbers behind it, and any
    surprise. Routine no-surprise calibration values already live in the GUI's
    MetaDict, not here.

ids are namespace-relative paths without .md. A record id is its FOLDER (e.g.
'records/Q5_2D/Q1/2026-06-08-reset-bath'); a solution is
'troubleshooting/reset/bath/low-contrast'; a checklist is 'checklists/reset-bath'.
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
        "checklist_get": lambda p: store.checklist_get(**p),
        "record_measurement": lambda p: store.record_measurement(**p),
        "checklist_set": lambda p: store.checklist_set(**p),
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
