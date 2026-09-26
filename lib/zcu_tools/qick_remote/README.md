# QICK remote connection

**Last updated:** 2026-09-27 — board package relocation

`zcu_tools.qick_remote` supplies the Pyro connection to a QICK SoC. Workstation callers use `make_soc_proxy` to connect; the GUI connection service owns the connection lifecycle and reports failures. This is separate from GUI remote-control transport and MCP.

The board entry points, [`scripts/start_server.py`](../../../scripts/start_server.py) and [`scripts/start_server.ipynb`](../../../scripts/start_server.ipynb), start the Pyro nameserver and SoC server. The server resolves its bitfile through the repo-root [`bitfiles`](../../../bitfiles/README.md) package. `get_bitfile` is imported inside `start_server`, so importing the workstation client does not require the board-only asset package. The board entry points must make the repo root and `lib` importable; the Python script supplies the repo root from its own path, while the notebook uses paths relative to `scripts/`.

Board files retain Python 3.8 syntax. The server and bitfile path have not been exercised on a board; a successful static path check does not establish hardware or deployed-environment compatibility.
