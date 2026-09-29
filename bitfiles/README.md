# Board bitfiles

**Last updated:** 2026-09-27 — board assets at repository root

This directory contains QICK board `.bit` and `.hwh` assets and the Python `get_bitfile(version)` path selector. `get_bitfile("v1")` returns the path to `qick_216.bit`; `get_bitfile("v2")` returns the path to `qick_216_v2.bit`. Other values raise `ValueError`. The helper returns a string path relative to this package; it does not load hardware or verify existence and compatibility.

The board server imports `bitfiles` from the repository root when `start_server` runs. Deploy this directory with the repository and make the repository root importable on the board. The Python entry point adds that path using its own location; the notebook adds `../` relative to `scripts/`. `lib` and QICK must also be importable in the board environment. This is a board-only package, not part of installed workstation `zcu_tools`.

Asset provenance and version compatibility: **待核實**. Python 3.8 syntax and local path resolution are checked, but board deployment and execution are **待核實**; no hardware validation has been performed.
