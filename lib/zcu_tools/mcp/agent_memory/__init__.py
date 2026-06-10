"""Agent lab notebook — file-backed records + solutions over ``agent_memory/<ns>/``.

Two entry kinds, two trees:
  - ``records/``   episodic measurement records (append-mostly), keyed by chip/qub/date.
  - ``solutions/`` context-free problem -> fix (mutable, self-improving), keyed by
    exp_type/symptom.

The measuring agent reaches these only through the MCP tools (see ``server``); it
never reads the files directly.
"""
