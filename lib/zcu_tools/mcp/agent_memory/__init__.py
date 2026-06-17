"""Agent lab notebook — a human-readable, file-backed notebook over
``agent_memory/<ns>/`` with three functions and three trees:

  - ``records/<chip>/<qub>/<date>-<slug>/`` — one measurement per FOLDER, holding
    ``record.md`` (the per-item pass/fail verdict + the numbers) plus any copied
    figures. Episodic and IMMUTABLE (never overwritten, never deleted).
  - ``troubleshooting/<exp_slug>/<symptom_slug>.md`` — context-free problem -> fix
    (``type: solution``), mutable and self-improving across qubits.
  - ``checklists/<exp_slug>.md`` — one acceptance checklist per experiment type,
    items kept as a hand-editable markdown bullet list in the body.

The files are plain markdown + YAML frontmatter, meant to be read and hand-edited
by humans. The measuring agent reaches them only through the MCP tools (see
``server``); it never reads the files directly.
"""
