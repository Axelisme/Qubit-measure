# Smoke Harness

Read this when asked to verify the GUI loop without an MCP client.

## Run (smoke harness — verify the loop without an MCP client)

`smoke.py` launches the GUI itself and drives the mock pipeline over the raw
socket. Use it to prove a fresh checkout works:

```bash
# desktop session (DISPLAY set), Codex skill path:
.venv/bin/python .codex/skills/run-measure-gui/smoke.py
# desktop session, agents skill path:
.venv/bin/python .agents/skills/run-measure-gui/smoke.py
# headless example:
xvfb-run -a .venv/bin/python .codex/skills/run-measure-gui/smoke.py
```

Expected tail (≈30–60s):

```
[smoke] ready: {'project': True, 'context': True, 'active_context': True, 'soc': True}
[smoke] tab: fake-freq-09dbdc70
[smoke] edited reps/rounds
[smoke] progress: rounds %v/%m [0:00<0:04] (3.3%)
[smoke] run finished (saw_live_progress=True)
[smoke] analyzed + saved data
[smoke] screenshot -> /tmp/measure_gui_smoke.png (137535 bytes)
[smoke] tab closed
[smoke] SMOKE OK
```

It writes the screenshot to the OS temp dir (`$TMPDIR/measure_gui_smoke.png` —
the Analysis tab with the fitted resonator dip + writeback panel). It uses
control port **8799** so a live MCP session on the default 8765 is undisturbed.
