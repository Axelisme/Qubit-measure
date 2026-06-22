#!/usr/bin/env python
"""Standalone smoke driver for the measure-gui control socket.

This is the harness behind the run-measure-gui skill. It launches the GUI
exactly the way the MCP bridge does (``run_measure_gui.py --control-port 0`` under
xvfb), then speaks the *same* newline-delimited JSON RPC the bridge speaks —
so it verifies the whole experiment loop without needing an MCP client.

What it proves (mock SoC, no hardware):
  soc.connect(mock) -> startup.apply -> context -> tab.new(fake/freq)
  -> editor.set_field(reps/rounds) -> tab.run_start -> operation.progress (live)
  -> wait for run_finished event -> tab.analyze -> tab.save_data
  -> tab.close -> clean shutdown.

Run it (from the repo root):
    xvfb-run -a .venv/bin/python .claude/skills/run-measure-gui/smoke.py

The RPC method names here are the *wire* names (dotted: 'tab.run_start'), not the
MCP tool aliases ('gui_tab_run_start'). An agent driving via MCP uses the aliases;
this script talks the socket directly.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PORT = 8799  # avoid the default 8765 so a live MCP session is undisturbed


class Rpc:
    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        self._buf = b""

    def call(self, method: str, params: dict | None = None, timeout: float = 30.0):
        rid = uuid.uuid4().hex
        line = json.dumps({"id": rid, "method": method, "params": params or {}}) + "\n"
        self._sock.sendall(line.encode())
        deadline = time.monotonic() + timeout
        # The socket also pushes unsolicited events (no "id"); skip them until
        # our reply id comes back.
        while True:
            while b"\n" in self._buf:
                raw, self._buf = self._buf.split(b"\n", 1)
                if not raw.strip():
                    continue
                msg = json.loads(raw)
                if msg.get("id") != rid:
                    continue  # an event / another reply
                if not msg.get("ok", False):
                    err = msg.get("error", {})
                    raise RuntimeError(f"{method} failed: {err}")
                return msg.get("result", {})
            if time.monotonic() > deadline:
                raise TimeoutError(f"{method} timed out")
            self._sock.settimeout(deadline - time.monotonic())
            self._buf += self._sock.recv(65536)


def log(msg: str) -> None:
    print(f"[smoke] {msg}", flush=True)


def main() -> int:
    proc = subprocess.Popen(
        [
            sys.executable,
            str(REPO / "script" / "run_measure_gui.py"),
            "--control-port",
            str(PORT),
            "--no-log",
        ],
        cwd=str(REPO),
    )
    sock = None
    try:
        # 1. Wait for the control port to accept connections.
        for _ in range(100):
            try:
                sock = socket.create_connection(("127.0.0.1", PORT), timeout=1.0)
                break
            except OSError:
                time.sleep(0.2)
        if sock is None:
            raise RuntimeError("control port never came up")
        rpc = Rpc(sock)

        # 2. Mock SoC + project + context — the same RPC path a GUI user takes
        #    (soc.connect + startup.apply + context), no mock-only shortcut.
        #    soc.connect is synchronous: it returns once the SoC is connected
        #    (and the FLUX-AWARE-MOCK fake_flux provisioning has been kicked off),
        #    so the state.has_soc poll below settles immediately.
        soc_reply = rpc.call("soc.connect", {"kind": "mock"})
        assert soc_reply["soc"]["is_mock"] is True, soc_reply
        rpc.call(
            "startup.apply",
            {
                "chip_name": "Q1_Chip",
                "qub_name": "Q1",
                "res_name": "R1",
                "result_dir": str(REPO / "result"),
                "database_path": str(REPO / "Database"),
            },
        )
        for _ in range(50):
            if rpc.call("state.has_soc").get("value"):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("mock SoC never became ready")
        labels = rpc.call("context.labels").get("labels", [])
        if labels:
            rpc.call("context.use", {"label": labels[0]})
        else:
            rpc.call("context.new", {})
        flags = {
            k: rpc.call(f"state.has_{k}").get("value")
            for k in ("project", "context", "active_context", "soc")
        }
        assert all(flags.values()), f"not ready: {flags}"
        log(f"ready: {flags}")

        # 3b. SoC hardware summary (soc.info) — the agent's window into the board.
        #     include_cfg=True is required to receive the full QICK cfg: soc.info
        #     omits the ~2 KB cfg unless explicitly asked (default false).
        info = rpc.call("soc.info", {"include_cfg": True})
        assert "QICK" in info["description"] and info["cfg"]["gens"], info
        log(f"soc: mock={info['is_mock']}, gens={len(info['cfg']['gens'])}")

        # 4. New tab on the mock-friendly fake/freq adapter.
        tab_id = rpc.call("tab.new", {"adapter_name": "fake/freq"})["tab_id"]
        log(f"tab: {tab_id}")
        # tab.snapshot always returns {tabs: [...]} (a single tab_id yields a
        # one-element list); index reply["tabs"][0] uniformly.
        editor_id = rpc.call("tab.snapshot", {"tab_id": tab_id})["tabs"][0]["editor_id"]

        # 5. Edit a couple of fields through the cfg-editor session.
        rpc.call(
            "editor.set_field", {"editor_id": editor_id, "path": "rounds", "value": 30}
        )
        rpc.call(
            "editor.set_field", {"editor_id": editor_id, "path": "reps", "value": 50}
        )
        log("edited reps/rounds")

        # 6. Start the run, then read live progress at least once (progress is
        # queried by the run's operation_id — the unified operation.progress).
        op = rpc.call("tab.run_start", {"tab_id": tab_id})
        operation_id = op.get("operation_id")
        saw_progress = False
        for _ in range(60):
            prog = rpc.call("operation.progress", {"operation_id": operation_id})
            if prog.get("active") and prog.get("bars"):
                bar = prog["bars"][0]
                log(f"progress: {bar['format']} ({bar['percent']}%)")
                saw_progress = True
                break
            if rpc.call("run.running_tab").get("tab_id") is None:
                break  # already finished (too fast to catch a bar)
            time.sleep(0.2)

        # 7. Wait for completion (poll running_tab; events would also work).
        for _ in range(300):
            if rpc.call("run.running_tab").get("tab_id") is None:
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("run never finished")
        snap = rpc.call("tab.snapshot", {"tab_id": tab_id})["tabs"][0]
        assert snap["interaction"]["has_run_result"], "no run result after finish"
        log(f"run finished (saw_live_progress={saw_progress})")

        # 8. Analyze, wait for it to settle (it is its own operation — saving
        # while the tab is still analyzing returns precondition_failed/busy),
        # then save.
        rpc.call("tab.analyze", {"tab_id": tab_id})
        for _ in range(300):
            if not rpc.call("tab.snapshot", {"tab_id": tab_id})["tabs"][0][
                "interaction"
            ]["is_analyzing"]:
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("analyze never finished")
        rpc.call("tab.save_data", {"tab_id": tab_id})
        log("analyzed + saved data")

        # 8b. Screenshot the tab's current figure (base64 PNG over the socket) to
        # the OS temp dir (transient proof the GUI rendered; not committed).
        import base64
        import tempfile

        shot = rpc.call("tab.get_current_figure", {"tab_id": tab_id})
        out = Path(tempfile.gettempdir()) / "measure_gui_smoke.png"
        out.write_bytes(base64.b64decode(shot["png_b64"]))
        log(f"figure screenshot -> {out} ({shot['bytes']} bytes)")

        # 9. Close the tab (exercises the View detach path).
        rpc.call("tab.close", {"tab_id": tab_id})
        log("tab closed")

        log("SMOKE OK")
        return 0
    finally:
        if sock is not None:
            sock.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
