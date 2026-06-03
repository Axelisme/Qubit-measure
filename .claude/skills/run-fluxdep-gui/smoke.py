#!/usr/bin/env python
"""Standalone smoke driver for fluxdep-gui.

Launches the GUI itself (offscreen) with a control port, then drives the whole
v1 pipeline over the raw NDJSON socket (wire method names, no MCP client) against
the converted real fixtures next to this file. Use it to prove a fresh checkout's
RPC path works end to end.

    .venv/bin/python .claude/skills/run-fluxdep-gui/smoke.py
    xvfb-run -a .venv/bin/python .claude/skills/run-fluxdep-gui/smoke.py   # headless

Uses control port 8788 so a live MCP session on the default 8766 is undisturbed.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
RUN_GUI = REPO / "script" / "run_fluxdep_gui.py"
PORT = 8788

ONETONE = HERE / "fixtures" / "onetone_flux_Q2_1.hdf5"
TWOTONE = HERE / "fixtures" / "twotone_flux_Q1_1.hdf5"


class Client:
    def __init__(self, port: int) -> None:
        self._sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        self._buf = b""
        self._id = 0

    def call(self, method: str, **params: object) -> dict:
        self._id += 1
        rid = str(self._id)
        line = json.dumps({"id": rid, "method": method, "params": params}) + "\n"
        self._sock.sendall(line.encode())
        # read until we see the reply for this id (skip event pushes)
        while True:
            obj = self._read_line()
            if obj.get("id") == rid:
                if not obj.get("ok"):
                    raise RuntimeError(f"{method} failed: {obj.get('error')}")
                return obj.get("result") or {}

    def _read_line(self) -> dict:
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise RuntimeError("socket closed")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return json.loads(line.decode())

    def close(self) -> None:
        self._sock.close()


def _wait_for_port(port: int, proc: subprocess.Popen, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"GUI exited early (code {proc.returncode})")
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return
        except OSError:
            time.sleep(0.3)
    raise RuntimeError(f"GUI control port {port} never opened")


def main() -> int:
    for f in (ONETONE, TWOTONE):
        if not f.exists():
            print(f"[smoke] missing fixture: {f}", file=sys.stderr)
            return 2

    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = subprocess.Popen(
        [sys.executable, str(RUN_GUI), "--no-log", "--control-port", str(PORT)],
        env=env,
    )
    try:
        _wait_for_port(PORT, proc)
        c = Client(PORT)
        print(f"[smoke] connected on {PORT}")

        c.call("project.setup", chip_name="Q3_2D", qub_name="Q2")
        print("[smoke] project set:", c.call("project.info"))

        # OneTone: load → align → auto-detected points (threshold default) → done
        r = c.call("spectrum.load", filepath=str(ONETONE), spec_type="OneTone")
        one = r["name"]
        print(f"[smoke] loaded OneTone: {one}")
        c.call("alignment.set", name=one, flux_half=0.0, flux_int=10.0)

        # TwoTone: load (inherit alignment from the OneTone), align, points
        r = c.call(
            "spectrum.load",
            filepath=str(TWOTONE),
            spec_type="TwoTone",
            inherit_from=one,
        )
        two = r["name"]
        print(f"[smoke] loaded TwoTone: {two}")
        c.call("alignment.set", name=two, flux_half=0.0, flux_int=2.0)

        # feed explicit points (agent path — bypasses the in-figure picking)
        c.call("points.set", name=one, dev_values=[-10.0, 0.0, 10.0], freqs=[7.45, 7.45, 7.45])
        c.call("points.set", name=two, dev_values=[-2.0, 1.0], freqs=[4.0, 4.5])

        print("[smoke] spectra:", c.call("spectrum.list"))
        print("[smoke] versions:", c.call("resources.versions"))

        cloud = c.call("selection.pointcloud")
        n = len(cloud["fluxs"])
        print(f"[smoke] joint cloud: {n} points")
        c.call("selection.set", selected=[True] * n)

        out = REPO / "fluxdep_smoke_out.hdf5"
        if out.exists():
            out.unlink()
        res = c.call("export.spectrums", filepath=str(out))
        print(f"[smoke] exported -> {res['path']}")
        assert Path(res["path"]).exists(), "export file missing"
        out.unlink()

        print("[smoke] state:", c.call("state.check"))
        c.close()
        print("[smoke] SMOKE OK")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
