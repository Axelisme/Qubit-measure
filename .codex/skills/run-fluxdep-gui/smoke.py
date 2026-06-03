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
import tempfile
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


def _make_tiny_database(path: Path) -> None:
    """Write a tiny synthetic fluxonium database (params/fluxs/energies) for the
    v2 search step — avoids depending on the multi-hundred-MB real database."""
    import h5py
    import numpy as np

    M, L = 21, 4
    fluxs = np.linspace(0.0, 0.5, M).astype(np.float64)
    params = np.array(
        [[3.0, 1.0, 0.5], [5.0, 1.2, 0.4], [6.0, 0.9, 0.6]], dtype=np.float64
    )
    energies = np.zeros((len(params), M, L), dtype=np.float64)
    for n, (EJ, EC, EL) in enumerate(params):
        for lvl in range(L):
            energies[n, :, lvl] = lvl * (EC + EL) + EJ * np.cos(2 * np.pi * fluxs) * 0.1
    with h5py.File(path, "w") as f:
        f.create_dataset("fluxs", data=fluxs)
        f.create_dataset("params", data=params)
        f.create_dataset("energies", data=energies)


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
        c.call(
            "points.set",
            name=one,
            dev_values=[-10.0, 0.0, 10.0],
            freqs=[7.45, 7.45, 7.45],
        )
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

        # v2: database search over the selected joint cloud → params.json
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "tiny_db.h5"
            _make_tiny_database(db)
            c.call(
                "fit.set_params",
                database_path=str(db),
                EJb=[0.1, 50.0],
                ECb=[0.01, 10.0],
                ELb=[0.01, 10.0],
                transitions={"transitions": [[0, 1], [0, 2]]},
            )
            best = c.call("fit.search")
            print(
                f"[smoke] search: EJ={best['EJ']:.3f} "
                f"EC={best['EC']:.3f} EL={best['EL']:.3f}"
            )
            params_out = Path(tmp) / "params.json"
            res = c.call("fit.export_params", savepath=str(params_out))
            assert Path(res["path"]).exists(), "params.json missing"
            print(f"[smoke] params -> {res['path']}")
            print("[smoke] fit.result:", c.call("fit.result")["has_result"])

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
