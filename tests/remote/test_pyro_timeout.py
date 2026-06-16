"""make_soc_proxy sets a 1s Pyro4 COMMTIMEOUT so an unreachable board fails fast.

The COMMTIMEOUT is the single Pyro4 chokepoint covering both the nameserver
lookup (locateNS) and the proxy call (get_cfg); setting it to 1.0 is what bounds
the synchronous soc.connect main-thread block.

Two invariants must hold (see pyro.py for the full explanation):
  A. The 1s cap is in effect for every network call during connect (locateNS +
     first get_cfg).
  B. After make_soc_proxy returns, BOTH the process-global COMMTIMEOUT and the
     returned proxy's _pyroTimeout are restored to their pre-call values so that
     subsequent measurement RPCs (which can take minutes) are not capped at 1s.
"""

from __future__ import annotations

import pytest

Pyro4 = pytest.importorskip("Pyro4")

from zcu_tools.remote.pyro import make_soc_proxy


def test_make_soc_proxy_sets_commtimeout_before_locate_ns(monkeypatch):
    seen: dict[str, object] = {}

    def fake_locate_ns(host, port):
        # Capture the COMMTIMEOUT in force at the moment the first network call
        # would run — it must already be the 1s fail-fast cap.
        seen["commtimeout"] = Pyro4.config.COMMTIMEOUT
        raise Pyro4.errors.NamingError("no nameserver (test)")

    monkeypatch.setattr(Pyro4, "locateNS", fake_locate_ns)

    with pytest.raises(Pyro4.errors.NamingError):
        make_soc_proxy("127.0.0.1", 65000)

    assert seen["commtimeout"] == 1.0


def test_make_soc_proxy_restores_timeout_after_success(monkeypatch):
    """Regression: returned proxy must NOT carry the 1s connect-time cap.

    Without the per-proxy reset in make_soc_proxy, the returned soc._pyroTimeout
    would permanently be 1.0 because Pyro4.Proxy.__init__ snapshots COMMTIMEOUT at
    construction.  Any measurement RPC longer than 1s would then time out on a real
    board.
    """

    # Sentinel value that represents "no prior cap" (the typical default).
    initial_timeout: float | None = None

    # Record the COMMTIMEOUT seen during the connect phase.
    seen: dict[str, object] = {}

    class FakeProxy:
        """Minimal Pyro4.Proxy stand-in that snapshots COMMTIMEOUT at construction."""

        def __init__(self, uri: object) -> None:
            # Mirror the real Pyro4.Proxy behaviour: snapshot global at construction.
            self._pyroTimeout: float | None = Pyro4.config.COMMTIMEOUT
            seen["timeout_at_construction"] = self._pyroTimeout

        def get_cfg(self) -> dict:  # type: ignore[return]
            seen["timeout_at_get_cfg"] = self._pyroTimeout
            return {}  # minimal dict that QickConfig can receive

    class FakeNS:
        def lookup(self, name: str) -> str:
            return "PYRO:fake@127.0.0.1:0"

    def fake_locate_ns(host: str, port: int) -> FakeNS:
        seen["timeout_at_locate_ns"] = Pyro4.config.COMMTIMEOUT
        return FakeNS()

    class FakeQickConfig:
        def __init__(self, cfg: dict) -> None:
            pass

    monkeypatch.setattr(Pyro4, "locateNS", fake_locate_ns)
    monkeypatch.setattr(Pyro4, "Proxy", FakeProxy)
    monkeypatch.setattr("zcu_tools.remote.pyro.QickConfig", FakeQickConfig)

    # Set a known global state before the call.
    monkeypatch.setattr(Pyro4.config, "COMMTIMEOUT", initial_timeout)

    soc, _ = make_soc_proxy("127.0.0.1", 65000)

    # (a) The connect phase used the 1s cap for locateNS and get_cfg.
    assert seen["timeout_at_locate_ns"] == 1.0, "locateNS must run under the 1s cap"
    assert seen["timeout_at_construction"] == 1.0, (
        "Proxy was constructed under the 1s cap"
    )
    assert seen["timeout_at_get_cfg"] == 1.0, "get_cfg must run under the 1s cap"

    # (b) The process-global is restored so that future proxy constructions are not capped.
    assert (
        Pyro4.config.COMMTIMEOUT == initial_timeout  # type: ignore[attr-defined]
    ), "process-global COMMTIMEOUT must be restored after make_soc_proxy returns"

    # (c) CRITICAL: the returned proxy itself must NOT retain the 1s cap — without
    #     the explicit soc._pyroTimeout reset in make_soc_proxy, this would be 1.0
    #     and every measurement RPC on a real board would time out after 1s.
    assert soc._pyroTimeout != 1.0, (
        "returned proxy _pyroTimeout must not be 1.0 — measurement RPCs would time out"
    )
    assert soc._pyroTimeout == (initial_timeout or None), (
        "returned proxy _pyroTimeout must equal the pre-call value (None = no cap)"
    )
