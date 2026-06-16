from __future__ import annotations

import socket
import sys

# board-side file (Python 3.8): typing.Any/Literal exist since 3.8, so the
# stdlib import keeps the board free of a typing_extensions requirement.
from typing import Any, Literal

import IPython
import psutil
import Pyro4
import Pyro4.naming
from qick import QickConfig

from zcu_tools.bitfiles import get_bitfile


def setup_pyro4() -> None:
    # use dill instead of pickle
    Pyro4.config.SERIALIZER = "pickle"  # type: ignore
    Pyro4.config.SERIALIZERS_ACCEPTED = set(["pickle"])  # type: ignore
    Pyro4.config.DILL_PROTOCOL_VERSION = 5
    Pyro4.config.PICKLE_PROTOCOL_VERSION = 4  # type: ignore
    Pyro4.config.REQUIRE_EXPOSE = False  # type: ignore
    Pyro4.config.ONEWAY_THREADED = True


setup_pyro4()


def get_localhost_ip(ns: Pyro4.naming.NameServer, iface: str) -> str:
    # if we have multiple network interfaces, we want to register the daemon using the IP address that faces the nameserver
    host = Pyro4.socketutil.getInterfaceAddress(ns._pyroUri.host)  # type: ignore
    # if the nameserver is running on the QICK, the above will usually return the loopback address - not useful
    if host == "127.0.0.1":
        # if the eth0 interface has an IPv4 address, use that
        # otherwise use the address of any interface starting with "eth0" (e.g. eth0:1, which is typically a static IP)
        # unless you have an unusual network config (e.g. VPN), this is the interface clients will want to connect to
        for name, addrs in psutil.net_if_addrs().items():
            addrs_v4 = [
                addr.address
                for addr in addrs
                if addr.family == socket.AddressFamily.AF_INET
            ]
            if len(addrs_v4) == 1:
                if name.startswith(iface):
                    host = addrs_v4[0]
                if name == iface:
                    break

    return host


def start_nameserver(ns_port: int) -> None:
    Pyro4.naming.startNSloop(host="0.0.0.0", port=ns_port)


def start_server(
    port: int, ns_port: int, version: Literal["v1", "v2"] = "v1", iface="eth0", **kwargs
) -> None:
    from qick import QickSoc

    print("looking for nameserver . . .")
    ns = Pyro4.locateNS(host="0.0.0.0", port=ns_port)
    print("found nameserver")

    host = get_localhost_ip(ns, iface)

    print(f"starting daemon on {host}:{port}")
    daemon = Pyro4.Daemon(host=host, port=port)

    # create and register the QickSoc
    soc = QickSoc(bitfile=get_bitfile(version), **kwargs)
    uri = daemon.register(soc)
    ns.register("myqick", uri)
    print(f"registered QICK at {uri}")
    print(soc)

    for obj in soc.autoproxy:
        daemon.register(obj)
        print("registered member " + str(obj))

    print("starting daemon")
    daemon.requestLoop()  # this will run forever until interrupted


def make_soc_proxy(
    ns_host: str,
    ns_port: int,
    remote_traceback: bool = True,
    lookup_name: str = "myqick",
) -> tuple[Any, QickConfig]:
    # CONNECT-TIME FAIL-FAST: scope the 1s cap to this function only.
    #
    # Two independent pieces must both be undone after connect:
    #   1. Pyro4.config.COMMTIMEOUT is process-global — it must be restored so that
    #      *other* proxies created later (or the same proxy reused) are not affected.
    #   2. Pyro4.Proxy.__init__ SNAPSHOTS the current COMMTIMEOUT into _pyroTimeout
    #      at construction time.  That means the returned `soc` would permanently carry
    #      a 1s cap even after the global is restored — killing any measurement RPC
    #      that takes more than 1s (program.acquire, sweeps, etc. can take minutes).
    #      We must therefore explicitly reset soc._pyroTimeout after restoring the
    #      global so the returned proxy is uncapped.
    prev_timeout: float | None = Pyro4.config.COMMTIMEOUT  # type: ignore[attr-defined]
    Pyro4.config.COMMTIMEOUT = 1.0  # type: ignore[attr-defined]
    soc: Any = None
    try:
        ns = Pyro4.locateNS(host=ns_host, port=ns_port)
        soc = Pyro4.Proxy(ns.lookup(lookup_name))  # snapshots 1.0 into _pyroTimeout
        soccfg = QickConfig(soc.get_cfg())  # first call is under the 1s fail-fast cap
    finally:
        # Restore process-global so nothing outside this function sees the cap.
        Pyro4.config.COMMTIMEOUT = prev_timeout  # type: ignore[attr-defined]
        if soc is not None:
            # Drop the 1s snapshot on the returned proxy; None means "no cap" (default).
            soc._pyroTimeout = prev_timeout or None

    # adapted from https://pyro4.readthedocs.io/en/stable/errors.html and https://stackoverflow.com/a/70433500
    if remote_traceback:
        try:
            ip = IPython.get_ipython()  # type: ignore
            if ip is not None:

                def exception_handler(*_):
                    sys.stderr.write("".join(Pyro4.util.getPyroTraceback()))  # type: ignore

                ip.set_custom_exc(
                    (Exception,), exception_handler
                )  # register your handler
        except Exception as e:
            raise RuntimeError(f"Failed to set up Pyro exception handler: {e}") from e

    return soc, soccfg
