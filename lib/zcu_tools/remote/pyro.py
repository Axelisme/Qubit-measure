import socket
from types import ModuleType
from typing import Any, Literal, Tuple

import psutil
import Pyro4
import Pyro4.naming

from qick import QickConfig
from zcu_tools.program.bitfiles import get_bitfile


def setup_pyro4():
    # use dill instead of pickle
    # Pyro4.config.SERIALIZER = "pickle"
    Pyro4.config.SERIALIZER = "dill"
    Pyro4.config.SERIALIZERS_ACCEPTED = set(["dill", "pickle"])
    Pyro4.config.DILL_PROTOCOL_VERSION = 5
    Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
    Pyro4.config.REQUIRE_EXPOSE = False
    Pyro4.config.ONEWAY_THREADED = True


setup_pyro4()


def get_program_module(version: Literal["v1", "v2"]) -> ModuleType:
    import importlib

    if version == "v1":
        return importlib.import_module("zcu_tools.program.v1")
    elif version == "v2":
        return importlib.import_module("zcu_tools.program.v2")
    else:
        raise ValueError(f"Invalid version {version}")


def get_localhost_ip(ns: Pyro4.naming.NameServer, iface: str) -> str:
    # if we have multiple network interfaces, we want to register the daemon using the IP address that faces the nameserver
    host = Pyro4.socketutil.getInterfaceAddress(ns._pyroUri.host)
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


def start_server(port: int, ns_port: int, version="v1", iface="eth0", **kwargs) -> None:
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


def make_proxy(
    ns_host: str,
    ns_port: int,
    remote_traceback: bool = True,
    lookup_name: str = "myqick",
) -> Tuple[Any, QickConfig]:
    ns = Pyro4.locateNS(host=ns_host, port=ns_port)

    soc = Pyro4.Proxy(ns.lookup(lookup_name))
    soccfg = QickConfig(soc.get_cfg())

    # adapted from https://pyro4.readthedocs.io/en/stable/errors.html and https://stackoverflow.com/a/70433500
    if remote_traceback:
        try:
            import sys

            import IPython

            ip = IPython.get_ipython()
            if ip is not None:

                def exception_handler(*_):
                    sys.stderr.write("".join(Pyro4.util.getPyroTraceback()))

                ip.set_custom_exc(
                    (Exception,), exception_handler
                )  # register your handler
        except Exception as e:
            raise RuntimeError("Failed to set up Pyro exception handler: ", e)

    return soc, soccfg
