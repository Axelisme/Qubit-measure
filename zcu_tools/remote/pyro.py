import os

import Pyro4
import Pyro4.naming

import qick
from qick import QickConfig


def get_bitfile(version):
    version_dict = {
        "v1": "qick_216.bit",
        "v2": "qick_216_v2.bit",
    }
    if version not in version_dict:
        raise ValueError(f"Invalid version {version}")
    return os.path.join(os.path.dirname(qick.__file__), version_dict[version])


def get_program_module(version):
    import importlib

    if version == "v1":
        return importlib.import_module("zcu_tools.program.v1")
    elif version == "v2":
        return importlib.import_module("zcu_tools.program.v2")
    else:
        raise ValueError(f"Invalid version {version}")


def start_nameserver(ns_port):
    Pyro4.config.SERIALIZERS_ACCEPTED = set(["pickle"])
    Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
    Pyro4.naming.startNSloop(host="0.0.0.0", port=ns_port)


def start_server(host: str, port: int, ns_port: int, version="v1", **kwargs):
    from qick import QickSoc
    from zcu_tools.remote.server import ProgramServer

    Pyro4.config.REQUIRE_EXPOSE = False
    Pyro4.config.SERIALIZER = "pickle"
    Pyro4.config.SERIALIZERS_ACCEPTED = set(["pickle"])
    Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

    print("looking for nameserver . . .")
    ns = Pyro4.locateNS(host="0.0.0.0", port=ns_port)
    print("found nameserver")

    if host in ["localhost", "0.0.0.0", "127.0.0.1"]:
        print(
            "WARNING: using localhost as host, this will only work on the local machine"
        )
    print(f"starting daemon on {host}:{port}")
    daemon = Pyro4.Daemon(host=host, port=port)

    # create and register the QickSoc

    soc = QickSoc(bitfile=get_bitfile(version), **kwargs)
    uri = daemon.register(soc)
    ns.register("myqick", uri)
    print(f"registered QICK at {uri}")

    for obj in soc.autoproxy:
        daemon.register(obj)
        print("registered member " + str(obj))

    # create and register the program server
    prog_server = ProgramServer(soc, get_program_module(version))
    uri = daemon.register(prog_server)
    ns.register("prog_server", uri)
    print(f"registered program server at {uri}")

    print("starting daemon")
    daemon.requestLoop()  # this will run forever until interrupted


def make_proxy(ns_host, ns_port, remote_traceback=True):
    Pyro4.config.SERIALIZER = "pickle"
    Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

    ns = Pyro4.locateNS(host=ns_host, port=ns_port)

    # print the nameserver entries: you should see the QickSoc proxy
    for k, v in ns.list().items():
        print(k, v)

    soc = Pyro4.Proxy(ns.lookup("myqick"))
    soccfg = QickConfig(soc.get_cfg())
    prog_server = Pyro4.Proxy(ns.lookup("prog_server"))

    # adapted from https://pyro4.readthedocs.io/en/stable/errors.html and https://stackoverflow.com/a/70433500
    if remote_traceback:
        try:
            import sys

            import IPython

            ip = IPython.get_ipython()
            if ip is not None:

                def exception_handler(self, etype, evalue, tb, tb_offset=None):
                    sys.stderr.write("".join(Pyro4.util.getPyroTraceback()))
                    # self.showtraceback((etype, evalue, tb), tb_offset=tb_offset)  # standard IPython's printout

                ip.set_custom_exc(
                    (Exception,), exception_handler
                )  # register your handler
        except Exception as e:
            raise RuntimeError("Failed to set up Pyro exception handler: ", e)

    return soc, soccfg, prog_server
