import Pyro4
import Pyro4.naming

from qick import QickConfig

from ..tools import get_ip_address
from .server import ProgramServer


def start_nameserver(ns_host="0.0.0.0", ns_port=8888):
    Pyro4.config.SERIALIZERS_ACCEPTED = set(["pickle"])
    Pyro4.config.PICKLE_PROTOCOL_VERSION = 4
    Pyro4.naming.startNSloop(host=ns_host, port=ns_port)


def start_server(
    ns_host,
    ns_port=8888,
    iface="eth0",
):
    from qick import QickSoc

    Pyro4.config.REQUIRE_EXPOSE = False
    Pyro4.config.SERIALIZER = "pickle"
    Pyro4.config.SERIALIZERS_ACCEPTED = set(["pickle"])
    Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

    print("looking for nameserver . . .")
    ns = Pyro4.locateNS(host=ns_host, port=ns_port)
    print("found nameserver")

    # if we have multiple network interfaces, we want to register the daemon using the IP address that faces the nameserver
    # if the nameserver is running on the QICK, the above will usually return the loopback address - not useful
    host = Pyro4.socketutil.getInterfaceAddress(ns._pyroUri.host)
    if host == "127.0.0.1":
        host = get_ip_address(iface)
    daemon = Pyro4.Daemon(host=host)

    soc = QickSoc()
    print("initialized QICK")

    # register the QickSoc in the daemon (so the daemon exposes the QickSoc over Pyro4)
    # and in the nameserver (so the client can find the QickSoc)
    ns.register("myqick", daemon.register(soc))
    print("registered QICK")

    for obj in soc.autoproxy:
        daemon.register(obj)
        print("registered member " + str(obj))

    prog_server = ProgramServer(soc)
    ns.register("prog_server", daemon.register(prog_server))
    print("registered ProgramServer")

    print("starting daemon")
    daemon.requestLoop()  # this will run forever until interrupted


def make_proxy(ns_host, ns_port=8888, remote_traceback=True):
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

    return (soc, soccfg, prog_server)
