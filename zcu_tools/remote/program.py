import Pyro4

proxy = None  # global variable to store the Pyro4.Proxy object


def init_proxy(proxy_):
    global proxy
    proxy = proxy_


class RemoteProgram:
    def __init__(self, soccfg, cfg):
        self.cfg = cfg

        if proxy is None:
            raise RuntimeError("Please call init_proxy to initialize the proxy first")

    def acquire(self, soc, *args, **kwargs):
        name = self.__class__.__name__
        kwargs["progress"] = False
        kwargs["round_callback"] = None
        try:
            return proxy.run_program(name, self.cfg, *args, **kwargs)
        except Pyro4.errors.CommunicationError as e:
            print("Error: ", e)
            return None


class OneToneProgram(RemoteProgram):
    pass


class TwoToneProgram(RemoteProgram):
    pass
