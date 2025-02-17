from typing import Callable, Optional

import Pyro4


class RemoteCallback:
    def __init__(self, client, func: Optional[Callable]):
        self.client = client
        self.func = func

    def __enter__(self):
        if self.func is None:
            return None  # do nothing

        self.daemon = self.client.get_daemon()
        self.daemon.register(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.func is None:
            return  # do nothing

        self.daemon.unregister(self)

    @Pyro4.expose
    @Pyro4.callback
    @Pyro4.oneway
    def oneway_callback(self, ir, *args, **kwargs):
        assert self.func is not None, "This method should not be called if func is None"
        self.func(ir, *args, **kwargs)
