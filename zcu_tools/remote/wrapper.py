from typing import Callable, Optional

import Pyro4

CALLBACK_TIMEOUT = 2  # seconds


class RemoteCallback:
    def __init__(self, client, func: Optional[Callable]):
        self.client = client
        self.func = func

    def __enter__(self):
        if self.func is None:
            return None

        self.daemon = self.client.get_daemon()
        self.daemon.register(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.func is None:
            return

        self.daemon.unregister(self)

    @Pyro4.expose
    @Pyro4.oneway
    def do_callback(self, *args, **kwargs):
        if self.func is None:
            raise RuntimeError("This method should not be called if func is None")
        self.func(*args, **kwargs)


def unwrap_callback(cb: Optional[RemoteCallback]):
    """
    Unwrap the callback function from RemoteCallback object
    The return function is guaranteed to have no exception
    """
    if cb is None:
        return None  # do nothing

    def unwrapped_cb(*args, **kwargs):
        # timeout is set to prevent connecting forever
        cb._pyroTimeout, old = CALLBACK_TIMEOUT, cb._pyroTimeout  
        try:
            cb.do_callback(*args, **kwargs)
        except Exception as e:
            print(f"Error during calling client-side callback: {e}")
        finally:
            cb._pyroTimeout = old  

    return unwrapped_cb
