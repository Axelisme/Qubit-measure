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
    def do_callback(self, ir, *args, **kwargs):
        if self.func is None:
            raise RuntimeError("This method should not be called if func is None")
        self.func(ir, *args, **kwargs)


def unwrap_callback(cb: Optional[RemoteCallback]):
    if cb is None:
        return None  # do nothing

    def unwrapped_func(*args, **kwargs):
        # timeout is set to prevent connecting forever
        cb._pyroTimeout, old = CALLBACK_TIMEOUT, cb._pyroTimeout  # type: ignore
        try:
            cb.do_callback(*args, **kwargs)
        except Exception as e:
            print(f"Error during callback execution: {e}")
        finally:
            cb._pyroTimeout = old  # type: ignore

    return unwrapped_func
