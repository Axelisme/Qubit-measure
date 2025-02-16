import threading
from typing import Any, Dict

import Pyro4
from tqdm.auto import tqdm

from ..config import config
from .pyro import *  # noqa , initialize Pyro4.config
from .server import ProgramServer
from .wrapper import CallbackWrapper


class ProgramClient:
    callback_daemon = None  # lazy init
    callback_thread = None  # lazy init

    def __init__(self, prog_server: ProgramServer):
        self.prog_server = prog_server

    @classmethod
    def get_daemon(cls):
        if cls.callback_daemon is None:
            print(
                f"Client Pyro4 daemon started at {config.LOCAL_IP}:{config.LOCAL_PORT}"
            )
            Pyro4.config.ONEWAY_THREADED = False  # type: ignore
            cls.callback_daemon = Pyro4.Daemon(
                host=config.LOCAL_IP, port=config.LOCAL_PORT
            )
            # 將 daemon.requestLoop 放在背景執行緒執行
            cls.callback_thread = threading.Thread(
                target=cls.callback_daemon.requestLoop, daemon=True
            )
            cls.callback_thread.start()

        return cls.callback_daemon

    @classmethod
    def clear_daemon(cls):
        if cls.callback_daemon is not None:
            print("Client Pyro4 daemon stopped")
            cls.callback_daemon.shutdown()
            cls.callback_daemon = None
            cls.callback_thread.join()  # type: ignore
            cls.callback_thread = None

    def overwrite_kwargs_for_remote(self, prog, kwargs: Dict[str, Any]):
        # before send to remote server, override some kwargs

        soft_avgs = prog.cfg["soft_avgs"]

        kwargs.setdefault("progress", False)
        kwargs.setdefault("round_callback", None)

        # remote progress bar
        if kwargs["progress"]:
            # replace tqdm progress with callback
            # to make remote progress bar work
            kwargs["progress"] = False

            bar = tqdm(total=soft_avgs, desc="Soft_avgs", leave=True)

            # wrap existing callback
            def callback_with_bar(ir: int, *args, **kwargs):
                bar.update(max(ir + 1 - bar.n, 0))
                if kwargs["round_callback"] is not None:
                    kwargs["round_callback"](ir, *args, **kwargs)

            kwargs["round_callback"] = callback_with_bar
        else:
            bar = None

        return kwargs, bar

    def _remote_call(self, func_name: str, *args, **kwargs):
        # call server-side method with kwargs
        try:
            return getattr(self.prog_server, func_name)(*args, **kwargs)
        except BaseException as e:
            import sys

            # if find '_pyroTraceback' in error value, it's a remote error
            # if not, need to raise it on remote side
            if not hasattr(sys.exc_info()[1], "_pyroTraceback"):
                print("Client-side error, raise it on remote side...")
                prog_server = self.prog_server
                prog_server._pyroTimeout, old = 1, prog_server._pyroTimeout  # type: ignore
                prog_server.set_interrupt(str(e))
                prog_server._pyroTimeout = old  # type: ignore

            raise e

    def test_remote_callback(self) -> bool:
        success_flag = False

        def set_success(_):
            nonlocal success_flag
            success_flag = True

        print("Sending callback to server...", end="   ")
        with CallbackWrapper(self, set_success) as cb:
            self._remote_call("test_callback", cb)
        print("Callback test ", "passed" if success_flag else "failed", "!")
        return success_flag

    def _remote_acquire(self, prog, decimated: bool, **kwargs):
        kwargs, bar = self.overwrite_kwargs_for_remote(prog, kwargs)

        with CallbackWrapper(self, kwargs["round_callback"]) as cb:
            kwargs["round_callback"] = cb
            ret = self._remote_call(
                "run_program", type(prog).__name__, prog.cfg, decimated, **kwargs
            )

        if bar is not None:
            bar.update(bar.total - bar.n)
            bar.refresh()
            bar.close()

        return ret

    def acquire(self, prog, **kwargs):
        return self._remote_acquire(prog, decimated=False, **kwargs)

    def acquire_decimated(self, prog, **kwargs):
        return self._remote_acquire(prog, decimated=True, **kwargs)
