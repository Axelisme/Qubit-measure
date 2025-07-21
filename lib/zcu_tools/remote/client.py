import threading
import time
from typing import Any, Dict, Optional, Tuple

import Pyro4
from Pyro4.core import Daemon
from tqdm.auto import tqdm

from zcu_tools.program.base import AbsProxy, MyProgram

from ..config import config
from .pyro import *  # noqa , initialize Pyro4.config
from .server import ProgramServer
from .wrapper import RemoteCallback


class ProgramClient(AbsProxy):
    callback_daemon = None  # lazy init
    daemon_thread = None  # lazy init

    def __init__(self, prog_server: ProgramServer) -> None:
        self.prog_server = prog_server

    @classmethod
    def get_daemon(cls) -> Daemon:
        if cls.callback_daemon is None:
            print(
                f"Client Pyro4 daemon started at {config.LOCAL_IP}:{config.LOCAL_PORT}"
            )
            Pyro4.config.ONEWAY_THREADED = True
            cls.callback_daemon = Pyro4.Daemon(config.LOCAL_IP, config.LOCAL_PORT)
            # 將 daemon.requestLoop 放在背景執行緒執行
            cls.daemon_thread = threading.Thread(
                target=cls.callback_daemon.requestLoop, daemon=True
            )
            cls.daemon_thread.start()

        return cls.callback_daemon

    @classmethod
    def clear_daemon(cls) -> None:
        if cls.callback_daemon is not None:
            print("Client Pyro4 daemon stopped")
            cls.callback_daemon.shutdown()
            cls.callback_daemon = None
            cls.daemon_thread.join()
            cls.daemon_thread = None

    def _overwrite_kwargs_for_remote(
        self, prog: MyProgram, kwargs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Optional[Any]]:
        # before send to remote server, override some kwargs

        rounds = prog.cfg["rounds"]

        kwargs.setdefault("progress", False)
        kwargs.setdefault("callback", None)

        # remote progress bar
        if kwargs["progress"]:
            # replace tqdm progress with callback
            # to make remote progress bar work
            kwargs["progress"] = False

            bar = tqdm(total=rounds, desc="Rounds", leave=True)

            # wrap existing callback
            cb = kwargs["callback"]

            def callback_with_bar(ir: int, *args, **kwargs) -> None:
                bar.update(max(ir + 1 - bar.n, 0))
                if cb is not None:
                    cb(ir, *args, **kwargs)

            kwargs["callback"] = callback_with_bar
        else:
            bar = None

        return kwargs, bar

    def _remote_call(
        self,
        func_name: str,
        *args,
        timeout=None,
        copy_=False,
        **kwargs,
    ) -> Any:
        # call server-side method with kwargs
        prog_s = self.prog_server
        if copy_:
            prog_s = Pyro4.Proxy(prog_s._pyroUri)

        if timeout is None:
            timeout = prog_s._pyroTimeout

        try:
            prog_s._pyroTimeout, old = timeout, prog_s._pyroTimeout
            return getattr(prog_s, func_name)(*args, **kwargs)
        except Pyro4.errors.CommunicationError as e:
            print("Connection error:", e)
            raise e
        except BaseException as e:
            import sys

            # if find '_pyroTraceback' in error value, it's a remote error
            # if not, need to interrupt remote side
            if not hasattr(sys.exc_info()[1], "_pyroTraceback"):
                print("Client-side error, raise it on remote side...")
                prog_s._pyroTimeout = 1
                prog_s.set_early_stop()

            raise e
        finally:
            prog_s._pyroTimeout = old

    def test_remote_callback(self) -> bool:
        success_flag = False

        def set_success(_: int) -> None:
            nonlocal success_flag
            success_flag = True

        print("Sending callback to server...", end="   ")
        with RemoteCallback(self, set_success) as cb:
            assert cb is not None, "Callback should not be None in test_callback"
            self._remote_call("test_callback", cb)
            time.sleep(0.5)
        print("Callback test ", "passed" if success_flag else "failed", "!")
        return success_flag

    def _remote_acquire(self, prog: MyProgram, decimated: bool, **kwargs) -> list:
        kwargs, bar = self._overwrite_kwargs_for_remote(prog, kwargs)

        with RemoteCallback(self, kwargs["callback"]) as cb:
            kwargs["callback"] = cb

            # acquiring on server-side and return result
            ret = self._remote_call(
                "run_program", type(prog).__name__, prog.cfg, decimated, **kwargs
            )

        # force update progress bar
        if bar is not None:
            bar.update(bar.total - bar.n)
            bar.refresh()
            bar.close()

        return ret

    def get_raw(self) -> list:
        return self._remote_call("get_raw", timeout=5)

    def get_shots(self) -> list:
        return self._remote_call("get_shots", timeout=5)

    def get_round_data(self) -> Tuple[Optional[list], Optional[list]]:
        return self._remote_call("get_round_data", timeout=5)

    def set_early_stop(self) -> None:
        self._remote_call("set_early_stop", copy_=True, timeout=4)

    def acquire(self, prog: MyProgram, **kwargs) -> list:
        return self._remote_acquire(prog, decimated=False, **kwargs)

    def acquire_decimated(self, prog: MyProgram, **kwargs) -> list:
        return self._remote_acquire(prog, decimated=True, **kwargs)
