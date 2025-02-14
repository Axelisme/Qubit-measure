import threading
import time

import Pyro4
from tqdm.auto import tqdm

from ..config import config
from . import pyro  # noqa , 初始化Pyro4.config
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
            cls.callback_thread.join()
            cls.callback_thread = None

    @classmethod
    def wrap_callback(cls, func: callable) -> CallbackWrapper:
        daemon = cls.get_daemon()

        # 用 CallbackWrapper 包裝原始函數
        callback = CallbackWrapper(func)

        # 將 callback 物件註冊到 daemon 中，取得其 URI
        daemon.register(callback)

        return callback

    @classmethod
    def drop_callback(cls, callback: CallbackWrapper):
        cls.callback_daemon.unregister(callback)
        callback.close()

    def overwrite_kwargs_for_remote(self, prog, kwargs: dict):
        # before send to remote server, override some kwargs

        soft_avgs = prog.cfg["soft_avgs"]

        # remote progress bar
        if kwargs.get("progress", False):
            # replace tqdm progress with callback
            # to make remote progress bar work
            kwargs["progress"] = False

            bar = tqdm(total=soft_avgs, desc="soft_avgs", leave=True)
            if kwargs.get("round_callback") is not None:
                # wrap existing callback
                orig_callback = kwargs["round_callback"]

                def callback_with_bar(ir, *args, **kwargs):
                    bar.update(max(ir + 1 - bar.n, 0))
                    bar.refresh()
                    orig_callback(ir, *args, **kwargs)
            else:

                def callback_with_bar(ir, *args, **kwargs):
                    bar.update(max(ir + 1 - bar.n, 0))
                    bar.refresh()

            kwargs["round_callback"] = callback_with_bar
        else:
            bar = None

        # remote callback
        if kwargs.get("round_callback") is not None:
            kwargs["round_callback"] = type(self).wrap_callback(
                kwargs["round_callback"]
            )

        return kwargs, bar

    def _remote_call(self, func_name, *args, **kwargs):
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
                prog_server._pyroTimeout, old = 1, prog_server._pyroTimeout
                prog_server.set_interrupt(str(e))
                prog_server._pyroTimeout = old

            raise e

    def test_remote_callback(self) -> bool:
        success_flag = False

        def oneway_callback():
            nonlocal success_flag
            success_flag = True

        boxed_callback = type(self).wrap_callback(oneway_callback)
        print("Sending callback to server...", end="   ")
        self._remote_call("test_callback", boxed_callback)
        time.sleep(0.5)
        type(self).drop_callback(boxed_callback)
        print("Callback test ", "passed" if success_flag else "failed", "!")
        return success_flag

    def acquire(self, prog, **kwargs):
        kwargs, bar = self.overwrite_kwargs_for_remote(prog, kwargs)
        prog_name = type(prog).__name__
        ret = self._remote_call("run_program", prog_name, prog.cfg, **kwargs)
        boxed_callback = kwargs.get("round_callback")
        if boxed_callback is not None:
            type(self).drop_callback(boxed_callback)
        if bar is not None:
            bar.update(bar.total - bar.n)  # force to finish
            bar.close()
        return ret

    def acquire_decimated(self, prog, **kwargs):
        kwargs, bar = self.overwrite_kwargs_for_remote(prog, kwargs)
        prog_name = type(prog).__name__
        ret = self._remote_call("run_program_decimated", prog_name, prog.cfg, **kwargs)
        boxed_callback = kwargs.get("round_callback")
        if boxed_callback is not None:
            type(self).drop_callback(boxed_callback)
        if bar is not None:
            bar.update(bar.total - bar.n)  # force to finish
            bar.close()
        return ret
