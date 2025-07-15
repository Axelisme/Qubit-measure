import threading
import time
from copy import deepcopy
from functools import wraps
from typing import Callable, Generic, Optional, ParamSpec

from zcu_tools.utils.debug import print_traceback

P = ParamSpec("P")


class AsyncFunc(Generic[P]):
    """
    讓函數在非同步線程中執行，並確保最小間隔時間。
    """

    def __init__(
        self, func: Optional[Callable[P, None]], min_interval: float = 0.1
    ) -> None:
        """
        初始化 AsyncFunc 類別。

        Args:
            func (Optional[Callable]): 要執行的函數。
            min_interval (float, optional): 兩次執行之間的最小間隔時間。預設為 0.1 秒。
        """
        self.func = func
        self.min_interval = min_interval

        if min_interval <= 0:
            raise ValueError("min_interval must be greater than 0")

        if func is not None and not callable(func):
            raise TypeError("func must be a callable")

    def _init_worker_thread(self) -> None:
        self.lock = threading.Lock()

        # these variables are protected by lock
        self.acquiring = True
        self.have_new_job = threading.Event()
        self.last_job = None

        # start worker thread
        self.worker_t = threading.Thread(target=self.work_loop, daemon=True)
        self.worker_t.start()  # start worker thread

    def __enter__(self) -> Optional[Callable[P, None]]:
        """
        啟動非同步執行環境，並初始化相關資源。
        """
        if self.func is None:
            return None  # do nothing

        self._init_worker_thread()

        @wraps(self.func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> None:
            # this method may be called concurrently, so we need to protect it
            # also, make it executed in worker thread, to avoid blocking main thread
            with self.lock:
                # only keep the latest job
                if self.acquiring:
                    self.last_job = deepcopy((args, kwargs))
                    self.have_new_job.set()  # notify worker thread

            return None

        return wrapper

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        結束非同步執行環境，釋放相關資源。
        """
        if self.func is None:
            return  # do nothing

        self.last_job = None
        with self.lock:
            self.acquiring = False
            self.have_new_job.set()  # notify worker thread to exit
        self.worker_t.join()

    def work_loop(self) -> None:
        """
        工作執行緒的主迴圈，負責執行非同步任務。
        """
        assert self.func is not None, "This method should not be called if func is None"
        prev_start = time.time() - 2 * self.min_interval
        while True:
            self.have_new_job.wait()  # wait for new job

            # this don't need to be protected by lock
            # because it is only be set before event is set
            if not self.acquiring:
                break  # if not acquiring, exit

            # check if min_interval is satisfied
            if time.time() - prev_start < self.min_interval:
                time.sleep(self.min_interval / 10)
                continue

            with self.lock:  # get job
                job, self.last_job = self.last_job, None
                self.have_new_job.clear()  # clear flag

            # do not raise exception in this thread
            try:
                assert job is not None, "Job should not be None"
                args, kwargs = job
                self.func(*args, **kwargs)
            except Exception:
                print("Error in async func:")
                print_traceback()
            finally:
                prev_start = time.time()
