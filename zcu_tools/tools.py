from numbers import Number
from typing import Any, Callable, Dict, Literal, Optional

import threading


def deepupdate(
    d: Dict[str, Any],
    u: Dict[str, Any],
    behavior: Literal["error", "force", "ignore"] = "error",
):
    def conflict_handler(d: Dict[str, Any], u: Dict[str, Any], k: Any):
        if behavior == "error":
            raise KeyError(f"Key {k} already exists in {d}.")
        elif behavior == "force":
            d[k] = u[k]

    for k, v in u.items():
        if k not in d:
            d[k] = v
        elif isinstance(v, dict) and isinstance(d[k], dict):
            deepupdate(d[k], v, behavior)
        else:
            conflict_handler(d, u, k)


# type cast all numpy types to python types
def numpy2number(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    if isinstance(obj, dict):
        return {k: numpy2number(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy2number(v) for v in obj]
    if hasattr(obj, "item"):
        return obj.item()
    return obj


def make_sweep(
    start: Number,
    stop: Optional[Number] = None,
    expts: Optional[int] = None,
    step: Optional[Number] = None,
    force_int: bool = False,
) -> dict:
    err_str = "Not enough information to define a sweep."
    if expts is None:
        assert stop is not None, err_str
        assert step is not None, err_str
        expts = int((stop - start) / step + 0.99)  # pyright: ignore[reportOperatorIssue]
    elif step is None:
        assert stop is not None, err_str
        assert expts is not None, err_str
        step = (stop - start) / expts  # pyright: ignore[reportOperatorIssue]

    if force_int:
        start = int(start)  # pyright: ignore
        step = int(step)  # pyright: ignore
        expts = int(expts)

    stop = start + step * expts  # pyright: ignore[reportOperatorIssue]

    assert expts > 0, f"expts must be greater than 0, but got {expts}"
    assert step != 0, f"step must not be zero, but got {step}"

    return {"start": start, "stop": stop, "expts": expts, "step": step}


def get_ip_address(iface):
    import fcntl
    import socket
    import struct

    # get the IP address of the interface
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(
        fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack("256s", bytes(iface[:15], "utf-8")),
        )[20:24]
    )


class AsyncFunc:
    def __init__(self, func: Optional[Callable]):
        self.func = func

    def __enter__(self):
        if self.func is None:
            return None  # do nothing

        self.lock = threading.Lock()

        # these variables are protected by lock
        self.acquiring = True
        self.have_new_job = threading.Event()
        self.last_ir = -1  # initial to -1 to accept the first job
        self.last_job = None

        # start worker thread
        self.worker_t = threading.Thread(target=self.work_loop, daemon=True)
        self.worker_t.start()  # start worker thread

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.func is None:
            return  # do nothing

        with self.lock:
            self.acquiring = False
            self.have_new_job.set()  # notify worker thread to exit
        self.worker_t.join()

    def work_loop(self):
        assert self.func is not None, "This method should not be called if func is None"
        while True:
            self.have_new_job.wait()  # wait for new job

            if not self.acquiring:
                break  # if not acquiring, exit

            with self.lock:  # get job
                job, self.last_job = self.last_job, None
                self.have_new_job.clear()  # clear flag

            # do not raise exception in this thread
            try:
                assert job is not None, "Job should not be None"
                ir, args, kwargs = job
                self.func(ir, *args, **kwargs)
            except BaseException as e:
                print(f"Error in callback: {e}")

    def __call__(self, ir: int, *args, **kwargs):
        # this method may be called concurrently, so we need to protect it
        # also, make it executed in worker thread, to avoid blocking main thread
        with self.lock:
            # only keep the latest job
            if ir > self.last_ir and self.acquiring:
                self.last_ir = ir
                self.last_job = (ir, args, kwargs)
                self.have_new_job.set()  # notify worker thread
