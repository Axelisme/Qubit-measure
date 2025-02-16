from typing import Callable, Optional
import threading
from queue import Queue

import Pyro4


class CallbackWrapper:
    def __init__(self, client, func: Optional[Callable]):
        self.client = client
        self.func = func

    def __enter__(self):
        if self.func is None:
            return self.func  # do nothing

        self.daemon = self.client.get_daemon()

        self.lock = threading.Lock()

        # these variables are protected by lock
        self.alive = True
        self.last_ir = -1  # initial to -1 to accept the first job

        # start worker thread
        self.WORKER_T = threading.Thread(target=self.exec, daemon=True)
        self.JOB_Q = Queue(maxsize=1)

        self.WORKER_T.start()

        self.daemon.register(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.func is None:
            return  # do nothing

        self.daemon.unregister(self)

        self.alive = False
        with self.lock:
            self.clear_last_job()
            self.JOB_Q.put_nowait(None)  # tell worker no more job
        self.WORKER_T.join(2)

    def clear_last_job(self):
        assert self.lock.locked(), "This method should be called within lock"
        while not self.JOB_Q.empty():
            self.JOB_Q.get_nowait()

    def exec(self):
        assert self.func is not None, "This method should not be called"
        while True:
            job = self.JOB_Q.get()
            if job is None:
                break  # end of job

            # do not raise exception in this thread
            try:
                ir, args, kwargs = job
                self.func(ir, *args, **kwargs)
            except BaseException as e:
                print(f"Error in callback: {e}")
                pass

    @Pyro4.expose
    @Pyro4.callback
    @Pyro4.oneway
    def oneway_callback(self, ir, *args, **kwargs):
        # this method is called by remote, so it may be called concurrently
        with self.lock:
            # only keep the latest job
            if ir > self.last_ir and self.alive:
                self.last_ir = ir
                self.clear_last_job()
                self.JOB_Q.put_nowait((ir, args, kwargs))
