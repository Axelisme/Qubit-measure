import threading
from queue import Queue

import Pyro4


class CallbackWrapper:
    def exec(self):
        while True:
            job = self.JOB_Q.get()
            if job is None:
                break

            ir, args, kwargs = job
            try:
                # do not raise exception in this thread
                self.func(ir, *args, **kwargs)
            except Exception:
                pass

    def __init__(self, func: callable):
        self.func = func
        self.lock = threading.Lock()

        # these variables are protected by lock
        self.alive = True
        self.last_ir = -1  # initial to -1 to accept the first job

        # start worker thread
        self.WORKER_T = threading.Thread(target=self.exec, daemon=True)
        self.JOB_Q = Queue(maxsize=1)

        self.WORKER_T.start()

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

    def clear_last_job(self):
        assert self.lock.locked(), "This method should be called within lock"
        while not self.JOB_Q.empty():
            self.JOB_Q.get_nowait()

    def close(self):
        with self.lock:
            self.alive = False

            self.clear_last_job()
            self.JOB_Q.put_nowait(None)  # tell worker no more job
            self.WORKER_T.join(2)
