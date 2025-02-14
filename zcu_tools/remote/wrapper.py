import threading
from queue import Queue

import Pyro4


class CallbackWrapper:
    def __init__(self, func: callable):
        self.func = func
        self.lock = threading.Lock()
        self.worker_t = threading.Thread(target=self.exec)
        self.worker_t.start()

        self.last_job = Queue(maxsize=1)

        # these variables are protected by lock
        self.alive = True
        self.last_ir = -1  # initial to -1 to accept the first job

    def exec(self):
        while True:
            job = self.last_job.get()
            if job is None:
                break  # exit

            ir, args, kwargs = job
            self.func(ir, *args, **kwargs)

    @Pyro4.expose
    @Pyro4.callback
    @Pyro4.oneway
    def oneway_callback(self, ir, *args, **kwargs):
        with self.lock:
            # only keep the latest job
            if ir > self.last_ir and self.alive:
                self.last_ir = ir
                self.clear_last_job()
                self.last_job.put_nowait((ir, args, kwargs))

    def clear_last_job(self):
        assert self.lock.locked(), "This method should be called within lock"
        while not self.last_job.empty():
            self.last_job.get_nowait()

    def close(self):
        with self.lock:
            self.alive = False

            self.clear_last_job()
            self.last_job.put_nowait(None)  # tell worker to exit

            if self.worker_t.is_alive():
                self.worker_t.join(timeout=2)
