import Pyro4


class CallbackWrapper:
    def __init__(self, func: callable):
        self.func = func

    @Pyro4.expose
    @Pyro4.callback
    @Pyro4.oneway
    def oneway_callback(self, *args, **kwargs):
        return self.func(*args, **kwargs)
