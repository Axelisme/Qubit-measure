import threading

import Pyro4

from .config import config

Pyro4.config.SERIALIZER = "pickle"
Pyro4.config.SERIALIZERS_ACCEPTED = set(["pickle"])
Pyro4.config.PICKLE_PROTOCOL_VERSION = 4

# 取得或建立全域的 Daemon 實例，並以背景執行緒啟動 requestLoop
_daemon = None
_daemon_thread = None


def get_daemon():
    global _daemon, _daemon_thread
    if _daemon is None:
        _daemon = Pyro4.Daemon(host=config.LOCAL_IP, port=config.LOCAL_PORT)
        # 將 daemon.requestLoop 放在背景執行緒執行
        _daemon_thread = threading.Thread(target=_daemon.requestLoop, daemon=True)
        _daemon_thread.start()
    return _daemon


# 定義一個包裝器類別，內含你欲當作 callback 的函數。
# 加上 @Pyro4.oneway 表示此方法為 one-way 呼叫，遠端不等待回傳值
class CallbackWrapper(object):
    def __init__(self, func):
        self.func = func

    @Pyro4.expose
    @Pyro4.callback
    @Pyro4.oneway
    def oneway_callback(self, *args, **kwargs):
        return self.func(*args, **kwargs)


# 定義 decorator，將原函數包裝成 callback proxy 物件
def pyro_callback(func):
    daemon = get_daemon()

    # 用 CallbackWrapper 包裝原始函數
    callback = CallbackWrapper(func)

    # 將 callback 物件註冊到 daemon 中，取得其 URI
    daemon.register(callback)
    # uri = daemon.register(callback)
    # print(f"Registered callback {func.__name__} at {uri}")

    return callback
