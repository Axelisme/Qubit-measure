#!/usr/bin/env python3
"""This file starts a pyro nameserver and the proxying server."""

import atexit
import os
import subprocess
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from zcu_tools.remote.pyro import start_server

############
# parameters
############

proxy_name = "myqick"
ns_port = 8887
# set to 0.0.0.0 to allow access from outside systems
ns_host = "localhost"
# ns_host = "0.0.0.0"
# iface = "tailscale0"
iface = "eth0"

############

# start the nameserver process
try:
    ns_proc = subprocess.Popen(
        [
            f"PYRO_SERIALIZERS_ACCEPTED=pickle PYRO_PICKLE_PROTOCOL_VERSION=4 pyro4-ns -n {ns_host} -p {ns_port}"
        ],
        shell=True,
    )
except subprocess.SubprocessError as e:
    print(f"Failed to start nameserver: {e}", file=sys.stderr)
    sys.exit(1)


# cleanup the nameserver process on exit
def cleanup():
    try:
        ns_proc.terminate()  # 先嘗試溫和地終止
        try:
            ns_proc.wait(timeout=2)  # 等待最多 2 秒
        except subprocess.TimeoutExpired:
            ns_proc.kill()  # 如果超時，強制終止
            ns_proc.wait()  # 確保程序完全結束
    except Exception:
        print("Error during nameserver cleanup", file=sys.stderr)


# register the cleanup function
atexit.register(cleanup)


# wait for the nameserver to start up
time.sleep(2)

# start the qick proxy server
start_server(ns_host, ns_port, iface=iface)
