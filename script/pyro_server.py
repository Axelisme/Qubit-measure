#!/usr/bin/env python3
"""This file starts a pyro nameserver and the proxying server."""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.parent.absolute()

print(HERE)
sys.path.append(HERE.as_posix())

from qick.pyro import start_server  # noqa


############
# parameters
############

bitfile = "qick/qick_216.bit"
proxy_name = "myqick"
ns_port = 8887
# set to 0.0.0.0 to allow access from outside systems
# ns_host = 'localhost'
ns_host = "0.0.0.0"
iface = "tailscale0"

############

# start the nameserver process
ns_proc = subprocess.Popen(
    [
        f"PYRO_SERIALIZERS_ACCEPTED=pickle PYRO_PICKLE_PROTOCOL_VERSION=4 pyro4-ns -n {ns_host} -p {ns_port}"
    ],
    shell=True,
)

# wait for the nameserver to start up
time.sleep(5)

# start the qick proxy server
start_server(
    bitfile=str(HERE / bitfile),
    proxy_name=proxy_name,
    ns_host=ns_host,
    ns_port=ns_port,
    iface=iface,
)