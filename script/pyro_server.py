#!/usr/bin/env python3
"""This file starts a pyro nameserver and the proxying server."""

import argparse
import os
import sys
import threading
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from zcu_tools.remote.pyro import start_nameserver, start_server

############
# default parameters
ns_port = 8887
ns_host = "0.0.0.0"
# iface = "tailscale0"
iface = "eth0"

############
# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--port", "-p", type=int, default=ns_port, help="The nameserver port"
)
parser.add_argument("--host", "-H", default=ns_host, help="The nameserver host ip")
parser.add_argument(
    "--iface", "-i", default=iface, help="The network interface to bind to"
)

args = parser.parse_args()
ns_host = args.host
ns_port = args.port
iface = args.iface

############
# start the nameserver process
try:
    ns_t = threading.Thread(
        target=start_nameserver, args=(ns_host, ns_port), daemon=True
    )
    ns_t.start()

except Exception as e:
    print(f"Failed to start nameserver: {e}", file=sys.stderr)
    sys.exit(1)

time.sleep(2)  # wait for the nameserver to start up

############
# start the qick proxy server
start_server(ns_host, ns_port, iface=iface)
