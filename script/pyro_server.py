#!/usr/bin/env python3
"""This file starts a pyro nameserver and the proxying server."""

import argparse
import os
import sys
import threading
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

############
# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--host", "-H", default="localhost", help="Host ip")
parser.add_argument("--port", "-p", type=int, default=0, help="Daemon port")
parser.add_argument("--ns-port", "-np", type=int, default=8080, help="Nameserver port")
parser.add_argument(
    "--soc", "-s", default="v2", choices=["v1", "v2"], help="bitfile version"
)

args = parser.parse_args()

############
# start the nameserver process
import os  # noqa
import sys  # noqa

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from zcu_tools.remote.pyro import start_nameserver, start_server  # noqa

ns_t = threading.Thread(target=start_nameserver, args=(args.ns_port,), daemon=True)
ns_t.start()
time.sleep(2)  # wait for the nameserver to start up

############
# start the qick proxy server
start_server(args.host, args.port, args.ns_port, version=args.soc)
