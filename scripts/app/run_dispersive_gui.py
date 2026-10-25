"""User entrypoint for the dispersive GUI composition root."""

import sys

from apps.dispersive import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
