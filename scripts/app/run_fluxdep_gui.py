"""User entrypoint for the fluxdep GUI composition root."""

import sys

from apps.fluxdep import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
