"""User entrypoint for the measure GUI composition root."""

import sys

from apps.measure import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
