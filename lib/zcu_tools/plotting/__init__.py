"""Named figure management and caller-defined owner updates."""

from pathlib import Path

# Pass this path to matplotlib.style.use or matplotlib.style.context.
DEFAULT_STYLE = Path(__file__).with_name("experiment.mplstyle")
