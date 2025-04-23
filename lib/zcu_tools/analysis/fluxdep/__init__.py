"""Flux-dependent analysis submodules.

This package provides tools for analyzing flux-dependent spectroscopy data,
including interactive tools, visualization, data processing, physical models,
fitting algorithms, and persistence functions.
"""

# Interactive tools
# Fitting algorithms
from .fitting import fit_spectrum, search_in_database
from .interactive import (
    InteractiveFindPoints,
    InteractiveLines,
    InteractiveSelector,
    VisualizeSpet,
)

# Physical models
from .models import calculate_energy, energy2linearform, energy2transition

# Persistence functions
from .persistence import dump_result, dump_spects, load_result, load_spects

# Data processing
from .processing import flx2mA, format_rawdata, mA2flx, spectrum_analyze

__all__ = [
    # Interactive tools
    "InteractiveFindPoints",
    "InteractiveLines",
    "InteractiveSelector",
    "VisualizeSpet",
    # Data processing
    "flx2mA",
    "format_rawdata",
    "mA2flx",
    "spectrum_analyze",
    # Physical models
    "calculate_energy",
    "energy2linearform",
    "energy2transition",
    # Fitting algorithms
    "fit_spectrum",
    "search_in_database",
    # Persistence functions
    "dump_result",
    "load_result",
    "dump_spects",
    "load_spects",
]
