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
from .models import energy2linearform, energy2transition
from .onetone import InteractiveOneTone

# Data processing
from .processing import spectrum2d_findpoint

__all__ = [
    # Interactive tools
    "InteractiveFindPoints",
    "InteractiveLines",
    "InteractiveSelector",
    "VisualizeSpet",
    "InteractiveOneTone",
    # Data processing
    "spectrum2d_findpoint",
    # Physical models
    "energy2linearform",
    "energy2transition",
    # Fitting algorithms
    "fit_spectrum",
    "search_in_database",
]
