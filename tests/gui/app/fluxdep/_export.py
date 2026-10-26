"""Synthetic aligned inputs shared by export seam and UI scenarios."""

import numpy as np
from zcu_tools.analysis.fluxdep.models import TransitionDict
from zcu_tools.analysis.fluxdep.search import ParamBounds
from zcu_tools.gui.app.analysis_project.entry import AnalysisTarget
from zcu_tools.gui.app.fluxdep.state import FluxDepState, SpectrumEntry


def aligned_export_state(target: AnalysisTarget) -> FluxDepState:
    """Return a fresh fit with native anchors and an unlocked Search stage."""
    state = FluxDepState(target)
    state.put_spectrum(
        SpectrumEntry(
            name="aligned",
            spec_type="TwoTone",
            raw={
                "dev_values": np.array([0.0, 1.0]),
                "fluxs": np.array([0.0, 0.5]),
                "freqs": np.array([4.0, 5.0]),
                "signals": np.ones((2, 2), dtype=complex),
            },
            points={
                "dev_values": np.array([0.0, 1.0]),
                "fluxs": np.array([0.0, 0.5]),
                "freqs": np.array([4.0, 5.0]),
            },
            flux_half=0.7,
            flux_int=1.7,
            flux_period=2.0,
            aligned=True,
            points_completed=True,
        )
    )
    state.set_selection(np.array([True, True]))
    state.set_fit_params(
        "",
        ParamBounds(EJ=(3.0, 12.0), EC=(0.3, 1.5), EL=(0.2, 1.2)),
        TransitionDict(transitions=[(0, 1)]),
        None,
        None,
    )
    state.set_fit_result((5.0, 1.2, 0.4))
    return state
