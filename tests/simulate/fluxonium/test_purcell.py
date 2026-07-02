from __future__ import annotations

import zcu_tools.simulate.fluxonium as fluxonium
import zcu_tools.simulate.fluxonium.coherence as coherence


def test_purcell_export_uses_correct_spelling() -> None:
    assert (
        fluxonium.calculate_purcell_t1_vs_flux is coherence.calculate_purcell_t1_vs_flux
    )
    assert "calculate_purcell_t1_vs_flux" in fluxonium.__all__
    assert "calculate_percell_t1_vs_flux" not in fluxonium.__all__
    assert not hasattr(fluxonium, "calculate_percell_t1_vs_flux")
