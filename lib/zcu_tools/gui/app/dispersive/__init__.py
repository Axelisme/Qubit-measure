"""dispersive-fit-gui — a standalone Qt GUI for fluxonium dispersive-shift fitting.

An optional analysis tool, a sibling of fluxdep-gui (``zcu_tools.gui.app.fluxdep``)
and measure-gui (``zcu_tools.gui.app.main``). It ports the dispersive analysis
notebook (``zcu_tools.notebook.analysis.dispersive``): read a fluxonium fit from
``params.json`` (the ``fluxdep_fit`` section fluxdep-gui writes), load a one-tone
spectrum, preprocess it (edelay / circle / phase / normalize), tune the coupling
``g`` and resonator frequency ``bare_rf`` against the spectrum (live, or scipy
auto-fit), then write the ``dispersive`` section back to ``params.json``.

The app skeleton (state / event bus / RPC transport / mpl + progress backends) is
shared with or copied from fluxdep-gui; the dispersive domain (preprocessing /
prediction / fitting / visualisation) is written here. The two coexist in one
``params.json``: dispersive reads ``fluxdep_fit`` and writes ``dispersive`` —
fluxdep-gui must be run first to produce the fit inputs.
"""
