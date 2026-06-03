"""Interactive matplotlib widgets for the fluxdep pipeline.

Each widget owns its own ``FigureCanvasQTAgg`` and wires mouse events via
``mpl_connect`` — the same interaction the notebook's ipywidgets versions used,
with the ipywidgets controls replaced by Qt widgets. The numerical core of each
tool is reused verbatim from ``zcu_tools.notebook.analysis.fluxdep.processing``;
only the UI shell is rewritten.
"""
