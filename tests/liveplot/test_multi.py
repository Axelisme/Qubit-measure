from __future__ import annotations

from matplotlib.figure import Figure
from zcu_tools.liveplot import AbsLivePlot, MultiLivePlot


class _Plotter(AbsLivePlot):
    def clear(self) -> None:
        pass

    def refresh(self) -> None:
        pass


def test_multi_liveplot_has_no_update_dispatch() -> None:
    group = MultiLivePlot(Figure(), {"a": _Plotter()})

    assert not hasattr(group, "update")
