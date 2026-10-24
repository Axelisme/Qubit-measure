"""Workflow and current-operation figure ownership, without frontend dependencies."""

from collections.abc import Mapping

from matplotlib.figure import Figure

from zcu_tools.plotting.plots import PlotHost, Plots


class WorkflowPlots(Plots):
    """Own workflow trends and the latest operation's independent plotting lifetime.

    Engine authors use this collection for trends across commits. The engine
    starts one operation at a time and finishes it before starting another.
    Pane and MCP readers use figures() for the same detached name projection,
    without acquiring Figure ownership or moving canvases. Workflow names may
    not start with ``operation/``; that namespace identifies current-operation
    figures. Finish this workflow producer before release; tab replacement and
    close release both groups. Terminal completion retains the latest group.
    """

    def __init__(self, host: PlotHost) -> None:
        super().__init__(host)
        self._operation: Plots | None = None

    def adopt(self, name: str, figure: Figure) -> None:
        """Register a workflow figure; reject the reserved operation/ prefix.

        All native FigureCollection ownership and name-conflict checks still
        apply. The restriction also covers subplots and live factories.
        """
        if name.startswith("operation/"):
            raise ValueError("Workflow figure names must not start with operation/")
        super().adopt(name, figure)

    def start_operation(self) -> Plots:
        """Release the previous finished group and publish a fresh same-host Plots.

        Call on the producer worker before invoking a core operation. The caller
        must finish the returned producer on every exit path. Release failures
        propagate without replacing the old group, so cleanup can be retried.
        Operation names do not inherit the workflow namespace restriction.
        """
        self.ensure_active()
        if self._operation is not None:
            self._operation.release()
        operation = self.fork()
        self._operation = operation
        return operation

    def figures(self) -> Mapping[str, Figure]:
        """Return workflow names and operation/<name> for the latest group.

        This detached mapping borrows native Figures; it never adopts them.
        Re-read when membership changes. Names within each group retain creation
        order, workflow first. Artist reads/rendering still require host ownership.
        """
        operation = self._operation
        figures = dict(self.items())
        if operation is not None:
            figures.update(
                (f"operation/{name}", figure) for name, figure in operation.items()
            )
        return figures

    def release(self) -> None:
        """Release the current operation, then workflow presentation after finish.

        Neither group loses its retained Figures. A failed release propagates;
        retrying only repeats presentations the host has not released yet.
        """
        if self._operation is not None:
            self._operation.release()
        super().release()
