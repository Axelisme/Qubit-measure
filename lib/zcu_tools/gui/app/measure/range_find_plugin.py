"""Captured numeric spectrum selection shared by desktop and remote analysis."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from zcu_tools.analysis.dips import DipFinderOptions, SpectrumDip, find_spectrum_dips
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.interactive import Action, Command, PluginDefinition, Session
from zcu_tools.gui.remote.errors import RemoteError
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec, validate_params


@dataclass(frozen=True)
class DipSelection:
    """A sampled MHz position, optional caller component and numerical estimate.

    estimate is None for a manual point that is not an automatic candidate.
    component is None for an unassigned point; labels belong to the caller.
    """

    freq: float
    component: str | None
    estimate: SpectrumDip | None


@dataclass(frozen=True)
class RangeFindState:
    """Accepted controls, score-ranked candidates and frequency-ordered selections.

    options controls automatic selection only; manual edits may exceed max_dips.
    candidates contains all separated dips passing the current numerical filters.
    selected retains manual points and caller component assignments until recompute.
    """

    options: DipFinderOptions
    candidates: tuple[SpectrumDip, ...]
    selected: tuple[DipSelection, ...]


class RangeFindPlugin[R](PluginDefinition[RangeFindState, R]):
    """One captured magnitude spectrum with caller-owned component labels.

    freqs are absolute MHz and magnitude is nonnegative, matching 1-D arrays
    with at least 11 finite points. Arrays are copied and sorted. components
    contains unique nonblank labels, initially paired in ascending frequency.
    options=None uses DipFinderOptions defaults. build_result receives the final
    detached state after accepted Finish; at least one selected point is required.

    set_parameters accepts DipFinderOptions and replaces the automatic guess,
    retaining assignments at surviving frequencies. add_dip accepts finite MHz
    within the scan and snaps to the nearest measured sample. remove_dip accepts
    a zero-based selected index. assign_dip accepts (index, caller label or None),
    clearing a label's old assignment when moving it to another dip. Invalid
    edits raise InvalidInputError without publication. Commands expose the same
    actions and declare their wire arguments in commands. These actions belong
    to the Session owner loop; input closes on Finish or host cancellation.
    """

    freqs: NDArray[np.float64]
    magnitude: NDArray[np.float64]
    components: tuple[str, ...]
    set_parameters: Action[RangeFindState, DipFinderOptions]
    add_dip: Action[RangeFindState, float]
    remove_dip: Action[RangeFindState, int]
    assign_dip: Action[RangeFindState, tuple[int, str | None]]

    def __init__(
        self,
        freqs: NDArray[np.float64],
        magnitude: NDArray[np.float64],
        *,
        components: tuple[str, ...] = (),
        options: DipFinderOptions | None = None,
        build_result: Callable[[RangeFindState], R],
    ) -> None:
        if options is None:
            options = DipFinderOptions()
        candidates = find_spectrum_dips(
            freqs, magnitude, replace(options, max_dips=None)
        )
        if any(not name.strip() for name in components) or len(set(components)) != len(
            components
        ):
            raise ValueError("components must contain unique nonblank labels")
        chosen = sorted(candidates[: options.max_dips], key=lambda dip: dip.freq)
        selected = tuple(
            DipSelection(
                dip.freq, components[index] if index < len(components) else None, dip
            )
            for index, dip in enumerate(chosen)
        )
        action = Action(self._recompute)
        specs = (
            ParamSpec("smoothing_sigma", JsonType.NUMBER),
            ParamSpec("prominence_factor", JsonType.NUMBER),
            ParamSpec("max_fwhm", JsonType.NUMBER),
            ParamSpec("min_separation", JsonType.NUMBER),
            ParamSpec("max_dips", JsonType.INTEGER, nullable=True),
        )

        def parameters_command(
            session: Session[RangeFindState], params: Mapping[str, object]
        ) -> RangeFindState:
            values = _validated(specs, params)
            count = values["max_dips"]
            if count is not None and (
                isinstance(count, bool) or not isinstance(count, int)
            ):
                raise InvalidInputError("max_dips must be an integer or null")
            try:
                controls = DipFinderOptions(
                    smoothing_sigma=_number(values["smoothing_sigma"]),
                    prominence_factor=_number(values["prominence_factor"]),
                    max_fwhm=_number(values["max_fwhm"]),
                    min_separation=_number(values["min_separation"]),
                    max_dips=count,
                )
            except ValueError as exc:
                raise InvalidInputError(str(exc)) from exc
            return action.execute(session, controls)

        object.__setattr__(self, "add_dip", Action(self._add))
        object.__setattr__(self, "remove_dip", Action(_remove))
        object.__setattr__(self, "assign_dip", Action(self._assign))
        super().__init__(
            plugin_id="range_find",
            seed=RangeFindState(options, candidates, selected),
            commands=(
                Command("set_parameters", specs, parameters_command),
                *_selection_commands(self),
            ),
            can_finish=_require_selection,
            build_result=build_result,
        )
        order = np.argsort(freqs)
        object.__setattr__(self, "freqs", freqs[order].copy())
        object.__setattr__(self, "magnitude", magnitude[order].copy())
        self.freqs.flags.writeable = False
        self.magnitude.flags.writeable = False
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "set_parameters", action)

    def _recompute(
        self, state: RangeFindState, options: DipFinderOptions
    ) -> RangeFindState:
        candidates = find_spectrum_dips(
            self.freqs, self.magnitude, replace(options, max_dips=None)
        )
        assignments = {dip.freq: dip.component for dip in state.selected}
        selected = tuple(
            DipSelection(dip.freq, assignments.get(dip.freq), dip)
            for dip in sorted(candidates[: options.max_dips], key=lambda dip: dip.freq)
        )
        return RangeFindState(options, candidates, selected)

    def _add(self, state: RangeFindState, frequency: float) -> RangeFindState:
        if (
            isinstance(frequency, bool)
            or not isfinite(frequency)
            or not self.freqs[0] <= frequency <= self.freqs[-1]
        ):
            raise InvalidInputError("frequency must be finite and within the spectrum")
        frequency = float(self.freqs[np.argmin(np.abs(self.freqs - frequency))])
        if any(dip.freq == frequency for dip in state.selected):
            raise InvalidInputError("This sample is already selected")
        estimate = next(
            (dip for dip in state.candidates if dip.freq == frequency), None
        )
        selected = (*state.selected, DipSelection(frequency, None, estimate))
        return replace(
            state, selected=tuple(sorted(selected, key=lambda dip: dip.freq))
        )

    def _assign(
        self, state: RangeFindState, pair: tuple[int, str | None]
    ) -> RangeFindState:
        index, component = pair
        _require_index(state, index)
        if component is not None and component not in self.components:
            raise InvalidInputError(
                "component must be one of the caller labels or null"
            )
        selected = tuple(
            replace(dip, component=component)
            if position == index
            else replace(dip, component=None)
            if component is not None and dip.component == component
            else dip
            for position, dip in enumerate(state.selected)
        )
        return replace(state, selected=selected)


def _require_index(state: RangeFindState, index: int) -> None:
    if isinstance(index, bool) or not 0 <= index < len(state.selected):
        raise InvalidInputError("index must identify a selected dip (zero based)")


def _remove(state: RangeFindState, index: int) -> RangeFindState:
    _require_index(state, index)
    return replace(state, selected=state.selected[:index] + state.selected[index + 1 :])


def _require_selection(state: RangeFindState) -> None:
    if not state.selected:
        raise FailedPreconditionError("Select at least one dip before Done")


def _selection_commands[R](
    plugin: RangeFindPlugin[R],
) -> tuple[Command[RangeFindState], ...]:
    add_specs = (ParamSpec("frequency", JsonType.NUMBER),)
    remove_specs = (ParamSpec("index", JsonType.INTEGER),)
    assign_specs = (
        *remove_specs,
        ParamSpec("component", JsonType.STRING, nullable=True),
    )

    def add(
        session: Session[RangeFindState], params: Mapping[str, object]
    ) -> RangeFindState:
        return plugin.add_dip.execute(
            session, _number(_validated(add_specs, params)["frequency"])
        )

    def remove(
        session: Session[RangeFindState], params: Mapping[str, object]
    ) -> RangeFindState:
        return plugin.remove_dip.execute(
            session, _index(_validated(remove_specs, params)["index"])
        )

    def assign(
        session: Session[RangeFindState], params: Mapping[str, object]
    ) -> RangeFindState:
        values = _validated(assign_specs, params)
        component = values["component"]
        if component is not None and not isinstance(component, str):
            raise InvalidInputError("component must be a string or null")
        return plugin.assign_dip.execute(session, (_index(values["index"]), component))

    return (
        Command("add_dip", add_specs, add),
        Command("remove_dip", remove_specs, remove),
        Command("assign_dip", assign_specs, assign),
    )


def _index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidInputError("index must be an integer")
    return value


def _validated(
    specs: tuple[ParamSpec, ...], params: Mapping[str, object]
) -> Mapping[str, object]:
    try:
        return validate_params(specs, params)
    except RemoteError as exc:
        raise InvalidInputError(str(exc)) from exc


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInputError("expected a number")
    return float(value)
