"""Reset adapters accumulate scalars and propose tested finisher modules."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.cfg_model import (
    BathResetCfg,
    BathResetSpec,
    PulseResetCfg,
    PulseResetSpec,
    TwoPulseResetCfg,
    TwoPulseResetSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import RunRecord
from zcu_tools.format_version import YamlValue
from zcu_tools.gui.app.measure.adapter import (
    AnalyzeResult,
    ModuleWriteback,
    ParameterWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import (
    ContextSourceFault,
    ContextSourceUnavailableError,
    EntryDraft,
    PointUseRequest,
)
from zcu_tools.gui.session.types import ParameterContext, SessionEnv
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import AcceptedPayload, LedgerEvent, RoleResolutionError
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral

from tests.zcu_lab._writeback_support import WritebackSession
from tests.zcu_lab._writeback_support import writeback_registry as _registry
from zcu_lab.components import QubitSchema
from zcu_lab.v2.twotone.reset.bath.freq.core import FreqGainCfg, FreqGainResult
from zcu_lab.v2.twotone.reset.bath.freq.gui import BathFreqGainAnalyzeResult
from zcu_lab.v2.twotone.reset.bath.length.core import LengthCfg as BathLengthCfg
from zcu_lab.v2.twotone.reset.bath.length.core import LengthResult as BathLengthResult
from zcu_lab.v2.twotone.reset.bath.length.gui import BathLengthAnalyzeResult
from zcu_lab.v2.twotone.reset.bath.phase.core import PhaseCfg, PhaseResult
from zcu_lab.v2.twotone.reset.bath.phase.gui import BathPhaseAnalyzeResult
from zcu_lab.v2.twotone.reset.dual_tone.freq.core import FreqCfg as DualFreqCfg
from zcu_lab.v2.twotone.reset.dual_tone.freq.core import FreqResult as DualFreqResult
from zcu_lab.v2.twotone.reset.dual_tone.freq.gui import DualToneFreqAnalyzeResult
from zcu_lab.v2.twotone.reset.dual_tone.length.core import LengthCfg as DualLengthCfg
from zcu_lab.v2.twotone.reset.dual_tone.length.core import (
    LengthResult as DualLengthResult,
)
from zcu_lab.v2.twotone.reset.dual_tone.length.gui import DualToneLengthAnalyzeResult
from zcu_lab.v2.twotone.reset.dual_tone.power.core import PowerCfg, PowerResult
from zcu_lab.v2.twotone.reset.dual_tone.power.gui import DualTonePowerAnalyzeResult
from zcu_lab.v2.twotone.reset.single_tone.freq.core import FreqCfg as SingleFreqCfg
from zcu_lab.v2.twotone.reset.single_tone.freq.core import (
    FreqResult as SingleFreqResult,
)
from zcu_lab.v2.twotone.reset.single_tone.freq.gui import SingleToneFreqAnalyzeResult
from zcu_lab.v2.twotone.reset.single_tone.length.core import (
    LengthCfg as SingleLengthCfg,
)
from zcu_lab.v2.twotone.reset.single_tone.length.core import (
    LengthResult as SingleLengthResult,
)
from zcu_lab.v2.twotone.reset.single_tone.length.gui import (
    SingleToneLengthAnalyzeResult,
)

_RESET_NAMES = {
    "single": "twotone/reset/single_tone/freq",
    "dual": "twotone/reset/dual_tone/power",
    "bath": "twotone/reset/bath/phase",
    "single-freq": "twotone/reset/single_tone/freq",
    "dual-freq": "twotone/reset/dual_tone/freq",
    "dual-power": "twotone/reset/dual_tone/power",
    "bath-freq-gain": "twotone/reset/bath/freq_gain",
    "bath-phase": "twotone/reset/bath/phase",
    "single-length": "twotone/reset/single_tone/length",
    "dual-length": "twotone/reset/dual_tone/length",
    "bath-length": "twotone/reset/bath/length",
}
_RESET_ROOTS = {
    "single": ("reset_10",),
    "dual": ("reset_120",),
    "bath": ("reset_bath", "reset_bath_e"),
}
_RESET_EXT: dict[str, YamlValue] = {
    "reset_f": 4200.0,
    "reset_fwhm": 8.0,
    "reset_f1": 4300.0,
    "reset_f2": 4400.0,
    "reset_gain1": 0.61,
    "reset_gain2": 0.71,
    "reset_bath_freq": 6300.0,
    "reset_bath_gain": 0.25,
    "reset_bath_max_phase": 31.0,
    "reset_bath_min_phase": 41.0,
}


@dataclass
class ResetExample:
    source: RunRecord[ExpCfgModel, object]
    analysis: AnalyzeResult
    scalars: tuple[tuple[str, float], ...]
    expected: dict[str, dict[str, YamlValue]]
    tested: PulseResetCfg | TwoPulseResetCfg | BathResetCfg


def _reset_template(
    family: str, *, shift: float = 0.0
) -> PulseResetCfg | TwoPulseResetCfg | BathResetCfg:
    pulse: dict[str, YamlValue] = {
        "type": "pulse",
        "ch": 0,
        "nqz": 1,
        "freq": 3000.0 + shift,
        "gain": 0.3,
        "waveform": {"style": "const", "length": 0.5},
        "phase": 17.0,
        "post_delay": 0.2,
    }
    if family == "single":
        return PulseResetCfg.model_validate({"type": "reset/pulse", "pulse_cfg": pulse})
    if family == "dual":
        return TwoPulseResetCfg.model_validate(
            {
                "type": "reset/two_pulse",
                "pulse1_cfg": pulse,
                "pulse2_cfg": {
                    **pulse,
                    "freq": 3100.0 + shift,
                    "gain": 0.4,
                    "waveform": {"style": "const", "length": 0.6},
                },
            }
        )
    return BathResetCfg.model_validate(
        {
            "type": "reset/bath",
            "cavity_tone_cfg": {**pulse, "freq": 6000.0 + shift, "gain": 0.2},
            "qubit_tone_cfg": {**pulse, "freq": 4000.0 + shift},
            "pi2_cfg": {**pulse, "freq": 4000.0 + shift, "phase": 11.0},
        }
    )


def _reset_example(name: str, *, shift: float = 0.0) -> ResetExample:
    family = (
        "single"
        if name.startswith("single")
        else "dual"
        if name.startswith("dual")
        else "bath"
    )
    tested = _reset_template(family, shift=shift)
    readout = {
        "type": "readout/pulse",
        "pulse_cfg": {
            "type": "pulse",
            "ch": 1,
            "nqz": 1,
            "freq": 6100.0,
            "gain": 0.2,
            "waveform": {"style": "const", "length": 1.0},
        },
        "ro_cfg": {"ro_ch": 0, "ro_length": 1.0},
    }
    modules = {"tested_reset": tested.model_dump(mode="json"), "readout": readout}
    sweep = {"start": 0.0, "stop": 1.0, "expts": 2, "step": 1.0}
    axis = np.array([0.0, 1.0], dtype=np.float64)
    signals = np.array([1.0 + 0.0j, 0.0 + 1.0j])
    match name:
        case "single" | "single-freq":
            cfg = SingleFreqCfg.model_validate(
                {"modules": modules, "sweep": {"freq": sweep}}
            )
            data = SingleFreqResult(axis, signals)
            analysis = SingleToneFreqAnalyzeResult(4300.0, 8.0)
            scalars = (("$qubit.ext.reset_f", 4300.0), ("$qubit.ext.reset_fwhm", 8.0))
        case "dual-freq":
            cfg = DualFreqCfg.model_validate(
                {"modules": modules, "sweep": {"freq1": sweep, "freq2": sweep}}
            )
            data = DualFreqResult(axis, axis, np.zeros((2, 2), dtype=np.complex128))
            analysis = DualToneFreqAnalyzeResult(4300.0, 4400.0)
            scalars = (("$qubit.ext.reset_f1", 4300.0), ("$qubit.ext.reset_f2", 4400.0))
        case "dual" | "dual-power":
            cfg = PowerCfg.model_validate(
                {"modules": modules, "sweep": {"gain1": sweep, "gain2": sweep}}
            )
            data = PowerResult(axis, axis, np.zeros((2, 2), dtype=np.complex128))
            analysis = DualTonePowerAnalyzeResult(0.21, 0.31)
            scalars = (
                ("$qubit.ext.reset_gain1", 0.21),
                ("$qubit.ext.reset_gain2", 0.31),
            )
        case "bath-freq-gain":
            cfg = FreqGainCfg.model_validate(
                {"modules": modules, "sweep": {"freq": sweep, "gain": sweep}}
            )
            data = FreqGainResult(axis, axis, np.zeros((2, 2, 4), dtype=np.complex128))
            analysis = BathFreqGainAnalyzeResult(0.25, 6300.0)
            scalars = (
                ("$qubit.ext.reset_bath_gain", 0.25),
                ("$qubit.ext.reset_bath_freq", 6300.0),
            )
        case "bath" | "bath-phase":
            cfg = PhaseCfg.model_validate(
                {"modules": modules, "sweep": {"phase": sweep}}
            )
            data = PhaseResult(axis, signals)
            analysis = BathPhaseAnalyzeResult(75.0, -105.0)
            scalars = (
                ("$qubit.ext.reset_bath_max_phase", 75.0),
                ("$qubit.ext.reset_bath_min_phase", -105.0),
            )
        case "single-length":
            cfg = SingleLengthCfg.model_validate(
                {"modules": modules, "sweep": {"length": sweep}}
            )
            data = SingleLengthResult(axis, signals)
            analysis = SingleToneLengthAnalyzeResult()
            scalars = ()
        case "dual-length":
            cfg = DualLengthCfg.model_validate(
                {"modules": modules, "sweep": {"length": sweep}}
            )
            data = DualLengthResult(axis, signals)
            analysis = DualToneLengthAnalyzeResult()
            scalars = ()
        case "bath-length":
            cfg = BathLengthCfg.model_validate(
                {"modules": modules, "sweep": {"length": sweep}}
            )
            data = BathLengthResult(axis, np.zeros((2, 4), dtype=np.complex128))
            analysis = BathLengthAnalyzeResult()
            scalars = ()
        case _:
            raise ValueError(name)
    return ResetExample(
        RunRecord[ExpCfgModel, object](cfg, data),
        analysis,
        scalars,
        _reset_expected(name, tested),
        tested,
    )


def _reset_expected(
    name: str, tested: PulseResetCfg | TwoPulseResetCfg | BathResetCfg
) -> dict[str, dict[str, YamlValue]]:
    expected: dict[str, dict[str, YamlValue]] = {}
    # Independent input models give the entire expected schema, including defaults.
    if name in ("single", "single-freq") and isinstance(tested, PulseResetCfg):
        expected["reset_10"] = tested.with_updates(
            pulse_cfg=tested.pulse_cfg.with_updates(freq=4300.0)
        ).to_dict()
    if name in ("dual", "dual-power") and isinstance(tested, TwoPulseResetCfg):
        expected["reset_120"] = tested.with_updates(
            pulse1_cfg=tested.pulse1_cfg.with_updates(gain=0.21),
            pulse2_cfg=tested.pulse2_cfg.with_updates(gain=0.31),
        ).to_dict()
    if name in ("bath", "bath-phase") and isinstance(tested, BathResetCfg):
        expected["reset_bath"] = tested.with_updates(
            pi2_cfg=tested.pi2_cfg.with_updates(phase=75.0)
        ).to_dict()
        expected["reset_bath_e"] = tested.with_updates(
            pi2_cfg=tested.pi2_cfg.with_updates(phase=-105.0)
        ).to_dict()
    return expected


def _reset_capture() -> ParameterContext:
    old = {
        "ch": 1,
        "nqz": 1,
        "freq": 4120.0,
        "gain": 0.7,
        "waveform": {"style": "const", "length": 0.9},
    }
    single = PulseResetSpec.model_validate({"pulse_cfg": old})
    dual = TwoPulseResetSpec.model_validate(
        {"pulse1_cfg": old, "pulse2_cfg": {**old, "freq": 4220.0, "gain": 0.8}}
    )
    ground = BathResetSpec.model_validate(
        {
            "cavity_tone_cfg": {**old, "freq": 6100.0},
            "qubit_tone_cfg": old,
            "pi2_cfg": {**old, "phase": 31.0},
        }
    )
    excited = ground.model_copy(
        update={
            "cavity_tone_cfg": ground.cavity_tone_cfg.model_copy(
                update={"freq": 6200.0}
            )
            if not isinstance(ground.cavity_tone_cfg, str)
            else ground.cavity_tone_cfg,
            "pi2_cfg": ground.pi2_cfg.model_copy(update={"phase": 41.0})
            if not isinstance(ground.pi2_cfg, str)
            else ground.pi2_cfg,
        }
    )
    library = LibrarySnapshot(
        ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
    )
    components: dict[str, QubitSchema] = {}
    for index in (1, 2):
        qubit = f"Q{index}"
        slots: dict[str, str] = {}
        for slot, spec in (
            ("sideband", single),
            ("dual_tone", dual),
            ("bath_g", ground),
            ("bath_e", excited),
            ("reset", single),
        ):
            root = f"{qubit}.reset.old_{slot}"
            slots[slot] = root
            library = library.replaced_module(root, spec)
        components[qubit] = QubitSchema(
            kind="qubit/transmon", module=slots, ext=deepcopy(_RESET_EXT)
        )
    return ParameterContext(
        point=PointDocument.model_construct(
            format="zcu.parameter-container",
            format_version="1.0",
            general=PointGeneral(created_at="2026-10-09T00:00:00Z"),
            components=components,
        ),
        library=library,
        target=EntryDraft(entry_id="reset-contract"),
        roles=None,
    ).bind_aliases((AliasRow("qubit", "Q1"),), {})


def _reset_items(
    name: str, capture: ParameterContext | None, example: ResetExample | None = None
) -> list[WritebackItem]:
    sample = _reset_example(name) if example is None else example
    adapter = _registry().create(_RESET_NAMES[name])
    return list(
        adapter.get_writeback_items(
            WritebackRequest(
                sample.source,
                sample.analysis,
                SessionEnv(parameters=capture, soc=None, soccfg=None),
            )
        )
    )


def _scalar(item: WritebackItem) -> ParameterWriteback:
    if not isinstance(item, ParameterWriteback):
        raise AssertionError("Expected a parameter proposal")
    return item


def _module(item: WritebackItem) -> ModuleWriteback:
    if not isinstance(item, ModuleWriteback) or item.edit_schema is None:
        raise AssertionError("Expected an editable module proposal")
    return item


def _assert_reset_collection(
    family: str,
    items: list[WritebackItem],
    capture: ParameterContext,
    sample: ResetExample,
) -> None:
    assert [
        (item.target_name, _scalar(item).proposed_value) for item in items[:2]
    ] == list(sample.scalars)
    roots = _RESET_ROOTS[family]
    assert [item.target_name for item in items] == [
        path for path, _ in sample.scalars
    ] + [f"$qubit.reset.{root}" for root in roots] + ["$qubit.module.reset"]
    for item, root in zip(items[2:-1], roots, strict=True):
        proposal = _module(item)
        assert proposal.edit_schema is not None
        assert (
            schema_to_raw_dict(proposal.edit_schema, capture) == sample.expected[root]
        )
    assert _scalar(items[-1]).proposed_value == capture.resolve_path(
        f"$qubit.reset.{roots[0]}"
    )


def _install_reset_capture(
    session: WritebackSession, *, existing: bool = False
) -> ParameterContext:
    source = _reset_capture()
    for point in ("run-point", "later-point"):
        session.control.use_point(PointUseRequest(point))
        with session.entry.use_point(point).edit() as draft:
            for index in (1, 2):
                qubit = f"Q{index}"
                draft.set(f"{qubit}.ext", deepcopy(_RESET_EXT))
                component = source.document().components[qubit]
                if not isinstance(component, QubitSchema):
                    raise AssertionError("Expected fixture qubit")
                slots: dict[str, YamlValue] = dict(component.module)
                draft.set(f"{qubit}.module", slots)
        with session.entry.module_library(point).edit() as draft:
            for root in source.library.module_keys():
                draft.set_module(root, source.library.module_spec(root))
            if existing:
                for index in (1, 2):
                    for root, slot in (
                        ("reset_10", "sideband"),
                        ("reset_120", "dual_tone"),
                        ("reset_bath", "bath_g"),
                        ("reset_bath_e", "bath_e"),
                    ):
                        draft.set_module(
                            f"Q{index}.reset.{root}",
                            source.library.module_spec(f"Q{index}.reset.old_{slot}"),
                        )
    session.control.use_point(PointUseRequest("run-point"))
    session.control.set_alias_rows(
        (AliasRow("qubit", "Q1"), AliasRow("resonator", "$qubit.resonator"))
    )
    return session.control.read_context()


def _accepted(event: LedgerEvent) -> AcceptedPayload:
    if event.kind != "accepted" or not isinstance(event.payload, AcceptedPayload):
        raise AssertionError("Expected accepted evidence")
    return event.payload


def _module_schema(item: WritebackItem) -> CfgSchema:
    schema = _module(item).edit_schema
    if schema is None:
        raise AssertionError("Expected complete module schema")
    return schema


def _expected_child(
    sample: ResetExample, root: str, child: str
) -> dict[str, YamlValue]:
    value = sample.expected[root][child]
    if not isinstance(value, dict):
        raise AssertionError("Expected a pulse in the independent input fixture")
    return value


def _assert_reset_applied(
    session: WritebackSession,
    capture: ParameterContext,
    sample: ResetExample,
    run_id: str,
) -> None:
    destination = session.control.read_target(capture.target)
    ledger = session.context.entry_ledger(capture.target.entry_id)
    accepted = [event for event in ledger.events() if event.kind == "accepted"]
    assert _accepted(accepted[-1]).run_id == run_id
    for path, value in sample.scalars:
        concrete = capture.resolve_path(path)
        assert destination.value(concrete) == value
        source = destination.meta(concrete)
        assert (
            source is not None
            and source.run_id == run_id
            and source.source == accepted[-1].id
        )
    for root, expected in sample.expected.items():
        concrete = capture.resolve_path(f"$qubit.reset.{root}")
        assert destination.library.get_module(concrete).to_dict() == expected
        leaf = {
            "reset_10": "pulse_cfg.freq",
            "reset_120": "pulse1_cfg.gain",
            "reset_bath": "pi2_cfg.phase",
            "reset_bath_e": "pi2_cfg.phase",
        }[root]
        source = destination.library.meta(f"{concrete}.{leaf}")
        assert (
            source is not None
            and source.run_id == run_id
            and source.source == accepted[-1].id
        )


@pytest.mark.io
@pytest.mark.parametrize(
    "name", ["single-freq", "dual-freq", "dual-power", "bath-freq-gain", "bath-phase"]
)
def test_reset_scalar_calibrations_apply_to_captured_extensions(
    session: WritebackSession, name: str
) -> None:
    capture = _install_reset_capture(session)
    sample = _reset_example(name)
    items = _reset_items(name, capture, sample)
    assert [
        (item.target_name, _scalar(item).proposed_value) for item in items[:2]
    ] == list(sample.scalars)
    session.control.use_point(PointUseRequest("later-point"))
    session.control.set_alias_rows((AliasRow("qubit", "Q2"),))
    selected = session.control.read_context().target
    draft = session.service.create_draft(items, parameters=capture, run_id="scalar-run")
    try:
        draft.apply(item_ids=tuple(item.session_id for item in draft.items[:2]))
        destination = session.control.read_target(capture.target)
        for path, value in sample.scalars:
            assert destination.value(capture.resolve_path(path)) == value
        assert session.control.read_context().target == selected
        ledger = session.context.entry_ledger(capture.target.entry_id)
        assert _accepted(ledger.events()[-1]).run_id == "scalar-run"
        for path, _ in sample.scalars:
            source = destination.meta(capture.resolve_path(path))
            assert (
                source is not None
                and source.run_id == "scalar-run"
                and source.source == ledger.events()[-1].id
            )
    finally:
        draft.teardown()


@pytest.mark.io
@pytest.mark.parametrize(
    ("family", "root_state", "change"),
    [
        (family, state, change)
        for family in _RESET_ROOTS
        for state in ("existing", "new")
        for change in ("rebind", "rebind-and-point")
    ],
    ids=[
        f"{family}-{state}-{change}"
        for family in _RESET_ROOTS
        for state in ("existing", "new")
        for change in ("rebind", "rebind-and-point")
    ],
)
def test_reset_finisher_preserves_tested_configuration_and_applies_analysis(
    session: WritebackSession, family: str, root_state: str, change: str
) -> None:
    capture = _install_reset_capture(session, existing=root_state == "existing")
    sample = _reset_example(family)
    original = (
        sample.source.cfg.model_dump(mode="json")
        if sample.source.cfg is not None
        else None
    )
    session.control.set_alias_rows((AliasRow("qubit", "Q2"),))
    if change == "rebind-and-point":
        session.control.use_point(PointUseRequest("later-point"))
    selected = session.control.read_context().target
    items = _reset_items(family, capture, sample)
    _assert_reset_collection(family, items, capture, sample)
    draft = session.service.create_draft(
        items, parameters=capture, run_id="finisher-run"
    )
    try:
        draft.apply()
        _assert_reset_applied(session, capture, sample, "finisher-run")
        assert (
            session.control.read_target(capture.target).value("Q1.module.reset")
            == f"Q1.reset.{_RESET_ROOTS[family][0]}"
        )
        assert session.control.read_context().target == selected
        assert (
            sample.source.cfg is not None
            and sample.source.cfg.model_dump(mode="json") == original
        )
    finally:
        draft.teardown()


_RESET_FINISHER_FIELDS = {
    "single": ("reset_f", "reset_fwhm"),
    "dual": ("reset_f1", "reset_f2", "reset_gain1", "reset_gain2"),
    "bath": (
        "reset_bath_freq",
        "reset_bath_gain",
        "reset_bath_max_phase",
        "reset_bath_min_phase",
    ),
}
_RESET_MISSING_CASES = [
    (family, field, state)
    for family, fields in _RESET_FINISHER_FIELDS.items()
    for field in fields
    for state in ("absent", "null")
] + [(family, "all", "absent") for family in _RESET_ROOTS]


@pytest.mark.parametrize(
    ("family", "field", "state"),
    _RESET_MISSING_CASES,
    ids=[f"{family}-{field}-{state}" for family, field, state in _RESET_MISSING_CASES],
)
def test_reset_finisher_proposes_tested_module_with_missing_extensions(
    family: str, field: str, state: str
) -> None:
    base = _reset_capture()
    point = base.document()
    component = QubitSchema.model_validate(point.components["Q1"].model_dump())
    for key in _RESET_FINISHER_FIELDS[family] if field == "all" else (field,):
        if state == "absent":
            component.ext.pop(key)
        else:
            component.ext[key] = None
    point.components["Q1"] = component
    capture = ParameterContext(
        point=point, library=base.library, target=base.target, roles=None
    ).bind_aliases((AliasRow("qubit", "Q1"),), {})
    sample = _reset_example(family)
    _assert_reset_collection(
        family, _reset_items(family, capture, sample), capture, sample
    )


@pytest.mark.parametrize("name", ["single-length", "dual-length", "bath-length"])
def test_reset_length_proposals_are_empty(name: str) -> None:
    assert _reset_items(name, _reset_capture()) == []


@pytest.mark.parametrize("family", list(_RESET_ROOTS))
def test_reset_finisher_without_capture_keeps_scalar_proposals(family: str) -> None:
    sample = _reset_example(family)
    items = _reset_items(family, None, sample)
    assert [(item.target_name, _scalar(item).proposed_value) for item in items] == list(
        sample.scalars
    )


@pytest.mark.parametrize("family", list(_RESET_ROOTS))
def test_reset_finisher_without_cfg_keeps_scalar_proposals(family: str) -> None:
    sample = _reset_example(family)
    sample.source = RunRecord[ExpCfgModel, object](None, sample.source.result)
    items = _reset_items(family, _reset_capture(), sample)
    assert [(item.target_name, _scalar(item).proposed_value) for item in items] == list(
        sample.scalars
    )


@pytest.mark.parametrize("family", list(_RESET_ROOTS))
def test_reset_finisher_propagates_capture_source_fault(
    tmp_path: Path, family: str
) -> None:
    base = _reset_capture()
    fault = ContextSourceFault(
        document="point",
        path=tmp_path / "point.yaml",
        error_type="OSError",
        message="gone",
    )
    capture = ParameterContext(
        point=base.document(),
        library=base.library,
        target=base.target,
        roles=base.roles,
        source_faults=(fault,),
    )
    with pytest.raises(ContextSourceUnavailableError) as error:
        _reset_items(family, capture)
    assert fault in error.value.faults


@pytest.mark.parametrize("family", list(_RESET_ROOTS))
def test_reset_finisher_propagates_unbound_capture_alias(family: str) -> None:
    # Authored empty string is unbound; resolved None is not an AliasRow input.
    capture = _reset_capture().bind_aliases((AliasRow("qubit", ""),), {})
    with pytest.raises(RoleResolutionError) as error:
        _reset_items(family, capture)
    assert error.value.role == "qubit"


@pytest.mark.io
@pytest.mark.parametrize("family", list(_RESET_ROOTS))
def test_reset_library_application_leaves_unselected_downstream_slot(
    session: WritebackSession, family: str
) -> None:
    capture = _install_reset_capture(session)
    sample = _reset_example(family)
    items = _reset_items(family, capture, sample)
    _assert_reset_collection(family, items, capture, sample)
    draft = session.service.create_draft(
        items, parameters=capture, run_id="library-only"
    )
    try:
        draft.edit(draft.items[-1].session_id, selected=False)
        draft.apply()
        _assert_reset_applied(session, capture, sample, "library-only")
        assert (
            session.control.read_target(capture.target).value("Q1.module.reset")
            == "Q1.reset.old_reset"
        )
    finally:
        draft.teardown()


@pytest.mark.io
def test_bath_reset_slot_can_select_excited_module(session: WritebackSession) -> None:
    capture = _install_reset_capture(session)
    sample = _reset_example("bath")
    items = _reset_items("bath", capture, sample)
    _assert_reset_collection("bath", items, capture, sample)
    draft = session.service.create_draft(
        items, parameters=capture, run_id="excited-run"
    )
    try:
        draft.edit(draft.items[-1].session_id, proposed_value="Q1.reset.reset_bath_e")
        draft.apply(item_ids=(draft.items[-2].session_id, draft.items[-1].session_id))
        destination = session.control.read_target(capture.target)
        assert destination.value("Q1.module.reset") == "Q1.reset.reset_bath_e"
        assert (
            destination.library.get_module("Q1.reset.reset_bath_e").to_dict()
            == sample.expected["reset_bath_e"]
        )
        assert (
            schema_to_raw_dict(_module_schema(draft.preview()[2]), capture)
            == sample.expected["reset_bath"]
        )
    finally:
        draft.teardown()


@pytest.mark.io
@pytest.mark.parametrize("family", list(_RESET_ROOTS))
def test_reset_drafts_keep_each_run_values_and_ledger_identity(
    session: WritebackSession, family: str
) -> None:
    first = _install_reset_capture(session)
    sample_a = _reset_example(family)
    items_a = _reset_items(family, first, sample_a)
    _assert_reset_collection(family, items_a, first, sample_a)
    draft_a = session.service.create_draft(items_a, parameters=first, run_id="run-A")
    session.control.use_point(PointUseRequest("later-point"))
    session.control.set_alias_rows((AliasRow("qubit", "Q2"),))
    target = session.control.read_context().target
    session.control.set_parameter(target, "Q2.ext.reset_f", 5200.0)
    second = session.control.read_context()
    sample_b = _reset_example(family, shift=1000.0)
    # Each Run owns a separate complete cfg and its own independently fitted values.
    if family == "single":
        sample_b.analysis = SingleToneFreqAnalyzeResult(5300.0, 9.0)
        sample_b.scalars = (
            ("$qubit.ext.reset_f", 5300.0),
            ("$qubit.ext.reset_fwhm", 9.0),
        )
        _expected_child(sample_b, "reset_10", "pulse_cfg")["freq"] = 5300.0
    elif family == "dual":
        sample_b.analysis = DualTonePowerAnalyzeResult(0.41, 0.51)
        sample_b.scalars = (
            ("$qubit.ext.reset_gain1", 0.41),
            ("$qubit.ext.reset_gain2", 0.51),
        )
        _expected_child(sample_b, "reset_120", "pulse1_cfg")["gain"] = 0.41
        _expected_child(sample_b, "reset_120", "pulse2_cfg")["gain"] = 0.51
    else:
        sample_b.analysis = BathPhaseAnalyzeResult(85.0, -95.0)
        sample_b.scalars = (
            ("$qubit.ext.reset_bath_max_phase", 85.0),
            ("$qubit.ext.reset_bath_min_phase", -95.0),
        )
        _expected_child(sample_b, "reset_bath", "pi2_cfg")["phase"] = 85.0
        _expected_child(sample_b, "reset_bath_e", "pi2_cfg")["phase"] = -95.0
    items_b = _reset_items(family, second, sample_b)
    _assert_reset_collection(family, items_b, second, sample_b)
    draft_b = session.service.create_draft(items_b, parameters=second, run_id="run-B")
    try:
        session.control.use_point(PointUseRequest("run-point"))
        session.control.set_alias_rows((AliasRow("qubit", "Q1"),))
        for draft, capture, sample, identity in (
            (draft_b, second, sample_b, "run-B"),
            (draft_a, first, sample_a, "run-A"),
        ):
            _assert_reset_collection(family, draft.preview(), capture, sample)
            draft.apply()
            _assert_reset_applied(session, capture, sample, identity)
        assert session.control.read_context().target == first.target
    finally:
        draft_b.teardown()
        draft_a.teardown()


@pytest.mark.io
def test_reset_apply_retains_accepted_and_successful_prefix_on_validation_failure(
    session: WritebackSession,
) -> None:
    capture = _install_reset_capture(session)
    sample = _reset_example("single")
    items = _reset_items("single", capture, sample)
    _assert_reset_collection("single", items, capture, sample)
    draft = session.service.create_draft(
        items, parameters=capture, run_id="partial-run"
    )
    try:
        draft.edit(
            draft.items[1].session_id, target_name="Q1.freq", proposed_value=True
        )
        with pytest.raises(ValueError, match="freq"):
            draft.apply()
        destination = session.control.read_target(capture.target)
        assert destination.value("Q1.ext.reset_f") == 4300.0
        assert not destination.lookup("Q1.freq").present
        assert "Q1.reset.reset_10" not in destination.library.module_keys()
        assert destination.value("Q1.module.reset") == "Q1.reset.old_reset"
        accepted = session.context.entry_ledger(capture.target.entry_id).events()[-1]
        assert _accepted(accepted).run_id == "partial-run"
        source = destination.meta("Q1.ext.reset_f")
        assert source is not None and source.source == accepted.id
    finally:
        draft.teardown()
