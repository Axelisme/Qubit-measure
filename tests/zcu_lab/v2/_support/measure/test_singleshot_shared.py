"""Single-shot analysis consumes calibration from a fixed parameter capture."""

import numpy as np
import pytest
from zcu_tools.experiment.records import RunRecord
from zcu_tools.format_version import YamlValue
from zcu_tools.gui.app.measure.adapter import PostWritebackRequest, WritebackRequest
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import EntryDraft
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral

from tests.zcu_lab._writeback_support import WritebackSession
from zcu_lab.components import QubitSchema, ResonatorSchema
from zcu_lab.v2._support.measure.singleshot_shared import (
    read_chi_kappa,
    read_ge_centers,
)
from zcu_lab.v2.singleshot.ge.core import GE_Result
from zcu_lab.v2.singleshot.ge.gui import (
    GEAdapter,
    GEAnalyzeResult,
    GEPostAnalyzeResult,
)


@pytest.fixture
def calibration() -> ParameterContext:
    point = PointDocument.model_construct(
        format="zcu.parameter-container",
        format_version="1.0",
        general=PointGeneral(created_at="2026-10-09T00:00:00Z"),
        components={
            "Q1": QubitSchema(
                kind="qubit/fluxonium",
                resonator="R1",
                ext={
                    "chi": 2.0,
                    "readout_cal": {
                        "g_center": "1+2j",
                        "e_center": "3+4j",
                        "radius": 0.5,
                    },
                },
            ),
            "R1": ResonatorSchema(kind="resonator", kappa=4.0),
        },
    )
    return ParameterContext(
        point=point,
        library=LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
        ),
        target=EntryDraft(entry_id="entry"),
        roles=None,
    ).bind_aliases(
        (AliasRow("qubit", "Q1"), AliasRow("resonator", "$qubit.resonator")), {}
    )


def test_classification_reads_canonical_capture(calibration: ParameterContext) -> None:
    assert read_ge_centers(calibration) == (1 + 2j, 3 + 4j, 0.5)


def test_classification_reads_applied_ge_calibration(session: WritebackSession) -> None:
    captured = session.context.read_context()
    env = SessionEnv(parameters=captured, soc=None, soccfg=None)
    run = RunRecord(
        None,
        GE_Result(np.zeros((2, 2), dtype=np.complex128), np.arange(2), np.arange(2)),
    )
    analysis = GEAnalyzeResult("ground", 0.95, 0.0, 0.5, 0.1, 1 + 2j, 3 + 4j, np.eye(2))
    adapter = GEAdapter()
    proposals = list(adapter.get_writeback_items(WritebackRequest(run, analysis, env)))
    proposals.extend(
        adapter.get_post_writeback_items(
            PostWritebackRequest(
                run, analysis, GEPostAnalyzeResult(0.3, np.eye(3).tolist()), env
            )
        )
    )
    draft = session.service.create_draft(proposals, parameters=captured)
    try:
        session.service.apply_draft(draft)
        applied = session.context.read_context()
        raw_center = applied.lookup("$qubit.ext.readout_cal.g_center").value
        try:
            centers = read_ge_centers(applied)
        except RuntimeError as exc:
            pytest.fail(f"Classification must accept applied GE calibration: {exc}")
        assert centers == (1 + 2j, 3 + 4j, 0.3)
        assert applied.lookup("$qubit.ext.readout_cal.g_center").value == raw_center
        assert applied.lookup("$qubit.ext.readout_cal.confusion_matrix").value == [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    finally:
        draft.teardown()


def test_ac_stark_reads_qubit_chi_and_resonator_kappa(
    calibration: ParameterContext,
) -> None:
    assert read_chi_kappa(calibration) == (2.0, 4.0)


@pytest.mark.parametrize("reader", [read_ge_centers, read_chi_kappa])
def test_analysis_requires_explicit_parameter_source(reader) -> None:
    with pytest.raises(RuntimeError, match="parameter"):
        reader(None)


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        [],
        {},
        {"real": 1.0, "imaginary": 2.0},
        {"__complex__": [1.0]},
        {"__complex__": [True, 2.0]},
        {"__complex__": ["1", 2.0]},
        {"__complex__": [1.0, 2.0], "extra": 3.0},
    ],
)
def test_classification_rejects_invalid_center_without_changing_capture(
    calibration: ParameterContext, value: YamlValue
) -> None:
    point = calibration.document()
    point.components["Q1"] = QubitSchema(
        kind="qubit/fluxonium",
        ext={"readout_cal": {"g_center": value, "e_center": "3+4j", "radius": 0.5}},
    )
    invalid = ParameterContext(
        point=point, library=calibration.library, target=calibration.target, roles=None
    ).bind_aliases(calibration.aliases, {})
    with pytest.raises(RuntimeError, match=r"\$qubit.ext.readout_cal.g_center"):
        read_ge_centers(invalid)
    assert invalid.lookup("$qubit.ext.readout_cal.g_center").value == value


def test_classification_names_missing_canonical_calibration(
    calibration: ParameterContext,
) -> None:
    point = calibration.document()
    point.components["Q1"] = point.components["Q1"].model_copy(update={"ext": {}})
    missing = ParameterContext(
        point=point, library=calibration.library, target=calibration.target, roles=None
    ).bind_aliases(calibration.aliases, {})
    with pytest.raises(RuntimeError, match=r"\$qubit.ext.readout_cal.g_center"):
        read_ge_centers(missing)
