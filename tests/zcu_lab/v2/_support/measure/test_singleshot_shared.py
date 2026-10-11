"""Single-shot analysis consumes calibration from a fixed parameter capture."""

import pytest
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral

from zcu_lab.components import QubitSchema, ResonatorSchema
from zcu_lab.v2._support.measure.singleshot_shared import (
    read_chi_kappa,
    read_ge_centers,
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
        target=SavedPoint(entry_id="entry", point="working"),
        roles=None,
    ).bind_aliases(
        (AliasRow("qubit", "Q1"), AliasRow("resonator", "$qubit.resonator")), {}
    )


def test_classification_reads_canonical_capture(calibration: ParameterContext) -> None:
    assert read_ge_centers(calibration) == (1 + 2j, 3 + 4j, 0.5)


def test_ac_stark_reads_qubit_chi_and_resonator_kappa(
    calibration: ParameterContext,
) -> None:
    assert read_chi_kappa(calibration) == (2.0, 4.0)


@pytest.mark.parametrize("reader", [read_ge_centers, read_chi_kappa])
def test_analysis_requires_explicit_parameter_source(reader) -> None:
    with pytest.raises(RuntimeError, match="parameter"):
        reader(None)


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
