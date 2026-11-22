"""Verify the Program simulation boundary at the import-linter contract seam."""

import pytest
from grimp import ImportGraph

from tools.check_program_simulation_imports import ProgramSimulationImportsContract


@pytest.fixture
def contract() -> ProgramSimulationImportsContract:
    return ProgramSimulationImportsContract("Program simulation imports", {}, {})


@pytest.mark.parametrize(
    ("importer", "imported"),
    [
        ("zcu_tools.program", "zcu_tools.simulate"),
        ("zcu_tools.program.future.nested", "zcu_tools.simulate.boundary"),
        ("zcu_tools.program.v2.simulation_extra", "zcu_tools.simulators"),
        ("zcu_tools.program.future", "zcu_tools.simulators.ftf"),
    ],
)
def test_program_cannot_directly_import_simulation_outside_the_mixin(
    contract: ProgramSimulationImportsContract,
    capsys: pytest.CaptureFixture[str],
    importer: str,
    imported: str,
) -> None:
    graph = ImportGraph()
    graph.add_import(
        importer=importer,
        imported=imported,
        line_number=17,
        line_contents=f"import {imported}",
    )

    check = contract.check(graph, verbose=False)

    assert not check.kept
    contract.render_broken_contract(check)
    diagnostic = " ".join(capsys.readouterr().out.split())
    assert f"{importer} -> {imported} (l. 17)" in diagnostic


def test_mixin_and_its_callers_keep_their_direct_imports(
    contract: ProgramSimulationImportsContract,
) -> None:
    graph = ImportGraph()
    graph.add_import(
        importer="zcu_tools.program.v2.base",
        imported="zcu_tools.program.v2.simulation",
    )
    graph.add_import(
        importer="zcu_tools.program.v2.simulation",
        imported="zcu_tools.simulate",
    )
    graph.add_import(
        importer="zcu_tools.program.v2.simulation.future.nested",
        imported="zcu_tools.simulate.boundary",
    )
    graph.add_import(
        importer="zcu_tools.program.v2.base",
        imported="zcu_tools.simulate_extra",
    )
    graph.add_import(
        importer="zcu_tools.program_extra",
        imported="zcu_tools.simulate.boundary",
    )

    assert contract.check(graph, verbose=False).kept
