"""Import-linter contract for Program's direct imports of simulation packages."""

from typing import override

from grimp import ImportGraph
from importlinter.application import output
from importlinter.domain.contract import Contract, ContractCheck

_PROGRAM = "zcu_tools.program"
_MIXIN_PACKAGE = f"{_PROGRAM}.v2.simulation"
_SIMULATION_TARGETS = ("zcu_tools.simulate", "zcu_tools.simulators")


def _in_package(module: str, package: str) -> bool:
    return module == package or module.startswith(f"{package}.")


class ProgramSimulationImportsContract(Contract):
    """Keep direct simulator imports inside the Program simulation mixin package.

    This contract has no options. It does not constrain callers of Program's
    simulation entry or other contracts that apply inside the mixin package.
    """

    type_name = "program-simulation-imports"

    @override
    def check(self, graph: ImportGraph, verbose: bool) -> ContractCheck:
        """Return kept status and offending edges with line numbers for this graph.

        The import-linter graph is not modified. Verbose mode does not change the
        check or print progress.
        """
        program = graph.find_matching_modules(f"{_PROGRAM}.**")
        if _PROGRAM in graph.modules:
            program.add(_PROGRAM)

        violations: list[str] = []
        for importer in sorted(program):
            if _in_package(importer, _MIXIN_PACKAGE):
                continue
            for imported in sorted(graph.find_modules_directly_imported_by(importer)):
                if not any(
                    _in_package(imported, target) for target in _SIMULATION_TARGETS
                ):
                    continue
                lines = ", ".join(
                    str(detail["line_number"])
                    for detail in graph.get_import_details(
                        importer=importer, imported=imported
                    )
                )
                violations.append(f"{importer} -> {imported} (l. {lines})")
        return ContractCheck(
            kept=not violations, metadata={"diagnostic": "\n".join(violations)}
        )

    @override
    def render_broken_contract(self, check: ContractCheck) -> None:
        """Print the offending source, target, and line numbers from a failed check."""
        output.print(check.metadata["diagnostic"])
