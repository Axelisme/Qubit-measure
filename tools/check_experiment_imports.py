"""Import-linter contract for independent experiment directories in zcu_lab.v2."""

from typing import override

from grimp import ImportGraph
from importlinter.application import output
from importlinter.domain.contract import Contract, ContractCheck


def _experiment(module: str, experiments: set[str]) -> str | None:
    while module:
        if module in experiments:
            return module
        module = module.rpartition(".")[0]
    return None


class ExperimentImportsContract(Contract):
    """Enforce direct-import boundaries below zcu_lab.v2, with no contract options.

    A directory with a core module owns an experiment, including its descendants.
    Its family is the first directory below v2. Imports to another experiment or
    another experiment family are rejected; own-family helpers and support remain
    available. Family modules may still export their child experiments.
    """

    type_name = "experiment-imports"

    @override
    def check(self, graph: ImportGraph, verbose: bool) -> ContractCheck:
        """Check a source graph and return kept status plus offending edges and lines.

        ``graph`` is the unsquashed graph supplied by import-linter; it is not
        mutated. ``verbose`` is accepted for the plugin protocol; output is unchanged.
        """
        experiments = {
            module.removesuffix(".core")
            for module in graph.modules
            if module.startswith("zcu_lab.v2.") and module.endswith(".core")
        }
        families = {".".join(experiment.split(".")[:3]) for experiment in experiments}
        violations: list[str] = []
        for importer in sorted(graph.modules):
            source = _experiment(importer, experiments)
            if source is None:
                continue
            foreign_families = families - {
                ".".join(source.split(".")[:3]),
                "zcu_lab.v2.support",
            }
            for imported in sorted(graph.find_modules_directly_imported_by(importer)):
                target = _experiment(imported, experiments)
                target_family = ".".join(imported.split(".")[:3])
                if (
                    target is not None and target != source
                ) or target_family in foreign_families:
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
        """Print source, target and line numbers from this contract's failed check."""
        output.print(check.metadata["diagnostic"])
