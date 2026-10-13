"""Recording source selection at the public Reset capture boundary."""

from copy import deepcopy

from ._recipe_support import LookbackGui


class TargetGui(LookbackGui):
    """Record requested captures and fixed parameter/library queries."""

    def __init__(self) -> None:
        super().__init__()
        self.md = {"$resonator.freq": 6200.0}
        self.created: list[dict[str, object]] = []
        self.reads: list[dict[str, object]] = []
        self.modules: dict[str, str] = {}
        self.overrides: list[dict[str, str] | None] = []
        self.queried_paths: list[str] = []
        self.queried_modules: list[str] = []
        self.fixed_values: dict[str, object] = {}

    def __call__(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "context.snapshot":
            raise AssertionError("Recipe sources must come from the tab capture")
        if method == "tab.new":
            self.created.append(dict(params))
            return {"tab_id": "t"}
        if method == "tab.reset_cfg":
            raw = params.get("aliases")
            if raw is not None and not isinstance(raw, dict):
                raise ValueError("Expected aliases object")
            self.overrides.append(raw)
            if raw and raw.get("qubit") == "Q2":
                self.md["$resonator.freq"] = 8200.0
            self.fixed_values = deepcopy(self.md)
        if method == "tab.parameter_read":
            assert params["tab_id"] == "t"
            self.reads.append(dict(params))
            paths, references = params["paths"], params["module_refs"]
            if not isinstance(paths, list) or not isinstance(references, list):
                raise ValueError("Expected query arrays")
            self.queried_paths.extend(path for path in paths if isinstance(path, str))
            self.queried_modules.extend(
                ref for ref in references if isinstance(ref, str)
            )
            return {
                "values": {
                    path: self.fixed_values[path]
                    for path in paths
                    if isinstance(path, str) and path in self.fixed_values
                },
                "modules": {
                    reference: self.modules.get(reference)
                    for reference in references
                    if isinstance(reference, str)
                },
            }
        return super().__call__(method, params)
