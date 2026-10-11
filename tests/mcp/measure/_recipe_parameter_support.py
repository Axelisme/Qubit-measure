"""Shared parameter-source and sweep recording collaborators."""

from copy import deepcopy

from ._recipe_support import LookbackGui, scalar


class TargetGui(LookbackGui):
    """Record current/temporary source queries through the public recipe wire."""

    def __init__(self) -> None:
        super().__init__()
        self.md = {"$resonator.freq": 6200.0}
        self.created: list[dict[str, object]] = []
        self.reads: list[dict[str, object]] = []
        self.modules: dict[str, str] = {}
        self.overrides: list[dict[str, str] | None] = []
        self.queried_paths: list[str] = []
        self.queried_modules: list[str] = []

    def __call__(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "context.snapshot":
            raise AssertionError("Recipe sources must come from the tab capture")
        if method == "tab.new":
            self.created.append(dict(params))
            return {"tab_id": "t"}
        if method == "tab.get_cfg" and params.get("aliases") is not None:

            projected = deepcopy(self.publication)
            projected["status"] = "Valid"
            readout = projected["tree"]["children"]["modules"]["children"]["readout"]
            for node in readout["children"].values():
                for child in node.get("children", {}).values():
                    if child.get("input", {}).get("raw") == "$resonator.freq":
                        child["input"]["resolved"] = 8200.0
            projected["source_basis"] = [{"source_id": "target", "revision": "1"}]
            return projected
        if method == "tab.run_start":
            aliases = params.get("aliases")
            assert aliases is None or isinstance(aliases, dict)
            self.overrides.append(aliases)
            return super().__call__(
                method,
                {key: value for key, value in params.items() if key != "aliases"},
            )
        if method == "tab.parameter_read":
            assert params["tab_id"] == "t"
            self.reads.append(dict(params))
            paths = params["paths"]
            references = params["module_refs"]
            assert isinstance(paths, list) and isinstance(references, list)
            self.queried_paths.extend(path for path in paths if isinstance(path, str))
            self.queried_modules.extend(
                ref for ref in references if isinstance(ref, str)
            )
            return {
                "values": {
                    path: self.md[path]
                    for path in paths
                    if isinstance(path, str) and path in self.md
                },
                "modules": {
                    reference: self.modules.get(reference)
                    for reference in references
                    if isinstance(reference, str)
                },
            }
        return super().__call__(method, params)


class SweepGui(TargetGui):
    """Publish known GUI-resolved sweep inputs, without mocking recipe decisions."""

    def __init__(
        self, start_expr: str, stop_expr: str, start: float, stop: float
    ) -> None:
        super().__init__()
        inputs = {}
        for key, expression, value in (
            ("start", start_expr, start),
            ("stop", stop_expr, stop),
        ):
            state = scalar(value)["input"]
            state.update(mode="expression", raw=expression)
            inputs[key] = state
        inputs["expts"] = scalar(51)["input"]
        self.publication["tree"]["children"]["sweep"] = {
            "kind": "sweep",
            "valid": True,
            "readonly": False,
            "inputs": inputs,
        }
        self.resolved: dict[str, float] = {}

    def __call__(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method == "tab.edit_cfg":
            edits = params["edits"]
            assert isinstance(edits, list)
            if len(edits) == 1 and edits[0]["path"] == ["sweep"]:

                inputs = self.publication["tree"]["children"]["sweep"]["inputs"]
                for key, value in edits[0]["value"].items():
                    if isinstance(value, dict):
                        expression = value["__expr"]
                        state = scalar(self.resolved[expression])["input"]
                        state.update(mode="expression", raw=expression)
                    else:
                        state = scalar(value)["input"]
                    inputs[key] = state
                return self.publication
        return super().__call__(method, params)
