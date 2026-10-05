"""Two-stage native analysis collaborator for author handle contracts."""

import base64
from copy import deepcopy
from threading import Event

from zcu_tools.mcp.measure.analysis_execution import AnalysisStage, AnalysisWriteback

from ._recipe_support import PNG, LookbackGui


class AnalysisRecipeGui(LookbackGui):
    """Record Run/Primary/Post provenance and independent interactive outcomes."""

    def __init__(self, *, interactive: bool = False) -> None:
        super().__init__()
        self.interactive = interactive
        self.current_stage: AnalysisStage = "primary"
        self.done = {"primary": Event(), "post": Event()}
        self.awaited = {"primary": Event(), "post": Event()}
        self.outcomes = {"primary": "finished", "post": "finished"}
        self.params: dict[AnalysisStage, dict[str, object]] = {
            "primary": {"threshold": 0.5},
            "post": {"model": "fit"},
        }
        self.names = {"primary": ["fit", "residual"], "post": ["post-fit"]}
        stages: tuple[tuple[AnalysisStage, str], ...] = (
            ("primary", "frequency"),
            ("post", "linewidth"),
        )
        self.writebacks: dict[AnalysisStage, AnalysisWriteback] = {
            stage: AnalysisWriteback(
                has_draft=True,
                items=[
                    {
                        "id": f"{stage}-draft",
                        "kind": "metadict",
                        "target_name": name,
                        "current": 1.0,
                        "proposed": 2.0,
                        "selected": False,
                    }
                ],
                destination_context={"active_label": "sample"},
            )
            for stage, name in stages
        }

    def stage_for_operation(self, params: dict[str, object]) -> AnalysisStage:
        """Resolve one fixed wire ID; reject any unexpected native operation."""
        operation = params["operation_id"]
        assert operation in (93, 104)
        return "primary" if operation == 93 else "post"

    def __call__(self, method: str, params: dict[str, object]) -> dict[str, object]:
        if method in ("tab.analyze", "tab.post_analyze"):
            stage: AnalysisStage = "primary" if method == "tab.analyze" else "post"
            assert self.ran
            assert params["run_operation_id"] == 71
            if stage == "post":
                assert params["operation_id"] == 93
            else:
                assert "operation_id" not in params
            assert params["tab_id"] == "t"
            self.current_stage = stage
            return {
                "operation_id": 93 if stage == "primary" else 104,
                "interactive": self.interactive,
                "params": deepcopy(self.params[stage]),
                "invalidated_on_success": ["post.writeback"]
                if stage == "primary"
                else [],
            }
        if method == "operation.await" and params["operation_id"] in (93, 104):
            stage = self.stage_for_operation(params)
            self.awaited[stage].set()
            timeout = params["timeout"]
            assert isinstance(timeout, (int, float)) and 0 < timeout <= 0.25
            settled = not self.interactive or self.done[stage].is_set()
            return {
                "reason": "completed" if settled else "timeout",
                "status": self.outcomes[stage] if settled else "interactive",
            }
        if method == "operation.cancel":
            stage = self.stage_for_operation(params)
            self.outcomes[stage] = "cancelled"
            self.done[stage].set()
            return {"status": "cancelling"}
        if method == "tab.interact":
            stage = self.current_stage
            payload = params.get("payload")
            if isinstance(payload, dict) and payload.get("command") == "done":
                self.done[stage].set()
            return {
                "operation_id": 93 if stage == "primary" else 104,
                "plugin": "generic-test",
                "state": {"stage": stage},
                "info": {"label": f"{stage}-picker"},
                "commands": [{"name": f"select-{stage}"}, {"name": "done"}],
                "preview_active": True,
                "figure": {"png_b64": base64.b64encode(PNG).decode()}
                if params.get("include_figure", True)
                else None,
            }
        if method in (
            "tab.get_analyze_result",
            "tab.get_post_analyze_result",
            "tab.save_image",
            "tab.get_figure",
            "tab.writeback_preview",
        ):
            return self._completed_analysis(method, params)
        return super().__call__(method, params)

    def _completed_analysis(
        self, method: str, params: dict[str, object]
    ) -> dict[str, object]:
        if method in ("tab.get_analyze_result", "tab.get_post_analyze_result"):
            stage = self.stage_for_operation(params)
            pane = "analysis" if stage == "primary" else "post_analysis"
            assert method == (
                "tab.get_analyze_result"
                if stage == "primary"
                else "tab.get_post_analyze_result"
            )
            return {
                "summary": {"frequency": 5.0, "frequency_error": None},
                "invalid": [
                    {"path": "summary.frequency_error", "reason": "non_finite"}
                ],
                "params": deepcopy(self.params[stage]),
                "operation_state": {
                    f"{pane}_state": {"figure_names": list(self.names[stage])}
                },
            }
        if method in ("tab.save_image", "tab.get_figure", "tab.writeback_preview"):
            stage = self.stage_for_operation(params)
            pane = "analysis" if stage == "primary" else "post_analysis"
            assert params["subtab_id"] == pane
            if method == "tab.save_image":
                return {"image_path": f"/actual/{params['figure_name']}.png"}
            if method == "tab.get_figure":
                return {"png_b64": base64.b64encode(PNG).decode()}
            preview = self.writebacks[stage]
            return {
                "has_draft": preview["has_draft"],
                "items": deepcopy(preview["items"]),
                "destination_context": deepcopy(preview["destination_context"]),
            }
        raise AssertionError(method)
