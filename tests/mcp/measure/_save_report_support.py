"""Native Save report builders for recording analysis collaborators."""

from collections.abc import Sequence

from zcu_tools.mcp.measure.raw_save import SaveArtifact, SaveOutput, SaveReport


def analysis_report(pane: str, names: Sequence[str] = ()) -> SaveReport:
    """Build one successful results artifact with JSON and all named PNG outputs."""
    stage = "primary" if pane == "analysis" else "post"
    members = {
        f"analysis:{stage}": [
            SaveOutput("json", f"/actual/{pane}/results.json", "saved")
        ]
    }
    for name in names:
        members[f"figure:{name}"] = [SaveOutput("png", f"/actual/{name}.png", "saved")]
    return SaveReport("run-a", {pane: {"results": SaveArtifact("Saved", members)}})
