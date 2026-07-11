from __future__ import annotations

import json
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_NAMES = {
    "contract-planner",
    "lane-implementer",
    "lane-reviewer",
    "integration-reviewer",
}


def _frontmatter(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    _, header, body = text.split("---", maxsplit=2)
    fields = {}
    for line in header.splitlines():
        if ":" in line and not line.startswith(" "):
            key, value = line.split(":", maxsplit=1)
            fields[key] = value.strip()
    return fields, body


def test_generic_profiles_are_registered() -> None:
    config = json.loads((REPO_ROOT / ".agents" / "opencode.json").read_text())
    for name in PROFILE_NAMES:
        entry = config["agent"][name]
        assert entry["mode"] == "subagent"
        assert entry["options"]["source"] == f".agents/agents/{name}.md"


def test_runtime_profile_schemas_and_core_contracts() -> None:
    required_body = ("Outcome", "Evidence", "Open risks")
    for name in PROFILE_NAMES:
        generic_fields, generic_body = _frontmatter(
            REPO_ROOT / ".agents" / "agents" / f"{name}.md"
        )
        claude_fields, claude_body = _frontmatter(
            REPO_ROOT / ".claude" / "agents" / f"{name}.md"
        )
        codex = tomllib.loads(
            (REPO_ROOT / ".codex" / "agents" / f"{name}.toml").read_text()
        )
        assert generic_fields["name"] == name
        assert generic_fields["mode"] == "subagent"
        assert claude_fields["name"] == name
        assert claude_fields["model"] in {"opus", "sonnet"}
        assert codex["name"] == name
        assert codex["model_reasoning_effort"] in {"medium", "high"}
        for body in (generic_body, claude_body, codex["developer_instructions"]):
            for anchor in required_body:
                assert anchor in body


def test_orchestrate_skill_routes_all_profiles() -> None:
    text = (REPO_ROOT / ".agents" / "skills" / "orchestrate" / "SKILL.md").read_text()
    for name in PROFILE_NAMES:
        assert f"`{name}`" in text


def test_reviewers_enforce_identity_and_frozen_target_gates() -> None:
    for name in ("lane-reviewer", "integration-reviewer"):
        bodies = [
            _frontmatter(REPO_ROOT / root / "agents" / f"{name}.md")[1]
            for root in (".agents", ".claude")
        ]
        bodies.append(
            tomllib.loads(
                (REPO_ROOT / ".codex" / "agents" / f"{name}.toml").read_text()
            )["developer_instructions"]
        )
        for body in bodies:
            for anchor in (
                "identity",
                "target SHA",
                "frozen contract",
                "write scope",
                "blocked",
                "needs_decision",
                "Scope changes requested",
            ):
                assert anchor in body


def test_mcct_uses_profile_review_gate() -> None:
    text = (
        REPO_ROOT
        / ".agents"
        / "skills"
        / "orchestrate"
        / "references"
        / "merge-protocol.md"
    ).read_text()
    assert "profile/trigger 要求的 review" in text
    assert "只有 target 需要獨立 review 時缺 identity 才阻擋" in text
