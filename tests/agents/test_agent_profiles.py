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


def test_v18_parallel_burst_contract_is_stateless_and_bounded() -> None:
    skill = (REPO_ROOT / ".agents" / "skills" / "orchestrate" / "SKILL.md").read_text()
    burst = (
        REPO_ROOT
        / ".agents"
        / "skills"
        / "orchestrate"
        / "references"
        / "parallel-burst.md"
    ).read_text()
    validation = (
        REPO_ROOT
        / ".agents"
        / "skills"
        / "orchestrate"
        / "references"
        / "validation.md"
    ).read_text()
    for anchor in (
        "skill_version: 18",
        "`parallel-burst` 是無狀態",
        "loop_authority",
        "`refuse`、`adopt_existing` 或 `artifact_only`",
    ):
        assert anchor in skill
    for anchor in (
        "stateless execution capability",
        "Parallel Burst",
        "dependency matrix",
        "Decided / Rejected / Risks / Files / Remaining",
        "existing_authority_id",
        "completion authority仍屬原owner",
        "只產生plan/handoff artifact，不執行workers",
        "只有既有authority或使用者可決定",
    ):
        assert anchor in burst
    for anchor in (
        "`implementation|contract|environment`",
        "`light=1`",
        "`standard=2`",
        "`critical=3`",
        "Behavioral evidence",
        "舊review receipt失效",
        "重審新SHA",
        "不得以換reviewer關閉finding",
        "Contract/environment failure不消耗implementation",
    ):
        assert anchor in validation


def test_worker_profiles_use_events_and_context_budgets() -> None:
    for name in PROFILE_NAMES:
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
            assert "artifact" in body
            if name != "contract-planner":
                assert "event" in body
        if name == "lane-implementer":
            for body in bodies:
                assert "不得spawn sub-agent" in body
