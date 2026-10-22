"""Await guardrail contracts using synthetic lab sources, never lab imports."""

from __future__ import annotations

import json
from pathlib import Path

import check_script_awaits
import pytest


@pytest.mark.parametrize(
    "target",
    [
        "asyncio.sleep(0)",
        "create_task(tab.run())",
        "gather(tab.run())",
        "wait_for(tab.run(), 1)",
        "TaskGroup()",
        "external_helper(env)",
        "helper(env)",
        "pending",
        "tab.run",
    ],
)
def test_only_framework_or_defined_lab_calls_can_be_awaited(target: str) -> None:
    sources = {
        "zcu_lab/recipes/demo.py": f"async def run(env):\n    await {target}\n"
    }

    violations = check_script_awaits.check_sources(sources)

    assert [(v.path, v.line, v.target) for v in violations] == [
        ("zcu_lab/recipes/demo.py", 2, target)
    ]


@pytest.mark.parametrize(
    "statement",
    [
        "asyncio.create_task(tab.run())",
        "create_task(tab.run())",
        "gather(tab.run())",
        "wait_for(tab.run(), 1)",
        "async with asyncio.TaskGroup() as group:\n        pass",
        "async with TaskGroup() as group:\n        pass",
        "aio.sleep(0)",
    ],
)
def test_scheduling_calls_are_rejected_even_without_await(statement: str) -> None:
    source = (
        "import asyncio as aio\n"
        f"async def unused(tab):\n    {statement}\n"
    )

    violations = check_script_awaits.check_sources({"zcu_lab/helpers.py": source})

    assert len(violations) == 1


def test_other_async_contexts_and_iterators_are_not_blanket_banned() -> None:
    source = (
        "async def run(env):\n"
        "    async with env.context():\n"
        "        async for item in env.stream():\n"
        "            await env.run(item)\n"
    )

    assert check_script_awaits.check_sources({"zcu_lab/workflows/demo.py": source}) == ()


def test_uniform_rule_allows_framework_calls_and_real_lab_helpers() -> None:
    sources = {
        "zcu_lab/recipes/demo.py": (
            "from zcu_lab.helpers import step as helper\n"
            "async def recipe(session, tab, env):\n"
            "    await helper(env)\n"
            "    await session.ask(question)\n"
            "    await tab.run()\n"
            "    await env.run(cfg)\n"
        ),
        "zcu_lab/helpers.py": (
            "from .nested import step\n"
            "async def step_one(env):\n"
            "    await env.run(cfg)\n"
            "async def step(env):\n"
            "    await step_one(env)\n"
        ),
        "zcu_lab/nested.py": "async def step(tab):\n    await tab.analyze('fit')\n",
    }

    assert check_script_awaits.check_sources(sources) == ()


def test_unreferenced_helpers_are_checked_without_decorators_or_reachability() -> None:
    sources = {
        "zcu_lab/helpers.py": (
            "async def unused(env):\n"
            "    async def nested():\n"
            "        await external()\n"
            "    await asyncio.sleep(0)\n"
        ),
    }

    violations = check_script_awaits.check_sources(sources)

    assert [(v.line, v.target) for v in violations] == [
        (3, "external()"),
        (4, "asyncio.sleep(0)"),
    ]


@pytest.mark.parametrize(
    "import_line",
    [
        "from external import helper",
        "from external import helper as lab_helper",
        "from zcu_lab.not_defined import helper",
    ],
)
def test_import_spelling_is_not_evidence_of_a_lab_definition(import_line: str) -> None:
    name = "lab_helper" if " as " in import_line else "helper"
    source = f"{import_line}\nasync def run(env):\n    await {name}(env)\n"

    violations = check_script_awaits.check_sources({"zcu_lab/workflows/demo.py": source})

    assert len(violations) == 1


def test_imported_module_helpers_resolve_to_their_definitions() -> None:
    sources = {
        "zcu_lab/workflows/demo.py": (
            "import zcu_lab.helpers as helpers\n"
            "async def run(env):\n"
            "    await helpers.step(env)\n"
        ),
        "zcu_lab/helpers.py": "async def step(env):\n    await env.run(cfg)\n",
    }

    assert check_script_awaits.check_sources(sources) == ()


def test_cli_checks_fixture_sources_and_reports_rejections(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lab = tmp_path / "zcu_lab"
    lab.mkdir()
    (lab / "helper.py").write_text(
        "raise RuntimeError('source must not be imported')\n"
        "async def unused():\n    await external()\n"
    )

    code = check_script_awaits.main(["--root", str(tmp_path)])
    receipt = json.loads(capsys.readouterr().out)

    assert code == 1
    assert receipt["violation_count"] == 1
    assert receipt["violations"] == [
        {"path": "zcu_lab/helper.py", "line": 3, "target": "external()"}
    ]
