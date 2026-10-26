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
    sources = {"zcu_lab/recipes/demo.py": f"async def run(env):\n    await {target}\n"}

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
    source = f"import asyncio as aio\nasync def unused(tab):\n    {statement}\n"

    violations = check_script_awaits.check_sources({"zcu_lab/helpers.py": source})

    assert len(violations) == 1


def test_other_async_contexts_and_iterators_are_not_blanket_banned() -> None:
    source = (
        "async def run(env):\n"
        "    async with env.context():\n"
        "        async for item in env.stream():\n"
        "            await env.run(item)\n"
    )

    assert (
        check_script_awaits.check_sources({"zcu_lab/workflows/demo.py": source}) == ()
    )


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


@pytest.mark.parametrize("receiver", ["scope", "iteration"])
def test_explicit_scope_receivers_allow_operations_but_not_scheduling(
    receiver: str,
) -> None:
    source = (
        f"async def step({receiver}):\n"
        f"    await {receiver}.run(core, cfg, name='T1')\n"
        f"    await {receiver}.call(blocking)\n"
        f"    await {receiver}.setup_device('flux', cfg)\n"
        f"    await {receiver}.wait_until(target)\n"
        f"    await {receiver}.commit(record)\n"
        f"    await gather({receiver}.run(core, cfg))\n"
    )
    violations = check_script_awaits.check_sources(
        {"zcu_lab/workflows/demo.py": source}
    )
    assert [(violation.line, violation.target) for violation in violations] == [
        (7, f"gather({receiver}.run(core, cfg))")
    ]


def test_lab_helpers_defined_with_def_can_be_awaited() -> None:
    sources = {
        "zcu_lab/workflows/demo.py": (
            "from zcu_lab.helpers import helper\n"
            "def local_helper(env):\n"
            "    return env.run(cfg)\n"
            "async def run(env):\n"
            "    await local_helper(env)\n"
            "    await helper(env)\n"
        ),
        "zcu_lab/helpers.py": "def helper(env):\n    return env.run(cfg)\n",
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

    violations = check_script_awaits.check_sources(
        {"zcu_lab/workflows/demo.py": source}
    )

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


_CALLER = "from zcu_lab.facade import step\nasync def run(env):\n    await step(env)\n"
_STEP = "async def step(env):\n    await env.run(cfg)\n"


@pytest.mark.parametrize(
    "reexport",
    [
        "from zcu_lab.impl import step as step\n",
        "from .impl import step\n",
    ],
)
def test_reexported_lab_helpers_resolve_to_their_definitions(reexport: str) -> None:
    sources = {
        "zcu_lab/demo.py": _CALLER,
        "zcu_lab/facade.py": reexport,
        "zcu_lab/impl.py": _STEP,
    }

    assert check_script_awaits.check_sources(sources) == ()


def test_chained_reexports_resolve_to_their_definitions() -> None:
    sources = {
        "zcu_lab/demo.py": _CALLER,
        "zcu_lab/facade.py": "from zcu_lab.middle import step as step\n",
        "zcu_lab/middle.py": "from zcu_lab.impl import step as step\n",
        "zcu_lab/impl.py": _STEP,
    }

    assert check_script_awaits.check_sources(sources) == ()


def test_reexports_of_external_callables_stay_rejected() -> None:
    sources = {
        "zcu_lab/demo.py": _CALLER,
        "zcu_lab/facade.py": "from external import step as step\n",
    }

    violations = check_script_awaits.check_sources(sources)

    assert [(v.path, v.line) for v in violations] == [("zcu_lab/demo.py", 3)]


def test_circular_reexports_terminate_as_rejections() -> None:
    sources = {
        "zcu_lab/demo.py": _CALLER.replace("facade", "a"),
        "zcu_lab/a.py": "from zcu_lab.b import step as step\n",
        "zcu_lab/b.py": "from zcu_lab.a import step as step\n",
    }

    assert len(check_script_awaits.check_sources(sources)) == 1


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
