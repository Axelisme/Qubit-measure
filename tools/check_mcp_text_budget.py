#!/usr/bin/env python
"""Check the text sent by the measure MCP server against character budgets."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Iterator, Sequence, Set
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NotRequired, TypedDict


class WireTool(TypedDict):
    """A tools/list item: name, optional description, and JSON input schema."""

    name: str
    description: NotRequired[str | None]
    inputSchema: dict[str, Any]


@dataclass(frozen=True)
class Overage:
    """One excess: item identifies the wire location; length/limit count characters."""

    item: str
    length: int
    limit: int


@dataclass(frozen=True)
class BudgetReport:
    """Measured instructions length, tool count, and every over-budget wire item."""

    instructions_length: int
    tool_count: int
    overages: tuple[Overage, ...]

    @property
    def exit_code(self) -> int:
        """Return 1 for any overage, otherwise 0."""
        return int(bool(self.overages))


def check_text_budget(
    instructions: str | None,
    tools: Sequence[WireTool],
    recipe_names: Set[str],
) -> BudgetReport:
    """Measure decoded wire characters without changing inputs or loading a lab.

    instructions is the initialize text (None means absent). tools is the complete
    tools/list result. recipe_names contains exact catalog declaration names, not
    prefixes. Limits are inclusive: instructions 2000, recipes 200, other tools
    400, and schema property descriptions 100. Missing descriptions count as zero.
    Invalid wire items raise rather than being treated as empty input.
    """
    overages: list[Overage] = []
    instructions_length = len(instructions or "")
    if instructions_length > 2000:
        overages.append(Overage("instructions", instructions_length, 2000))
    for tool in tools:
        path = f"tools/{tool['name']}"
        limit = 200 if tool["name"] in recipe_names else 400
        length = len(tool.get("description") or "")
        if length > limit:
            overages.append(Overage(f"{path}/description", length, limit))
        overages.extend(_schema_overages(tool["inputSchema"], f"{path}/inputSchema"))
    return BudgetReport(instructions_length, len(tools), tuple(overages))


def _schema_overages(schema: dict[str, Any], path: str) -> Iterator[Overage]:
    for child_path, child, is_property in _subschemas(schema, path):
        length = len(child.get("description") or "")
        if is_property and length > 100:
            yield Overage(f"{child_path}/description", length, 100)
        yield from _schema_overages(child, child_path)


def _subschemas(
    schema: dict[str, Any], path: str
) -> Iterator[tuple[str, dict[str, Any], bool]]:
    # Visit schema keywords, not JSON instance data in defaults/examples/const/enum.
    for keyword in (
        "properties",
        "$defs",
        "definitions",
        "patternProperties",
        "dependentSchemas",
    ):
        for name, child in schema.get(keyword, {}).items():
            if isinstance(child, dict):
                yield f"{path}/{keyword}/{name}", child, keyword == "properties"
    for keyword in (
        "items",
        "additionalProperties",
        "unevaluatedProperties",
        "unevaluatedItems",
        "propertyNames",
        "contains",
        "not",
        "if",
        "then",
        "else",
    ):
        child = schema.get(keyword)
        if isinstance(child, dict):
            yield f"{path}/{keyword}", child, False
    for keyword in ("anyOf", "allOf", "oneOf", "prefixItems"):
        for index, child in enumerate(schema.get(keyword, [])):
            if isinstance(child, dict):
                yield f"{path}/{keyword}/{index}", child, False


def _read_wire(timeout: float) -> tuple[str | None, list[WireTool]]:
    requests = (
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "check-mcp-text-budget", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    )
    root = Path(__file__).resolve().parents[1]
    # The stdio loop handles requests in order, then drains on EOF. communicate
    # bounds the whole exchange and kills/reaps the server if the timeout expires.
    result = subprocess.run(
        [sys.executable, str(root / "mcp_servers/measure.py")],
        cwd=root,
        input="".join(json.dumps(request) + "\n" for request in requests),
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=timeout,
    )
    responses: dict[int, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        response = json.loads(line)
        if "error" in response:
            raise ValueError(f"MCP request failed: {response['error']}")
        if "id" in response:
            responses[response["id"]] = response["result"]
    return responses[1].get("instructions"), responses[2]["tools"]


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive and finite")
    return seconds


def main(argv: Sequence[str] | None = None) -> int:
    """Check this checkout's measure stdio server without starting a GUI.

    argv accepts --timeout SECONDS (positive, default 60) for the wire subprocess.
    Print a JSON BudgetReport plus overage_count to stdout and each excess to
    stderr. Return 0 within budget, 1 for overages, or 2 for collection failure.
    argparse handles --help and invalid arguments with its normal SystemExit.
    Lab declaration/import failures propagate; no registration or recipe runs here.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=_positive_seconds, default=60.0)
    args = parser.parse_args(argv)

    # Use the same lab assembly as the server, never a guessed recipe prefix.
    from zcu_lab.assembly import lab_definitions

    recipe_names = {recipe.name for recipe in lab_definitions().recipes}
    try:
        instructions, tools = _read_wire(args.timeout)
        report = check_text_budget(instructions, tools, recipe_names)
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(f"MCP text budget collection failed: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {**asdict(report), "overage_count": len(report.overages)},
            ensure_ascii=False,
        )
    )
    for overage in report.overages:
        print(
            f"{overage.item}: {overage.length} characters (limit {overage.limit})",
            file=sys.stderr,
        )
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
