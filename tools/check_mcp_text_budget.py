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
    Null descriptions also count as zero. Invalid text or required tool structure
    raises ValueError rather than being treated as empty input. Visited schema
    nodes must be objects or booleans, with string/null descriptions and correctly
    shaped schema containers. Instance data and other schema keywords are not
    validated; this is not a general JSON Schema validator.
    """
    overages: list[Overage] = []
    instructions_length = _text_length(instructions, "instructions")
    if instructions_length > 2000:
        overages.append(Overage("instructions", instructions_length, 2000))
    for tool in _validated_tools(tools):
        path = f"tools/{tool['name']}"
        limit = 200 if tool["name"] in recipe_names else 400
        length = _text_length(tool.get("description"), f"{path}/description")
        if length > limit:
            overages.append(Overage(f"{path}/description", length, limit))
        overages.extend(_schema_overages(tool["inputSchema"], f"{path}/inputSchema"))
    return BudgetReport(instructions_length, len(tools), tuple(overages))


def _validated_tools(value: object) -> list[WireTool]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("tools must be a sequence of tool objects")
    tools: list[WireTool] = []
    for index, tool in enumerate(value):
        if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
            raise ValueError(f"tools/{index} must be an object with a string name")
        name = tool["name"]
        schema = tool.get("inputSchema")
        if not isinstance(schema, dict):
            raise ValueError(f"tools/{name}/inputSchema must be an object")
        description = tool.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError(f"tools/{name}/description must be a string or null")
        tools.append(WireTool(name=name, description=description, inputSchema=schema))
    return tools


def _text_length(value: object, path: str) -> int:
    if value is None:
        return 0
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string or null")
    return len(value)


def _schema_overages(
    schema: object, path: str, *, is_property: bool = False
) -> Iterator[Overage]:
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise ValueError(f"{path} must be a schema object or boolean")
    length = _text_length(schema.get("description"), f"{path}/description")
    if is_property and length > 100:
        yield Overage(f"{path}/description", length, 100)
    for child_path, child, child_is_property in _subschemas(schema, path):
        yield from _schema_overages(child, child_path, is_property=child_is_property)


def _subschemas(
    schema: dict[str, Any], path: str
) -> Iterator[tuple[str, object, bool]]:
    # Visit schema keywords, not JSON instance data in defaults/examples/const/enum.
    for keyword in (
        "properties",
        "$defs",
        "definitions",
        "patternProperties",
        "dependentSchemas",
    ):
        if keyword not in schema:
            continue
        children = schema[keyword]
        if not isinstance(children, dict):
            raise ValueError(f"{path}/{keyword} must be an object of schemas")
        for name, child in children.items():
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
        if keyword in schema:
            yield f"{path}/{keyword}", schema[keyword], False
    for keyword in ("anyOf", "allOf", "oneOf", "prefixItems"):
        if keyword not in schema:
            continue
        children = schema[keyword]
        if not isinstance(children, list):
            raise ValueError(f"{path}/{keyword} must be an array of schemas")
        for index, child in enumerate(children):
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
    # This fixed, read-only collector targets the known same-repo stdio server,
    # not arbitrary MCP peers. Its three messages are processed in order and EOF
    # drains the server. subprocess.run bounds the exchange and kills/reaps on
    # timeout. A full async MCP SDK dependency adds little for this use; reconsider
    # it for interactive tool calls, heterogeneous servers, or notification routing.
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
        if not isinstance(response, dict):
            raise ValueError("MCP response must be an object")
        if "error" in response:
            raise ValueError(f"MCP request failed: {response['error']}")
        if "id" not in response:
            continue
        response_id = response["id"]
        if type(response_id) is not int or response_id not in (1, 2):
            raise ValueError("MCP response id must be 1 or 2")
        payload = response.get("result")
        if not isinstance(payload, dict):
            raise ValueError(f"MCP response {response_id} result must be an object")
        responses[response_id] = payload
    if 1 not in responses or 2 not in responses:
        raise ValueError("MCP initialize and tools/list responses are required")
    tools = responses[2].get("tools")
    if not isinstance(tools, list):
        raise ValueError("MCP tools/list result/tools must be an array")
    return responses[1].get("instructions"), tools


def collect_text_budget(
    recipe_names: Set[str], *, timeout: float = 60.0
) -> BudgetReport:
    """Collect and measure the same-repo measure server's decoded stdio text.

    recipe_names contains exact catalog names used for the recipe budget. timeout
    is a positive, finite number of seconds bounding the whole subprocess exchange.
    Start the server with this interpreter, send three read-only messages, and
    close stdin with EOF. A timeout kills and reaps the child. No GUI connection,
    tool call, or hardware operation occurs. Return the measured BudgetReport.
    Raise ValueError for invalid timeout, decoded wire structure, or text;
    OSError for launch failures; subprocess.SubprocessError for timeout/nonzero exit.
    """
    instructions, tools = _read_wire(_validated_seconds(timeout))
    return check_text_budget(instructions, tools, recipe_names)


def _validated_seconds(seconds: float) -> float:
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("timeout must be positive and finite")
    return seconds


def _positive_seconds(value: str) -> float:
    try:
        return _validated_seconds(float(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


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
        report = collect_text_budget(recipe_names, timeout=args.timeout)
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
