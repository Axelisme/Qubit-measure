"""MCP exposure projection for measure-gui wire methods."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from zcu_tools.gui.remote.method_spec import McpExposure, MethodSpec
from zcu_tools.mcp.core.bridge import McpServerConfig, Tool


@dataclass(frozen=True)
class McpExposurePlan:
    """Derived MCP generation plan from wire method policy."""

    generated_methods: frozenset[str]
    non_generated_methods: frozenset[str]
    override_methods: Mapping[str, tuple[str, ...]]
    generated_tool_names: Mapping[str, str]


def _generated_tool_name(config: McpServerConfig, method: str, spec: MethodSpec) -> str:
    return spec.tool_name or config.tool_prefix + method.replace(".", "_")


def build_mcp_exposure_plan(
    config: McpServerConfig,
    method_specs: Mapping[str, MethodSpec],
    manual_tools: Mapping[str, Tool],
) -> McpExposurePlan:
    """Project wire method MCP policies into generated/non-generated sets."""
    manual_tool_names = frozenset(manual_tools)
    generated_methods: set[str] = set()
    non_generated_methods: set[str] = set()
    override_methods: dict[str, tuple[str, ...]] = {}
    generated_tool_names: dict[str, str] = {}
    generated_tool_owner: dict[str, str] = {}
    generated_collisions: dict[str, list[str]] = {}
    generated_manual_collisions: dict[str, str] = {}

    for method, spec in method_specs.items():
        policy = spec.mcp
        if policy.exposure is McpExposure.GENERATED:
            tool_name = _generated_tool_name(config, method, spec)
            previous = generated_tool_owner.get(tool_name)
            if previous is not None:
                generated_collisions.setdefault(tool_name, [previous]).append(method)
            generated_tool_owner[tool_name] = method
            if tool_name in manual_tool_names:
                generated_manual_collisions[tool_name] = method
            generated_methods.add(method)
            generated_tool_names[method] = tool_name
            continue

        if spec.tool_name:
            raise RuntimeError(
                f"{method}: {policy.exposure.value} MCP policy cannot use "
                f"MethodSpec.tool_name={spec.tool_name!r}"
            )

        non_generated_methods.add(method)
        if policy.exposure is McpExposure.INTERNAL:
            continue
        if policy.exposure is McpExposure.OVERRIDE:
            missing = [
                name
                for name in policy.override_tool_names
                if name not in manual_tool_names
            ]
            if missing:
                raise RuntimeError(
                    f"{method}: override MCP tools are not registered: {missing}"
                )
            override_methods[method] = policy.override_tool_names
            continue
        raise RuntimeError(f"{method}: unsupported MCP exposure {policy.exposure!r}")

    if generated_collisions:
        details = {
            name: sorted(methods) for name, methods in generated_collisions.items()
        }
        raise RuntimeError(f"generated MCP tool name collision: {details}")
    if generated_manual_collisions:
        raise RuntimeError(
            "generated MCP tools collide with manual tools: "
            f"{dict(sorted(generated_manual_collisions.items()))}"
        )

    return McpExposurePlan(
        generated_methods=frozenset(generated_methods),
        non_generated_methods=frozenset(non_generated_methods),
        override_methods=MappingProxyType(dict(sorted(override_methods.items()))),
        generated_tool_names=MappingProxyType(
            dict(sorted(generated_tool_names.items()))
        ),
    )
