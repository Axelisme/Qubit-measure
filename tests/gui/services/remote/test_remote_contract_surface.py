from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from pathlib import Path

from zcu_tools.gui.app.main.services.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.app.main.services.remote.method_specs import METHOD_SPECS
from zcu_tools.gui.remote.method_spec import McpExposure

EXPECTED_WIRE_METHODS = {
    "adapter.guide",
    "adapter.list",
    "analyze.cancel",
    "app.shutdown",
    "arb_waveform.list",
    "arb_waveform.preview",
    "arb_waveform.set",
    "context.active",
    "context.labels",
    "context.md_del_attr",
    "context.md_get",
    "context.md_get_attr",
    "context.md_set_attr",
    "context.ml_create_from_role",
    "context.ml_del_module",
    "context.ml_del_waveform",
    "context.ml_get",
    "context.ml_list_roles",
    "context.ml_rename_module",
    "context.ml_rename_waveform",
    "context.new",
    "context.use",
    "device.active_operations",
    "device.cancel_operation",
    "device.connect",
    "device.disconnect",
    "device.forget",
    "device.list",
    "device.reconnect",
    "device.setup",
    "device.setup_spec",
    "device.snapshot",
    "dialog.screenshot",
    "editor.commit",
    "editor.discard",
    "editor.get",
    "editor.new",
    "editor.set_field",
    "notify.await",
    "notify.open",
    "operation.await",
    "operation.progress",
    "predictor.clear",
    "predictor.info",
    "predictor.load",
    "predictor.predict",
    "predictor.set_model_params",
    "project.info",
    "resources.versions",
    "result_scope.list",
    "run.running_tab",
    "soc.connect",
    "soc.info",
    "startup.apply",
    "state.has_active_context",
    "state.has_context",
    "state.has_project",
    "state.has_soc",
    "tab.analyze",
    "tab.close",
    "tab.get_analyze_params",
    "tab.get_analyze_result",
    "tab.get_cfg",
    "tab.get_current_figure",
    "tab.get_post_analyze_params",
    "tab.get_post_analyze_result",
    "tab.list_all",
    "tab.load_data",
    "tab.new",
    "tab.post_analyze",
    "tab.run_cancel",
    "tab.run_start",
    "tab.save_data",
    "tab.save_image",
    "tab.save_post_image",
    "tab.save_result",
    "tab.save_set_paths",
    "tab.set_active",
    "tab.set_cfg",
    "tab.snapshot",
    "tab.writeback_apply",
    "tab.writeback_preview",
    "tab.writeback_set",
    "value.list",
    "value.read",
    "view.screenshot",
    "view.snapshot",
}

EXPECTED_OFF_MAIN_METHODS = {"operation.await", "notify.await"}

EXPECTED_HIDDEN_EXPECTED_VERSIONS_METHODS = {
    "tab.run_start",
    "tab.load_data",
    "tab.save_data",
    "tab.save_image",
    "tab.save_post_image",
    "tab.save_result",
    "tab.save_set_paths",
    "arb_waveform.set",
    "tab.writeback_set",
    "tab.writeback_apply",
    "editor.commit",
}

EXPECTED_INTERNAL_MCP_METHODS = {
    "app.shutdown",
    "context.active",
    "context.labels",
    "operation.progress",
    "project.info",
    "run.running_tab",
    "state.has_active_context",
    "state.has_context",
    "state.has_project",
    "state.has_soc",
    "view.snapshot",
}

EXPECTED_OVERRIDE_MCP_METHODS = {
    "context.md_del_attr": ("gui_context_md_delete",),
    "context.md_get": ("gui_context_md_read",),
    "context.md_get_attr": ("gui_context_md_read",),
    "context.md_set_attr": ("gui_context_md_write",),
    "device.connect": ("gui_device_connect",),
    "device.disconnect": ("gui_device_disconnect",),
    "device.reconnect": ("gui_device_connect",),
    "device.setup": ("gui_device_apply",),
    "dialog.screenshot": ("gui_screenshot",),
    "editor.get": ("gui_editor_get_cfg",),
    "editor.new": ("gui_editor_open",),
    "editor.set_field": ("gui_editor_set",),
    "notify.await": ("gui_prompt_user",),
    "notify.open": ("gui_prompt_user",),
    "operation.await": ("gui_op_wait", "gui_op_poll"),
    "resources.versions": ("gui_debug_resource_versions",),
    "soc.connect": ("gui_soc_connect",),
    "tab.analyze": ("gui_tab_analyze_start",),
    "tab.get_current_figure": ("gui_tab_get_current_figure",),
    "tab.post_analyze": ("gui_tab_post_analyze_start",),
    "tab.run_start": ("gui_tab_run_start",),
    "tab.save_data": ("gui_tab_save",),
    "tab.save_image": ("gui_tab_save",),
    "tab.save_post_image": ("gui_tab_save",),
    "tab.save_result": ("gui_tab_save",),
    "tab.set_cfg": ("gui_tab_set_cfg",),
    "view.screenshot": ("gui_screenshot",),
}

_BROAD_CATCH_NAMES = frozenset({"RuntimeError", "Exception", "BaseException"})
CatchViolation = tuple[str, int, str]
AliasTargets = dict[str, set[str]]


def _record_import_aliases(
    node: ast.Import | ast.ImportFrom, aliases: AliasTargets
) -> set[str]:
    bindings: set[str] = set()
    if isinstance(node, ast.Import):
        for imported in node.names:
            binding = imported.asname or imported.name.partition(".")[0]
            target = imported.name if imported.asname else binding
            bindings.add(binding)
            aliases.setdefault(binding, set()).add(target)
        return bindings
    if node.module is None:
        return bindings
    for imported in node.names:
        if imported.name == "*":
            continue
        binding = imported.asname or imported.name
        bindings.add(binding)
        aliases.setdefault(binding, set()).add(f"{node.module}.{imported.name}")
    return bindings


class _ModuleScopeImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.aliases: AliasTargets = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        del node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        del node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        del node

    def visit_Lambda(self, node: ast.Lambda) -> None:
        del node

    def visit_ListComp(self, node: ast.ListComp) -> None:
        del node

    def visit_SetComp(self, node: ast.SetComp) -> None:
        del node

    def visit_DictComp(self, node: ast.DictComp) -> None:
        del node

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        del node

    def visit_Import(self, node: ast.Import) -> None:
        _record_import_aliases(node, self.aliases)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        _record_import_aliases(node, self.aliases)


def _module_import_aliases(tree: ast.Module) -> AliasTargets:
    visitor = _ModuleScopeImportVisitor()
    visitor.visit(tree)
    return visitor.aliases


class _FunctionScopeBindingVisitor(ast.NodeVisitor):
    """Collect lexical bindings in one function without crossing child scopes."""

    def __init__(self, root: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._root = root
        self.bindings: set[str] = {
            arg.arg
            for arg in (
                *root.args.posonlyargs,
                *root.args.args,
                *root.args.kwonlyargs,
            )
        }
        if root.args.vararg is not None:
            self.bindings.add(root.args.vararg.arg)
        if root.args.kwarg is not None:
            self.bindings.add(root.args.kwarg.arg)
        self.import_aliases: AliasTargets = {}

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is self._root:
            self.generic_visit(node)
        else:
            self.bindings.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node is self._root:
            self.generic_visit(node)
        else:
            self.bindings.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.bindings.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        del node

    def visit_ListComp(self, node: ast.ListComp) -> None:
        del node

    def visit_SetComp(self, node: ast.SetComp) -> None:
        del node

    def visit_DictComp(self, node: ast.DictComp) -> None:
        del node

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        del node

    def visit_Import(self, node: ast.Import) -> None:
        self.bindings.update(_record_import_aliases(node, self.import_aliases))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.bindings.update(_record_import_aliases(node, self.import_aliases))

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.bindings.add(node.id)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self.bindings.add(node.name)
        self.generic_visit(node)


def _qualified_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _qualified_name(node.value)
        if owner is not None:
            return f"{owner}.{node.attr}"
    return None


def _normalized_catch_names(
    node: ast.expr, aliases: AliasTargets, shadowed_locals: set[str]
) -> set[str]:
    if isinstance(node, ast.Tuple):
        return {
            name
            for item in node.elts
            for name in _normalized_catch_names(item, aliases, shadowed_locals)
        }
    qualified = _qualified_name(node)
    if qualified is None:
        return set()
    root, separator, suffix = qualified.partition(".")
    if root in shadowed_locals:
        return set()
    normalized: set[str] = set()
    for normalized_root in aliases.get(root, {root}):
        target = (
            f"{normalized_root}.{suffix}" if separator and suffix else normalized_root
        )
        if target.startswith("builtins."):
            target = target.removeprefix("builtins.")
        normalized.add(target)
    return normalized


class _RequestCatchVisitor(ast.NodeVisitor):
    def __init__(
        self,
        root: ast.FunctionDef | ast.AsyncFunctionDef,
        aliases: AliasTargets,
        shadowed_locals: set[str],
    ) -> None:
        self._root = root
        self._aliases = aliases
        self._shadowed_locals = shadowed_locals
        self.violations: list[CatchViolation] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is self._root:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node is self._root:
            self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        del node

    def visit_Lambda(self, node: ast.Lambda) -> None:
        del node

    def visit_ListComp(self, node: ast.ListComp) -> None:
        del node

    def visit_SetComp(self, node: ast.SetComp) -> None:
        del node

    def visit_DictComp(self, node: ast.DictComp) -> None:
        del node

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        del node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.violations.append((self._root.name, node.lineno, "bare"))
        else:
            broad = _normalized_catch_names(
                node.type, self._aliases, self._shadowed_locals
            )
            self.violations.extend(
                (self._root.name, node.lineno, name)
                for name in sorted(broad & _BROAD_CATCH_NAMES)
            )
        self.generic_visit(node)


def _broad_request_catches(
    source: str, request_handler_names: set[str], *, filename: str = "<source>"
) -> list[CatchViolation]:
    tree = ast.parse(source, filename=filename)
    module_aliases = _module_import_aliases(tree)
    violations: list[CatchViolation] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in request_handler_names:
            continue
        binding_visitor = _FunctionScopeBindingVisitor(node)
        binding_visitor.visit(node)
        aliases = {name: set(targets) for name, targets in module_aliases.items()}
        for binding in binding_visitor.bindings:
            aliases.pop(binding, None)
        aliases.update(binding_visitor.import_aliases)
        shadowed_locals = binding_visitor.bindings - set(binding_visitor.import_aliases)
        visitor = _RequestCatchVisitor(node, aliases, shadowed_locals)
        visitor.visit(node)
        violations.extend(visitor.violations)
    return violations


def test_exact_remote_wire_method_surface() -> None:
    assert set(METHOD_SPECS) == EXPECTED_WIRE_METHODS
    assert set(METHOD_REGISTRY) == EXPECTED_WIRE_METHODS


def test_exact_off_main_method_surface() -> None:
    off_main = {
        name
        for name, bound_method in METHOD_REGISTRY.items()
        if bound_method.off_main_thread
    }
    assert off_main == EXPECTED_OFF_MAIN_METHODS


def test_request_handlers_do_not_translate_broad_runtime_exceptions() -> None:
    handlers_by_path: dict[Path, set[str]] = {}
    for bound_method in METHOD_REGISTRY.values():
        handler = bound_method.handler
        source_path = inspect.getsourcefile(handler)
        assert source_path is not None
        handlers_by_path.setdefault(Path(source_path), set()).add(handler.__name__)

    violations: list[str] = []
    for path, handler_names in sorted(handlers_by_path.items()):
        catches = _broad_request_catches(
            path.read_text(encoding="utf-8"), handler_names, filename=str(path)
        )
        violations.extend(
            f"{path.name}:{line}:{handler_name}: {catch_name}"
            for handler_name, line, catch_name in catches
        )

    assert violations == []


def test_broad_request_catch_analyzer_normalizes_aliases_and_ownership() -> None:
    source = """
from builtins import RuntimeError as RE
import builtins as bi

def _h_alias():
    try:
        pass
    except RE:
        pass

def _h_qualified():
    try:
        pass
    except bi.Exception:
        pass

def _h_tuple():
    try:
        pass
    except (ValueError, RuntimeError):
        pass

def _h_bare():
    try:
        pass
    except:
        pass

def _h_local_alias():
    from builtins import RuntimeError as LocalRE
    try:
        pass
    except LocalRE:
        pass

def _h_local_qualified():
    import builtins as local_bi
    try:
        pass
    except local_bi.BaseException:
        pass

def _nonrequest_helper():
    try:
        pass
    except BaseException:
        pass
"""
    violations = _broad_request_catches(
        source,
        {
            "_h_alias",
            "_h_qualified",
            "_h_tuple",
            "_h_bare",
            "_h_local_alias",
            "_h_local_qualified",
        },
    )

    assert {(handler, catch) for handler, _, catch in violations} == {
        ("_h_alias", "RuntimeError"),
        ("_h_qualified", "Exception"),
        ("_h_tuple", "RuntimeError"),
        ("_h_bare", "bare"),
        ("_h_local_alias", "RuntimeError"),
        ("_h_local_qualified", "BaseException"),
    }


def test_broad_request_catch_analyzer_ignores_nonrequest_helper_boundary() -> None:
    source = """
def _h_request():
    try:
        pass
    except ValueError:
        pass

def _nonrequest_helper():
    try:
        pass
    except Exception:
        pass
"""

    assert _broad_request_catches(source, {"_h_request"}) == []


def test_broad_request_catch_analyzer_keeps_child_scope_aliases_isolated() -> None:
    source = """
from builtins import RuntimeError as RE

def _h_request():
    from builtins import ValueError as RE

    def nested_function():
        from builtins import Exception as NestedError
        try:
            pass
        except NestedError:
            pass

    class NestedClass:
        from builtins import BaseException as NestedError
        try:
            pass
        except NestedError:
            pass

    shadowed = [RE for RE in ()]
    try:
        pass
    except RE:
        pass

def _nonrequest_helper():
    try:
        pass
    except Exception:
        pass
"""

    assert _broad_request_catches(source, {"_h_request"}) == []


def test_broad_request_catch_analyzer_local_bindings_shadow_module_aliases() -> None:
    source = """
from builtins import RuntimeError as RE
from builtins import Exception as Error

def _h_assignment():
    RE = ValueError
    try:
        pass
    except RE:
        pass

def _h_parameter(Error):
    try:
        pass
    except Error:
        pass
"""

    assert _broad_request_catches(source, {"_h_assignment", "_h_parameter"}) == []


def test_broad_request_catch_analyzer_preserves_conditional_import_targets() -> None:
    source = """
if condition:
    from builtins import RuntimeError as ModuleError
else:
    from builtins import ValueError as ModuleError

def _h_module_conditional():
    try:
        pass
    except ModuleError:
        pass

def _h_function_conditional():
    if condition:
        from builtins import RuntimeError as LocalError
    else:
        from builtins import ValueError as LocalError
    try:
        pass
    except LocalError:
        pass
"""
    violations = _broad_request_catches(
        source, {"_h_module_conditional", "_h_function_conditional"}
    )

    assert {(handler, catch) for handler, _, catch in violations} == {
        ("_h_module_conditional", "RuntimeError"),
        ("_h_function_conditional", "RuntimeError"),
    }


def test_exact_hidden_expected_versions_surface() -> None:
    guarded = {
        method
        for method, spec in METHOD_SPECS.items()
        for param in spec.params
        if param.name == "expected_versions" and param.mcp_hidden
    }
    assert guarded == EXPECTED_HIDDEN_EXPECTED_VERSIONS_METHODS


def test_method_entries_build_specs_and_registry_from_same_source() -> None:
    assert set(METHOD_SPECS) == set(METHOD_REGISTRY) == EXPECTED_WIRE_METHODS
    for method, bound_method in METHOD_REGISTRY.items():
        assert bound_method.spec is METHOD_SPECS[method]


def test_exact_mcp_exposure_policy_surface() -> None:
    internal = {
        method
        for method, spec in METHOD_SPECS.items()
        if spec.mcp.exposure is McpExposure.INTERNAL
    }
    overrides = {
        method: spec.mcp.override_tool_names
        for method, spec in METHOD_SPECS.items()
        if spec.mcp.exposure is McpExposure.OVERRIDE
    }
    generated = {
        method
        for method, spec in METHOD_SPECS.items()
        if spec.mcp.exposure is McpExposure.GENERATED
    }
    assert internal == EXPECTED_INTERNAL_MCP_METHODS
    assert overrides == EXPECTED_OVERRIDE_MCP_METHODS
    assert generated == EXPECTED_WIRE_METHODS - internal - set(overrides)


def test_method_specs_import_does_not_pull_dispatch_or_service() -> None:
    code = (
        "import sys\n"
        "import zcu_tools.gui.app.main.services.remote.method_specs\n"
        "bad_exact = {\n"
        "    'qtpy',\n"
        "    'zcu_tools.gui.app.main.controller',\n"
        "    'zcu_tools.gui.app.main.services.remote.dispatch',\n"
        "    'zcu_tools.gui.app.main.services.remote.service',\n"
        "}\n"
        "bad = sorted(name for name in sys.modules if name in bad_exact)\n"
        "bad.extend(\n"
        "    sorted(\n"
        "        name for name in sys.modules\n"
        "        if name.startswith('zcu_tools.gui.app.main.services.remote.handlers')\n"
        "    )\n"
        ")\n"
        "assert not bad, bad\n"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )


def test_services_package_lazy_exports_preserve_explicit_imports() -> None:
    code = (
        "from zcu_tools.gui.app.main.services import (\n"
        "    create_persistence_caretaker,\n"
        "    StartupProjectRequest,\n"
        ")\n"
        "import zcu_tools.gui.app.main.services as services\n"
        "assert services.create_persistence_caretaker is create_persistence_caretaker\n"
        "assert services.StartupProjectRequest is StartupProjectRequest\n"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
