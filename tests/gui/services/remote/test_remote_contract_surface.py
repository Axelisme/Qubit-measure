from __future__ import annotations

import subprocess
import sys

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
        "    PersistenceCaretaker,\n"
        "    StartupProjectRequest,\n"
        ")\n"
        "import zcu_tools.gui.app.main.services as services\n"
        "assert services.PersistenceCaretaker is PersistenceCaretaker\n"
        "assert services.StartupProjectRequest is StartupProjectRequest\n"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
