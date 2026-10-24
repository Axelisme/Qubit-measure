"""Core metadata controls flat groups without changing canonical input paths."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated, Literal

import pytest
from pydantic import Field
from qtpy.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
from zcu_tools.cfg_model import ConfigBase, FieldMeta, Range
from zcu_tools.gui.form import Display, FormEnv, FormInput, FormModel, FormState
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.program.v2 import SweepCfg

from tests.gui._form_converters import RebuildingForm
from tests.gui._user_input import ComponentDriver

FOLDED, EXPANDED = "\u25b8 ", "\u25be "
type Host = Callable[[FormInput[FormModel]], ComponentDriver]


class DriveCfg(ConfigBase):
    level: float = Field(default=1.0, gt=0)


class FlatCfg(ConfigBase):
    drive: Annotated[DriveCfg, FieldMeta(group="Modules")] = Field(
        default_factory=DriveCfg
    )
    scan_window: Annotated[
        SweepCfg, FieldMeta(group="Sweep", label="Frequency", unit="MHz")
    ]


@pytest.fixture
def host(qapp: QApplication) -> Iterator[Host]:
    windows: list[QWidget] = []

    def make(initial: FormInput[FormModel]) -> ComponentDriver:
        env = FormEnv()
        state = FormState(initial)
        editor = FormEditor(state, env)
        window = QWidget()
        windows.append(window)
        layout = QVBoxLayout(window)
        layout.addWidget(editor)
        refresh = QPushButton("Refresh", window)
        refresh.clicked.connect(lambda: editor.refresh(env))
        layout.addWidget(refresh)
        reset = QPushButton("Reset", window)
        reset.clicked.connect(lambda: state.reset(initial))
        layout.addWidget(reset)
        window.resize(800, 760)
        window.show()
        qapp.processEvents()
        return ComponentDriver(window)

    yield make
    for window in windows:
        window.close()
        window.deleteLater()
    qapp.processEvents()


def test_single_sweep_group_merges_title_and_preserves_commands_and_paths(
    host: Host, tmp_path: Path
) -> None:
    form = Form(FlatCfg, {"scan_window": Range.between(10.0, 20.0, expts=11)})
    driver = host(form.default_input(FormEnv()))
    driver.screenshot(tmp_path / "01-flat-single-axis.png")
    assert driver.observe(object_name="form.fold.scan_window").text == (
        EXPANDED + "Sweep · Frequency (MHz)"
    )
    assert (
        driver.observe(object_name="form.fold.group.Modules").text == FOLDED + "Modules"
    )
    assert driver.observe(object_name="form.input.scan_window.start").text == "10.0"
    driver.enter_text(object_name="form.input.scan_window.start", value="12")
    driver.screenshot(tmp_path / "02-edited-axis.png")
    driver.invoke_context_action(
        object_name="form.fold.scan_window", action="Switch to Center / Span"
    )
    driver.screenshot(tmp_path / "03-center-span.png")
    assert driver.observe(object_name="form.input.scan_window.center").text == "16.0"
    assert driver.observe(object_name="form.input.scan_window.span").text == "8.0"
    assert driver.observe(object_name="form.fold.scan_window").text == (
        EXPANDED + "Sweep · Frequency (MHz)"
    )
    driver.click(object_name="form.fold.group.Modules")
    driver.screenshot(tmp_path / "04-modules-group-open.png")
    driver.click(object_name="form.fold.drive")
    driver.screenshot(tmp_path / "05-drive-open.png")
    driver.click(object_name="form.fold.scan_window")
    driver.screenshot(tmp_path / "06-axis-folded.png")
    driver.click("Refresh")
    driver.screenshot(tmp_path / "07-refresh-user-folds.png")
    assert (
        driver.observe(object_name="form.fold.group.Modules").text
        == EXPANDED + "Modules"
    )
    assert driver.observe(object_name="form.input.drive.level").visible
    assert driver.observe(object_name="form.fold.scan_window").text == (
        FOLDED + "Sweep · Frequency (MHz)"
    )
    assert not driver.observe(object_name="form.input.scan_window.center").visible


def test_multiple_sweep_axes_start_open_and_keep_user_folds_on_refresh(
    host: Host, tmp_path: Path
) -> None:
    class DualCfg(FlatCfg):
        bias_window: Annotated[SweepCfg, FieldMeta(group="Sweep", unit="A")]

    form = Form(
        DualCfg,
        {
            "scan_window": Range.between(10.0, 20.0, expts=11),
            "bias_window": Range.between(0.0, 0.2, expts=3),
        },
    )
    driver = host(form.default_input(FormEnv()))
    driver.screenshot(tmp_path / "01-two-axes.png")
    assert (
        driver.observe(object_name="form.fold.group.Sweep").text == EXPANDED + "Sweep"
    )
    assert (
        driver.observe(object_name="form.fold.scan_window").text
        == EXPANDED + "Frequency"
    )
    assert (
        driver.observe(object_name="form.fold.bias_window").text
        == EXPANDED + "Bias window"
    )
    assert driver.observe(object_name="form.input.scan_window.start").visible
    assert driver.observe(object_name="form.input.bias_window.start").visible
    driver.click(object_name="form.fold.scan_window")
    driver.screenshot(tmp_path / "02-axis-folded.png")
    driver.click(object_name="form.fold.group.Sweep")
    driver.screenshot(tmp_path / "03-sweep-folded.png")
    driver.click("Refresh")
    driver.screenshot(tmp_path / "04-refresh-folded.png")
    assert driver.observe(object_name="form.fold.group.Sweep").text == FOLDED + "Sweep"
    driver.click(object_name="form.fold.group.Sweep")
    driver.screenshot(tmp_path / "05-group-reopened.png")
    assert not driver.observe(object_name="form.input.scan_window.start").visible
    assert driver.observe(object_name="form.input.bias_window.start").visible


def test_groups_and_merged_axis_keep_declared_visible_order_after_publications(
    host: Host, tmp_path: Path
) -> None:
    class OrderedCfg(ConfigBase):
        tail: int = 9
        diagnostic: Annotated[int, FieldMeta(group="Diagnostics")] = 3
        sample: Annotated[int, FieldMeta(group="Sensors")] = 1
        plain: int = 2
        scan_window: Annotated[
            SweepCfg, FieldMeta(group="Sweep", label="Frequency", unit="MHz")
        ]

    form = Form(
        OrderedCfg,
        {"scan_window": Range.between(10.0, 20.0, expts=11)},
        field_order=("sample", "plain", "diagnostic", "scan_window", "tail"),
    )
    driver = host(form.default_input(FormEnv()))
    expected = (
        "form.fold.group.Sensors",
        "form.input.plain",
        "form.fold.group.Diagnostics",
        "form.fold.scan_window",
        "form.input.tail",
    )

    def check_order() -> None:
        observations = {name: driver.observe(object_name=name) for name in expected}
        positions = {
            name: observed.bounds[1]
            for name, observed in observations.items()
            if observed.bounds is not None
        }
        assert len(positions) == len(expected), observations
        assert tuple(sorted(positions, key=lambda name: positions[name])) == expected

    driver.screenshot(tmp_path / "01-initial-order.png")
    check_order()
    driver.click(object_name="form.fold.group.Sensors")
    driver.screenshot(tmp_path / "02-expanded-order.png")
    check_order()
    driver.enter_text(object_name="form.input.plain", value="5")
    driver.screenshot(tmp_path / "03-published-order.png")
    check_order()
    driver.click("Refresh")
    driver.screenshot(tmp_path / "04-refreshed-order.png")
    check_order()
    assert driver.observe(object_name="form.input.sample").visible
    assert driver.observe(object_name="form.input.plain").text == "5"
    driver.click("Reset")
    driver.screenshot(tmp_path / "05-reset-order.png")
    check_order()
    assert driver.observe(object_name="form.input.sample").visible
    assert driver.observe(object_name="form.input.plain").text == "2"


def test_initial_errors_expand_sections_even_when_named_modules(
    host: Host, tmp_path: Path
) -> None:
    class InvalidCfg(ConfigBase):
        modules: DriveCfg
        renamed: DriveCfg
        healthy: DriveCfg = Field(default_factory=DriveCfg)

    form = Form(InvalidCfg, {"modules": {"level": -1.0}, "renamed": {"level": -1.0}})
    driver = host(form.default_input(FormEnv()))
    driver.screenshot(tmp_path / "01-initial-errors.png")
    for name in ("modules", "renamed"):
        assert driver.observe(object_name=f"form.fold.{name}").text == (
            EXPANDED + name.capitalize()
        )
        assert driver.observe(object_name=f"form.input.{name}.level").visible
    assert driver.observe(object_name="form.fold.healthy").text == FOLDED + "Healthy"
    driver.click(object_name="form.fold.modules")
    driver.screenshot(tmp_path / "02-error-folded.png")
    driver.click("Refresh")
    driver.screenshot(tmp_path / "03-error-refresh.png")
    assert driver.observe(object_name="form.fold.modules").text == FOLDED + "Modules"
    assert not driver.observe(object_name="form.input.modules.level").visible
    assert driver.observe(object_name="form.input.renamed.level").visible


class EndpointsAxis(FormModel):
    kind: Literal["endpoints"] = "endpoints"
    start: float = 1.0


class CenterAxis(FormModel):
    kind: Literal["center"] = "center"
    center: float = 2.0


class SelectableForm(RebuildingForm):
    branch_models = {("window",): (EndpointsAxis, CenterAxis)}
    window: Annotated[
        EndpointsAxis | CenterAxis,
        Field(discriminator="kind"),
        Display(group="Sweep", label="Detuning"),
    ] = Field(default_factory=EndpointsAxis)


def test_merged_axis_keeps_selector_and_omits_missing_unit(
    host: Host, tmp_path: Path
) -> None:
    driver = host(SelectableForm.default_input(FormEnv()))
    driver.screenshot(tmp_path / "01-axis-selector.png")
    assert driver.observe(object_name="form.fold.window").text == (
        EXPANDED + "Sweep · Detuning"
    )
    assert driver.observe(object_name="form.input.window").text == "endpoints"
    driver.select_combo("center", object_name="form.input.window")
    driver.screenshot(tmp_path / "02-axis-selected.png")
    assert driver.observe(object_name="form.input.window").text == "center"
    assert driver.observe(object_name="form.input.window.center").text == "2.0"
    assert driver.observe(object_name="form.fold.window").text == (
        EXPANDED + "Sweep · Detuning"
    )
