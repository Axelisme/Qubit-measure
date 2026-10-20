"""Expression completion through the public editor and real component input.

The synthetic host uses refresh(env) and Reset replacement, not the measure app.
Screenshots are kept under pytest's caller-selected --basetemp directory.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget
from zcu_tools.gui.form import Display, FormEnv, FormInput, FormModel, FormState
from zcu_tools.gui.session.aliases import AliasRow, ResolvedAliasRow
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext, RoleBinding
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import ComponentSchema
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral

from tests.gui._user_input import ComponentDriver
from tests.gui.widgets.form._clock import Clock

INPUT = "form.input.amount"


class Qubit(ComponentSchema):
    """Synthetic captured component with a numeric field and nested data."""

    freq: float = 4.0
    wiring: dict[str, int] = {"channel": 2}


class CountingContext(ParameterContext):
    """Record public candidate reads while preserving real capture behaviour."""

    alias_reads: int = 0
    document_reads: int = 0

    def resolved_aliases(self) -> tuple[ResolvedAliasRow, ...]:
        self.alias_reads += 1
        return super().resolved_aliases()

    def document(self) -> PointDocument:
        self.document_reads += 1
        return super().document()


def capture(alias: str = "qubit") -> CountingContext:
    """Build an isolated fixed point with two components and one device alias."""
    return CountingContext(
        point=PointDocument.model_construct(
            format="zcu.parameter-container",
            format_version="2.0",
            general=PointGeneral(created_at="2026-10-20T00:00:00Z"),
            components={"Q1": Qubit(kind="completion/test")},
        ),
        library=LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
        ),
        target=SavedPoint(entry_id="entry", point="working"),
        roles=RoleBinding(
            None,
            {},
            {},
            aliases=(
                AliasRow(alias, "Q1"),
                AliasRow(f"{alias}1", "Q1"),
                AliasRow("flux_dev", "bias"),
            ),
        ),
    )


class CompletionForm(FormModel):
    """Editable expression, literal text and a declared read-only expression."""

    amount: float = 1.0
    direct: str = "$qu"
    locked: Annotated[float, Display(readonly=True)] = 1.0


class Host(QWidget):
    """Minimal visible host using editor Refresh and state/editor Reset replacement."""

    def __init__(self, parameters: ParameterContext) -> None:
        super().__init__()
        self.next_capture = parameters
        self.env = FormEnv(parameters=parameters)
        self.layout_rows = QVBoxLayout(self)
        self.state = FormState(
            FormInput(
                CompletionForm,
                {"amount": {"expr": "1"}, "direct": "$qu", "locked": {"expr": "$qu"}},
            )
        )
        self.editor = FormEditor(self.state, self.env)
        self.layout_rows.addWidget(self.editor)
        for caption, callback in (
            ("Refresh", self.refresh),
            ("Reset", self.reset),
            ("Read only", self.read_only),
        ):
            button = QPushButton(caption, self)
            button.clicked.connect(callback)
            self.layout_rows.addWidget(button)
        self.resize(840, 220)

    def refresh(self) -> None:
        """Install the host's next fixed capture while retaining authored input."""
        self.env = FormEnv(parameters=self.next_capture)
        self.editor.refresh(self.env)

    def reset(self) -> None:
        """Replace editor/state with defaults from the next fixed capture."""
        self.editor.detach()
        self.layout_rows.removeWidget(self.editor)
        self.editor.setParent(None)
        self.editor.deleteLater()
        self.env = FormEnv(parameters=self.next_capture)
        self.state = FormState(CompletionForm.default_input(self.env))
        self.editor = FormEditor(self.state, self.env)
        self.layout_rows.insertWidget(0, self.editor)

    def read_only(self) -> None:
        """Keep controls visible but prohibit writes."""
        self.editor.set_editing_enabled(enabled=False)


@pytest.fixture
def host(qapp: QApplication, request: pytest.FixtureRequest) -> Iterator[Host]:
    window = Host(capture(request.param if hasattr(request, "param") else "qubit"))
    window.show()
    window.activateWindow()
    qapp.processEvents()
    # Public preconditions: all controls exist, are visible, and Expression is on.
    driver = ComponentDriver(window)
    assert driver.observe(object_name=INPUT).fully_visible
    assert driver.text_input(object_name=INPUT).text == "1"
    assert not driver.text_input(object_name=INPUT).read_only
    assert driver.text_input(object_name="form.input.locked").read_only
    yield window
    window.close()
    window.editor.detach()
    window.deleteLater()
    qapp.processEvents()


def shot(driver: ComponentDriver, folder: Path, name: str) -> None:
    """Save each visible step to the caller's evidence folder."""
    driver.screenshot(folder / f"{name}.png")


def key(driver: ComponentDriver, value: Qt.Key, name: str = INPUT) -> None:
    """Send one actual key to a public named control."""
    driver.press_key(value, object_name=name)


def test_expression_completion_preserves_surrounding_text_and_focus(
    host: Host, tmp_path: Path
) -> None:
    driver = ComponentDriver(host)
    driver.enter_text(value="0.8 * $qubit.fre + 5", object_name=INPUT, commit=False)
    # Move over the existing right-hand expression without changing it.
    for _ in range(4):
        key(driver, Qt.Key.Key_Left)
    shot(driver, tmp_path, "01-middle-ghost")
    assert (
        driver.text_input(object_name=INPUT).description
        == "Completion: q. Tab accepts; repeated Tab cycles; Esc dismisses."
    )
    key(driver, Qt.Key.Key_Tab)
    shot(driver, tmp_path, "02-middle-accepted")
    assert driver.text_input(object_name=INPUT).text == "0.8 * $qubit.freq + 5"
    assert driver.text_input(object_name=INPUT).focused
    assert driver.observe(object_name="form.value.amount").text == "= 1"
    key(driver, Qt.Key.Key_Return)
    shot(driver, tmp_path, "03-committed")
    assert driver.observe(object_name="form.value.amount").text == "= 8.2"


@pytest.mark.parametrize(
    "host,prefix,completed,suffix,tail",
    [
        pytest.param("量子位", "$量", "$量子位", "子位", "", id="cjk-alias"),
        pytest.param("量子位", "$量子位.fre", "$量子位.freq", "q", "", id="cjk-field"),
        pytest.param(
            "q\u0301ubit", "$q\u0301", "$q\u0301ubit", "ubit", "", id="combining-alias"
        ),
        pytest.param(
            "q\u0301ubit",
            "$q\u0301ubit.fre",
            "$q\u0301ubit.freq",
            "q",
            "",
            id="combining-field",
        ),
        pytest.param("𐐀ubit", "$𐐀", "$𐐀ubit", "ubit", " + 5", id="nonbmp-alias"),
        pytest.param(
            "𐐀ubit", "$𐐀ubit.fre", "$𐐀ubit.freq", "q", " + 5", id="nonbmp-field"
        ),
    ],
    indirect=["host"],
)
def test_unicode_alias_and_field_completion(
    host: Host, tmp_path: Path, prefix: str, completed: str, suffix: str, tail: str
) -> None:
    driver = ComponentDriver(host)
    driver.enter_text(value=prefix + tail, object_name=INPUT, commit=False)
    for _ in tail:
        key(driver, Qt.Key.Key_Left)
    shot(driver, tmp_path, "01-unicode-ghost")
    observed = driver.text_input(object_name=INPUT)
    assert observed.text == prefix + tail
    assert observed.description == (
        f"Completion: {suffix}. Tab accepts; repeated Tab cycles; Esc dismisses."
    )
    key(driver, Qt.Key.Key_Tab)
    shot(driver, tmp_path, "02-unicode-accepted")
    assert driver.text_input(object_name=INPUT).text == completed + tail
    assert driver.text_input(object_name=INPUT).focused


def test_tab_cycle_survives_idle_timeout_without_submitting(
    clock: Clock, host: Host, tmp_path: Path
) -> None:
    driver = ComponentDriver(host)
    driver.enter_text(value="$qu", object_name=INPUT, commit=False)
    shot(driver, tmp_path, "01-pending-idle")
    key(driver, Qt.Key.Key_Tab)
    shot(driver, tmp_path, "02-tab-accepted")
    assert driver.text_input(object_name=INPUT).text == "$qubit"
    assert driver.observe(object_name="form.value.amount").text == "= 1"
    clock.advance(500)
    shot(driver, tmp_path, "03-after-idle")
    key(driver, Qt.Key.Key_Tab)
    shot(driver, tmp_path, "04-cycle-after-idle")
    assert (
        driver.text_input(object_name=INPUT).text,
        driver.observe(object_name="form.value.amount").text,
    ) == ("$qubit1", "= 1")
    assert driver.text_input(object_name=INPUT).focused
    key(driver, Qt.Key.Key_Tab)
    shot(driver, tmp_path, "05-wrapped")
    assert driver.text_input(object_name=INPUT).text == "$qubit"
    # A new ordinary edit must restore the regular idle-commit policy.
    driver.enter_text(value="$qubit.fre", object_name=INPUT, commit=False)
    key(driver, Qt.Key.Key_Tab)
    clock.advance(500)
    shot(driver, tmp_path, "06-field-not-submitted")
    assert driver.observe(object_name="form.value.amount").text == "= 1"
    driver.enter_text(value="$qubit.freq + 1", object_name=INPUT, commit=False)
    clock.advance(500)
    shot(driver, tmp_path, "07-new-edit-idle-committed")
    assert driver.observe(object_name="form.value.amount").text == "= 5.0"


def test_hint_requires_whitespace_or_end_after_cursor(
    host: Host, tmp_path: Path
) -> None:
    driver = ComponentDriver(host)
    for step, (text, moves, expected) in enumerate(
        (
            ("$qubit.fre", 0, "Completion: q."),
            ("0.8 * $qubit.fre + 5", 4, "Completion: q."),
            ("($qubit.fre)", 1, ""),
            ("$qubit.fre+5", 2, ""),
        )
    ):
        driver.enter_text(value=text, object_name=INPUT, commit=False)
        for _ in range(moves):
            key(driver, Qt.Key.Key_Left)
        shot(driver, tmp_path, f"{step:02}-next-character")
        observed = driver.text_input(object_name=INPUT)
        assert observed.focused
        assert observed.text == text
        if expected:
            assert expected in observed.description
        else:
            assert observed.description == ""


def test_token_middle_and_selection_hide_the_ghost(host: Host, tmp_path: Path) -> None:
    driver = ComponentDriver(host)
    driver.enter_text(value="$qubit.fre", object_name=INPUT, commit=False)
    shot(driver, tmp_path, "01-token-end")
    assert "Completion: q." in driver.text_input(object_name=INPUT).description
    key(driver, Qt.Key.Key_Left)
    shot(driver, tmp_path, "02-token-middle")
    assert driver.text_input(object_name=INPUT).description == ""
    key(driver, Qt.Key.Key_Right)
    shot(driver, tmp_path, "03-token-end-again")
    assert "Completion: q." in driver.text_input(object_name=INPUT).description
    driver.press_key(
        Qt.Key.Key_A, object_name=INPUT, modifiers=Qt.KeyboardModifier.ControlModifier
    )
    shot(driver, tmp_path, "04-selected")
    assert driver.text_input(object_name=INPUT).description == ""


def test_alias_tab_cycles_and_escape_dismisses_until_new_input(
    host: Host, tmp_path: Path
) -> None:
    driver = ComponentDriver(host)
    driver.enter_text(value="$qu", object_name=INPUT, commit=False)
    shot(driver, tmp_path, "01-alias-ghost")
    key(driver, Qt.Key.Key_Tab)
    shot(driver, tmp_path, "02-alias-accepted")
    assert driver.text_input(object_name=INPUT).text == "$qubit"
    assert driver.text_input(object_name=INPUT).focused
    key(driver, Qt.Key.Key_Tab)
    shot(driver, tmp_path, "03-alias-cycled")
    assert driver.text_input(object_name=INPUT).text == "$qubit1"
    key(driver, Qt.Key.Key_Tab)
    shot(driver, tmp_path, "04-alias-wrapped")
    assert driver.text_input(object_name=INPUT).text == "$qubit"
    driver.enter_text(value="$qu", object_name=INPUT, commit=False)
    key(driver, Qt.Key.Key_Escape)
    shot(driver, tmp_path, "05-dismissed")
    assert driver.text_input(object_name=INPUT).description == ""
    key(driver, Qt.Key.Key_Tab)
    shot(driver, tmp_path, "06-normal-tab")
    assert driver.text_input(object_name=INPUT).text == "$qu"
    assert not driver.text_input(object_name=INPUT).focused
    driver.press_key(Qt.Key.Key_Tab, modifiers=Qt.KeyboardModifier.ShiftModifier)
    shot(driver, tmp_path, "06-back-to-expression")
    assert driver.text_input(object_name=INPUT).focused
    driver.enter_text(value="$qubit.fre", object_name=INPUT, commit=False)
    shot(driver, tmp_path, "07-new-input")
    assert "Completion: q." in driver.text_input(object_name=INPUT).description


def test_only_editable_expressions_offer_one_level_and_device_names(
    host: Host, tmp_path: Path
) -> None:
    driver = ComponentDriver(host)
    driver.enter_text(value="$flux_d", object_name=INPUT, commit=False)
    shot(driver, tmp_path, "01-device-name")
    key(driver, Qt.Key.Key_Tab)
    shot(driver, tmp_path, "02-device-accepted")
    assert driver.text_input(object_name=INPUT).text == "$flux_dev"
    for index, text in enumerate(
        ("$flux_dev.", "$qubit.wiring.c", "plain", "qubit.fre")
    ):
        driver.enter_text(value=text, object_name=INPUT, commit=False)
        shot(driver, tmp_path, f"03-no-candidate-{index}")
        assert driver.text_input(object_name=INPUT).description == ""
    driver.press_key(Qt.Key.Key_Tab)
    assert driver.text_input(object_name="form.input.direct").focused
    driver.enter_text(value="$qu", object_name="form.input.direct", commit=False)
    shot(driver, tmp_path, "04-direct")
    assert driver.text_input(object_name="form.input.direct").description == ""
    driver.click(object_name="form.input.locked")
    shot(driver, tmp_path, "05-declared-readonly")
    assert driver.text_input(object_name="form.input.locked").description == ""
    driver.press_key(Qt.Key.Key_Tab, modifiers=Qt.KeyboardModifier.ShiftModifier)
    assert driver.text_input(object_name=INPUT).focused
    driver.enter_text(value="$qu", object_name=INPUT, commit=False)
    shot(driver, tmp_path, "06-editable")
    assert "Completion: bit." in driver.text_input(object_name=INPUT).description
    driver.click("Read only")
    driver.click(object_name=INPUT)
    shot(driver, tmp_path, "07-host-readonly")
    assert driver.text_input(object_name=INPUT).read_only
    assert driver.text_input(object_name=INPUT).description == ""


def test_candidates_are_captured_once_and_refresh_and_reset_replace_them(
    host: Host, tmp_path: Path
) -> None:
    driver = ComponentDriver(host)
    original = host.env.parameters
    assert isinstance(original, CountingContext)
    assert (original.alias_reads, original.document_reads) == (1, 1)
    driver.enter_text(value="$qubit.fre", object_name=INPUT, commit=False)
    key(driver, Qt.Key.Key_Tab)
    shot(driver, tmp_path, "01-original")
    assert driver.text_input(object_name=INPUT).text == "$qubit.freq"
    assert (original.alias_reads, original.document_reads) == (1, 1)
    replacement = capture("resonator")
    host.next_capture = replacement
    driver.enter_text(value="$res", object_name=INPUT, commit=False)
    shot(driver, tmp_path, "02-before-refresh")
    assert driver.text_input(object_name=INPUT).description == ""
    driver.click("Refresh")
    driver.enter_text(value="$res", object_name=INPUT, commit=False)
    shot(driver, tmp_path, "03-refreshed")
    key(driver, Qt.Key.Key_Tab)
    shot(driver, tmp_path, "04-refreshed-accepted")
    assert driver.text_input(object_name=INPUT).text == "$resonator"
    assert (replacement.alias_reads, replacement.document_reads) == (1, 1)
    host.next_capture = capture("drive")
    driver.click("Reset")
    shot(driver, tmp_path, "05-reset-defaults")
    assert driver.text_input(object_name=INPUT).text == "1.0"
    for step in range(8):
        if driver.text_input(object_name=INPUT).focused:
            break
        driver.press_key(Qt.Key.Key_Tab)
        shot(driver, tmp_path, f"05-reset-tab-{step}")
    assert driver.text_input(object_name=INPUT).focused
    driver.invoke_context_action(
        object_name=INPUT,
        action="Expression",
        evidence=tmp_path / "06-reset-mode-menu.png",
    )
    driver.enter_text(value="$dri", object_name=INPUT, commit=False)
    shot(driver, tmp_path, "07-reset-ghost")
    key(driver, Qt.Key.Key_Tab)
    shot(driver, tmp_path, "08-reset-accepted")
    assert driver.text_input(object_name=INPUT).text == "$drive"
    assert (original.alias_reads, original.document_reads) == (1, 1)
    assert (replacement.alias_reads, replacement.document_reads) == (1, 1)
