"""Qt rendering and local text drafts for the caller-owned form state."""

from PyQt6.QtCore import pyqtSignal
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QWidget

from zcu_tools.gui.form import FormEnv, FormErrors, FormModel, FormState


class FormEditor[F: FormModel](QWidget):
    """Edit a caller-owned FormState on the Qt GUI thread, without submitting it.

    The editor borrows state and the host-captured env. Parsed input writes the
    state immediately; unparseable text remains local and blocks editor validity.
    Evaluation, choices, branch defaults and model validation belong to state.
    No Engine update, revision check, source capture, persistence or I/O occurs.

    Public child object names use model-field paths joined by dots:
    form.input.<path> is a QLineEdit for text, numeric, complex, expression or
    whole-container input; a QCheckBox for bool; or a QComboBox for choices and
    discriminators. Combo item data is the ordinary input value.
    form.mode.<path> is a numeric/complex QComboBox with data "direct" and "expr".
    form.enabled.<path> is a nullable-model activation QCheckBox.
    form.value.<path> is a QLabel with the current evaluated value.
    form.error.<path> is a QLabel with current field errors; form.error displays
    model-wide errors. Mapping-key/index errors appear at their container field.
    Only current active fields have input controls. Sequences and typed mappings
    use JSON text for the whole leaf; they do not expose key/index write paths.
    Complex direct text uses Python complex-number notation, not an expression.

    Build finishes before subscribing. detach or close removes the subscription,
    disables this view and drops local text without changing caller state.
    QObject destruction also removes the subscription without touching state.
    refresh and error queries after detach raise RuntimeError.
    validity_changed emits a bool only when combined validity changes; it is not
    an Apply request. GUI-thread use and a live QApplication are required.
    """

    validity_changed = pyqtSignal(bool)

    def __init__(
        self, state: FormState[F], env: FormEnv, parent: QWidget | None = None
    ) -> None:
        """Build controls for state using env, then subscribe to state changes.

        State/source validation errors are displayed, not replaced by defaults.
        Structural/schema or source failures propagate; failed construction
        leaves no state subscription. parent controls the ordinary Qt lifetime.
        """
        super().__init__(parent)
        self._state = state
        self._env = env
        raise NotImplementedError

    def refresh(self, env: FormEnv) -> None:
        """Re-evaluate with env and refresh controls, choices, values and errors.

        Keep still-active local unparseable text. Never reset state, run domain
        defaults, poll sources or echo a write. External state publications
        refresh parsed controls through the subscription. Inactive branch text
        is discarded. Raises RuntimeError after detach.
        """
        raise NotImplementedError

    def set_editing_enabled(self, *, enabled: bool) -> None:
        """Enable or disable input controls without changing state or local text.

        External state changes remain visible while inputs are disabled.
        Raises RuntimeError after detach.
        """
        raise NotImplementedError

    def local_parse_errors(self) -> FormErrors | None:
        """Return detached local text errors, or None when every input parsed.

        Paths identify active controls. This query does not parse, write, or
        imply that the model is valid. Raises RuntimeError after detach.
        """
        raise NotImplementedError

    def errors(self) -> FormErrors | None:
        """Return merged local parse and current state errors, or None if valid.

        Local errors at a path take precedence over state errors at that path.
        Querying performs no state write. Raises RuntimeError after detach.
        """
        raise NotImplementedError

    def detach(self) -> None:
        """Idempotently unsubscribe, disable the view and discard local text.

        Do not reset, close or submit caller-owned state. Later publications do
        not update this view. The QWidget may still be destroyed normally.
        """
        raise NotImplementedError

    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Detach and accept Qt's close event a0, preserving caller state."""
        raise NotImplementedError
