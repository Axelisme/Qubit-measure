"""Reusable value-source token completion for QLineEdit inputs.

The controller treats value sources as an input convenience only:

- ``@{prefix`` opens a key-only completion popup.
- Choosing a completion inserts ``@{full.key}``; accepting an all-namespace
  candidate list drills into the next segment by appending ``.``.
- Typing a space after a complete token, ``@{full.key} ``, resolves once and
  replaces the token with the current native value formatted as text.

It does not write cfg/md state and does not know about ContextService or
LiveModel. Callers decide what a resolved value means by listening to the
``resolved`` signal or by reading the edited text.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from qtpy.QtCore import (  # type: ignore[attr-defined]
    QEvent,
    QObject,
    QStringListModel,
    Qt,
    Signal,  # type: ignore[reportPrivateImportUsage]
)
from qtpy.QtWidgets import QCompleter, QLineEdit  # type: ignore[attr-defined]


@dataclass(frozen=True)
class _TokenSpan:
    start: int
    end: int
    key: str


class ValueSourceInputHost(Protocol):
    """Narrow host interface for value-source input helpers."""

    def list_value_source_keys(self) -> Sequence[str]: ...

    def resolve_value_source(self, key: str) -> object: ...

    def format_resolved_value(self, value: object) -> str: ...


class SessionValueSourcePort(Protocol):
    """Narrow controller surface needed by SessionValueSourceInputHost."""

    def list_value_sources(self) -> Sequence[Any]: ...

    def read_value_source(
        self, key: str, type_name: str | None = None
    ) -> tuple[Any, object]: ...


class SessionValueSourceInputHost:
    """Adapter from SessionControllerPort to ValueSourceInputHost."""

    def __init__(self, ctrl: SessionValueSourcePort) -> None:
        self._ctrl = ctrl

    def list_value_source_keys(self) -> Sequence[str]:
        return tuple(info.key for info in self._ctrl.list_value_sources())

    def resolve_value_source(self, key: str) -> object:
        _, value = self._ctrl.read_value_source(key)
        return value

    def format_resolved_value(self, value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)


class ValueSourceInputController(QObject):
    """Attach value-source completion and resolve-on-space to a QLineEdit."""

    resolved = Signal(object)
    resolve_failed = Signal(str)

    def __init__(
        self,
        line_edit: QLineEdit,
        host: ValueSourceInputHost,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent or line_edit)
        self._line_edit = line_edit
        self._host = host
        self._model = QStringListModel(self)
        self._completer = QCompleter(self._model, self)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)  # type: ignore[attr-defined]
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)  # type: ignore[attr-defined]
        self._completer.setWidget(line_edit)
        self._updating = False
        self._popup_filter_targets: list[QObject] = []

        line_edit.textEdited.connect(self._on_text_edited)  # type: ignore[attr-defined]
        line_edit.installEventFilter(self)
        self._install_popup_event_filters()
        self._completer.activated[str].connect(self._on_completion_activated)  # type: ignore[index]

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if (
            (watched is self._line_edit or watched in self._popup_filter_targets)
            and event.type() == QEvent.Type.KeyPress
            and self._is_completion_accept_key(event)
            and self._complete_first_candidate()
        ):
            event.accept()
            return True
        return super().eventFilter(watched, event)

    def detach(self) -> None:
        """Disconnect from the line edit and hide any popup."""

        self._hide_completion_popup()
        self._line_edit.removeEventFilter(self)
        for target in self._popup_filter_targets:
            target.removeEventFilter(self)
        self._popup_filter_targets.clear()
        try:
            self._line_edit.textEdited.disconnect(self._on_text_edited)  # type: ignore[attr-defined]
        except (RuntimeError, TypeError):
            pass
        try:
            self._completer.activated[str].disconnect(self._on_completion_activated)  # type: ignore[index]
        except (RuntimeError, TypeError):
            pass

    def _on_text_edited(self, _text: str) -> None:
        if self._updating:
            return
        if self._resolve_committed_token():
            return
        self._refresh_completion_popup()

    def _refresh_completion_popup(self) -> None:
        text = self._line_edit.text()
        cursor = self._line_edit.cursorPosition()
        active = _active_token(text, cursor)
        if active is None:
            self._hide_completion_popup()
            return
        candidates = self._completion_candidates(active.key)
        if not candidates:
            self._hide_completion_popup()
            return
        self._model.setStringList(candidates)
        width = self._completion_popup_width(candidates)
        popup = self._completer.popup()
        if popup is not None:
            self._install_popup_event_filters()
            popup.setMinimumWidth(width)
        rect = self._line_edit.rect()
        rect.setWidth(width)
        self._completer.complete(rect)
        self._select_first_completion()

    def _completion_candidates(self, token_prefix: str) -> list[str]:
        keys = sorted(str(key) for key in self._host.list_value_source_keys())
        return _completion_candidates(keys, token_prefix)

    def _completion_popup_width(self, candidates: Sequence[str]) -> int:
        metrics = self._line_edit.fontMetrics()
        text_width = max(metrics.horizontalAdvance(key) for key in candidates)
        return max(self._line_edit.width(), text_width + 36)

    def _install_popup_event_filters(self) -> None:
        popup = self._completer.popup()
        if popup is None:
            return
        targets: list[QObject] = [popup]
        viewport = getattr(popup, "viewport", lambda: None)()
        if isinstance(viewport, QObject):
            targets.append(viewport)
        for target in targets:
            if target in self._popup_filter_targets:
                continue
            target.installEventFilter(self)
            self._popup_filter_targets.append(target)

    def _is_completion_accept_key(self, event: QEvent) -> bool:
        key = getattr(event, "key", lambda: None)()
        return key in {
            Qt.Key.Key_Tab,  # type: ignore[attr-defined]
            Qt.Key.Key_Backtab,  # type: ignore[attr-defined]
        }

    def _select_first_completion(self) -> None:
        if not self._completer.setCurrentRow(0):
            return
        popup = self._completer.popup()
        if popup is not None:
            popup.setCurrentIndex(self._completer.completionModel().index(0, 0))

    def _complete_first_candidate(self) -> bool:
        text = self._line_edit.text()
        cursor = self._line_edit.cursorPosition()
        active = _active_token(text, cursor)
        if active is None:
            return False
        candidates = self._completion_candidates(active.key)
        if not candidates:
            return False
        self._on_completion_activated(
            candidates[0],
            drill_down=_all_candidates_are_non_leaf(
                self._registered_keys(),
                candidates,
            ),
        )
        return True

    def _on_completion_activated(self, key: str, *, drill_down: bool = False) -> None:
        text = self._line_edit.text()
        cursor = self._line_edit.cursorPosition()
        active = _active_token(text, cursor)
        if active is None:
            return
        keys = self._registered_keys()
        should_drill_down = drill_down or (key not in keys and _has_children(keys, key))
        if key in keys:
            replacement = f"@{{{key}}}"
        elif should_drill_down:
            replacement = f"@{{{key}."
        else:
            replacement = f"@{{{key}"
        self._replace_text(active.start, active.end, replacement)
        self._refresh_completion_popup()

    def _registered_keys(self) -> set[str]:
        return set(str(item) for item in self._host.list_value_source_keys())

    def _resolve_committed_token(self) -> bool:
        text = self._line_edit.text()
        cursor = self._line_edit.cursorPosition()
        token = _committed_token(text, cursor)
        if token is None:
            return False
        try:
            value = self._host.resolve_value_source(token.key)
        except Exception as exc:
            message = str(exc)
            self._hide_completion_popup()
            self._line_edit.setToolTip(message)
            self.resolve_failed.emit(message)
            return True
        replacement = self._host.format_resolved_value(value)
        self._replace_text(token.start, token.end, replacement)
        self._hide_completion_popup()
        self._line_edit.setToolTip("")
        self.resolved.emit(value)
        return True

    def _hide_completion_popup(self) -> None:
        popup = self._completer.popup()
        if popup is not None:
            popup.hide()

    def _replace_text(self, start: int, end: int, replacement: str) -> None:
        text = self._line_edit.text()
        next_text = text[:start] + replacement + text[end:]
        next_cursor = start + len(replacement)
        self._updating = True
        try:
            self._line_edit.setText(next_text)
            self._line_edit.setCursorPosition(next_cursor)
        finally:
            self._updating = False


def _active_token(text: str, cursor: int) -> _TokenSpan | None:
    """Return the in-progress ``@{prefix`` token at cursor, if any."""

    if cursor < 2:
        return None
    start = text.rfind("@{", 0, cursor)
    if start < 0:
        return None
    prefix = text[start + 2 : cursor]
    if "}" in prefix or any(ch.isspace() for ch in prefix):
        return None
    return _TokenSpan(start=start, end=cursor, key=prefix)


def _committed_token(text: str, cursor: int) -> _TokenSpan | None:
    """Return the complete ``@{key} `` token ending at cursor, if any."""

    if cursor < 4 or text[cursor - 1] != " ":
        return None
    close = cursor - 2
    if close < 0 or text[close] != "}":
        return None
    start = text.rfind("@{", 0, close + 1)
    if start < 0:
        return None
    key = text[start + 2 : close]
    if not key or any(ch.isspace() for ch in key) or "}" in key:
        return None
    return _TokenSpan(start=start, end=cursor, key=key)


def _completion_candidates(keys: Sequence[str], token_prefix: str) -> list[str]:
    """Return immediate segment completions for the current value-source prefix."""

    normalized = token_prefix.strip()
    if "." in normalized:
        base, segment_prefix = normalized.rsplit(".", 1)
        base_prefix = f"{base}."
    else:
        base_prefix = ""
        segment_prefix = normalized

    candidates: set[str] = set()
    for key in keys:
        if base_prefix and not key.lower().startswith(base_prefix.lower()):
            continue
        remainder = key[len(base_prefix) :]
        if not remainder:
            continue
        segment = remainder.split(".", 1)[0]
        if not segment.lower().startswith(segment_prefix.lower()):
            continue
        candidates.add(f"{base_prefix}{segment}")

    if normalized and candidates == {normalized}:
        return []
    return sorted(candidates)


def _all_candidates_are_non_leaf(keys: set[str], candidates: Sequence[str]) -> bool:
    return bool(candidates) and all(candidate not in keys for candidate in candidates)


def _has_children(keys: set[str], candidate: str) -> bool:
    prefix = f"{candidate}."
    return any(key.startswith(prefix) for key in keys)
