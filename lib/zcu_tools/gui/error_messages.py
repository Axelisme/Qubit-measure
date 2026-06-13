"""Shared skeleton for turning low-level exceptions into friendly UI messages.

Qt-free pure helpers shared by the analysis GUIs' ``error_messages`` modules. The
*structure* of a friendly message is common — a "what went wrong + how to fix it"
head, then the raw error on a ``Details:`` line (the full traceback is already in
the debug log) — while the domain rules (which substring maps to which head) differ
per app and stay there.

Each app's ``friendly_fit_message`` is a substring-rule ladder: it normalises the
error to ``raw`` (an exception or a worker-marshalled string), routes a params.json
write failure to ``friendly_io_message`` via ``fit_io_redirect``, then matches its
own ordered rules with ``friendly_from_rules``. ``friendly_io_message`` keeps its
per-app branch ladder (the conditions there depend on ``isinstance`` / file
existence, not just substrings) but shares the ``details_tail`` formatting.
"""

from __future__ import annotations

from collections.abc import Callable

# An ordered domain rule: (matches the lowercased raw error?) → the friendly head.
FriendlyRule = tuple[Callable[[str], bool], str]


def details_tail(raw: str, fallback: str) -> str:
    """The shared ``\\n\\nDetails: …`` suffix (the raw error, or a fallback label)."""
    return f"\n\nDetails: {raw if raw else fallback}"


def normalize_raw(exc: Exception | str) -> str:
    """The error text to match/show — works for an exception or a marshalled string.

    Worker failures cross a Qt signal as a plain string; locally-caught ones are the
    exception. Both reduce to the same stripped message here.
    """
    return (str(exc) if not isinstance(exc, str) else exc).strip()


def friendly_from_rules(
    action: str, raw: str, rules: list[FriendlyRule], fallback: str
) -> str:
    """Match ``raw`` (lowercased) against the ordered ``rules`` and format a message.

    The first rule whose predicate matches supplies the head; if none match, the
    head is ``"{action} failed."``. The ``Details:`` tail is always appended.
    """
    low = raw.lower()
    head = f"{action} failed."
    for predicate, candidate in rules:
        if predicate(low):
            head = candidate
            break
    return head + details_tail(raw, fallback)


def fit_io_redirect(
    exc: Exception | str,
    raw: str,
    io_message: Callable[[str, str, Exception], str],
) -> str | None:
    """Redirect a params.json *write* failure to the app's ``friendly_io_message``.

    A fit/export action that fails to open the params.json is really a file-IO
    problem, so detect that ("unable to open file") and delegate, wrapping a
    marshalled string back into an ``OSError`` for the IO formatter. Returns None
    when it is not an IO failure (the caller then runs its domain rules).
    """
    low = raw.lower()
    if "unable to open file" in low or ("unable to" in low and "file" in low):
        wrapped = exc if isinstance(exc, Exception) else OSError(raw)
        return io_message("Export", "params.json", wrapped)
    return None
