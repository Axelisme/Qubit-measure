"""Call declaration admission with detached trees and preserve request paths."""

from collections.abc import Iterable

from .model import FormInput, FormInputRejected, FormModel, FormWritePath


def check_change(
    before: FormInput[FormModel],
    after: FormInput[FormModel],
    paths: Iterable[FormWritePath],
) -> None:
    """Check the complete candidate, mapping a directed refusal to its request."""
    try:
        before.model.check_input_change(before.data, after.data)
    except FormInputRejected as exc:
        requested = next(
            (
                path
                for path in paths
                if path[: len(exc.path)] == exc.path or exc.path[: len(path)] == path
            ),
            exc.path,
        )
        raise FormInputRejected(requested, str(exc)) from exc
