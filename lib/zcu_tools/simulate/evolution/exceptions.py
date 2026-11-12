"""Public routing failures from the evolution engine."""


class NoValidKernel(RuntimeError):
    """No candidate accepted the first failing segment group.

    message is the legacy ``no valid kernel`` diagnostic. The read-only fields
    describe that group; later groups in the same acquire have not been checked.

    domain: structural row domain, such as ``idle``, ``plateau`` or ``shaped``,
        matching the diagnostic message.
    points: affected caller point IDs, matching the message, in sorted order.
    segments: affected segment rows, matching the message, in row order.
    attempted: family names offered in order by the final dispatch candidate
        round, none of which accepted these rows. With a ``families=`` allowlist,
        this includes all allowed candidates for these rows, including families
        skipped for cost or tried earlier. The order is the final round's offer
        order, not necessarily their first-attempt order. An empty tuple means
        the allowlist contains no candidate for these rows. With
        ``families=None``, only the final round's candidates are listed; families
        tried or skipped by earlier cost shortcuts are not included.

    A listed family failed a numerical guard or an input precondition, possibly
    via a retained guard rejection. Under an allowlist, an unlisted family is
    not a candidate for these rows. Preparation and apply failures propagate
    instead.
    """

    def __init__(
        self,
        message: str,
        *,
        domain: str,
        points: tuple[int, ...],
        segments: tuple[int, ...],
        attempted: tuple[str, ...],
    ) -> None:
        super().__init__(message)
        self._domain = domain
        self._points = points
        self._segments = segments
        self._attempted = attempted

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def points(self) -> tuple[int, ...]:
        return self._points

    @property
    def segments(self) -> tuple[int, ...]:
        return self._segments

    @property
    def attempted(self) -> tuple[str, ...]:
        return self._attempted
