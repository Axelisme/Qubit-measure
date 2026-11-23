"""Caller-scoped storage for exact sampled multitone RWA comparisons."""

from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True, slots=True)
class ProbeCertificate:
    """One sampled probe's immutable comparison result.

    difference is the maximum absolute density-entry difference of the RWA
    and full-drive endpoints. accepted is true when it is within the probe's
    configured fraction of the row budget.
    """

    difference: float
    accepted: bool


class SampledRWAProbeMemo:
    """Borrowed storage for up to 4096 exact sampled guard certificates.

    Create one per FTF preparation, lend it to its sequential Engine.evolve
    calls, and close it after they drain, including on cancel or failure
    (ADR-0078). The memo starts empty, evicts least recently used certificates,
    and retains no model, prepared payload, density or production result.
    Engine owns the keys and certificates. Lookups and inserts from its workers
    are synchronized; integration runs outside the memo lock.
    """

    def __init__(self) -> None:
        self._certificates: OrderedDict[bytes, ProbeCertificate] = OrderedDict()
        self._lock = Lock()
        self._closed = False

    def close(self) -> None:
        """Discard all certificates and permanently close the memo; idempotent."""
        with self._lock:
            self._certificates.clear()
            self._closed = True

    def get(self, key: bytes) -> ProbeCertificate | None:
        """Return the certificate for an Engine probe digest, or None on a miss.

        Refreshes recency on a hit. Raises RuntimeError after close.
        """
        with self._lock:
            self._check_open()
            certificate = self._certificates.get(key)
            if certificate is not None:
                self._certificates.move_to_end(key)
            return certificate

    def put(self, key: bytes, certificate: ProbeCertificate) -> None:
        """Store a completed certificate for its probe digest; raise after close."""
        with self._lock:
            self._check_open()
            self._certificates[key] = certificate
            self._certificates.move_to_end(key)
            if len(self._certificates) > 4096:
                self._certificates.popitem(last=False)

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("sampled RWA probe memo is closed")
