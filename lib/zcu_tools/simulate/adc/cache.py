"""Bounded lifetime for physical ADC solvers and their exact response contents."""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from itertools import islice
from sys import getsizeof
from types import FunctionType, MemberDescriptorType, MethodType, ModuleType

import numpy as np

from ._constant import ConstantLine
from ._integrated import IntegratedLine
from .model import ReadoutModel, Receiver, readout_model_key
from .readout_batch import ConditionalSums, ReadoutBatch
from .request import PiecewiseLinearEnvelope, ReadoutPulse
from .response import ConditionalReadout


class ReadoutCache:
    """Own byte-bounded solvers and exact deterministic content, never noisy IQ.

    max_bytes is a nonnegative resident byte limit. Full numeric model/receiver
    content and rtol/atol select a solver; equivalent models share its immutable
    canonical model. Pulse/grid keys and absolute phase origins remain unchanged.
    Opaque callbacks remain referenced and identity-keyed, as in ConditionalReadout.
    A changed receiver naturally misses; restoring its original content can hit
    until eviction. The owner must clear on settings, DC snapshot or numeric
    revision changes, rather than mixing those physical generations.
    Oldest solvers are evicted whole after each response, including an oversized
    current solver. Active solve workspaces and returned arrays are not residents.
    Accounting includes the incremental ledger, Python containers/keys, traces,
    ConstantLine subcaches and entire backing allocations of retained views,
    deduplicated by identity.
    Callback instance attributes, closure cells and defaults are included, but
    module globals and external resources owned by opaque callbacks are not.
    Calls must be serialized; the cache is not thread-safe.

    builds/hits are cumulative physical solver misses/hits (not content hits),
    including across clear(). resident_bytes is the retained size after the last
    call; zero means no resident solver. Diagnostics do not expose numeric truth.
    """

    def __init__(self, max_bytes: int = 8 * 1024**2) -> None:
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("readout cache max_bytes must be a nonnegative integer")
        self.max_bytes = max_bytes
        self.builds = 0
        self.hits = 0
        self.resident_bytes = 0
        self._solvers: dict[tuple[object, ...], ConditionalReadout] = {}
        self._ledger = _ByteLedger()
        self._opaque_callbacks = False

    def clear(self) -> None:
        """Release every solver, content and callback, preserving diagnostic counts."""
        self._solvers.clear()
        self._ledger = _ByteLedger()
        self._opaque_callbacks = False
        self.resident_bytes = 0

    def responses(
        self, batch: ReadoutBatch, *, rtol: float = 1e-10, atol: float = 1e-12
    ) -> ConditionalSums:
        """Return the same noiseless sums/traces as ConditionalReadout.responses.

        batch and solver tolerances obey that method's contract. Equivalent model
        objects use the resident canonical model rather than rebuilding the line.
        Errors propagate without retaining partial/new cache contents. No sampling
        occurs. A successful oversized response is returned but not kept resident.
        """
        key = (readout_model_key(batch.model), rtol, atol)
        solver = self._solvers.get(key)
        if solver is None:
            solver = ConditionalReadout(batch.model, rtol=rtol, atol=atol)
            self.builds += 1
            self._solvers[key] = solver
        else:
            self.hits += 1
        try:
            result = solver.responses(replace(batch, model=solver.model))
        except Exception:
            # A failed solve may have populated only part of a line/content cache.
            self.clear()
            raise
        self._opaque_callbacks |= any(
            p.envelope is not None
            and not isinstance(p.envelope, PiecewiseLinearEnvelope)
            for pulses in batch.pulses
            for p in pulses
        )
        # Opaque callbacks may mutate captures. Recount them rather than assuming
        # the append-only ownership of our immutable numeric contents.
        if self._opaque_callbacks:
            self._ledger = _ByteLedger()
        self.resident_bytes = self._ledger.measure(self._solvers)
        while self._solvers and self.resident_bytes > self.max_bytes:
            del self._solvers[next(iter(self._solvers))]
            self._ledger = _ByteLedger()
            self.resident_bytes = (
                self._ledger.measure(self._solvers) if self._solvers else 0
            )
        return result


@dataclass(slots=True)
class _MappingSize:
    value: dict[object, object]
    scan_all: bool
    length: int
    storage_bytes: int


class _ByteLedger:
    """Incrementally charge append-only numeric caches, including ledger storage.

    Numeric response entries are immutable and line caches only append. Mutable
    solver/line attribute dictionaries are rescanned for lazy facts. Seen objects
    stay referenced until whole-solver eviction, preventing object-id reuse. The
    ledger is rebuilt on eviction, failure, clear or opaque callback requests.
    """

    def __init__(self) -> None:
        self._seen: dict[int, object] = {}
        self._mappings: list[_MappingSize] = []
        self._total = 0

    def measure(self, root: object) -> int:
        self._visit(root)
        for record in tuple(self._mappings):
            mapping = record.value
            self._total += getsizeof(mapping) - record.storage_bytes
            offset = 0 if record.scan_all else record.length
            for key, value in islice(mapping.items(), offset, None):
                self._visit(key)
                self._visit(value)
            record.length = len(mapping)
            record.storage_bytes = getsizeof(mapping)
        # id keys and accounting records are resident too. Charge scalar fields
        # conservatively even when Python happens to share their small integers.
        return (
            self._total
            + getsizeof(self)
            + getsizeof(vars(self))
            + getsizeof(self._total)
            + sum(getsizeof(k) for k in vars(self))
            + getsizeof(self._seen)
            + len(self._seen) * getsizeof(id(self))
            + getsizeof(self._mappings)
            + sum(getsizeof(r) + 3 * getsizeof(0) for r in self._mappings)
        )

    def _visit(self, value: object, *, attributes: bool = False) -> None:
        if id(value) in self._seen:
            return
        self._seen[id(value)] = value
        self._total += getsizeof(value)
        if isinstance(value, dict):
            self._mappings.append(
                _MappingSize(value, attributes, len(value), getsizeof(value))
            )
            for key, child in value.items():
                self._visit(key)
                self._visit(child)
        elif isinstance(value, (ConditionalReadout, ConstantLine)):
            self._visit(vars(value), attributes=True)
        elif isinstance(
            value,
            (
                IntegratedLine,
                ReadoutModel,
                Receiver,
                ReadoutPulse,
                PiecewiseLinearEnvelope,
            ),
        ):
            # These records do not change after construction. Charge their
            # attribute dictionary once without scheduling repeated scans.
            data = vars(value)
            self._seen[id(data)] = data
            self._total += getsizeof(data)
            for key, child in data.items():
                self._visit(key)
                self._visit(child)
        else:
            for child in _owned_references(value):
                self._visit(child)


def _owned_references(value: object) -> Iterator[object]:
    """Follow only ADC-owned data and callback captures, never module globals."""
    if isinstance(value, np.ndarray):
        # Owning arrays' getsizeof includes data; views keep their base alive.
        if value.base is not None:
            yield value.base
    elif isinstance(value, memoryview):
        yield value.obj
    elif isinstance(value, dict):
        yield from value.keys()
        yield from value.values()
    elif isinstance(value, (tuple, list)):
        yield from value
    elif isinstance(value, FunctionType):
        yield from _function_captures(value)
    elif isinstance(value, MethodType):
        yield value.__self__
        yield value.__func__
    elif not isinstance(value, (type, ModuleType)):
        if hasattr(value, "__dict__"):
            yield vars(value)
        if hasattr(type(value), "__slots__"):
            yield from _slot_captures(value)


def _slot_captures(value: object) -> Iterator[object]:
    """Include Python slot-held callback captures without following class globals."""
    for cls in type(value).__mro__:
        for descriptor in vars(cls).values():
            if isinstance(descriptor, MemberDescriptorType):
                try:
                    yield descriptor.__get__(value, type(value))
                except AttributeError:
                    # A callback need not initialize every declared slot.
                    continue


def _function_captures(function: FunctionType) -> Iterator[object]:
    yield function.__dict__
    yield function.__defaults__
    yield function.__kwdefaults__
    if function.__closure__:
        yield function.__closure__
        for cell in function.__closure__:
            # A valid callback can close over a name that has since been deleted.
            try:
                yield cell.cell_contents
            except ValueError:
                continue
