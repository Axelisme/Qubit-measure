"""Bounded immutable routing facts for caller-owned exact-content preparations.

No geometry, WorkPlan resources, numerical payloads or source ownership survives.
Dispatch owns the shared price/choice operations; this cache only replays inventory
and compares their current selections before the caller reuses its populations.
"""

from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass
from sys import getsizeof
from weakref import ReferenceType, ref

from ._multitone_dispatch import (
    MultitoneCost,
    multitone_cost,
    populate_multitone_inventory,
    populate_multitone_plan,
)
from ._shaped_dispatch import (
    GateFamilyCost,
    populate_single_gate_plan,
    single_gate_costs,
)
from .kernel import KernelContext, SegmentTable
from .secular import ModelSource


@dataclass(frozen=True)
class RoutingSnapshot:
    source: ReferenceType[ModelSource]
    budget: float
    content: bytes
    inventory: tuple[tuple[bytes, int], ...]
    priced_usage: tuple[tuple[bytes, int], ...]
    single: tuple[GateFamilyCost, ...]
    multiple: tuple[tuple[int, bytes, MultitoneCost], ...]
    selected: tuple[bytes, ...]
    direct: frozenset[int]

    @classmethod
    def build(cls, table: SegmentTable, context: KernelContext) -> "RoutingSnapshot":
        source, plan, budget = context.source, context.work_plan, context.budget
        inventory, single = single_gate_costs(table, source, plan, budget)
        multiple = populate_multitone_inventory(table, source, plan, budget)
        counts = dict(inventory)
        costs = []
        for (row, _uses), key in zip(multiple.uses, multiple.keys, strict=True):
            counts[key] = counts.get(key, 0) + 1
            costs.append((row, key, multitone_cost(table, row, context)))
        choices = populate_single_gate_plan(table, source, plan, budget)
        selected = tuple(sorted({key for choice in choices for key in choice.keys}))
        direct = populate_multitone_plan(table, source, plan, budget)
        return cls(
            ref(source),
            budget,
            plan.content_key,
            tuple(counts.items()),
            tuple(
                source.projected_gate_usage(counts, content=plan.content_key).items()
            ),
            single,
            tuple(costs),
            selected,
            direct,
        )

    def reuse(self, source: ModelSource, budget: float) -> bool:
        if self.source() is not source or budget != self.budget:
            return False
        if not self.inventory:
            return True
        inventory = dict(self.inventory)
        usage = source.projected_gate_usage(inventory, content=self.content)
        # The pricing function is pure in these relevant totals and immutable
        # facts. Identical inputs reuse its answer; changed totals still reselect
        # actual keys/rows, rather than treating counter growth as a route change.
        if tuple(usage.items()) == self.priced_usage:
            source.record_gate_usage(inventory, content=self.content)
            return True
        selected = tuple(
            sorted({key for family in self.single for key in family.choose(usage).keys})
        )
        direct = frozenset(
            row for row, key, cost in self.multiple if cost.direct_cheaper(usage[key])
        )
        if selected != self.selected or direct != self.direct:
            # A fresh evolve will record this acquire once, using the new route.
            return False
        source.record_gate_usage(inventory, content=self.content)
        return True


class RoutingReuse:
    """Engine-local LRU; oversize snapshots stream without evicting residents."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.entries: OrderedDict[bytes, tuple[RoutingSnapshot, int]] = OrderedDict()
        self.resident_bytes = 0

    def remember_independent(
        self, key: bytes | None, source: ModelSource, budget: float, content: bytes
    ) -> None:
        # A raw-input result hit has no cumulative candidates. No geometry or
        # policy facts are needed to validate the caller's independent result.
        self.add(
            key,
            RoutingSnapshot(
                ref(source), budget, content, (), (), (), (), (), frozenset()
            ),
        )

    def snapshot(
        self, key: bytes | None, table: SegmentTable, context: KernelContext
    ) -> RoutingSnapshot | None:
        if key is None or not self.limit:
            return None
        return RoutingSnapshot.build(table, context)

    def add(self, key: bytes | None, snapshot: RoutingSnapshot | None) -> None:
        if key is None or snapshot is None:
            return
        size = _charge((key, snapshot)) + 1024
        if size > self.limit:
            return
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.resident_bytes -= previous[1]
        while self.entries and self.resident_bytes + size > self.limit:
            _, (_, charge) = self.entries.popitem(last=False)
            self.resident_bytes -= charge
        self.entries[key] = snapshot, size
        self.resident_bytes += size

    def reuse(self, key: bytes, source: ModelSource, budget: float) -> bool:
        entry = self.entries.get(key)
        if entry is None:
            return False
        self.entries.move_to_end(key)
        return entry[0].reuse(source, budget)

    def clear(self) -> None:
        self.entries.clear()
        self.resident_bytes = 0


def _charge(value: object) -> int:
    # Facts contain only frozen records, tuples and scalars. Charge shared fields
    # conservatively per occurrence; weakrefs never retain a model source.
    size = getsizeof(value)
    if isinstance(value, (tuple, frozenset)):
        size += sum(_charge(child) for child in value)
    elif is_dataclass(value) and not isinstance(value, type):
        size += getsizeof(vars(value))
        size += sum(_charge(getattr(value, item.name)) for item in fields(value))
    return size
