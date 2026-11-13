"""Input-only direct/channel pricing and complete constant-tone key inventory.

Dispatch owns strategy selection; kernels retain numerical preparation and apply
(ADR-0078). Key construction is shared with the t6 Kernel protocol.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .kernel import SegmentTable
from .secular import ModelSource
from .workplan import FactKey, WorkPlan


@dataclass(frozen=True)
class T6Price:
    """One complete-acquire prepare key's input-only strategy decision.

    key is the opaque inventory_key bytes for a prepare_keys scalar. rows counts
    all constant-tone rows sharing that key. size is the model's energy-level
    dimension. direct selects density integration when True, a reusable channel
    otherwise. No numerical payload or cache/history state enters this record.
    """

    key: bytes
    rows: int
    size: int
    direct: bool


@dataclass(frozen=True)
class T6Inventory:
    """Complete constant-tone strategy publication for one WorkPlan.

    prices contains one T6Price per distinct prepare key, in np.unique order.
    An empty tuple means the complete geometry has no constant-tone rows.
    """

    prices: tuple[T6Price, ...]


def publish_inventory(
    table: SegmentTable, source: ModelSource, work_plan: WorkPlan
) -> T6Inventory:
    """Publish input-only strategy choices from the complete acquire table.

    The first prepare calls this lazily, using work_plan.complete_table when
    present; otherwise table must be the standalone caller's complete geometry.
    source supplies models for that geometry. Each constant-tone prepare key
    counts all its rows, independently of histories, numerical admission and
    density. Explicit early publication is also allowed. Repeated calls borrow
    the first publication unchanged, sharing builder failures without retries.
    No numerical products are retained here; missing published keys fail at prepare.
    A closed plan raises RuntimeError; model construction failures propagate.
    """

    def build() -> T6Inventory:
        complete = work_plan.complete_table
        full_table = table if complete is None else complete
        rows = np.flatnonzero(
            (full_table.tones == 1)
            & full_table.constant_envelopes(
                np.arange(len(full_table.timeline), dtype=np.int64)
            )
        )
        keys, first, counts = np.unique(
            prepare_keys(full_table, rows), return_index=True, return_counts=True
        )
        prices = []
        for key, row, count in zip(keys, rows[first], counts, strict=True):
            size = len(full_table.model(source, row).energies_ghz)
            duration = float(full_table.duration_ns[row])
            frequency = abs(float(full_table.tone_frequency_ghz[row, 0]))
            # Budget-scaled direct prices use warm 21-level I/L aggregates.
            # Channel production aggregates give about 31 s per build.
            # Dimension powers and the carrier factor remain uncalibrated
            # outside those inputs. Prices do not model prefix lifetimes.
            direct = 0.0012 * duration * max(1.0, frequency / 0.582) * (size / 21) ** 3
            channel = 31.0 * (size / 21) ** 5
            prices.append(
                T6Price(inventory_key(key), int(count), size, count * direct < channel)
            )
        return T6Inventory(tuple(prices))

    return work_plan.fact(FactKey("t6-inventory", T6Inventory), build)


def inventory_key(key: np.void) -> bytes:
    """Return opaque lookup bytes for a scalar returned by prepare_keys.

    Normalize signed zero in floating fields on a copy, matching np.unique's
    numeric equality. Leave the caller's scalar and public prepare key unchanged.
    Use this for both publication and lookup; do not index by raw scalar bytes.
    """
    normalized = key.copy()
    for field in ("frequency", "amplitude", "duration"):
        if normalized[field] == 0:
            normalized[field] = 0.0
    return normalized.tobytes()


def prepare_keys(table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.void]:
    """Return structured t6 prepare keys aligned with constant-tone table rows.

    rows indexes valid rows with one constant-envelope tone. Keys contain flux
    level, shift content, port, signed carrier GHz, signed gain and duration ns;
    phase and start time are apply-only. The Engine supplies source namespacing.
    The returned array owns its storage and preserves signed-zero bits. Its
    layout is opaque to consumers; use inventory_key for inventory lookup.
    """
    keys = np.empty(
        len(rows),
        dtype=[
            ("level", "<i8", (table.level.shape[1],)),
            ("shift", "S64"),
            ("port", "<i8"),
            ("frequency", "<f8"),
            ("amplitude", "<f8"),
            ("duration", "<f8"),
        ],
    )
    if not len(rows):
        return keys
    keys["level"] = table.level[rows]
    keys["shift"] = table.shift_content[rows]
    keys["port"] = table.tone_port[rows, 0]
    keys["frequency"] = table.tone_frequency_ghz[rows, 0]
    keys["amplitude"] = table.tone_amplitude[rows, 0]
    keys["duration"] = table.duration_ns[rows]
    return keys
