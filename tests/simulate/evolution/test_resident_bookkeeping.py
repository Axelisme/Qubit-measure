"""Incremental retention work observed through Engine and injected products."""

from contextlib import closing
from dataclasses import replace
from threading import Event

import numpy as np
from zcu_tools.simulate.evolution import Engine
from zcu_tools.simulate.evolution.cache import RESIDENT_METADATA_BYTES

from ._engine import Payload, RecordingKernel, batch, source


class ChargeReadKernel(RecordingKernel):
    """Identity dynamics whose immutable product reports charge-read work."""

    def __init__(self):
        super().__init__("t4")
        self.charge_reads = 0

    def prepare(self, keys, table, rows, context):
        self.preparations += len(keys)
        owner = self
        size = 16 * len(keys)

        class Product(Payload):
            @property
            def nbytes(self) -> int:
                owner.charge_reads += 1
                return size

        return Product(len(keys))


def test_each_new_payload_charge_is_read_once_without_revisiting_history():
    kernel = ChargeReadKernel()
    src = source()
    # Include metadata for the intended resident working set.
    with closing(
        Engine(1, 100000 + 100 * RESIDENT_METADATA_BYTES, kernels=(kernel,))
    ) as engine:
        for duration in range(1, 17):
            engine.evolve(src, (batch(2, duration=float(duration)),), cancel=Event())
        reads = kernel.charge_reads
        builds = kernel.preparations
        engine.evolve(src, (batch(2, duration=17.0),), cancel=Event())
        assert kernel.preparations == builds + 1
        assert kernel.charge_reads - reads == 1
        # Point IDs bypass result memo, not prepare reuse.
        engine.evolve(
            src,
            (replace(batch(2, duration=1.0), points=np.array([20, 21])),),
            cancel=Event(),
        )
        assert kernel.preparations == builds + 1
        assert kernel.charge_reads - reads == 1
