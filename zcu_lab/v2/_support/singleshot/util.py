from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class RawShotProgram(Protocol):
    @property
    def ro_chs(self) -> object: ...

    def get_raw(self) -> Sequence[NDArray[Any]] | None: ...


def raw_shots_to_signal(program: RawShotProgram) -> NDArray[np.complex128]:
    acc_buf = program.get_raw()
    if acc_buf is None:
        raise RuntimeError("program did not expose raw shot buffer")
    ro_chs = program.ro_chs
    if not isinstance(ro_chs, Mapping):
        raise TypeError("program readout channels must be a mapping")
    if len(ro_chs) != 1:
        raise ValueError(
            "singleshot raw conversion requires exactly one readout channel"
        )

    (ro_info,) = ro_chs.values()
    if not isinstance(ro_info, Mapping):
        raise TypeError("readout channel info must be a mapping")
    length = ro_info.get("length")
    if not isinstance(length, int | float):
        raise TypeError("readout channel length must be numeric")

    avgiq = acc_buf[0] / float(length)
    i0, q0 = avgiq[..., 0, 0], avgiq[..., 0, 1]
    return np.asarray(i0 + 1j * q0, dtype=np.complex128)
