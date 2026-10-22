"""Labber export, separate from generic container serialization."""

from pathlib import Path
from typing import Literal

import numpy as np

from ._labber_layout import write_independent_labber_file
from .container_models import Container
from .container_nodes import validate_variable
from .grouped import write_grouped_labber_data_file
from .labber import write_labber_data_file
from .models import GroupedLabberData, LabberData, LabberMetadata, LabberPayload


def write_labber(
    destination: Path,
    container: Container,
    *,
    metadata: LabberMetadata | None = None,
    grouped_layout: Literal["common", "independent"] = "common",
) -> None:
    """Export container.vars at exactly destination, leaving docs/groups untouched.

    NumPy-ordered axes convert to Labber's inner-first order. Variable keys become
    channel names and grouped identities; grouped names obey Labber's lowercase
    snake-case identity contract. At least one real step axis is required.
    Caller explicitly supplies comment/tags/project/user/creation_time through
    metadata; no JSON documents are interpreted. Use encode_labber_comment at
    the caller when a cfg comment envelope is wanted.

    common requires identical grids for multiple variables; one variable writes
    a single log. independent explicitly writes each rank-1/2 variable to its own
    root/Log_N layout using Labber-only helpers, not a temporary native file.
    Existing paths raise FileExistsError. Bad arrays, grids, labels or unsupported
    layouts raise ValueError. I/O errors propagate and may leave a partial file.
    """
    if grouped_layout not in {"common", "independent"}:
        raise ValueError("grouped_layout must be common or independent")
    if not container.vars:
        raise ValueError("container vars must not be empty for Labber export")
    payloads: dict[str, LabberPayload] = {}
    for name, variable in container.vars.items():
        validate_variable(variable)
        if not variable.axes:
            raise ValueError(f"{name}: Labber requires at least one step axis")
        axes = []
        for axis in reversed(variable.axes):
            if np.iscomplexobj(axis.values) and np.any(np.imag(axis.values) != 0):
                raise ValueError(f"{name}/{axis.name}: Labber axes must be real")
            axes.append((axis.name, axis.unit, np.real(axis.values)))
        payloads[name] = LabberPayload((name, variable.unit, variable.values), axes)
    metadata = LabberMetadata() if metadata is None else metadata
    if grouped_layout == "independent":
        write_independent_labber_file(destination, payloads, metadata)
    elif len(payloads) > 1:
        write_grouped_labber_data_file(
            str(destination), GroupedLabberData(payloads, metadata=metadata)
        )
    else:
        write_labber_data_file(
            str(destination),
            LabberData(payload=next(iter(payloads.values())), metadata=metadata),
        )
