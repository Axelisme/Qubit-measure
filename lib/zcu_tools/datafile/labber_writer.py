"""Labber export, separate from generic container serialization."""

import logging
from pathlib import Path
from typing import Literal

import numpy as np

from ._labber_layout import write_independent_labber_file
from .container_models import Container
from .container_nodes import validate_variable
from .grouped import write_grouped_labber_data_file
from .labber import write_labber_data_file
from .models import GroupedLabberData, LabberData, LabberMetadata, LabberPayload

logger = logging.getLogger(__name__)


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
    snake-case identity contract. Each scalar is exported as one sample on a
    unitless index axis at 0, with one warning naming the variable. The axis is
    named point, with trailing underscores added to avoid channel names. This
    does not change the source variable.
    Caller explicitly supplies comment/tags/project/user/creation_time through
    metadata; no JSON documents are interpreted. Use encode_labber_comment at
    the caller when a cfg comment envelope is wanted.

    common requires identical grids for multiple variables; one variable writes
    a single log. independent writes each scalar or rank-1/2 variable to its own
    root/Log_N log. Rank-three variables split on the NumPy first axis into
    two-dimensional logs with identities <variable>_<zero-based index>. Their
    log labels are "<destination stem> - <variable> <first axis name>=<coordinate>". Readers
    return the slices separately, without reassembling the original array.
    An empty first axis in rank three, slice identities colliding with input
    variable names, and ranks above three raise ValueError before file creation.
    Rank-1/2 identities and log labels stay unchanged. This uses Labber-only helpers, not a temporary native file.
    Existing paths raise FileExistsError. Bad arrays, grids, labels or unsupported
    layouts raise ValueError. I/O errors propagate and may leave a partial file.
    """
    if grouped_layout not in {"common", "independent"}:
        raise ValueError("grouped_layout must be common or independent")
    if not container.vars:
        raise ValueError("container vars must not be empty for Labber export")
    index_axis_name = "point"
    while index_axis_name in container.vars:
        index_axis_name += "_"
    payloads: dict[str, LabberPayload] = {}
    for name, variable in container.vars.items():
        validate_variable(variable)
        if not variable.axes:
            logger.warning(
                "%s: added length-1 index axis %r for Labber export",
                name,
                index_axis_name,
            )
            payloads[name] = LabberPayload(
                (name, variable.unit, variable.values.reshape(1)),
                [(index_axis_name, "", np.array([0]))],
            )
            continue
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
