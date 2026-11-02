"""Complete independent-grid Labber layout; not a live acquisition writer."""

import time
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np

from .grouped import DATA_VARIABLE_ATTR, DATA_VARIABLES_ATTR, GROUPED_VERSION_ATTR
from .labber import write_payload_to_log
from .labber_format import labber_string_array
from .models import DataVariable, LabberMetadata, LabberPayload


def write_independent_labber_file(
    destination: Path, payloads: dict[str, LabberPayload], metadata: LabberMetadata
) -> None:
    """Write independent logs at exact destination, refusing existing files.

    Each payload is inner-first, with its own grid. Rank-three arrays split on
    the NumPy first axis into rank-two logs named <variable>_<index>. Log labels
    include that axis's name and coordinate. Rank-one/two logs keep their names.
    The first log occupies root; others occupy Log_2, Log_3, etc.
    Retain the marker-qualified grouped-v1 wire
    layout so the grouped reader can identify each variable. Mark finalized only
    after every log succeeds. Invalid rank/name, an empty first axis in rank
    three, or a slice identity colliding with an input variable raises ValueError
    before opening. I/O errors propagate and may leave a partial file.
    No intermediate file.
    """
    logs: list[tuple[str, LabberPayload, str]] = []
    for name, payload in payloads.items():
        DataVariable(name)
        values = np.asarray(payload.z)
        if values.ndim not in {1, 2, 3}:
            raise ValueError(
                "independent Labber layout requires rank one, two or three"
            )
        if values.ndim == 3:
            if values.shape[0] == 0:
                raise ValueError(f"{name}: rank-three Labber first axis is empty")
            outer = payload.axes[-1]
            for index, coordinate in enumerate(outer.values):
                identity = f"{name}_{index}"
                if identity in payloads:
                    raise ValueError(f"duplicate data variable {identity!r}")
                sliced = LabberPayload(
                    (payload.data.name, payload.data.unit, values[index]),
                    payload.axes[:-1],
                )
                label = f"{destination.stem} - {name} {outer.name}={coordinate}"
                logs.append((identity, sliced, label))
        else:
            logs.append((name, payload, destination.stem))
    created = time.time() if metadata.creation_time is None else metadata.creation_time
    effective = replace(metadata, creation_time=created)
    with h5py.File(destination, "x") as file:
        for index, (name, payload, label) in enumerate(logs):
            target = file if index == 0 else file.create_group(f"Log_{index + 1}")
            write_payload_to_log(
                target,
                payload,
                effective,
                log_name=label,
                creation_time=created,
                write_tags=index == 0,
            )
            target.attrs[DATA_VARIABLE_ATTR] = name
        file.attrs[GROUPED_VERSION_ATTR] = 1
        file.attrs[DATA_VARIABLES_ATTR] = labber_string_array(
            [name for name, _, _ in logs]
        )
        file.attrs["zcu_tools.streaming_grouped_dataset_version"] = 1
        file.attrs["zcu_tools.streaming_finalized"] = True
