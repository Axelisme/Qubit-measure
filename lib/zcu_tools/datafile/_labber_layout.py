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
    """Write complete rank-1/2 logs at exact destination, refusing existing files.

    Each payload is inner-first, with its own grid. The first log occupies root;
    others occupy Log_2, Log_3, etc. Retain the marker-qualified grouped-v1 wire
    layout so the grouped reader can identify each variable. Mark finalized only
    after every log succeeds. Invalid rank/name raises ValueError before opening;
    I/O errors propagate and may leave a partial file. No intermediate file.
    """
    for name, payload in payloads.items():
        DataVariable(name)
        if np.asarray(payload.z).ndim not in {1, 2}:
            raise ValueError("independent Labber layout requires rank one or two")
    created = time.time() if metadata.creation_time is None else metadata.creation_time
    effective = replace(metadata, creation_time=created)
    with h5py.File(destination, "x") as file:
        for index, (name, payload) in enumerate(payloads.items()):
            target = file if index == 0 else file.create_group(f"Log_{index + 1}")
            write_payload_to_log(
                target,
                payload,
                effective,
                log_name=destination.stem,
                creation_time=created,
                write_tags=index == 0,
            )
            target.attrs[DATA_VARIABLE_ATTR] = name
        file.attrs[GROUPED_VERSION_ATTR] = 1
        file.attrs[DATA_VARIABLES_ATTR] = labber_string_array(list(payloads))
        file.attrs["zcu_tools.streaming_grouped_dataset_version"] = 1
        file.attrs["zcu_tools.streaming_finalized"] = True
