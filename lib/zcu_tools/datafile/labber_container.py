"""Labber-to-container projection without experiment metadata conventions."""

from pathlib import Path

import h5py as h5
import numpy as np

from .container_models import Axis, Container, ContainerAttrs, GroupSource, Variable
from .grouped import GROUPED_VERSION_ATTR, load_grouped_labber_data
from .labber_reader import load_labber_data
from .models import LabberMetadata


def read_labber(path: Path) -> tuple[Container, LabberMetadata]:
    """Read single or marked grouped Labber into NumPy-ordered variables.

    Return (container, metadata). metadata retains comment, tags, project, user
    and creation_time explicitly, outside JSON docs. Variable keys are saved
    grouped identities, or the single channel name. docs is empty; groups/labber
    holds the detached original tree, preserving timestamps and other opaque
    Labber content for rewrite. Existing low-level load_labber_data and
    load_grouped_labber_data expose per-trace timestamps directly when needed.
    Unsupported or malformed layouts raise ValueError; I/O errors propagate.
    No cfg comment, units or experiment identity is inferred.
    """
    with h5.File(path, "r") as file:
        grouped = GROUPED_VERSION_ATTR in file.attrs
        image = file.id.get_file_image()
    if grouped:
        loaded = load_grouped_labber_data(str(path))
        payloads = loaded.variables
        metadata = loaded.metadata
    else:
        single = load_labber_data(str(path))
        payloads = {single.data.name: single.payload}
        metadata = single.metadata
    variables = {
        str(name): Variable(
            np.asarray(payload.z),
            tuple(
                Axis(axis.name, axis.unit, np.asarray(axis.values))
                for axis in reversed(payload.axes)
            ),
            payload.data.unit,
        )
        for name, payload in payloads.items()
    }
    return Container(
        variables,
        {},
        {"labber": GroupSource(image, "/")},
        ContainerAttrs("3.0", "zcu_tools Labber reader", None, {}),
    ), metadata
