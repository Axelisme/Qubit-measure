"""Write Labber 1.8.6 uniform-grid and variable-length trace HDF5 logs.

Axes are inner-first; array dimensions are outermost-first, inner axis last.
Save helpers normalize the Labber suffix through LabberData.save. Exact-path
writers and borrowed-log writers document their own path and handle ownership.
New files use exclusive creation and failures may leave partial new files.
Shared bookkeeping labels, dtypes and string conversion belong to labber_format;
loading belongs to labber_reader. Neither owner chooses caller artifact policy.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from .labber_format import (
    API_STEP_NAME,
    CHANNEL_NAMES_DTYPE,
    CHANNELS_DTYPE,
    INSTRUMENTS_DTYPE,
    LABBER_VERSION,
    LOG_INSTRUMENT,
    LOG_LIST_DTYPE,
    RELATION_PARAMETERS_DTYPE,
    STEP_INSTRUMENT,
    STEP_ITEMS_DTYPE,
    STEP_LIST_DTYPE,
    labber_string_array,
)
from .models import (
    LabberData,
    LabberMetadata,
    LabberPayload,
    as_tag_list,
    unpack_triple,
)
from .paths import format_ext

if TYPE_CHECKING:
    # array-like type alias (covers ndarray, list, tuple, scalars) for checkers
    from numpy.typing import ArrayLike
else:
    ArrayLike = Any

__all__ = ["save_labber_data", "save_labber_trace_data"]


def save_labber_data(
    path: str,
    z: tuple[str, str, ArrayLike],
    axes: Sequence[tuple[str, str, ArrayLike]],
    *,
    metadata: LabberMetadata | None = None,
    timestamps: ArrayLike | None = None,
) -> str:
    """Save complex measurement data to a Labber-compatible HDF5 log file.

    Both the log channel ``z`` and every axis are described by a
    ``(name, unit, values)`` triple, e.g.::

        save_labber_data("scan",
            z=("S21", "", z_data),
            axes=[("Frequency", "Hz", freq),     # inner axis (x)
                  ("Power", "dBm", power)])       # outer axis (y)

    The data array (the ``values`` part of ``z``) has the **inner axis last**;
    outer axes precede it in its natural index order, so ``z_data[i_power, :]``
    selects the inner trace at ``power[i_power]``.  Supports 1-D, 2-D, 3-D and
    arbitrary N-D by passing that many axes.

    Parameters
    ----------
    path : str
        Output file path.  ``.hdf5`` is appended if missing (``.h5`` -> ``.hdf5``).
    z : (name, unit, values)
        The complex log channel.  ``values`` is the data array, shaped
        ``(Nx,)`` for 1-D, ``(Ny, Nx)`` for 2-D, ``(Nw, Ny, Nx)`` for 3-D, ...
    axes : sequence of (name, unit, values)
        One triple per axis, **inner axis first** (``axes[0]`` is x).
        ``len(axes)`` must equal ``values.ndim`` and each ``len(axes[k][2])``
        must match the corresponding data dimension.
    metadata : LabberMetadata, optional
        Comment, tags, project, user and optional creation epoch. Omission uses
        default metadata. The supplied object is passed through without copying.
    timestamps : array_like, optional
        Per-entry absolute timestamps (epoch s), one per stored entry.  If
        omitted, the current time is used for all entries.

    Returns
    -------
    str
        The path actually written.
    """
    return LabberData(
        z,
        axes,
        metadata=metadata,
        timestamps=timestamps,
    ).save(path)


def write_labber_data_file(path: str, ld: LabberData) -> str:
    """Write ld's inner-first uniform data/metadata at exact path; return it.
    Existing paths raise FileExistsError; invalid shapes raise ValueError.
    I/O errors propagate and may leave a partial new file.
    """
    log_name = os.path.splitext(os.path.basename(path))[0]
    with h5py.File(path, "x") as f:
        _write_uniform_log_group(f, ld, log_name=log_name, write_tags=True)
    return path


def write_payload_to_log(
    target: h5py.File | h5py.Group,
    payload: LabberPayload,
    metadata: LabberMetadata,
    *,
    log_name: str,
    creation_time: float | None = None,
    write_tags: bool = False,
) -> None:
    """Populate a borrowed open log group from one inner-first payload.

    List-valued data selects trace layout; other values select uniform layout.
    log_name labels the log, creation_time overrides metadata's epoch origin,
    and write_tags includes metadata tags/project/user when true. Inputs are
    borrowed without mutation. The caller owns flush/close. Invalid shapes or
    timestamps raise ValueError; HDF5 failures propagate and may leave partial
    groups in target. This does not open a file or normalize a path.
    """
    data = LabberData(payload=payload, metadata=metadata)
    if isinstance(data.data.values, list):
        _write_trace_log_group(
            target,
            data,
            log_name=log_name,
            creation_time=creation_time,
            write_tags=write_tags,
        )
        return

    _write_uniform_log_group(
        target,
        data,
        log_name=log_name,
        creation_time=creation_time,
        write_tags=write_tags,
    )


def _write_uniform_log_group(
    target: h5py.File | h5py.Group,
    ld: LabberData,
    *,
    log_name: str,
    creation_time: float | None = None,
    write_tags: bool = False,
) -> None:
    """Write uniform-grid data into an open root or ``Log_N`` group."""
    z_name, z_unit, z_values = ld.data
    z_arr = np.asarray(z_values, dtype=complex)

    # axis list, inner-first, values raveled to 1-D float arrays
    axis_list = [
        (a.name, a.unit, np.asarray(a.values, dtype=float).ravel()) for a in ld.axes
    ]
    if len(axis_list) != z_arr.ndim:
        raise ValueError(
            f"len(axes) ({len(axis_list)}) must equal z data ndim ({z_arr.ndim})"
        )

    # z axis order is outer..inner; axis_list is inner..outer -> reverse to match.
    inner_to_outer_dims = z_arr.shape[::-1]
    for k, (nm, _un, val) in enumerate(axis_list):
        if val.shape[0] != inner_to_outer_dims[k]:
            raise ValueError(
                f"axis '{nm}' length {val.shape[0]} != z dim {inner_to_outer_dims[k]}"
            )

    # Step dimensions are inner-first: [Nx, Ny, Nw, ...].
    step_dims = np.array([a[2].shape[0] for a in axis_list], dtype=np.int64)
    # number of stored entries = product of all outer dims
    n_entry = int(np.prod(step_dims[1:])) if len(step_dims) > 1 else 1

    log_channels = [(z_name, z_unit)]
    t0, ts_rel = _resolve_timestamps(
        ld.timestamps,
        n_entry,
        ld.creation_time if creation_time is None else creation_time,
    )

    _write_root_attrs(target, log_name, step_dims, ld.comment, t0)
    _write_config(target, axis_list, log_channels)
    if write_tags:
        _write_tags(target, as_tag_list(ld.tags), ld.project, ld.user)
    _write_multi_channel_data_group(
        target, axis_list, log_channels, step_dims, [z_arr], ts_rel
    )


def write_uniform_multi_channel_log_group(
    target: h5py.File | h5py.Group,
    payloads: Sequence[LabberPayload],
    metadata: LabberMetadata,
    *,
    log_name: str,
    creation_time: float,
) -> None:
    """Populate a borrowed log group with parallel common-grid scalar channels.

    payloads is a nonempty ordered sequence already validated for equal
    inner-first axes, shapes and timestamps by the grouped owner. metadata
    provides shared tags and comment; log_name labels the log and creation_time
    is its epoch origin. Inputs are not mutated. Empty payloads or invalid
    timestamps raise ValueError; conversion/HDF5 failures propagate and may
    leave partial groups. The caller owns the open target and flush/close.
    """
    if not payloads:
        raise ValueError("at least one payload is required")

    reference = payloads[0]
    axis_list = [
        (axis.name, axis.unit, np.asarray(axis.values, dtype=float).ravel())
        for axis in reference.axes
    ]
    z_arrays = [np.asarray(payload.z, dtype=complex) for payload in payloads]
    step_dims = np.array([axis[2].shape[0] for axis in axis_list], dtype=np.int64)
    n_entry = int(np.prod(step_dims[1:])) if len(step_dims) > 1 else 1
    log_channels = [(payload.data.name, payload.data.unit) for payload in payloads]
    t0, ts_rel = _resolve_timestamps(reference.timestamps, n_entry, creation_time)

    _write_root_attrs(target, log_name, step_dims, metadata.comment, t0)
    _write_config(target, axis_list, log_channels)
    _write_tags(target, as_tag_list(metadata.tags), metadata.project, metadata.user)
    _write_multi_channel_data_group(
        target,
        axis_list,
        log_channels,
        step_dims,
        z_arrays,
        ts_rel,
    )


def _resolve_timestamps(timestamps, n_entry, creation_time: float | None = None):
    """Return ``(t0, ts_rel)`` for the file's Time stamp datasets.

    Labber stores per-entry timestamps *relative* to a base ``creation_time``
    (``t0``).  We use the first entry's absolute timestamp as ``t0`` (or
    ``time.time()`` when none is given), and store the rest as offsets.
    """
    if timestamps is None:
        t0 = time.time() if creation_time is None else float(creation_time)
        return t0, np.zeros(n_entry, dtype=float)
    ts_abs = np.asarray(timestamps, dtype=float).ravel()
    if ts_abs.shape[0] != n_entry:
        raise ValueError(
            f"len(timestamps) ({ts_abs.shape[0]}) must equal "
            f"number of entries ({n_entry})"
        )
    t0 = float(ts_abs[0]) if creation_time is None else float(creation_time)
    return t0, ts_abs - t0


def save_labber_trace_data(
    path: str,
    z: tuple[str, str, Sequence[ArrayLike]],
    x: tuple[str, str, ArrayLike | Sequence[ArrayLike]],
    y: tuple[str, str, ArrayLike] | None = None,
    *,
    metadata: LabberMetadata | None = None,
    timestamps: ArrayLike | None = None,
) -> str:
    """Save **variable-length** complex traces (Labber ``vector=True`` mode).

    Unlike :func:`save_labber_data` (which needs a uniform x-axis shared by all
    traces), this stores each trace in the ``Traces/`` group with its own
    x-axis, so entries may have **different lengths and/or non-uniform x**.

    Like :func:`save_labber_data`, the channel and axes use ``(name, unit,
    values)`` triples::

        save_labber_trace_data("scan",
            z=("S21", "", traces),                  # traces = list of arrays
            x=("Frequency", "Hz", xs),              # xs = shared array or per-trace
            y=("Power", "dBm", power))              # optional outer axis

    Parameters
    ----------
    path : str
        Output path (``.hdf5`` appended if missing).
    z : (name, unit, traces)
        ``traces`` is a sequence of complex arrays, one per entry; lengths may
        differ between entries.
    x : (name, unit, values)
        The inner x-axis.  ``values`` is either a single array shared by all
        traces, or a sequence with one x array per trace (matching each trace's
        length).
    y : (name, unit, values), optional
        Outer step axis, one value per trace (e.g. Power).  ``len(values)`` must
        equal ``len(traces)``.  If omitted, the traces are an unstructured list.
    metadata : LabberMetadata, optional
        File metadata (see :func:`save_labber_data`); passed without copying.
    timestamps : array_like, optional
        Per-entry absolute timestamps (epoch s), one per trace.  If omitted,
        the current time is used for all entries.

    Returns
    -------
    str
        The path actually written.
    """
    z_name, z_unit, traces = unpack_triple(z, "z")
    # store the trace list as data.values; LabberData.save() detects the list
    # and routes to the trace writer.  x keeps its per-trace-or-shared values.
    axes = [x] if y is None else [x, y]
    return LabberData(
        (z_name, z_unit, list(traces)),
        axes,
        metadata=metadata,
        timestamps=timestamps,
    ).save(path)


def write_labber_trace_file(path: str, ld: LabberData) -> str:
    """Write borrowed dataset ld as a new variable-length Traces-layout file.

    Normalize path to .hdf5 and return that path. ld supplies inner-first axes,
    trace values, metadata and timestamps without being mutated. The file is
    closed before return. Existing paths raise FileExistsError; invalid shapes
    or timestamps raise ValueError. I/O errors propagate and may leave a
    partial new file; the destination is never overwritten.
    """
    path = format_ext(path)
    log_name = os.path.splitext(os.path.basename(path))[0]

    with h5py.File(path, "x") as f:
        _write_trace_log_group(f, ld, log_name=log_name, write_tags=True)

    return path


def _write_trace_log_group(
    target: h5py.File | h5py.Group,
    ld: LabberData,
    *,
    log_name: str,
    creation_time: float | None = None,
    write_tags: bool = False,
) -> None:
    """Write variable-length trace data into an open root or ``Log_N`` group."""
    z_name, z_unit, traces = ld.data
    x_name, x_unit, x_values = ld.axes[0]

    trace_list = [np.asarray(t, dtype=complex).ravel() for t in traces]
    n_entry = len(trace_list)
    if n_entry == 0:
        raise ValueError("`traces` must contain at least one trace")

    # per-trace x-axis: either one shared array or one array per trace
    if _is_sequence_of_arrays(x_values):
        x_list = [np.asarray(xi, dtype=float).ravel() for xi in x_values]
        if len(x_list) != n_entry:
            raise ValueError("len(x values) must equal len(traces) when per-trace")
    else:
        x1 = np.asarray(x_values, dtype=float).ravel()
        x_list = [x1 for _ in range(n_entry)]
    for i, (t, xi) in enumerate(zip(trace_list, x_list, strict=False)):
        if len(t) != len(xi):
            raise ValueError(f"trace {i}: len(x) {len(xi)} != len(trace) {len(t)}")

    has_outer = len(ld.axes) > 1
    if has_outer:
        y_name, y_unit, y_raw = ld.axes[1]
        y_vals = np.asarray(y_raw, dtype=float).ravel()
        if y_vals.shape[0] != n_entry:
            raise ValueError(
                f"len(y) ({y_vals.shape[0]}) must equal len(traces) ({n_entry})"
            )
    else:
        y_name, y_unit, y_vals = "", "", None

    # Step dimensions: inner is variable (=1), then the outer entry axis.
    if has_outer:
        step_dims = np.array([1, n_entry], dtype=np.int64)
        step_channels = [(API_STEP_NAME, "", None), (y_name, y_unit, y_vals)]
    else:
        step_dims = np.array([1], dtype=np.int64)
        step_channels = [(API_STEP_NAME, "", None)]
    log_channels = [(z_name, z_unit)]

    t0, ts_rel = _resolve_timestamps(
        ld.timestamps,
        n_entry,
        ld.creation_time if creation_time is None else creation_time,
    )

    _write_root_attrs(target, log_name, step_dims, ld.comment, t0)
    _write_config(
        target,
        step_channels,
        log_channels,
        log_vector=True,
        trace_x_name=x_name,
        trace_x_unit=x_unit,
    )
    if write_tags:
        _write_tags(target, as_tag_list(ld.tags), ld.project, ld.user)
    # Data/ holds only the (dummy + outer) step columns, no z
    _write_trace_data_stub(target, step_channels, step_dims, y_vals, n_entry, ts_rel)
    _write_traces_group(
        target, z_name, trace_list, x_list, x_name, x_unit, n_entry, ts_rel
    )


def _is_sequence_of_arrays(x) -> bool:
    """True if x is a list/tuple whose elements are themselves array-like."""
    if isinstance(x, (list, tuple)) and len(x) > 0:
        first = x[0]
        return isinstance(first, (list, tuple, np.ndarray))
    return False


def _write_root_attrs(f, log_name, step_dims, comment, t0):
    a = f.attrs
    a["Step dimensions"] = step_dims
    a["version"] = LABBER_VERSION
    a["log_name"] = log_name
    a["comment"] = comment
    a["creation_time"] = float(t0)
    a["arm_trig_mode"] = False
    a["hardware_loop"] = False
    a["log_parallel"] = True
    a["logger_mode"] = False
    a["time_per_point"] = 0.0
    a["trig_channel"] = ""
    a["wait_between"] = 0.01


def _channels_rows(step_channels, log_channels):
    rows = []
    for name, unit, _vals in step_channels:
        rows.append(
            (
                name,
                STEP_INSTRUMENT,
                name,
                unit,
                unit,
                1.0,
                0.0,
                1.0,
                np.inf,
                -np.inf,
                "",
                "Nothing",
                False,
                "",
                False,
                1.0,
                False,
            )
        )
    for name, unit in log_channels:
        rows.append(
            (
                name,
                LOG_INSTRUMENT,
                name,
                unit,
                unit,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                "",
                "Nothing",
                False,
                "",
                False,
                1.0,
                False,
            )
        )
    return np.array(rows, dtype=CHANNELS_DTYPE)


def _write_config(
    f,
    step_channels,
    log_channels,
    *,
    log_vector=False,
    trace_x_name="Index",
    trace_x_unit="",
):
    """Write the bookkeeping datasets/groups Labber needs to open the file.

    ``log_vector`` selects how the log channel is registered in the instrument
    config: scalar (``0j`` default, no ``___x___`` attrs) for ``Data/Data``
    storage, or vector (empty-array default plus ``___<name>___x_name``/
    ``x_unit`` attrs) for ``Traces/`` storage.
    """
    f.create_dataset("Channels", data=_channels_rows(step_channels, log_channels))

    def instr_row(iid, name):
        return (
            "Generic",
            "1.0",
            iid,
            "",
            name,
            0,
            "",
            "",
            0,
            False,
            False,
            10.0,
            "Auto",
            True,
            False,
            False,
            False,
            0.0,
            False,
            9600.0,
            8.0,
            1.0,
            "No parity",
            0.0,
            False,
            1.0,
            False,
        )

    f.create_dataset(
        "Instruments",
        data=np.array(
            [
                instr_row(STEP_INSTRUMENT, "Step channels"),
                instr_row(LOG_INSTRUMENT, "Log channels"),
            ],
            dtype=INSTRUMENTS_DTYPE,
        ),
    )

    f.create_dataset(
        "Log list",
        data=np.array([(n,) for n, _u in log_channels], dtype=LOG_LIST_DTYPE),
    )

    f.create_dataset(
        "Step list",
        data=np.array(
            [
                (name, 0, 0.0, 0, 0.0, False, "x", False, 0, False, 0.0, False)
                for name, _u, _v in step_channels
            ],
            dtype=STEP_LIST_DTYPE,
        ),
    )

    g_step_cfg = f.create_group("Step config")
    for name, _unit, vals in step_channels:
        _write_step_config_entry(g_step_cfg, name, vals)

    g_ic = f.create_group("Instrument config")
    g_log = g_ic.create_group(LOG_INSTRUMENT)
    g_log.attrs["Installed options"] = np.array([], dtype=float)
    for name, _unit in log_channels:
        if log_vector:
            # vector channel: empty-array default + ___x___ attrs -> Traces group
            g_log.attrs[name] = np.array([], dtype=complex)
            g_log.attrs["___%s___x_name" % name] = trace_x_name
            g_log.attrs["___%s___x_unit" % name] = trace_x_unit
        else:
            # scalar complex log channel: scalar default (0j) and NO ___x___
            # attrs (those would mark it vector and trigger a Traces lookup).
            g_log.attrs[name] = 0j
    g_st = g_ic.create_group(STEP_INSTRUMENT)
    g_st.attrs["Installed options"] = np.array([], dtype=float)
    for name, _unit, _vals in step_channels:
        g_st.attrs[name] = 0.0

    f.create_group("Settings")


def _write_step_config_entry(parent, name, vals):
    g = parent.create_group(name)
    opt = g.create_group("Optimizer")
    # vals is None for the dummy 'Step index API' channel -> placeholder range
    v = None if vals is None else np.asarray(vals, dtype=float).ravel()
    if v is not None and v.size > 0:
        lo, hi = float(v.min()), float(v.max())
        start, stop, n = float(v[0]), float(v[-1]), int(v.size)
    else:
        lo, hi, start, stop, n = 1.0, 2.0, 1.0, 2.0, 1
    span = hi - lo
    opt.attrs["Enabled"] = False
    opt.attrs["Initial step size"] = span * 0.2 if span > 0 else 0.2
    opt.attrs["Max value"] = hi
    opt.attrs["Min value"] = lo
    opt.attrs["Precision"] = span * 1e-4 if span > 0 else 1e-4
    opt.attrs["Start value"] = lo

    g.create_dataset(
        "Relation parameters",
        data=np.array([("x", "Step values", False)], dtype=RELATION_PARAMETERS_DTYPE),
    )

    g.create_dataset(
        "Step items",
        data=np.array(
            [(1, 1, start, start, stop, 0.0, 0.0, 0.0, n, 0, 0.0)],
            dtype=STEP_ITEMS_DTYPE,
        ),
    )


def _write_tags(f, tags, project, user):
    g = f.create_group("Tags")
    g.attrs["Project"] = labber_string_array([project])
    g.attrs["User"] = labber_string_array([user])
    if tags:
        g.attrs["Tags"] = labber_string_array(tags)
    else:
        g.attrs["Tags"] = np.array([], dtype=float)


def _write_multi_channel_data_group(
    f,
    axis_list,
    log_channels,
    step_dims,
    z_arrays,
    ts_rel,
):
    """Write scalar log channels sharing one inner-first step grid."""
    if len(log_channels) != len(z_arrays):
        raise ValueError("log channel count does not match data array count")

    g = f.create_group("Data")
    names_info: list[tuple[str, str]] = [(name, "") for name, _u, _v in axis_list]
    for name, _u in log_channels:
        names_info.extend(((name, "Real"), (name, "Imaginary")))
    g.create_dataset(
        "Channel names", data=np.array(names_info, dtype=CHANNEL_NAMES_DTYPE)
    )

    reference = z_arrays[0]
    n_x = reference.shape[-1]
    n_entry = int(np.prod(step_dims[1:])) if len(step_dims) > 1 else 1
    data = np.zeros((n_x, len(names_info), n_entry), dtype=float)
    data[:, 0, :] = axis_list[0][2][:, None]

    z_outer_shape = reference.shape[:-1]
    if z_outer_shape:
        multi = np.unravel_index(np.arange(n_entry), z_outer_shape)
        n_outer = len(z_outer_shape)
        for j in range(n_outer):
            col = n_outer - j
            coord = axis_list[col][2][multi[j]]
            data[:, col, :] = coord[None, :]

    first_log_col = len(axis_list)
    for index, z in enumerate(z_arrays):
        zf = z.reshape(n_entry, n_x)
        col = first_log_col + 2 * index
        data[:, col, :] = zf.real.T
        data[:, col + 1, :] = zf.imag.T

    g.create_dataset("Data", data=data)
    g.create_dataset("Time stamp", data=np.asarray(ts_rel, dtype=float))
    g.attrs["Completed"] = True
    g.attrs["Step dimensions"] = step_dims
    g.attrs["Step index"] = np.arange(len(axis_list), dtype=np.int64)
    g.attrs["Fixed step index"] = np.array([], dtype=np.int64)
    g.attrs["Fixed step values"] = np.array([], dtype=float)


def _write_trace_data_stub(f, step_channels, step_dims, y_vals, n_entry, ts_rel):
    """Write the Data/ group for trace-mode logs: only the step columns (the
    dummy 'Step index API' plus any outer axis), no z columns."""
    g = f.create_group("Data")
    names_info = [(name, "") for name, _u, _v in step_channels]
    g.create_dataset(
        "Channel names", data=np.array(names_info, dtype=CHANNEL_NAMES_DTYPE)
    )

    n_cols = len(step_channels)
    data = np.zeros((1, n_cols, n_entry), dtype=float)
    data[0, 0, :] = 1.0  # dummy Step index API
    if y_vals is not None and n_cols > 1:
        data[0, 1, :] = np.asarray(y_vals, dtype=float)
    g.create_dataset("Data", data=data)
    g.create_dataset("Time stamp", data=np.asarray(ts_rel, dtype=float))

    g.attrs["Completed"] = True
    g.attrs["Step dimensions"] = step_dims
    g.attrs["Step index"] = np.arange(n_cols, dtype=np.int64)
    g.attrs["Fixed step index"] = np.array([], dtype=np.int64)
    g.attrs["Fixed step values"] = np.array([], dtype=float)
    if n_entry > 0:
        # mirror Labber: record length of the last trace (used for partial logs)
        g.attrs["Entries, last trace"] = 1


# tolerated: One trace-layout caller supplies ragged values, x labels and entry timestamps.
def _write_traces_group(  # noqa: PLR0913
    f, z_name, trace_list, x_list, x_name, x_unit, n_entry, ts_rel
):
    """Write the Traces/ group: <name> (maxN, 3, Nentry), <name>_N, <name>_t0dt.

    Column layout per entry: 0=real, 1=imag, 2=x (explicit axis).  ``_t0dt`` is
    ``(0, 0)`` to signal "use the explicit x column"; ``_N`` records each
    trace's true length so ragged traces round-trip exactly.
    """
    g = f.create_group("Traces")
    max_n = max(len(t) for t in trace_list)
    arr = np.full((max_n, 3, n_entry), np.nan, dtype=float)
    n_arr = np.zeros(n_entry, dtype=np.int32)
    for e, (t, xi) in enumerate(zip(trace_list, x_list, strict=False)):
        n = len(t)
        arr[:n, 0, e] = t.real
        arr[:n, 1, e] = t.imag
        arr[:n, 2, e] = xi
        n_arr[e] = n

    ds = g.create_dataset(z_name, data=arr)
    ds.attrs["complex"] = True
    ds.attrs["x, name"] = x_name
    ds.attrs["x, unit"] = x_unit
    g.create_dataset(z_name + "_t0dt", data=np.zeros((1, 2), dtype=float))
    g.create_dataset(z_name + "_N", data=n_arr)
    g.create_dataset("Time stamp", data=np.asarray(ts_rel, dtype=float))
