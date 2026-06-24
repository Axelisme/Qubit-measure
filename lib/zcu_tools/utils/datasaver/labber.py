"""Standalone read/write of Labber HDF5 log files (Python 3.9+).

This module saves and loads complex measurement data with axes in Labber's
HDF5 log-file format, *without* depending on the original ``Labber`` package
(which only runs on Python 3.6-3.9).  It uses only ``h5py`` and ``numpy`` and
therefore runs on Python 3.9 through 3.14+.  Files written here are
byte-compatible with the real Labber software (version tag ``1.8.6``) and can be
opened by Labber / the original API; conversely it reads files Labber wrote.

Quick start
-----------
Everything is described by ``(name, unit, values)`` triples.  Axes are listed
**inner axis first** (the inner axis is the last axis of the data array)::

    import numpy as np
    from zcu_tools.utils.datasaver import save_labber_data, load_labber_data

    freq  = np.linspace(4e9, 5e9, 201)
    power = np.linspace(-30, 0, 11)

    # 1-D: one complex trace,  data shape (Nx,)
    save_labber_data("scan_1d",
        z=("S21", "", s21),
        axes=[("Frequency", "Hz", freq)],
        comment="resonator dip", tags=["cooldown7"])

    # 2-D: one trace per power, data shape (Ny, Nx)  (inner axis last)
    save_labber_data("scan_2d",
        z=("S21", "", z2d),
        axes=[("Frequency", "Hz", freq),    # inner axis (x)
              ("Power", "dBm", power)])      # outer axis (y)

    # 3-D / N-D: just pass more axes (inner first), data shape (Nw, Ny, Nx)
    save_labber_data("scan_3d",
        z=("S21", "", z3d),
        axes=[("Frequency","Hz",freq), ("Power","dBm",power), ("Flux","",flux)])

    # read back
    d = load_labber_data("scan_2d")
    d.data.values        # the complex data array  (== d.z)
    d.axes[0].values     # the inner-axis array    (== d.x)
    d.axes[1].name       # "Power"
    d.comment, d.tags, d.timestamps
    z, x, y = d          # also unpacks as (z, x, y)

``LabberData`` is the hub
-------------------------
:func:`save_labber_data` / :func:`load_labber_data` are thin wrappers around the
:class:`LabberData` model, which can be used directly::

    from zcu_tools.utils.datasaver import Axis, LabberData

    ld = LabberData(("S21", "", z2d),
                    axes=[("Frequency","Hz",freq), ("Power","dBm",power)])
    ld.save("scan_2d")                 # == save_labber_data(...)
    ld = LabberData.load("scan_2d")    # == load_labber_data(...)

:data:`Axis` is a ``namedtuple(name, unit, values)`` -- and therefore a plain
3-tuple, so a ``(name, unit, values)`` triple and an ``Axis`` are
interchangeable everywhere.

Data-shape convention
---------------------
The data array's **last axis is always the inner (x) axis**; outer axes precede
it in the array's natural index order, so ``z[i_power, :]`` is the inner trace at
``power[i_power]``.  List ``axes`` inner-first to match.

Variable-length traces
----------------------
When each entry has a *different length* or a *non-uniform x-axis*, use
:func:`save_labber_trace_data` instead (Labber's ``vector=True`` mode).  The data
is stored in the ``Traces/`` group and :func:`load_labber_data` auto-detects it,
returning ``d.data.values`` as a list of ragged arrays (or a stacked array when
all lengths match).

Public API
----------
* :data:`Axis`                  -- ``(name, unit, values)`` namedtuple
* :class:`LabberData`           -- the data model, with ``.save`` / ``.load``
* :func:`save_labber_data`      -- save a uniform-grid scan (1-D / 2-D / N-D)
* :func:`save_labber_trace_data`-- save variable-length / non-uniform traces
* :func:`load_labber_data`      -- load any of the above into a ``LabberData``

On-disk format (for reference)
------------------------------
Uniform-grid data lives in the ``Data/Data`` dataset, shape
``(Nx, Ncol, Nentries)``: axis 0 = inner step points (x), axis 1 = channel
columns ordered ``[x, y, w, ..., z_real, z_imag]``, axis 2 = outer entries.  A
complex log channel occupies the last *two* columns (real, imaginary), recorded
in ``Data/Channel names`` with ``info`` = ``Real`` / ``Imaginary``.  Per-entry
timestamps live in ``Data/Time stamp`` relative to the root ``creation_time``.
Variable-length traces live in ``Traces/<name>`` of shape
``(maxN, ncol, Nentries)`` with a ``<name>_N`` length per entry.  Multiple log
configs are stored as the root plus ``Log_2``, ``Log_3`` groups.  The format was
reverse-engineered from the ``Labber._include38`` bytecode and ground-truth
files produced by the original package.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from .models import (
    Axis,
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

__all__ = [
    "Axis",
    "LabberMetadata",
    "LabberPayload",
    "LabberData",
    "save_labber_data",
    "save_labber_trace_data",
    "load_labber_data",
]

_VERSION = "1.8.6"
_INSTR_STEP = "Generic - GPIB: , Step channels at localhost"
_INSTR_LOG = "Generic - GPIB: , Log channels at localhost"
# dummy step channel Labber inserts for trace/vector logs with no real step axis
_STEP_NAME_API = "Step index API"

# ----------------------------------------------------------------------------
# h5py string / attribute helpers (mirror Labber's SR_HDF5)
# ----------------------------------------------------------------------------

_VLEN_STR = h5py.special_dtype(vlen=str)


def _decode(value: Any) -> Any:
    """Decode hdf5 bytes/arrays to python str (mirrors SR_HDF5.decodeAttribute)."""
    if isinstance(value, bytes):
        s = value.decode("utf-8")
        return None if s == "NoneValue" else s
    if isinstance(value, np.ndarray):
        if value.dtype.type is np.bytes_:
            return [el.decode("utf-8") for el in value]
        if value.dtype.type is np.object_:
            return [
                el.decode("utf-8") if isinstance(el, bytes) else str(el) for el in value
            ]
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, str):
        return None if value == "NoneValue" else value
    return value


def _str_array(values: Sequence[str]) -> np.ndarray:
    return np.array(list(values), dtype=_VLEN_STR)


# ----------------------------------------------------------------------------
# compound dtypes for the bookkeeping datasets (exact match to Labber 1.8.6)
# ----------------------------------------------------------------------------


def _enum_i2(mapping: dict[str, int]) -> np.dtype:
    return h5py.special_dtype(enum=(np.int16, mapping))


_DT_CHANNELS = np.dtype(
    [
        ("name", _VLEN_STR),
        ("instrument", _VLEN_STR),
        ("quantity", _VLEN_STR),
        ("unitPhys", _VLEN_STR),
        ("unitInstr", _VLEN_STR),
        ("gain", "<f8"),
        ("offset", "<f8"),
        ("amp", "<f8"),
        ("highLim", "<f8"),
        ("lowLim", "<f8"),
        ("outputChannel", _VLEN_STR),
        ("limit_action", _VLEN_STR),
        ("limit_run_script", "?"),
        ("limit_script", _VLEN_STR),
        ("use_log_interval", "?"),
        ("log_interval", "<f8"),
        ("limit_run_always", "?"),
    ]
)

_DT_INSTRUMENTS = np.dtype(
    [
        ("hardware", _VLEN_STR),
        ("version", _VLEN_STR),
        ("id", _VLEN_STR),
        ("model", _VLEN_STR),
        ("name", _VLEN_STR),
        (
            "interface",
            _enum_i2(
                {
                    "GPIB": 0,
                    "TCPIP": 1,
                    "USB": 2,
                    "PXI": 3,
                    "Serial": 4,
                    "VISA": 5,
                    "Other": 6,
                    "None": 7,
                }
            ),
        ),
        ("address", _VLEN_STR),
        ("server", _VLEN_STR),
        ("startup", _enum_i2({"Set config": 0, "Get config": 1, "Do nothing": 2})),
        ("lock", "?"),
        ("show_advanced", "?"),
        ("Timeout", "<f8"),
        ("Term. character", _VLEN_STR),
        ("Send end on write", "?"),
        ("Lock VISA resource", "?"),
        ("Suppress end bit termination on read", "?"),
        ("Use specific TCP port", "?"),
        ("TCP port", "<f8"),
        ("Use VICP protocol", "?"),
        ("Baud rate", "<f8"),
        ("Data bits", "<f8"),
        ("Stop bits", "<f8"),
        ("Parity", _VLEN_STR),
        ("GPIB board number", "<f8"),
        ("Send GPIB go to local at close", "?"),
        ("PXI chassis", "<f8"),
        ("Run in 32-bit mode", "?"),
    ]
)

_DT_LOG_LIST = np.dtype([("channel_name", _VLEN_STR)])

_DT_STEP_LIST = np.dtype(
    [
        ("channel_name", _VLEN_STR),
        ("step_unit", _enum_i2({"Instrument": 0, "Physical": 1})),
        ("wait_after", "<f8"),
        (
            "after_last",
            _enum_i2({"Goto first point": 0, "Stay at final": 1, "Goto value...": 2}),
        ),
        ("final_value", "<f8"),
        ("use_relations", "?"),
        ("equation", _VLEN_STR),
        ("show_advanced", "?"),
        ("sweep_mode", _enum_i2({"Off": 0, "Between points": 1, "Continuous": 2})),
        ("use_outside_sweep_rate", "?"),
        ("sweep_rate_outside", "<f8"),
        ("alternate_direction", "?"),
    ]
)

_DT_REL_PARAMS = np.dtype(
    [
        ("variable", _VLEN_STR),
        ("channel_name", _VLEN_STR),
        ("use_lookup", "?"),
    ]
)

_DT_STEP_ITEMS = np.dtype(
    [
        ("range_type", _enum_i2({"Single": 0, "Start - Stop": 1, "Center - Span": 2})),
        ("step_type", _enum_i2({"Fixed step": 0, "Fixed # of pts": 1})),
        ("single", "<f8"),
        ("start", "<f8"),
        ("stop", "<f8"),
        ("center", "<f8"),
        ("span", "<f8"),
        ("step", "<f8"),
        ("n_pts", "<i4"),
        (
            "interp",
            _enum_i2({"Linear": 0, "Log": 1, "Log, #/decade": 2, "Lorentzian": 3}),
        ),
        ("sweep_rate", "<f8"),
    ]
)

_DT_CHANNEL_NAMES = np.dtype([("name", _VLEN_STR), ("info", _VLEN_STR)])


# ============================================================================
# WRITE -- public save functions (thin wrappers) + private _save_* / _write_*
# ============================================================================


def save_labber_data(
    path: str,
    z: tuple[str, str, ArrayLike],
    axes: Sequence[tuple[str, str, ArrayLike]],
    *,
    comment: str = "",
    tags: str | Sequence[str] | None = None,
    project: str = "",
    user: str = "",
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
    comment : str
        Free-text comment.
    tags : str or sequence of str, optional
        Tag(s) to store.
    project, user : str
        Project / user metadata.
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
        comment=comment,
        tags=tags,
        project=project,
        user=user,
        timestamps=timestamps,
    ).save(path)


def _save_labber_data(path: str, ld: LabberData) -> str:
    """Core writer for uniform-grid data (the ``Data/Data`` scalar layout)."""
    path = format_ext(path)
    log_name = os.path.splitext(os.path.basename(path))[0]

    with h5py.File(path, "w") as f:
        _write_uniform_log_group(f, ld, log_name=log_name, write_tags=True)

    return path


def _write_payload_to_log(
    target: h5py.File | h5py.Group,
    payload: LabberPayload,
    metadata: LabberMetadata,
    *,
    log_name: str,
    creation_time: float | None = None,
    write_tags: bool = False,
) -> None:
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

    n_x = z_arr.shape[-1]
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
    _write_data_group(
        target, axis_list, log_channels, step_dims, z_arr, n_entry, n_x, ts_rel
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
    z: tuple[str, str, Sequence],
    x: tuple[str, str, ArrayLike | Sequence[ArrayLike]],
    y: tuple[str, str, ArrayLike] | None = None,
    *,
    comment: str = "",
    tags: str | Sequence[str] | None = None,
    project: str = "",
    user: str = "",
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
    comment, tags, project, user : metadata (see :func:`save_labber_data`).
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
        comment=comment,
        tags=tags,
        project=project,
        user=user,
        timestamps=timestamps,
    ).save(path)


def _save_labber_trace_data(path: str, ld: LabberData) -> str:
    """Core writer for variable-length traces (the ``Traces/`` layout)."""
    path = format_ext(path)
    log_name = os.path.splitext(os.path.basename(path))[0]

    with h5py.File(path, "w") as f:
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
    for i, (t, xi) in enumerate(zip(trace_list, x_list)):
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
        step_channels = [(_STEP_NAME_API, "", None), (y_name, y_unit, y_vals)]
    else:
        step_dims = np.array([1], dtype=np.int64)
        step_channels = [(_STEP_NAME_API, "", None)]
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
    a["version"] = _VERSION
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
                _INSTR_STEP,
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
                _INSTR_LOG,
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
    return np.array(rows, dtype=_DT_CHANNELS)


def _write_config(
    f,
    step_channels,
    log_channels,
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
                instr_row(_INSTR_STEP, "Step channels"),
                instr_row(_INSTR_LOG, "Log channels"),
            ],
            dtype=_DT_INSTRUMENTS,
        ),
    )

    f.create_dataset(
        "Log list", data=np.array([(n,) for n, _u in log_channels], dtype=_DT_LOG_LIST)
    )

    f.create_dataset(
        "Step list",
        data=np.array(
            [
                (name, 0, 0.0, 0, 0.0, False, "x", False, 0, False, 0.0, False)
                for name, _u, _v in step_channels
            ],
            dtype=_DT_STEP_LIST,
        ),
    )

    g_step_cfg = f.create_group("Step config")
    for name, _unit, vals in step_channels:
        _write_step_config_entry(g_step_cfg, name, vals)

    g_ic = f.create_group("Instrument config")
    g_log = g_ic.create_group(_INSTR_LOG)
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
    g_st = g_ic.create_group(_INSTR_STEP)
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
        data=np.array([("x", "Step values", False)], dtype=_DT_REL_PARAMS),
    )

    g.create_dataset(
        "Step items",
        data=np.array(
            [(1, 1, start, start, stop, 0.0, 0.0, 0.0, n, 0, 0.0)], dtype=_DT_STEP_ITEMS
        ),
    )


def _write_tags(f, tags, project, user):
    g = f.create_group("Tags")
    g.attrs["Project"] = _str_array([project])
    g.attrs["User"] = _str_array([user])
    if tags:
        g.attrs["Tags"] = _str_array(tags)
    else:
        g.attrs["Tags"] = np.array([], dtype=float)


def _write_data_group(f, axis_list, log_channels, step_dims, z, n_entry, n_x, ts_rel):
    """Write Data/ group with complex log channel as the last two columns.

    ``axis_list`` is inner-first: ``axis_list[0]`` is x, ``axis_list[1]`` is y,
    ...  Each outer axis gets one column whose value at entry ``e`` is that
    axis' coordinate for the entry.  Entries iterate with the first outer axis
    (y) fastest -- matching Labber and ``z.reshape(n_entry, n_x)`` in C-order.
    ``ts_rel`` holds per-entry timestamps relative to the file creation time.
    """
    g = f.create_group("Data")

    # column layout: step channels (1 col each), then log channels (2 cols)
    names_info: list[tuple[str, str]] = [(name, "") for name, _u, _v in axis_list]
    for name, _u in log_channels:
        names_info.append((name, "Real"))
        names_info.append((name, "Imaginary"))
    g.create_dataset(
        "Channel names", data=np.array(names_info, dtype=_DT_CHANNEL_NAMES)
    )

    n_cols = len(names_info)
    data = np.zeros((n_x, n_cols, n_entry), dtype=float)

    # inner x axis (column 0): same for every entry
    data[:, 0, :] = axis_list[0][2][:, None]

    # outer axes (columns 1..): coordinate per entry.
    # z outer shape (C-order, slowest-first) is z.shape[:-1] == (Nw, ..., Ny);
    # entry e = unravel over that shape, so y (z axis -2) varies fastest.
    z_outer_shape = z.shape[:-1]  # (Nw, ..., Ny)
    if z_outer_shape:
        ent = np.arange(n_entry)
        multi = np.unravel_index(ent, z_outer_shape)  # tuple, one per z-outer axis
        # z-outer axis j (0=slowest=outermost) is step column (n_axes-1 - j).
        n_outer = len(z_outer_shape)
        for j in range(n_outer):
            col = n_outer - j  # 1..n_outer (x is col 0)
            coord = axis_list[col][2][multi[j]]  # (n_entry,)
            data[:, col, :] = coord[None, :]

    # complex log channel -> last two columns
    zf = z.reshape(n_entry, n_x)  # entries, y fastest
    data[:, -2, :] = zf.real.T
    data[:, -1, :] = zf.imag.T

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
        "Channel names", data=np.array(names_info, dtype=_DT_CHANNEL_NAMES)
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


def _write_traces_group(f, z_name, trace_list, x_list, x_name, x_unit, n_entry, ts_rel):
    """Write the Traces/ group: <name> (maxN, 3, Nentry), <name>_N, <name>_t0dt.

    Column layout per entry: 0=real, 1=imag, 2=x (explicit axis).  ``_t0dt`` is
    ``(0, 0)`` to signal "use the explicit x column"; ``_N`` records each
    trace's true length so ragged traces round-trip exactly.
    """
    g = f.create_group("Traces")
    max_n = max(len(t) for t in trace_list)
    arr = np.full((max_n, 3, n_entry), np.nan, dtype=float)
    n_arr = np.zeros(n_entry, dtype=np.int32)
    for e, (t, xi) in enumerate(zip(trace_list, x_list)):
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


# ============================================================================
# READ -- public load function (thin wrapper) + private _load_* / _read_*
# ============================================================================


def load_labber_data(path: str) -> LabberData:
    """Load complex measurement data from a Labber HDF5 log file.

    Reads every log in the file (root + ``Log_2``, ``Log_3`` ...).  A single log
    returns ``z`` reshaped to its natural N-D grid: ``(Nx,)`` for 1-D,
    ``(Ny, Nx)`` for 2-D, ``(Nw, Ny, Nx)`` for 3-D, etc. (inner axis last).
    Multiple logs that share identical axes are stacked along a new leading
    axis, giving ``z`` shape ``(Nlog, ...)``.

    Parameters
    ----------
    path : str
        Path to the ``.hdf5`` log file (``.h5`` is also accepted).

    Returns
    -------
    LabberData
        Complex data, axes and metadata.  ``.axes`` lists ``(name, unit,
        values)`` inner-first; ``.x``/``.y``/``.w`` are convenience aliases.
        Also unpacks as ``z, x, y``.
    """
    return _load_labber_data(path)


def _load_labber_data(path: str) -> LabberData:
    """Core reader: build a :class:`LabberData` from a Labber HDF5 file."""
    path = _resolve_path(path)

    with h5py.File(path, "r") as f:
        if "zcu_tools.grouped_dataset_version" in f.attrs:
            raise ValueError(
                "file is a grouped Labber dataset; use load_grouped_labber_data"
            )

        comment = _decode(f.attrs.get("comment", "")) or ""
        tags, project, user = _read_tags(f)
        creation_time = float(f.attrs.get("creation_time", 0.0) or 0.0)

        logs = _all_log_refs(f)
        z0, axes0, ts0 = _read_single_log(f, logs[0])
        z_name, z_unit = _read_log_label(f, logs[0])

        if len(logs) == 1:
            z = z0
            ts_rel = ts0
        else:
            z_list = [z0]
            for log in logs[1:]:
                zi, axes_i, _tsi = _read_single_log(f, log)
                if len(axes_i) != len(axes0):
                    raise ValueError("logs have different number of axes")
                for (n0, _u0, v0), (ni, _ui, vi) in zip(axes0, axes_i):
                    if np.shape(v0) != np.shape(vi) or not np.allclose(v0, vi):
                        raise ValueError(f"axis '{n0}' differs across logs")
                z_list.append(zi)
            z = np.array(z_list)
            ts_rel = ts0

    timestamps = None if ts_rel is None else (creation_time + np.asarray(ts_rel))

    return LabberData(
        data=Axis(z_name, z_unit, z),
        axes=[Axis(n, u, v) for n, u, v in axes0],
        comment=comment,
        tags=tags,
        project=project,
        user=user,
        timestamps=timestamps,
        creation_time=creation_time,
    )


def _resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    cand = format_ext(path)
    return cand if os.path.exists(cand) else path


def _all_log_refs(f: h5py.File) -> list[Any]:
    """Root (if it holds Data) plus Log_2, Log_3 ... in sorted order."""
    refs: list[Any] = []
    if "Data" in f:
        refs.append(f)
    log_ns = sorted(
        int(k[4:]) for k in f.keys() if k.startswith("Log_") and k[4:].isdigit()
    )
    for n in log_ns:
        g = f["Log_%d" % n]
        if isinstance(g, h5py.Group) and "Data" in g:
            refs.append(g)
    if not refs:
        raise ValueError("file contains no data ('Data' group not found)")
    return refs


def _read_single_log(f, log):
    """Read one log into an N-D complex array, its axes and timestamps.

    Two storage modes are auto-detected:

    * **scalar** (``vector=False``): z lives in ``Data/Data`` columns
      ``[x, y, w, ..., z_real, z_imag]``; reshaped to ``(..., Ny, Nx)``.
    * **trace** (``vector=True``): z lives in ``Traces/<name>`` with a per-entry
      x-axis; entries may have different lengths.  The inner x-axis is read from
      the trace; outer axes come from ``Data/Data`` step columns.

    Returns ``(z, axes, ts_rel)``: ``axes`` is inner-first
    ``[(name, unit, values), ...]``; ``ts_rel`` is per-entry timestamps relative
    to ``creation_time``.  For trace mode, ``z`` is a list of ragged arrays when
    entry lengths differ.
    """
    z_name, _z_unit = _read_log_label(f, log)
    if "Traces" in log and z_name and z_name in log["Traces"]:
        return _read_trace_log(f, log, z_name)

    D = log["Data"]["Data"][()]  # (Nx, Ncol, Nentries)
    n_x, n_col, n_entry = D.shape
    col_names = [(_decode(n), _decode(i)) for n, i in log["Data"]["Channel names"][()]]
    infos = [i for _n, i in col_names]
    log_complex = "Imaginary" in infos

    # which columns are step axes (everything before the log channel columns)
    n_log_cols = 2 if log_complex else 1
    n_axes = n_col - n_log_cols
    step_dims = [int(v) for v in _data_step_dims(log, n_x, n_entry, n_axes)]

    # axis values + labels/units
    units = _channel_units(f, log)
    axes: list[tuple[str, str, np.ndarray]] = []
    for k in range(n_axes):
        name = col_names[k][0]
        if k == 0:
            vals = D[:, 0, 0]  # inner axis along axis 0
        else:
            # outer axis k: distinct coordinate per "block" of entries.
            # entry e has axis-k index = (e // prod(step_dims[1:k])) % step_dims[k]
            stride = int(np.prod(step_dims[1:k])) if k > 1 else 1
            idx = (np.arange(n_entry) // stride) % step_dims[k]
            # value for each unique index, in order
            vals = np.array([D[0, k, np.argmax(idx == j)] for j in range(step_dims[k])])
        axes.append((name, units.get(name, ""), np.asarray(vals)))

    # complex data, entries flattened with y (axis 1) fastest
    if log_complex:
        zf = D[:, -2, :] + 1j * D[:, -1, :]  # (Nx, Nentries)
    else:
        zf = D[:, -1, :].astype(complex)
    zf = zf.T  # (Nentries, Nx)

    # reshape to natural grid (..., Ny, Nx): outer dims are step_dims[1:] but
    # in z-order they are outermost-first == reversed(step_dims[1:]).
    outer = step_dims[1:][::-1]  # (Nw, ..., Ny)
    z = zf.reshape(tuple(outer) + (n_x,))
    if z.ndim == 1 or (len(outer) == 0):
        z = z.reshape(n_x)  # 1-D -> (Nx,)

    ts_rel = _read_timestamps(log["Data"])
    return z, axes, ts_rel


def _read_trace_log(f, log, z_name):
    """Read a vector log channel stored in Traces/<z_name>.

    ``Traces/<name>`` has shape ``(maxN, ncol, Nentries)`` -- col 0 real,
    col 1 imag (if complex), the next col the explicit x-axis (when
    ``<name>_t0dt`` is ``(0, 0)``).  ``<name>_N`` gives the true length of each
    entry's trace; when lengths differ, ``z`` is returned as a list of arrays.
    """
    g = log["Traces"]
    h5 = g[z_name]
    is_complex = bool(_decode(h5.attrs.get("complex", False)))
    x_name = _decode(h5.attrs.get("x, name", "")) or "Index"
    x_unit = _decode(h5.attrs.get("x, unit", "")) or ""
    raw = h5[()]  # (maxN, ncol, Nentries)
    max_n, n_col, n_entry = raw.shape
    i_x = 2 if is_complex else 1
    has_x_col = i_x <= n_col - 1

    if is_complex:
        zf = raw[:, 0, :] + 1j * raw[:, 1, :]  # (maxN, Nentries)
    else:
        zf = raw[:, 0, :].astype(complex)

    h5_N = g[z_name + "_N"][()]
    t0dt = g[z_name + "_t0dt"][()]

    def _x_for(e, n):
        if has_x_col and t0dt.shape[0] >= 1 and t0dt[0, 0] == 0.0 and t0dt[0, 1] == 0.0:
            return raw[:n, i_x, e]
        row = t0dt[e] if t0dt.shape[0] > e else t0dt[0]
        t0, dt = float(row[0]), float(row[1])
        return t0 + np.arange(n, dtype=float) * dt

    lengths = [int(h5_N[e]) if h5_N.shape[0] > e else max_n for e in range(n_entry)]
    ragged = len(set(lengths)) > 1

    if ragged:
        z = [zf[: lengths[e], e] for e in range(n_entry)]
        x = _x_for(0, lengths[0])  # representative inner axis
    else:
        n = lengths[0] if lengths else max_n
        z = zf[:n, :].T  # (Nentries, n)
        x = _x_for(0, n)

    # outer axes from Data/Data step columns (skip the dummy Step index API)
    axes: list[tuple[str, str, np.ndarray]] = [(x_name, x_unit, np.asarray(x))]
    outer_axes = _read_outer_step_axes(f, log)
    axes.extend(outer_axes)

    if not ragged:
        # reshape (Nentries, n) -> (..., n) by outer dims
        z_arr = np.asarray(z)
        n_inner = z_arr.shape[1]
        outer_dims = [len(a[2]) for a in outer_axes][::-1]  # outermost-first
        if outer_dims:
            z = z_arr.reshape(tuple(outer_dims) + (n_inner,))
        elif z_arr.shape[0] == 1:
            # single entry, no outer axis -> canonical 1-D (n,)
            z = z_arr.reshape(n_inner)
        else:
            # multiple equal-length entries, no outer axis -> stacked (Nentries, n)
            z = z_arr  # already (Nentries, n), no reshape needed

    ts_rel = _read_timestamps(g)
    return z, axes, ts_rel


def _read_outer_step_axes(f, log):
    """Return outer step axes (name, unit, values) from Data/Data, skipping the
    dummy 'Step index API' column.  Used for trace-mode logs."""
    if "Data" not in log:
        return []
    D = log["Data"]["Data"][()]
    col_names = [(_decode(n), _decode(i)) for n, i in log["Data"]["Channel names"][()]]
    units = _channel_units(f, log)
    _n_inner, _n_col, n_entry = D.shape

    # all step columns are scalar-info columns; drop the dummy API channel
    out = []
    for col, (name, _info) in enumerate(col_names):
        if name == _STEP_NAME_API:
            continue
        vals = D[0, col, :]  # one value per entry
        # collapse to unique values in order of first appearance
        uniq = []
        seen = set()
        for v in vals:
            key = round(float(v), 15)
            if key not in seen:
                seen.add(key)
                uniq.append(float(v))
        out.append((name, units.get(name, ""), np.asarray(uniq)))
    return out


def _read_timestamps(group):
    """Return the relative per-entry timestamps from a group's Time stamp ds."""
    if "Time stamp" in group:
        return np.asarray(group["Time stamp"][()], dtype=float)
    return None


def _data_step_dims(log, n_x, n_entry, n_axes):
    """Return step dimensions (inner-first), preferring the stored attr."""
    g = log["Data"]
    if "Step dimensions" in g.attrs:
        dims = [int(v) for v in g.attrs["Step dimensions"]]
        if len(dims) == n_axes and int(np.prod(dims[1:]) or 1) == n_entry:
            return dims
    # fall back: assume each outer column is one axis with all-distinct blocks
    if n_axes <= 1:
        return [n_x]
    # best effort: 2-D layout
    return [n_x, n_entry] + [1] * (n_axes - 2)


def _read_tags(f) -> tuple[list[str], str, str]:
    tags: list[str] = []
    project = user = ""
    if "Tags" in f:
        a = f["Tags"].attrs
        dec = _decode(a.get("Tags"))
        if isinstance(dec, list):
            tags = list(dec)
        proj = _decode(a.get("Project"))
        if isinstance(proj, list) and proj:
            project = proj[0]
        usr = _decode(a.get("User"))
        if isinstance(usr, list) and usr:
            user = usr[0]
    return tags, project or "", user or ""


def _channel_units(f, log) -> dict[str, str]:
    """Map channel name -> physical unit, from the root/log ``Channels`` table."""
    units: dict[str, str] = {}
    src = log if "Channels" in log else f
    if "Channels" in src:
        for row in src["Channels"][()]:
            units[_decode(row["name"])] = _decode(row["unitPhys"]) or ""
    return units


def _read_log_label(f, log) -> tuple[str, str]:
    """Return (z_name, z_unit) of the first log channel."""
    z_name = ""
    src = log if "Log list" in log else f
    if "Log list" in src and len(src["Log list"]):
        z_name = _decode(src["Log list"][()][0]["channel_name"]) or ""
    if not z_name:
        # fall back: the log-instrument channel in Channels
        csrc = log if "Channels" in log else f
        if "Channels" in csrc:
            for row in csrc["Channels"][()]:
                if _decode(row["instrument"]) == _INSTR_LOG:
                    z_name = _decode(row["name"])
                    break
    return z_name, _channel_units(f, log).get(z_name, "")
