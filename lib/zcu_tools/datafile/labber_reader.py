"""Read Labber uniform and trace logs, including strict grouped-v2 bookkeeping.

Path loading returns detached data and closes its file. Log readers borrow open
HDF5 handles without closing or mutating them; callers retain their lifetime.
Shared Labber labels and attribute conversion belong to labber_format.
"""

from __future__ import annotations

import os

import h5py
import numpy as np

from .labber_format import (
    API_STEP_NAME,
    LOG_INSTRUMENT,
    STEP_INSTRUMENT,
    decode_labber_attribute,
)
from .models import Axis, LabberData, LabberMetadata
from .paths import format_ext

__all__ = ["load_labber_data"]


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
    path = resolve_labber_path(path)

    with h5py.File(path, "r") as f:
        if "zcu_tools.grouped_dataset_version" in f.attrs:
            raise ValueError(
                "file is a grouped Labber dataset; use load_grouped_labber_data"
            )

        comment = decode_labber_attribute(f.attrs.get("comment", "")) or ""
        tags, project, user = read_labber_tags(f)
        creation_time = float(f.attrs.get("creation_time", 0.0) or 0.0)

        logs = labber_log_refs(f)
        z0, axes0, ts0 = read_single_log(f, logs[0])
        z_name, z_unit = read_log_label(f, logs[0])

        if len(logs) == 1:
            z = z0
            ts_rel = ts0
        else:
            z_list = [z0]
            for log in logs[1:]:
                zi, axes_i, _tsi = read_single_log(f, log)
                if len(axes_i) != len(axes0):
                    raise ValueError("logs have different number of axes")
                for (n0, _u0, v0), (_ni, _ui, vi) in zip(axes0, axes_i, strict=False):
                    if np.shape(v0) != np.shape(vi) or not np.allclose(v0, vi):
                        raise ValueError(f"axis '{n0}' differs across logs")
                z_list.append(zi)
            z = np.array(z_list)
            ts_rel = ts0

    timestamps = None if ts_rel is None else (creation_time + np.asarray(ts_rel))

    return LabberData(
        data=Axis(z_name, z_unit, z),
        axes=[Axis(n, u, v) for n, u, v in axes0],
        metadata=LabberMetadata(
            comment=comment,
            tags=tags,
            project=project,
            user=user,
            creation_time=creation_time,
        ),
        timestamps=timestamps,
    )


def resolve_labber_path(path: str) -> str:
    """Choose an existing exact path, then its .hdf5-normalized alternative.

    Return path unchanged when neither exists, leaving missing-file failures
    to the reader. This only checks existence and never opens or changes files.
    """
    if os.path.exists(path):
        return path
    cand = format_ext(path)
    return cand if os.path.exists(cand) else path


def labber_log_refs(f: h5py.File) -> list[h5py.File | h5py.Group]:
    """List borrowed logs: root if it has Data, then numbered Log_N groups.

    Numbered groups are sorted numerically and must contain Data. Raise
    ValueError when no logs exist; HDF5 failures propagate. No data is copied
    or changed. Returned handles remain valid only while caller-owned f is open.
    """
    refs: list[h5py.File | h5py.Group] = []
    if "Data" in f:
        refs.append(f)
    log_ns = sorted(
        int(k[4:])
        # tolerated: h5py's direct iterator has nullable names; keys() avoids new type errors.
        for k in f.keys()  # noqa: SIM118
        if k.startswith("Log_") and k[4:].isdigit()
    )
    for n in log_ns:
        g = f["Log_%d" % n]
        if isinstance(g, h5py.Group) and "Data" in g:
            refs.append(g)
    if not refs:
        raise ValueError("file contains no data ('Data' group not found)")
    return refs


def read_single_log(f, log):
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
    entry lengths differ. f is the borrowed root for fallback bookkeeping and
    log is a borrowed root or Log_N group. Results are detached from HDF5;
    neither input is mutated or closed. Missing or malformed bookkeeping
    propagates lookup/shape errors and HDF5 failures to the caller.
    """
    z_name, _z_unit = read_log_label(f, log)
    if "Traces" in log and z_name and z_name in log["Traces"]:
        return _read_trace_log(f, log, z_name)

    D = log["Data"]["Data"][()]  # (Nx, Ncol, Nentries)
    n_x, n_col, n_entry = D.shape
    col_names = [
        (decode_labber_attribute(n), decode_labber_attribute(i))
        for n, i in log["Data"]["Channel names"][()]
    ]
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
    zf = (
        D[:, -2, :] + 1j * D[:, -1, :] if log_complex else D[:, -1, :].astype(complex)
    )  # (Nx, Nentries)
    zf = zf.T  # (Nentries, Nx)

    # reshape to natural grid (..., Ny, Nx): outer dims are step_dims[1:] but
    # in z-order they are outermost-first == reversed(step_dims[1:]).
    outer = step_dims[1:][::-1]  # (Nw, ..., Ny)
    z = zf.reshape(tuple(outer) + (n_x,))
    if z.ndim == 1 or (len(outer) == 0):
        z = z.reshape(n_x)  # 1-D -> (Nx,)

    ts_rel = _read_timestamps(log["Data"])
    return z, axes, ts_rel


def _read_v2_log_channel_bookkeeping(
    source: h5py.File | h5py.Group, channel_source: h5py.File | h5py.Group
) -> tuple[list[str], h5py.Dataset, h5py.Group]:
    log_list = source.get("Log list")
    if not isinstance(log_list, h5py.Dataset):
        raise ValueError("grouped v2 log is missing Log list")
    channel_names = [
        decode_labber_attribute(row["channel_name"]) or "" for row in log_list[()]
    ]
    if not channel_names or any(not name for name in channel_names):
        raise ValueError("grouped v2 log channels must have non-empty labels")
    if len(set(channel_names)) != len(channel_names):
        raise ValueError("grouped v2 log channel labels must be unique")

    channels_dataset = channel_source.get("Channels")
    if not isinstance(channels_dataset, h5py.Dataset):
        raise ValueError("grouped v2 log is missing Channels")
    configured_channels = [
        decode_labber_attribute(row["name"]) or ""
        for row in channels_dataset[()]
        if decode_labber_attribute(row["instrument"]) == LOG_INSTRUMENT
    ]
    if configured_channels != channel_names:
        raise ValueError("grouped v2 Labber channel bookkeeping is inconsistent")

    instrument_config = source.get("Instrument config")
    if not isinstance(instrument_config, h5py.Group):
        raise ValueError("grouped v2 log is missing Instrument config")
    log_instrument = instrument_config.get(LOG_INSTRUMENT)
    if not isinstance(log_instrument, h5py.Group):
        raise ValueError("grouped v2 log is missing log-channel instrument config")
    configured_defaults = [
        name
        for name in log_instrument.attrs
        if isinstance(name, str)
        and name != "Installed options"
        and not name.startswith("___")
    ]
    if set(configured_defaults) != set(channel_names):
        raise ValueError("grouped v2 Labber channel bookkeeping is inconsistent")
    return channel_names, channels_dataset, instrument_config


def _require_v2_step_bookkeeping(
    source: h5py.File | h5py.Group,
    channels_dataset: h5py.Dataset,
    instrument_config: h5py.Group,
    step_names: list[str],
) -> None:
    """Require consistent ordered step names across all Labber metadata tables."""
    configured_steps = [
        decode_labber_attribute(row["name"]) or ""
        for row in channels_dataset[()]
        if decode_labber_attribute(row["instrument"]) == STEP_INSTRUMENT
    ]
    step_list = source.get("Step list")
    step_config = source.get("Step config")
    step_instrument = instrument_config.get(STEP_INSTRUMENT)
    if (
        configured_steps != step_names
        or not isinstance(step_list, h5py.Dataset)
        or [decode_labber_attribute(row["channel_name"]) or "" for row in step_list[()]]
        != step_names
        or not isinstance(step_config, h5py.Group)
        or set(step_config) != set(step_names)
        or not isinstance(step_instrument, h5py.Group)
        or {
            name
            for name in step_instrument.attrs
            if isinstance(name, str) and name != "Installed options"
        }
        != set(step_names)
    ):
        raise ValueError("grouped v2 step-channel bookkeeping is inconsistent")


def _decode_v2_step_grid(
    data: np.ndarray, step_dims: list[int], step_names: list[str], units: dict[str, str]
) -> list[tuple[str, str, np.ndarray]]:
    """Decode inner-first axes and require every stored coordinate to match the grid."""
    n_x, _n_col, n_entry = data.shape
    n_axes = len(step_names)
    axes: list[tuple[str, str, np.ndarray]] = []
    outer_strides = np.cumprod([1, *step_dims[1:-1]])
    for k, name in enumerate(step_names):
        values = (
            data[:, 0, 0]
            if k == 0
            else data[0, k, outer_strides[k - 1] * np.arange(step_dims[k])]
        )
        axes.append((name, units.get(name, ""), np.asarray(values)))

    expected_steps = np.zeros((n_x, n_axes, n_entry), dtype=float)
    expected_steps[:, 0, :] = axes[0][2][:, None]
    for col in range(n_axes - 1, 0, -1):
        indices = (np.arange(n_entry) // outer_strides[col - 1]) % step_dims[col]
        expected_steps[:, col, :] = axes[col][2][indices][None, :]
    if not np.array_equal(data[:, :n_axes, :], expected_steps, equal_nan=True):
        raise ValueError("grouped v2 step-coordinate columns do not match the grid")

    return axes


def read_uniform_multi_channel_log(
    f: h5py.File, log: h5py.File | h5py.Group
) -> tuple[
    dict[str, tuple[str, np.ndarray]],
    list[tuple[str, str, np.ndarray]],
    np.ndarray | None,
]:
    """Read a borrowed scalar log with strict grouped-v2 Labber bookkeeping.

    f is the open root for fallback metadata; log is a root or Log_N group.
    Return channel-name-to-(unit, values), inner-first (name, unit, values)
    axes, and per-entry timestamps relative to creation_time, or None.
    Results are detached arrays and inputs are not mutated or closed.
    Unsupported traces or inconsistent/missing bookkeeping raise ValueError;
    conversion, lookup and HDF5 errors propagate.
    """
    if "Traces" in log:
        raise ValueError("grouped v2 does not support vector or trace log channels")

    source = log if "Log list" in log else f
    channel_source = log if "Channels" in log else f
    channel_names, channels_dataset, instrument_config = (
        _read_v2_log_channel_bookkeeping(source, channel_source)
    )

    data_group = log.get("Data")
    if not isinstance(data_group, h5py.Group):
        raise ValueError("grouped v2 log is missing Data group")
    data_dataset = data_group.get("Data")
    channel_names_dataset = data_group.get("Channel names")
    if not isinstance(data_dataset, h5py.Dataset) or not isinstance(
        channel_names_dataset, h5py.Dataset
    ):
        raise ValueError("grouped v2 Data group is incomplete")
    data = np.asarray(data_dataset[()])
    n_x, n_col, n_entry = data.shape
    columns = [
        (decode_labber_attribute(name) or "", decode_labber_attribute(info) or "")
        for name, info in channel_names_dataset[()]
    ]
    n_axes = n_col - 2 * len(channel_names)
    if n_axes < 1:
        raise ValueError("grouped v2 requires at least one step axis")
    expected_log_columns = [
        item for name in channel_names for item in ((name, "Real"), (name, "Imaginary"))
    ]
    if columns[n_axes:] != expected_log_columns:
        raise ValueError("grouped v2 data columns do not match declared log channels")
    if any(info for _name, info in columns[:n_axes]):
        raise ValueError("grouped v2 step-channel columns have invalid bookkeeping")

    step_dims = _read_strict_v2_step_dimensions(
        log, data_group, n_x=n_x, n_entry=n_entry, n_axes=n_axes
    )
    step_names = [name for name, _info in columns[:n_axes]]
    _require_v2_step_bookkeeping(
        source, channels_dataset, instrument_config, step_names
    )

    # Decode inner-first axes, then verify every stored coordinate against them.
    units = _channel_units(f, log)
    axes = _decode_v2_step_grid(data, step_dims, step_names, units)

    values_by_channel: dict[str, tuple[str, np.ndarray]] = {}
    for index, name in enumerate(channel_names):
        col = n_axes + 2 * index
        flat = (data[:, col, :] + 1j * data[:, col + 1, :]).T
        values = flat.reshape(tuple(step_dims[::-1]))
        values_by_channel[name] = (units.get(name, ""), values)

    return values_by_channel, axes, _read_timestamps(data_group)


def _read_strict_v2_step_dimensions(
    log: h5py.File | h5py.Group,
    data_group: h5py.Group,
    *,
    n_x: int,
    n_entry: int,
    n_axes: int,
) -> list[int]:
    root_raw = log.attrs.get("Step dimensions")
    data_raw = data_group.attrs.get("Step dimensions")
    if root_raw is None or data_raw is None:
        raise ValueError("grouped v2 is missing Step dimensions")
    root_values = np.asarray(root_raw)
    data_values = np.asarray(data_raw)
    try:
        root_dims = root_values.astype(np.int64)
        data_dims = data_values.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError("grouped v2 has invalid Step dimensions") from exc
    if (
        root_values.ndim != 1
        or data_values.ndim != 1
        or len(root_dims) != n_axes
        or not np.array_equal(root_values, root_dims)
        or not np.array_equal(data_values, data_dims)
        or not np.array_equal(root_dims, data_dims)
        or np.any(root_dims <= 0)
        or int(root_dims[0]) != n_x
        or int(np.prod(root_dims[1:])) != n_entry
    ):
        raise ValueError("grouped v2 has inconsistent Step dimensions")
    return [int(value) for value in root_dims]


def _read_trace_log(f, log, z_name):
    """Read a vector log channel stored in Traces/<z_name>.

    ``Traces/<name>`` has shape ``(maxN, ncol, Nentries)`` -- col 0 real,
    col 1 imag (if complex), the next col the explicit x-axis (when
    ``<name>_t0dt`` is ``(0, 0)``).  ``<name>_N`` gives the true length of each
    entry's trace; when lengths differ, ``z`` is returned as a list of arrays.
    """
    g = log["Traces"]
    h5 = g[z_name]
    is_complex = bool(decode_labber_attribute(h5.attrs.get("complex", False)))
    x_name = decode_labber_attribute(h5.attrs.get("x, name", "")) or "Index"
    x_unit = decode_labber_attribute(h5.attrs.get("x, unit", "")) or ""
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
    dummy 'Step index API' column.  Used for trace-mode logs.

    Labber trace logs store one scalar step coordinate per entry. We reconstruct
    axes by first appearance after rounding to 15 decimals, so intentional
    repeated coordinates such as bidirectional sweeps cannot be represented as
    distinct axis points in this compatibility layout.
    """
    if "Data" not in log:
        return []
    D = log["Data"]["Data"][()]
    col_names = [
        (decode_labber_attribute(n), decode_labber_attribute(i))
        for n, i in log["Data"]["Channel names"][()]
    ]
    units = _channel_units(f, log)
    _n_inner, _n_col, _n_entry = D.shape

    # all step columns are scalar-info columns; drop the dummy API channel
    out = []
    for col, (name, _info) in enumerate(col_names):
        if name == API_STEP_NAME:
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


def read_labber_tags(f: h5py.File) -> tuple[list[str], str, str]:
    """Read tags, project and user from a borrowed open Labber root.

    Return a new tags list and text values, defaulting to empty values when
    absent. The file is not changed or closed. Invalid UTF-8 and HDF5 errors
    propagate to the caller.
    """
    tags: list[str] = []
    project = user = ""
    if "Tags" in f:
        a = f["Tags"].attrs
        dec = decode_labber_attribute(a.get("Tags"))
        if isinstance(dec, list):
            tags = list(dec)
        proj = decode_labber_attribute(a.get("Project"))
        if isinstance(proj, list) and proj:
            project = proj[0]
        usr = decode_labber_attribute(a.get("User"))
        if isinstance(usr, list) and usr:
            user = usr[0]
    return tags, project or "", user or ""


def _channel_units(f, log) -> dict[str, str]:
    """Map channel name -> physical unit, from the root/log ``Channels`` table."""
    units: dict[str, str] = {}
    src = log if "Channels" in log else f
    if "Channels" in src:
        for row in src["Channels"][()]:
            units[decode_labber_attribute(row["name"])] = (
                decode_labber_attribute(row["unitPhys"]) or ""
            )
    return units


def read_log_label(f, log) -> tuple[str, str]:
    """Read the first log channel's (name, unit) from borrowed open handles.

    Use log bookkeeping when present, otherwise root f. Missing labels/units
    are empty strings. Results are detached text and handles are not changed
    or closed. Malformed bookkeeping, decoding and HDF5 errors propagate.
    """
    z_name = ""
    src = log if "Log list" in log else f
    if "Log list" in src and len(src["Log list"]):
        z_name = decode_labber_attribute(src["Log list"][()][0]["channel_name"]) or ""
    if not z_name:
        # fall back: the log-instrument channel in Channels
        csrc = log if "Channels" in log else f
        if "Channels" in csrc:
            for row in csrc["Channels"][()]:
                if decode_labber_attribute(row["instrument"]) == LOG_INSTRUMENT:
                    z_name = decode_labber_attribute(row["name"])
                    break
    return z_name, _channel_units(f, log).get(z_name, "")
