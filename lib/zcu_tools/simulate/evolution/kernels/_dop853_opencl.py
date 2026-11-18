"""Engine-owned OpenCL runner for independent DOP853 interaction endpoints."""

from __future__ import annotations

import os
from hashlib import sha256
from threading import Lock
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..._numba_cache import user_cache_root
from ._dop853 import ROW_INITIAL_STEP_NS, Tolerances, tolerances
from ._dop853_data import Models, Segments
from ._dop853_opencl_source import opencl_source

if TYPE_CHECKING:
    import pyopencl as cl

    from .dop853 import EndpointData

# Bound the amount of work in one launch even for a 10 us row or an unusually
# stiff interval. Resume the same controller on device without a knot restart.
# 32 trials is no more work than the 32 complete intervals used in the dense
# 100 ns prototype (~30 ms per launch for 24 d=32 rows on Intel Arc).
_TRIALS_PER_LAUNCH = 32
_ROWS_PER_BATCH = 32


class _Playback(NamedTuple):
    knots: NDArray[np.float64]
    samples: NDArray[np.complex64]
    tones: NDArray[np.float64]
    interval_counts: NDArray[np.int32]


class _Model(NamedTuple):
    energies: NDArray[np.float64]
    operators: NDArray[np.complex64]
    bath: NDArray[np.float32]


def _pack_model(models: Models, key: int, ports: NDArray[np.int64]) -> _Model:
    return _Model(
        models.energies[key],
        np.stack((models.x[key, ports], models.y[key, ports]), axis=1).astype(
            np.complex64
        ),
        np.stack((models.rates[key], models.decay[key])).astype(np.float32),
    )


def _pack_playback(segments: Segments, rows: NDArray[np.int64]) -> _Playback:
    # Admission guarantees both pulses start at this row's origin. Their grids
    # need not match: stop at the union, using each tone's own linear segment.
    spans = segments.sample_offsets[
        segments.envelope[rows, :2][..., None] + np.arange(2)
    ]
    durations = segments.duration[rows]
    nodes = []
    for span, duration in zip(spans, durations, strict=True):
        a, b = span[0]
        c, d = span[1]
        grid = np.union1d(segments.sample_time[a:b], segments.sample_time[c:d])
        nodes.append(np.r_[grid[grid < duration], duration])
    width = max(map(len, nodes))
    knots = np.zeros((len(rows), width))
    samples = np.zeros((len(rows), width, 2), dtype=np.complex64)
    for i, (row, grid) in enumerate(zip(rows, nodes, strict=True)):
        knots[i, : len(grid)] = grid
        for tone, (a, b) in enumerate(spans[i]):
            samples[i, : len(grid), tone] = (
                np.interp(grid, segments.sample_time[a:b], segments.samples[a:b])
                * segments.gain[row, tone]
            )
    tones = np.column_stack((segments.frequency[rows, :2], segments.theta[rows, :2]))
    return _Playback(
        knots, samples, tones, np.array([len(g) - 1 for g in nodes], np.int32)
    )


class OpenCLDOP853:
    """Own the FP64-capable GPU, context and programs borrowed by one Engine.

    Construction loads the optional pyopencl dependency, finds a GPU with
    cl_khr_fp64 and creates a context; unavailable dependencies or devices
    raise RuntimeError. Programs build on demand. The Engine drains workers
    before dropping this owner, which releases the retained OpenCL handles.
    """

    def __init__(self) -> None:
        try:
            import pyopencl as cl
        except ImportError as error:
            raise RuntimeError(
                "OpenCL requires pyopencl (install the gpu extra)"
            ) from error
        try:
            device = next(
                (
                    device
                    for platform in cl.get_platforms()
                    for device in platform.get_devices(device_type=cl.device_type.GPU)
                    if "cl_khr_fp64" in device.extensions.split()
                ),
                None,
            )
        except cl.Error as error:
            raise RuntimeError(
                f"cannot discover an OpenCL GPU with cl_khr_fp64: {error}"
            ) from error
        if device is None:
            raise RuntimeError("OpenCL requires a GPU supporting cl_khr_fp64")
        try:
            self._context = cl.Context([device])
        except cl.Error as error:
            raise RuntimeError(
                f"cannot create an OpenCL context for {device.name}: {error}"
            ) from error
        self._cl = cl
        self._device = device
        self._programs: dict[int, cl.Program] = {}
        self._build_lock = Lock()

    def _width(self, dimension: int) -> int:
        # Use at least a full SIMD wave on Intel Arc even for small matrices:
        # its optimized OpenCL compiler loses lane writes for WG=8 here.
        limit = min(
            128,
            max(32, dimension**2),
            self._device.max_work_group_size,
            self._device.max_work_item_sizes[0],
        )
        return 1 << (limit.bit_length() - 1)

    def _program(self, dimension: int) -> cl.Program:
        with self._build_lock:
            old = self._programs.get(dimension)
            if old is not None:
                return old
            device = self._device
            source = opencl_source()
            options = [f"-DD={dimension}", f"-DWG={self._width(dimension)}"]
            fields = (
                source,
                *options,
                os.environ.get("PYOPENCL_BUILD_OPTIONS", ""),
                device.platform.vendor,
                device.platform.name,
                device.vendor,
                device.name,
                device.version,
                device.driver_version,
            )
            identity = sha256("\0".join(fields).encode()).hexdigest()
            directory = user_cache_root() / "opencl" / "dop853" / identity
            directory.mkdir(parents=True, exist_ok=True)
            program = self._cl.Program(self._context, source).build(
                options=options, devices=[device], cache_dir=str(directory)
            )
            self._programs[dimension] = program
            return program

    def advance(
        self,
        data: EndpointData,
        row_models: tuple[NDArray[np.int64], NDArray[np.int64]],
        states: NDArray[np.complex128],
        budget: float,
    ) -> NDArray[np.complex128]:
        """Return DC-eigenbasis endpoints for pre-admitted independent rows.

        row_models gives valid global row IDs and model indices into data.
        states has shape (rows,D,D) and contains the input in each row's model
        basis; it is not modified. budget is the positive DOP853 local error
        allowance. The caller admits rows with two full-support sampled tones.
        Build, driver and numerical failures raise RuntimeError; no CPU retry.
        Queue, buffers and mutable kernel arguments belong to the calling worker.
        """
        rows, keys = row_models
        result = np.empty_like(states)
        groups, membership = np.unique(
            np.column_stack((keys, data.segments.port[rows, :2])),
            axis=0,
            return_inverse=True,
        )
        controls = tolerances(budget)
        try:
            for group_id, group in enumerate(groups):
                model = _pack_model(data.models, int(group[0]), group[1:])
                members = np.flatnonzero(membership == group_id)
                for start in range(0, len(members), _ROWS_PER_BATCH):
                    batch = members[start : start + _ROWS_PER_BATCH]
                    playback = _pack_playback(data.segments, rows[batch])
                    result[batch] = self._solve(
                        model, playback, states[batch], controls
                    )
        except (self._cl.Error, OSError) as error:
            raise RuntimeError(
                f"DOP853 OpenCL failed on {self._device.name}: {error}"
            ) from error
        return result

    def _solve(
        self,
        model: _Model,
        playback: _Playback,
        states: NDArray[np.complex128],
        controls: Tolerances,
    ) -> NDArray[np.complex128]:
        state = states.astype(np.complex64)
        count, dimension, _ = state.shape
        cl = self._cl
        kernel = cl.Kernel(self._program(dimension), "advance")
        progress = np.zeros((count, 4), dtype=np.int32)
        inputs = (*model, *playback)
        flags = cl.mem_flags
        buffers = [
            cl.Buffer(
                self._context, flags.READ_WRITE | flags.COPY_HOST_PTR, hostbuf=array
            )
            for array in (
                state,
                progress,
                np.zeros(count),
                np.full(count, ROW_INITIAL_STEP_NS, dtype=np.float32),
            )
        ]
        slopes = cl.Buffer(self._context, flags.READ_WRITE, 12 * state.nbytes)
        immutable = [
            cl.Buffer(
                self._context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=array
            )
            for array in inputs
        ]
        width = self._width(dimension)
        with cl.CommandQueue(self._context) as queue:
            while np.any(progress[:, 0] < playback.interval_counts):
                kernel(
                    queue,
                    (count * width,),
                    (width,),
                    buffers[0],
                    slopes,
                    *buffers[1:],
                    *immutable,
                    np.int32(playback.knots.shape[1]),
                    np.float32(controls.rtol),
                    np.float32(controls.atol),
                    np.int32(_TRIALS_PER_LAUNCH),
                )
                cl.enqueue_copy(queue, progress, buffers[1], is_blocking=True)
                if np.any(progress[:, 3]):
                    code = int(progress[progress[:, 3] != 0, 3][0])
                    detail = (
                        "step underflow or iteration limit"
                        if code == 1
                        else "nonfinite solution or error estimate"
                    )
                    raise RuntimeError(f"DOP853 OpenCL {detail}")
            cl.enqueue_copy(queue, state, buffers[0], is_blocking=True)
        duration = playback.knots[np.arange(count), playback.interval_counts]
        return _handoff(state, duration, model.energies)


def _handoff(
    state: NDArray[np.complex64],
    duration: NDArray[np.float64],
    energies: NDArray[np.float64],
) -> NDArray[np.complex128]:
    phase = np.exp(-2j * np.pi * duration[:, None] * energies)
    density = state * phase[:, :, None] * phase[:, None, :].conj()
    # FP64 state accumulation alone left trace errors up to 2.33e-7 because
    # the RHS is still FP32. Projecting in the final frame meets both handoff
    # checks and changed any entry by at most 2.0e-6 on the measured cases.
    density = (density + density.conj().swapaxes(-1, -2)) * 0.5
    traces = np.trace(density, axis1=1, axis2=2).real
    if not np.isfinite(density).all() or np.any(traces <= 0):
        raise RuntimeError(
            "DOP853 OpenCL returned a nonfinite or nonpositive-trace density"
        )
    return density / traces[:, None, None]
