"""Engine-owned OpenCL runner for independent DOP853 interaction endpoints."""

from __future__ import annotations

import os
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..._numba_cache import user_cache_root
from ._dop853 import (
    DP54_TOLERANCE_DIVISOR,
    ROW_INITIAL_STEP_NS,
    Tolerances,
    endpoint_stage_count,
    tolerances,
)
from ._dop853_data import Models, Segments
from ._dop853_opencl_selftest import PROBE_DIMENSION, PROBE_REVISION, endpoint_error
from ._dop853_opencl_source import opencl_source

if TYPE_CHECKING:
    import pyopencl as cl

    from .dop853 import EndpointData

# Aim below both Windows' default 2 s TDR and Linux xe's
# preempt_timeout_us=640000 without relying on either driver setting.
_TARGET_DISPATCH_NS = 100_000_000
# 32 trials took at most the work of 32 intervals in the dense prototype
# (~30 ms for 24 d=32 rows on Intel Arc). Start here before any timing.
_INITIAL_TRIALS = 32
# The runner caps each device batch; dispatch prices work at the same cap.
ROWS_PER_BATCH = 32
# The production probe differed by 2.01e-6 on Intel Arc; 1e-5 leaves room for
# compiler rounding without admitting large deviations from the CPU endpoint.
_PROBE_TOLERANCE = 1e-5


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


class OpenCLBackend:
    """Own one Engine's optional DOP853 runner and its detection outcome."""

    forced: bool
    """True for explicit OpenCL; False for price-based automatic routing."""

    def __init__(self, *, forced: bool) -> None:
        self.forced = forced
        self._runner: OpenCLDOP853 | None = None
        self._detect_lock = Lock()
        self._probe_failure: str | None = None
        self._build_failures: dict[tuple[int, bool], str | None] = {}
        if forced:
            self._detect_device()

    def detect_execution_device(self) -> str:
        """Resolve this Engine's device, or return cpu if auto detection fails."""
        with self._detect_lock:
            self._detect_device()
            runner = self._runner
        return "cpu" if runner is None else runner.execution_device

    def failure(self, dimension: int, *, dp54: bool) -> str | None:
        """Return an observed probe or build failure for this dimension and tableau."""
        return self._probe_failure or self._build_failures.get((dimension, dp54))

    def candidate(self, dimension: int, *, dp54: bool) -> OpenCLDOP853 | None:
        """Detect on the first eligible row and read the chosen variant's failures.

        Workers can reach this together. Detection does not compile unpriced
        point programs. A forced backend has already passed device detection
        and does not consult cached compiler rejections for point programs.
        """
        key = (dimension, dp54)
        if self.failure(dimension, dp54=dp54) is not None:
            return None
        if self.forced or key in self._build_failures:
            return self._runner
        with self._detect_lock:
            self._detect_device()
            runner = self._runner
            if runner is None:
                return None
            if key not in self._build_failures:
                self._build_failures[key] = runner.cached_build_failure(
                    dimension, dp54=dp54
                )
            return runner if self._build_failures[key] is None else None

    def _detect_device(self) -> None:
        if self._runner is not None or self._probe_failure is not None:
            return
        runner = None
        try:
            runner = OpenCLDOP853()
            runner.check_device()
        except (RuntimeError, OSError, ValueError) as error:
            if runner is not None:
                runner.close()
            if self.forced:
                raise
            self._probe_failure = str(error)
        else:
            self._runner = runner

    def prepare(self, dimension: int, *, dp54: bool) -> OpenCLDOP853 | None:
        """Build an auto wave's dimension/tableau variant before running any point.

        Called after workers have joined. A failed auto build disables only
        this variant. A forced runner keeps its build errors in point apply.
        """
        runner = self.candidate(dimension, dp54=dp54)
        if runner is None or self.forced:
            return runner
        failure = runner.compile_variant(dimension, dp54=dp54)
        if failure is not None:
            self._build_failures[dimension, dp54] = failure
            return None
        return runner

    def close(self) -> None:
        """Release the runner and detection state after the Engine joins workers."""
        if self._runner is not None:
            self._runner.close()
            self._runner = None
        self._build_failures.clear()
        self._probe_failure = None


class OpenCLDOP853:
    """Own the FP64-capable GPU, context and programs borrowed by one Engine.

    Construction loads the optional pyopencl dependency, finds a GPU with
    cl_khr_fp64 and creates a context; unavailable dependencies or devices
    raise RuntimeError. Programs build on demand. The Engine drains workers
    before closing this owner and releasing its retained OpenCL handles.
    """

    execution_device: str
    """GPU identity: opencl:<name>/<driver>/<kernel source hash>/<pyopencl version>."""

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
                f"cannot discover an OpenCL GPU with cl_khr_fp64 (code {error.code}): {error}"
            ) from error
        if device is None:
            raise RuntimeError("OpenCL requires a GPU supporting cl_khr_fp64")
        try:
            self._context = cl.Context([device])
        except cl.Error as error:
            raise RuntimeError(
                f"cannot create an OpenCL context for {device.name} (code {error.code}): {error}"
            ) from error
        self._cl = cl
        self._device = device
        self._programs: dict[tuple[int, bool], cl.Program] = {}
        self._build_lock = Lock()
        self._source = opencl_source()
        kernel_version = sha256(self._source.encode()).hexdigest()
        self.execution_device = (
            f"opencl:{device.name}/{device.driver_version}/{kernel_version}/"
            f"{cl.VERSION_TEXT}"
        )

    def close(self) -> None:
        """Drop the retained programs and context once, after the last borrow."""
        # PyOpenCL releases these native handles when their wrappers are dropped.
        self._programs.clear()
        del self._context

    def check_device(self) -> None:
        """Validate the device using the persisted CPU/GPU comparison.

        A matching cached result avoids both compilation and the numerical
        probe. A build or self-test failure, including a cached one, raises.
        """
        # Both probes share one receipt. The common source hashes both tableaux.
        _, program = self._program_location(PROBE_DIMENSION, dp54=False)
        name = f"{program.name}-probe-{PROBE_REVISION}-{_PROBE_TOLERANCE:g}.txt"
        path = program.parent / "detection" / name
        record = _read_detection(path)
        if record is None:
            record = self._probe()
            # Persistence saves a later probe; it cannot change this result.
            with suppress(OSError):
                _write_detection(path, record)
        if record.startswith("failed\n"):
            raise RuntimeError(record.removeprefix("failed\n"))

    def _build_failure_path(self, dimension: int, *, dp54: bool) -> Path:
        _, program = self._program_location(dimension, dp54=dp54)
        return program.parent / "detection" / f"{program.name}-build.txt"

    def cached_build_failure(self, dimension: int, *, dp54: bool) -> str | None:
        """Read a known compiler rejection for this dimension and tableau."""
        try:
            record = _read_detection(self._build_failure_path(dimension, dp54=dp54))
        except (OSError, ValueError) as error:
            return f"DOP853 OpenCL dimension {dimension} build cache failed: {error}"
        if record is not None and record.startswith("failed\n"):
            return record.removeprefix("failed\n")
        return None

    def compile_variant(self, dimension: int, *, dp54: bool) -> str | None:
        """Build the chosen variant for auto and persist any rejection reason."""
        try:
            self._program(dimension, dp54=dp54)
        except (self._cl.Error, OSError, RuntimeError, ValueError) as error:
            method = "DP5(4)" if dp54 else "DOP853"
            reason = (
                f"DOP853 OpenCL build failed at dimension {dimension} "
                f"({method}) on {self._device.name}: {error}"
            )
            try:
                _write_detection(
                    self._build_failure_path(dimension, dp54=dp54), f"failed\n{reason}"
                )
            except (OSError, ValueError) as cache_error:
                return f"{reason} (could not store detection result: {cache_error})"
            return reason
        return None

    def _probe(self) -> str:
        device = self._device.name
        for dp54, method in ((False, "DOP853"), (True, "DP5(4)")):
            try:
                self._program(PROBE_DIMENSION, dp54=dp54)
            except (self._cl.Error, OSError, RuntimeError, ValueError) as error:
                return f"failed\nDOP853 OpenCL probe build failed on {device} ({method}): {error}"
            try:
                delta = endpoint_error(self, dp54=dp54)
            except (
                self._cl.Error,
                ArithmeticError,
                OSError,
                RuntimeError,
                ValueError,
            ) as error:
                return f"failed\nDOP853 OpenCL self-test failed on {device} ({method}): {error}"
            if not np.isfinite(delta) or delta > _PROBE_TOLERANCE:
                return (
                    f"failed\nDOP853 OpenCL self-test failed on {device} ({method}): "
                    f"max CPU–GPU density error {delta:.3g} exceeds {_PROBE_TOLERANCE:.3g}"
                )
        return "passed\n"

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

    def _program_location(
        self, dimension: int, *, dp54: bool
    ) -> tuple[list[str], Path]:
        device = self._device
        options = [
            f"-DD={dimension}",
            f"-DWG={self._width(dimension)}",
            f"-DDP54={int(dp54)}",
        ]
        fields = (
            self._source,
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
        return options, user_cache_root() / "opencl" / "dop853" / identity

    def _program(self, dimension: int, *, dp54: bool) -> cl.Program:
        with self._build_lock:
            key = (dimension, dp54)
            old = self._programs.get(key)
            if old is not None:
                return old
            options, directory = self._program_location(dimension, dp54=dp54)
            directory.mkdir(parents=True, exist_ok=True)
            try:
                program = self._cl.Program(self._context, self._source).build(
                    options=options, devices=[self._device], cache_dir=str(directory)
                )
            except self._cl.Error as error:
                raise RuntimeError(f"OpenCL code {error.code}: {error}") from error
            self._programs[key] = program
            return program

    def advance(
        self,
        data: EndpointData,
        row_models: tuple[NDArray[np.int64], NDArray[np.int64]],
        states: NDArray[np.complex128],
        budget: float,
        *,
        dp54: bool,
    ) -> NDArray[np.complex128]:
        """Return DC-eigenbasis endpoints for pre-admitted independent rows.

        row_models gives valid global row IDs and model indices into data.
        states has shape (rows,D,D) and contains the input in each row's model
        basis; it is not modified. dp54 selects DP5(4) with one third of the
        local tolerances derived from the positive budget. Otherwise use DOP853.
        The caller admits rows with two full-support sampled tones.
        Build, driver and numerical failures raise RuntimeError; no CPU retry.
        Queue, buffers and mutable kernel arguments belong to this call.
        """
        rows, keys = row_models
        result = np.empty_like(states)
        groups, membership = np.unique(
            np.column_stack((keys, data.segments.port[rows, :2])),
            axis=0,
            return_inverse=True,
        )
        controls = tolerances(budget)
        if dp54:
            controls = controls._replace(
                rtol=controls.rtol / DP54_TOLERANCE_DIVISOR,
                atol=controls.atol / DP54_TOLERANCE_DIVISOR,
            )
        try:
            for group_id, group in enumerate(groups):
                model = _pack_model(data.models, int(group[0]), group[1:])
                members = np.flatnonzero(membership == group_id)
                for start in range(0, len(members), ROWS_PER_BATCH):
                    batch = members[start : start + ROWS_PER_BATCH]
                    playback = _pack_playback(data.segments, rows[batch])
                    result[batch] = self._solve(
                        model, playback, states[batch], controls, dp54=dp54
                    )
        except self._cl.Error as error:
            raise RuntimeError(
                f"DOP853 OpenCL failed on {self._device.name} (code {error.code}): {error}"
            ) from error
        except OSError as error:
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
        *,
        dp54: bool,
    ) -> NDArray[np.complex128]:
        state = states.astype(np.complex64)
        count, dimension, _ = state.shape
        cl = self._cl
        kernel = cl.Kernel(self._program(dimension, dp54=dp54), "advance")
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
        slopes = cl.Buffer(
            self._context,
            flags.READ_WRITE,
            endpoint_stage_count(dp54=dp54) * state.nbytes,
        )
        immutable = [
            cl.Buffer(
                self._context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=array
            )
            for array in inputs
        ]
        width = self._width(dimension)
        active = np.arange(count, dtype=np.int32)
        row_map = cl.Buffer(
            self._context, flags.READ_ONLY | flags.COPY_HOST_PTR, hostbuf=active
        )
        trials = _INITIAL_TRIALS
        with cl.CommandQueue(
            self._context, properties=cl.command_queue_properties.PROFILING_ENABLE
        ) as queue:
            while active.size:
                event = kernel(
                    queue,
                    (active.size * width,),
                    (width,),
                    buffers[0],
                    slopes,
                    *buffers[1:],
                    *immutable,
                    np.int32(playback.knots.shape[1]),
                    np.float32(controls.rtol),
                    np.float32(controls.atol),
                    np.int32(trials),
                    row_map,
                )
                try:
                    cl.enqueue_copy(queue, progress, buffers[1], is_blocking=True)
                except cl.Error as error:
                    status = None
                    with suppress(cl.Error):
                        status = event.command_execution_status
                    raise RuntimeError(
                        f"DOP853 OpenCL failed on {self._device.name} "
                        f"(code {error.code}, advance execution status {status}): {error}"
                    ) from error
                if np.any(progress[:, 3]):
                    code = int(progress[progress[:, 3] != 0, 3][0])
                    detail = (
                        "step underflow or iteration limit"
                        if code == 1
                        else "nonfinite solution or error estimate"
                    )
                    raise RuntimeError(f"DOP853 OpenCL {detail}")
                remaining = active[
                    progress[active, 0] < playback.interval_counts[active]
                ]
                if remaining.size and remaining.size != active.size:
                    cl.enqueue_copy(queue, row_map, remaining, is_blocking=True)
                active = remaining
                # Size the next submission from measured device time. A trial
                # cap also bounds the number of intervals, without letting one
                # stiff interval run unbounded. Raise it by at most 2x at once.
                elapsed_ns = max(1, event.profile.end - event.profile.start)
                trials = max(
                    1, min(2 * trials, trials * _TARGET_DISPATCH_NS // elapsed_ns)
                )
            cl.enqueue_copy(queue, state, buffers[0], is_blocking=True)
        duration = playback.knots[np.arange(count), playback.interval_counts]
        return _handoff(state, duration, model.energies)


def _read_detection(path: Path) -> str | None:
    try:
        record = path.read_text(encoding="utf-8")
    except (FileNotFoundError, UnicodeError):
        return None
    if record == "passed\n" or (record.startswith("failed\n") and record[7:]):
        return record
    return None


def _write_detection(path: Path, record: str) -> None:
    # Readers in other processes must never pick up a half-written result.
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temp = Path(stream.name)
            stream.write(record)
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


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
