"""Engine-owned OpenCL runner for independent DOP853 interaction endpoints."""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import NDArray

from ..._numba_cache import user_cache_root
from ._dop853 import ROW_INITIAL_STEP_NS, Tolerances, tolerances
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

    def __init__(self, *, forced: bool) -> None:
        self.forced = forced
        self._runner = OpenCLDOP853() if forced else None
        self._detect_lock = Lock()
        self._used_dimensions: set[int] = set()
        self._probe_failure: str | None = None
        self._build_failures: dict[int, str | None] = {}

    def failure(self, dimension: int) -> str | None:
        """Return the device probe or dimension build failure already observed."""
        return self._probe_failure or self._build_failures.get(dimension)

    def candidate(self, dimension: int) -> OpenCLDOP853 | None:
        """Detect on the first eligible row and read recorded variant failures.

        Workers can reach this together. Detection does not compile unpriced
        point programs. A forced backend skips all auto availability checks.
        """
        if self.failure(dimension) is not None:
            return None
        if self.forced or dimension in self._build_failures:
            return self._runner
        with self._detect_lock:
            self._detect_device()
            runner = self._runner
            if runner is None:
                return None
            if dimension not in self._build_failures:
                self._build_failures[dimension] = runner.cached_build_failure(dimension)
            return runner if self._build_failures[dimension] is None else None

    def _detect_device(self) -> None:
        if self._runner is not None or self._probe_failure is not None:
            return
        try:
            runner = OpenCLDOP853()
            runner.check_device()
        except (RuntimeError, OSError, ValueError) as error:
            self._probe_failure = str(error)
        else:
            self._runner = runner

    def prepare(self, dimension: int) -> OpenCLDOP853 | None:
        """Build an auto wave's selected variant before running any point.

        Called after workers have joined. A failed auto build disables only
        this dimension. A forced runner keeps its build errors in point apply.
        """
        runner = self.candidate(dimension)
        if runner is None or self.forced:
            return runner
        failure = runner.compile_variant(dimension)
        if failure is not None:
            self._build_failures[dimension] = failure
            return None
        return runner

    def used_dimension(self, dimension: int) -> bool:
        """Whether this Engine has completed a GPU evolve at dimension."""
        return dimension in self._used_dimensions

    def record_use(self, dimension: int) -> bool:
        """Record a successful GPU evolve; return True on first use of dimension."""
        if self.used_dimension(dimension):
            return False
        self._used_dimensions.add(dimension)
        return True


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
        self._source = opencl_source()

    def check_device(self) -> None:
        """Validate automatic admission using the persisted CPU/GPU comparison.

        A matching cached result avoids both compilation and the numerical
        probe. A build or self-test failure, including a cached one, raises.
        """
        _, program = self._program_location(PROBE_DIMENSION)
        name = f"{program.name}-probe-{PROBE_REVISION}-{_PROBE_TOLERANCE:g}.txt"
        path = program.parent / "detection" / name
        record = _read_detection(path)
        if record is None:
            record = self._probe()
            _write_detection(path, record)
        if record.startswith("failed\n"):
            raise RuntimeError(record.removeprefix("failed\n"))

    def _build_failure_path(self, dimension: int) -> Path:
        _, program = self._program_location(dimension)
        return program.parent / "detection" / f"{program.name}-build.txt"

    def cached_build_failure(self, dimension: int) -> str | None:
        """Read a known compiler rejection for this variant, without compiling."""
        try:
            record = _read_detection(self._build_failure_path(dimension))
        except (OSError, ValueError) as error:
            return f"DOP853 OpenCL dimension {dimension} build cache failed: {error}"
        if record is not None and record.startswith("failed\n"):
            return record.removeprefix("failed\n")
        return None

    def compile_variant(self, dimension: int) -> str | None:
        """Build for automatic routing and return a persisted reason on failure."""
        try:
            self._program(dimension)
        except (self._cl.Error, OSError, RuntimeError, ValueError) as error:
            reason = (
                f"DOP853 OpenCL build failed at dimension {dimension} "
                f"on {self._device.name}: {error}"
            )
            try:
                _write_detection(
                    self._build_failure_path(dimension), f"failed\n{reason}"
                )
            except (OSError, ValueError) as cache_error:
                return f"{reason} (could not store detection result: {cache_error})"
            return reason
        return None

    def _probe(self) -> str:
        device = self._device.name
        try:
            self._program(PROBE_DIMENSION)
        except (self._cl.Error, OSError, RuntimeError, ValueError) as error:
            return f"failed\nDOP853 OpenCL probe build failed on {device}: {error}"
        try:
            error = endpoint_error(self)
        except (
            self._cl.Error,
            ArithmeticError,
            OSError,
            RuntimeError,
            ValueError,
        ) as error:
            return f"failed\nDOP853 OpenCL self-test failed on {device}: {error}"
        if not np.isfinite(error) or error > _PROBE_TOLERANCE:
            return (
                f"failed\nDOP853 OpenCL self-test failed on {device}: "
                f"max CPU–GPU density error {error:.3g} exceeds {_PROBE_TOLERANCE:.3g}"
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

    def _program_location(self, dimension: int) -> tuple[list[str], Path]:
        device = self._device
        options = [f"-DD={dimension}", f"-DWG={self._width(dimension)}"]
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

    def _program(self, dimension: int) -> cl.Program:
        with self._build_lock:
            old = self._programs.get(dimension)
            if old is not None:
                return old
            options, directory = self._program_location(dimension)
            directory.mkdir(parents=True, exist_ok=True)
            program = self._cl.Program(self._context, self._source).build(
                options=options, devices=[self._device], cache_dir=str(directory)
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
        try:
            for group_id, group in enumerate(groups):
                model = _pack_model(data.models, int(group[0]), group[1:])
                members = np.flatnonzero(membership == group_id)
                for start in range(0, len(members), ROWS_PER_BATCH):
                    batch = members[start : start + ROWS_PER_BATCH]
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
                cl.enqueue_copy(queue, progress, buffers[1], is_blocking=True)
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
