"""Pure graph scoring with known synthetic charges or limited T6 estimates."""

from dataclasses import dataclass
from heapq import heappop, heappush

from .cache import CacheEntry, CacheNamespace
from .declarations import AlgorithmIdentity, RepresentationTransition
from .kernel import SegmentTable
from .profiles import FamilyProfile, OneCoreProfile
from .shadow import ShadowCandidate, ShadowRequest, ShadowScore, ShadowTask


@dataclass(frozen=True)
class BuildKey:
    """Complete model build partition, including declared algorithm and representation."""

    identity: AlgorithmIdentity
    reuse_namespace: str
    namespace: CacheNamespace
    representation: RepresentationTransition
    prepare_key: bytes


def build_key(candidate: ShadowCandidate) -> BuildKey:
    """Return the declared build partition; never infer validity from this key."""
    declaration = candidate.declaration
    return BuildKey(
        declaration.identity,
        declaration.reuse.namespace,
        candidate.namespace,
        declaration.transition,
        candidate.prepare_key,
    )


def _known(value: float | None) -> float:
    if value is None:
        raise ValueError("shadow cost must be known")
    return value


class _Graph:
    def __init__(self, profile: OneCoreProfile, inventory: tuple[CacheEntry, ...]):
        self.profile = profile
        self.inventory = inventory
        self.tasks: list[ShadowTask] = []
        self.builds: dict[BuildKey, str] = {}
        self.bindings: dict[tuple[BuildKey, bytes], str] = {}
        self.hits: dict[int, CacheEntry] = {}
        self.acquisitions: dict[int, str] = {}
        self.resident_bytes = 0
        self.binding_bytes = 0
        self.task_bytes = 0
        self.memory_known = True

    def account(self, value: int | None) -> int:
        """Accumulate known charges without reporting an unknown total as zero."""
        if value is None:
            self.memory_known = False
            return 0
        return value

    def add(
        self,
        stage: str,
        name: str,
        dependencies: tuple[str, ...],
        seconds: float,
        candidate: ShadowCandidate,
        estimate: FamilyProfile,
    ) -> str:
        task_id = f"{stage}:{name}"
        overhead = _known(estimate.stages.task_overhead)
        self.tasks.append(
            ShadowTask(
                task_id,
                stage,
                dependencies,
                seconds + overhead,
                (candidate.candidate_id,),
            )
        )
        return task_id

    def _adopt(self, task_id: str, candidate: ShadowCandidate) -> None:
        for index, task in enumerate(self.tasks):
            if task.task_id == task_id:
                self.tasks[index] = ShadowTask(
                    task.task_id,
                    task.stage,
                    task.dependencies,
                    task.seconds,
                    task.candidate_ids + (candidate.candidate_id,),
                )
                return
        raise ValueError("missing shared task")

    def build(self, candidate: ShadowCandidate, estimate: FamilyProfile) -> str:
        key = build_key(candidate)
        if key in self.builds:
            self._adopt(self.builds[key], candidate)
            return self.builds[key]
        hits = [
            entry
            for entry in self.inventory
            if entry.namespace == candidate.namespace
            and entry.revision == candidate.declaration.identity.revision
            and candidate.prepare_key in entry.keys
        ]
        seconds = _known(estimate.stages.build)
        hit = min(
            hits, key=lambda entry: (entry.nbytes, entry.allocation_id), default=None
        )
        if hit is not None:
            self.hits[hit.allocation_id] = hit
            # Acquiring one opaque payload exposes all its keys at one task cost.
            if hit.allocation_id in self.acquisitions:
                task_id = self.acquisitions[hit.allocation_id]
                self._adopt(task_id, candidate)
                self.builds[key] = task_id
                return task_id
            seconds = 0.0
        else:
            self.resident_bytes += self.account(estimate.memory.resident)
        task_id = self.add(
            "build", str(len(self.builds)), (), seconds, candidate, estimate
        )
        self.builds[key] = task_id
        if hit is not None:
            self.acquisitions[hit.allocation_id] = task_id
        return task_id

    def bind(
        self, candidate: ShadowCandidate, estimate: FamilyProfile, build: str
    ) -> str:
        if candidate.binding_key is None:
            return build
        key = (build_key(candidate), candidate.binding_key)
        if key in self.bindings:
            self._adopt(self.bindings[key], candidate)
            return self.bindings[key]
        task_id = self.add(
            "bind",
            str(len(self.bindings)),
            (build,),
            _known(estimate.stages.bind),
            candidate,
            estimate,
        )
        self.bindings[key] = task_id
        self.binding_bytes += self.account(estimate.memory.input_bound)
        return task_id

    def row(self, candidate: ShadowCandidate, previous: str | None) -> str:
        estimate = self.profile.family(candidate.declaration.identity)
        build = self.build(candidate, estimate)
        bound = self.bind(candidate, estimate, build)
        dependencies = [bound]
        if previous is not None:
            dependencies.append(previous)
        if candidate.guard == "tentative":
            guard_dependencies = (
                (build,) if candidate.declaration.guard.builds_model else ()
            )
            guard = self.add(
                "guard",
                candidate.candidate_id,
                guard_dependencies,
                _known(estimate.stages.guard),
                candidate,
                estimate,
            )
            dependencies.append(guard)
        if estimate.stages.basis_conversion:
            basis = self.add(
                "basis",
                candidate.candidate_id,
                tuple(dependencies),
                estimate.stages.basis_conversion,
                candidate,
                estimate,
            )
            dependencies = [basis]
        apply = self.add(
            "apply",
            candidate.candidate_id,
            tuple(dict.fromkeys(dependencies)),
            _known(estimate.stages.apply),
            candidate,
            estimate,
        )
        self.task_bytes = max(
            self.task_bytes,
            self.account(estimate.memory.build_workspace),
            self.account(estimate.memory.apply_scratch),
        )
        if estimate.stages.lift:
            return self.add(
                "lift",
                candidate.candidate_id,
                (apply,),
                estimate.stages.lift,
                candidate,
                estimate,
            )
        return apply


def _schedule(tasks: tuple[ShadowTask, ...], workers: int) -> tuple[float, float]:
    """Return canonical list-schedule wall and work/chain/indivisible lower bound."""
    pending = {task.task_id: task for task in tasks}
    finished: dict[str, float] = {}
    longest: dict[str, float] = {}
    running: list[tuple[float, str]] = []
    now = 0.0
    while pending or running:
        ready = sorted(
            task_id
            for task_id, task in pending.items()
            if all(dependency in finished for dependency in task.dependencies)
        )
        for task_id in ready[: workers - len(running)]:
            task = pending.pop(task_id)
            longest[task_id] = task.seconds + max(
                (longest[d] for d in task.dependencies), default=0.0
            )
            heappush(running, (now + task.seconds, task_id))
        if not running:
            if pending:
                raise ValueError("shadow graph contains a cycle")
            break
        now = running[0][0]
        while running and running[0][0] == now:
            end, task_id = heappop(running)
            finished[task_id] = end
    work = sum(task.seconds for task in tasks)
    indivisible = max(
        (task.seconds for task in tasks if task.stage == "build"), default=0.0
    )
    return now, max(work / workers, max(longest.values(), default=0.0), indivisible)


def score_cover(
    table: SegmentTable,
    cover: tuple[ShadowCandidate, ...],
    profile: OneCoreProfile,
    inventory: tuple[CacheEntry, ...],
    request: ShadowRequest,
) -> ShadowScore:
    """Construct shared tasks once, preserve timeline dependencies, and score a full cover.

    cover must be validated and in timeline/time order. Known synthetic memory
    charges retain payloads through acquire end. Unknown totals stay None and
    only the dominant T6 channel estimate is limited. No cache handles or
    numerical kernel methods are used.
    """
    graph = _Graph(profile, inventory)
    previous: dict[int, str] = {}
    for candidate in cover:
        timeline = int(table.timeline[candidate.row])
        previous[timeline] = graph.row(candidate, previous.get(timeline))
    tasks = tuple(graph.tasks)
    wall, lower = _schedule(tasks, request.workers)
    required = (
        request.state_bytes
        + graph.resident_bytes
        + graph.binding_bytes
        + sum(entry.nbytes for entry in graph.hits.values())
        + request.workers * graph.task_bytes
    )
    # T6 retains phase-channel samples, Fourier values and coefficients. The
    # largest phase grid has 512 samples plus three probes. This checks the
    # dominant d**4 arrays, not SciPy/native workspace or process RSS. Other
    # unknown allocations retain legacy's unrestricted policy.
    large = sum(
        (4 * 515 + 64) * request.dimension**4 * 16
        for key in graph.builds
        if key.identity.name == "t6"
    )
    return ShadowScore(
        cover,
        tasks,
        wall,
        lower,
        required if graph.memory_known else None,
        (required if graph.memory_known else large) <= request.memory_allowance,
        tuple(c.candidate_id for c in cover if c.guard != "accepted"),
        tuple(sorted(graph.hits)),
        large,
    )
