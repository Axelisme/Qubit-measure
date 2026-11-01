"""Producer-owned cycle bounds and emission-time write provenance."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral

from qick.asm_v2 import CloseLoop, Macro, OpenLoop, QickParam, TimedMacro, WriteReg

from ._timeline_guard import bounds
from .ir.instructions import BaseInst
from .macro.structure import MarkedCloseLoop, MarkedOpenLoop
from .macro.table import TableSelection


@dataclass(frozen=True)
class CycleBounds:
    """Inclusive ordered bounds for integer tProc-cycle values or expressions.

    minimum and maximum are signed cycle endpoints. Producer bounds cover every
    executable write, including cases, back-edges and sweep resets; expression
    bounds cover every legal point and any unknown inputs conservatively.
    Values used as time must be nonnegative and no larger than 2147483647.
    Non-time words need not satisfy that restriction. Reversed endpoints raise
    ValueError during construction.
    """

    minimum: int
    maximum: int

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError("cycle bounds must be ordered")


@dataclass(frozen=True)
class FixedLoopIndex:
    """Owner metadata for a nonempty fixed Repeat's zero-based counter.

    name identifies the loop, counter names its native register, and count is
    its positive fixed iteration count. bounds covers every executable counter
    value, including the final conditional backedge and next-entry reset.
    """

    name: str
    counter: str
    count: int

    @property
    def bounds(self) -> CycleBounds:
        """Return the inclusive valid index range, zero through count minus one."""
        return CycleBounds(0, self.count - 1)


@dataclass(eq=False)
class SelectionProvenance:
    """Reaching table relations for one register at a control-flow position.

    selection is the exact loader relation at a leaf; None is an unknown write
    or uninitialized entry. incoming joins predecessor states at Branch/Repeat
    boundaries. Repeat adds its backedge after preparing the body, so consumers
    retain nodes rather than deciding validity from linear preprocessing order.
    This graph is proof metadata only; it allocates no runtime storage.
    """

    selection: TableSelection | None = None
    incoming: list["SelectionProvenance"] = field(default_factory=list)

    def matches(self, expected: TableSelection) -> bool:
        """Return whether every reaching definition has the expected mapping."""
        pending = [self]
        seen: set[SelectionProvenance] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            if current.incoming:
                pending.extend(current.incoming)
            elif current.selection != expected:
                return False
        return True


class ProducerBounds:
    """Track known producers and reject unregistered writes to timing dependencies.

    The program observes native constant writes and owner-certified operations
    before preprocessing. Translation supplies the originating macro for each
    instruction. This is provenance checking, not interval interpretation of ALU.
    """

    def __init__(self):
        self.ranges: dict[str, list[CycleBounds]] = {}
        self.fixed_loops: dict[str, FixedLoopIndex] = {}
        self.fixed_counters: dict[str, FixedLoopIndex] = {}
        self.counter_writers: dict[str, set[int]] = {}
        self.dependencies: dict[str, set[str]] = {}
        self.requirements: dict[str, list[tuple[str, CycleBounds]]] = {}
        self.input_errors: dict[str, list[str]] = {}
        self.allowed: dict[int, set[str]] = {}
        self.sources: set[str] = set()
        self.defined: set[str] = set()
        self.origins: dict[int, Macro | None] = {}
        self.generated: dict[str, set[int]] = {}
        self.sweep_times: set[str] = set()
        self.selection_writers: dict[
            str, list[tuple[TableSelection, tuple[Macro, ...]]]
        ] = {}
        self.selected: dict[str, list[TableSelection]] = {}
        self.selection_at_position: dict[str, SelectionProvenance] = {}
        self.selection_consumers: dict[
            int, list[tuple[str, TableSelection, SelectionProvenance]]
        ] = {}
        self.signed_sources: set[str] = set()
        self.author_writes: set[str] = set()
        self.framework_sources: set[str] = set()

    def author_source(self, name: str) -> bool:
        """Return whether a register has no certified range or arbitrary writes."""
        return name in self.author_writes or name not in self.ranges

    def register_selection(
        self, register: str, selection: TableSelection, writers: tuple[Macro, ...]
    ) -> None:
        """Bind exact loader writers to one declared mapping and captured index."""
        self.selection_writers.setdefault(register, []).append((selection, writers))

    def require_selection(self, prog, selection: TableSelection) -> None:
        """Activate table-relation auditing for actual playback and its capture.

        General constant producers remain legal elsewhere. For these two
        registers only, every translated write must belong to a consumed mapping.
        The captured index binds lookup, playback and finisher at execution.
        """
        for name in (selection.register, selection.selector):
            self.source(prog, name)
            current = self.selection_at_position.get(name, SelectionProvenance())
            self.selection_consumers.setdefault(id(prog._time_origin), []).append(
                (name, selection, current)
            )
            self.selected.setdefault(name, []).append(selection)

    def merge_selections(self, paths: list[dict[str, SelectionProvenance]]) -> None:
        """Join all executable paths, including unchanged empty Branch cases."""
        self.selection_at_position = {
            name: SelectionProvenance(
                incoming=[path.get(name, SelectionProvenance()) for path in paths]
            )
            for name in set().union(*paths)
        }

    def observe(self, macro: Macro) -> None:
        """Register the bounds and write identity of a native constant producer."""
        # Subclasses can replace translate/expand, so their src is not a proof.
        if (
            isinstance(getattr(macro, "dst", None), str)
            and type(macro) is not WriteReg
            and macro.dst not in self.allowed.get(id(macro), ())
        ):
            self.author_writes.add(macro.dst)
        if type(macro) is WriteReg and isinstance(macro.src, Integral):
            value = int(macro.src)
            # Native constants provide bounds, not framework ownership. Authors
            # may overwrite them with their own arithmetic or memory reads.
            self.ranges.setdefault(macro.dst, []).append(CycleBounds(value, value))
            self.allowed.setdefault(id(macro), set()).add(macro.dst)

    def register(
        self,
        name: str,
        limits: CycleBounds,
        writers: tuple[Macro, ...],
        dependencies: tuple[str, ...],
        dependency_ranges: Mapping[str, CycleBounds] | None = None,
    ) -> None:
        """Register a framework owner's value bounds, exact writers and inputs.

        Writers may write other registers, but certify only name. Dependencies
        must themselves have known producers. The owner guarantees these bounds
        across its entire operation, including lookup/word conversion steps.
        """
        self.framework_sources.add(name)
        self.ranges.setdefault(name, []).append(limits)
        self.dependencies.setdefault(name, set()).update(dependencies)
        if dependency_ranges is not None:
            self.dependencies[name].update(dependency_ranges)
            self.requirements.setdefault(name, []).extend(dependency_ranges.items())
        for writer in writers:
            self.allowed.setdefault(id(writer), set()).add(name)

    def register_fixed_loop(
        self, index: FixedLoopIndex, opening: OpenLoop, closing: CloseLoop
    ) -> None:
        """Certify only this loop's native initialization, backedge and reset."""
        if index.name in self.fixed_loops:
            raise ValueError(f"{index.name}: fixed Repeat loop name already used")
        self.fixed_loops[index.name] = index
        self.fixed_counters[index.counter] = index
        self.counter_writers[index.counter] = {id(opening), id(closing)}
        self.register(index.counter, index.bounds, (opening, closing), ())

    def activate(self, macro: Macro) -> None:
        """Mark values written by an observed producer as initialized on this path."""
        self.defined.update(self.allowed.get(id(macro), ()))

    def capture_inputs(
        self,
        prog,
        name: str,
        dependencies: tuple[str, ...],
        requirements: Mapping[str, CycleBounds],
    ) -> None:
        """Retain each producer's input proof at its actual execution position.

        Structured owners restore/intersect defined on Branch/Repeat paths.
        Snapshot errors include dependency failures and survive later initialization,
        overwrites and other cases. They become errors only when name participates
        in a time dependency. Non-time table consumers keep their runtime contract.
        """
        for dependency in set(dependencies) | requirements.keys():
            try:
                if errors := self.input_errors.get(dependency):
                    raise ValueError(errors[0])
                actual = self._resolve(prog, dependency, check_defined=True)
                allowed = requirements.get(dependency)
                if allowed is not None and (
                    actual.minimum < allowed.minimum or actual.maximum > allowed.maximum
                ):
                    raise ValueError(
                        f"{name}: index dependency {dependency} allowed "
                        f"[{allowed.minimum}, {allowed.maximum}], actual "
                        f"[{actual.minimum}, {actual.maximum}]"
                    )
            except ValueError as exc:
                self.input_errors.setdefault(name, []).append(str(exc))

    def describe(self, prog, name: str) -> CycleBounds:
        """Read current producer bounds without activating a timing audit.

        Metadata capture for non-time scalar lookups must not turn them into
        timing consumers. Actual consumers still call source() for full proof.
        Unknown or uninitialized sources raise ValueError.
        """
        return self._resolve(prog, name, check_defined=True)

    def source(self, prog, name: str, *, check_defined: bool = True) -> CycleBounds:
        """Resolve a nonnegative cycle source and all of its producer dependencies.

        Unknown registers, uninitialized paths and unproved/out-of-range values
        raise ValueError. Loop counters and QICK sweep registers are initialized
        by their native loop machinery, including back-edges and resets.
        """
        return self._source(prog, name, check_defined=check_defined, signed=False)

    def watch_signed(
        self, prog, name: str, *, check_defined: bool = True
    ) -> CycleBounds:
        """Audit a framework-generated signed32 intermediate, not a time consumer.

        Only exact lookup/arithmetic owners use this entry. Ordinary source()
        still rejects negative cycle registers, even if watched here earlier.
        Dependencies retain their ordinary nonnegative input proofs.
        """
        self.signed_sources.add(name)
        return self._source(prog, name, check_defined=check_defined, signed=True)

    def _source(
        self, prog, name: str, *, check_defined: bool, signed: bool
    ) -> CycleBounds:
        result = self._resolve(prog, name, check_defined=check_defined, signed=signed)
        pending = [name]
        seen: set[str] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            if not check_defined:
                self._resolve(
                    prog,
                    current,
                    check_defined=False,
                    signed=signed and current == name,
                )
            if errors := self.input_errors.get(current):
                raise ValueError(errors[0])
            self.sources.add(current)
            pending.extend(self.dependencies.get(current, ()))
            if check_defined:
                continue
            for dependency, allowed in self.requirements.get(current, ()):
                actual = self._resolve(prog, dependency, check_defined=check_defined)
                if actual.minimum < allowed.minimum or actual.maximum > allowed.maximum:
                    raise ValueError(
                        f"{current}: index dependency {dependency} allowed "
                        f"[{allowed.minimum}, {allowed.maximum}], actual "
                        f"[{actual.minimum}, {actual.maximum}]"
                    )
        return result

    def _resolve(
        self, prog, name: str, *, check_defined: bool, signed: bool = False
    ) -> CycleBounds:
        try:
            prog._get_reg(name)
        except KeyError as exc:
            raise ValueError(f"{name}: unknown time register") from exc
        register = prog.reg_dict.get(name)
        if name in self.fixed_counters:
            result = self.fixed_counters[name].bounds
            if check_defined and name not in self.defined:
                raise ValueError(f"{name}: uninitialized fixed Repeat counter")
        elif name in prog.loop_dict:
            result = CycleBounds(0, prog.loop_dict[name] - 1)
        elif register is not None and register.init is not None:
            raw = register.init
            lo, hi = bounds(QickParam(raw.start, raw.spans))
            result = CycleBounds(round(lo), round(hi))
        else:
            ranges = self.ranges.get(name)
            if not ranges or (check_defined and name not in self.defined):
                raise ValueError(
                    f"{name}: unknown or uninitialized time producer bounds"
                )
            result = CycleBounds(
                min(r.minimum for r in ranges), max(r.maximum for r in ranges)
            )
        minimum = -(2**31) if signed else 0
        if result.minimum < minimum or result.maximum > 2**31 - 1:
            contract = "signed 32-bit" if signed else "nonnegative signed 32-bit"
            raise ValueError(f"{name}: time producer exceeds {contract} cycles")
        return result

    def emission(
        self, raw, address: int, origin: Macro | None, *, arithmetic: bool
    ) -> None:
        """Record compiler arithmetic writers independently of author producers."""
        self.origins[address] = origin
        if arithmetic:
            for register in BaseInst.from_dict(raw).reg_write:
                # s14 is shared with native outputs, not compiler-owned storage.
                if register != "s14":
                    self.generated.setdefault(register, set()).add(address)

    def time_parameters(self, macro: TimedMacro) -> None:
        """Watch registers allocated by native time conversion, not arbitrary data."""
        self.sweep_times.update(
            name for name in macro.t_regs.values() if isinstance(name, str)
        )

    def _audit_sweeps(self, prog) -> None:
        for name, register in prog.reg_dict.items():
            raw = register.init
            if raw is None or name not in self.sweep_times | self.sources:
                continue
            lo = raw.start + sum(min(0, span) for span in raw.spans.values())
            hi = raw.start + sum(max(0, span) for span in raw.spans.values())
            increments = [step["step"] for step in raw.steps.values()]
            if (
                lo + min([0, *increments]) < -(2**31)
                or hi + max([0, *increments]) > 2**31 - 1
            ):
                raise ValueError(
                    f"{name}: signed 32-bit sweep backedge/reset intermediate overflow"
                )

    def _audit_selection(self, name: str, origin: Macro | None, address: int) -> None:
        if name not in self.selected:
            return
        valid = any(
            relation.table is expected.table
            and relation.selector == expected.selector
            and relation.register == expected.register
            and any(origin is writer for writer in writers)
            for relation, writers in self.selection_writers.get(name, ())
            for expected in self.selected[name]
        )
        if not valid:
            raise ValueError(
                f"{name}: invalid selected playback selection write at instruction {address}"
            )

    def _audit_selection_consumer(self, origin: Macro | None, address: int) -> None:
        """Reject a playback that reads a relation overwritten before its use."""
        for name, expected, current in self.selection_consumers.get(id(origin), ()):
            if not current.matches(expected):
                raise ValueError(
                    f"{name}: overwritten selected playback selection at instruction {address}"
                )

    def audit(self, prog) -> None:
        """Audit every translated write, including unreachable alternate paths.

        Report the logical register and original instruction position when an
        unregistered writer invalidates a time source or its dependencies.
        """
        for name in tuple(self.sources):
            if name in self.signed_sources:
                self.watch_signed(prog, name, check_defined=False)
            else:
                self.source(prog, name, check_defined=False)
        self._audit_sweeps(prog)
        watched = {
            register.full_addr(): name
            for name, register in prog.reg_dict.items()
            if register.init is not None and name in self.sweep_times | self.sources
        }
        watched.update({prog._get_reg(name): name for name in self.sources})
        watched.update({name: name for name in self.generated})
        for raw in prog.prog_list:
            inst = BaseInst.from_dict(raw)
            address = raw["P_ADDR"]
            origin = self.origins.get(address)
            self._audit_selection_consumer(origin, address)
            for physical in inst.reg_write & watched.keys():
                name = watched[physical]
                if physical in self.generated:
                    if address in self.generated[physical]:
                        continue
                    raise ValueError(
                        f"{name}: unregistered compiler time write at instruction {address}"
                    )
                if name in self.counter_writers:
                    if id(origin) in self.counter_writers[name]:
                        continue
                    raise ValueError(
                        f"{name}: unregistered fixed Repeat counter write at instruction {address}"
                    )
                self._audit_selection(name, origin, address)
                if name in self.allowed.get(id(origin), set()):
                    continue
                register = prog.reg_dict.get(name)
                native_sweep = register is not None and register.init is not None
                native_loop = name in prog.loop_dict
                if (native_sweep or native_loop) and (
                    # These exact wrappers only annotate the unchanged QICK shell.
                    origin is None
                    or type(origin)
                    in (OpenLoop, CloseLoop, MarkedOpenLoop, MarkedCloseLoop)
                ):
                    continue
                raise ValueError(
                    f"{name}: unregistered time producer write at instruction {address}"
                )
