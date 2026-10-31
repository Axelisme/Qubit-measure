from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .instructions import BaseInst, Instruction, LabelInst, MetaInst


@dataclass(frozen=True)
class IRCursor:
    """End of a positioned stream.

    final_p_addr is the zero-based word offset immediately after the stream.
    final_line counts labels and machine instructions; metadata adds no lines.
    Both fields are zero for an empty stream.
    """

    final_p_addr: int
    final_line: int


@dataclass(frozen=True)
class PositionedInstruction:
    """A linker positioning view used by QICK linking and compile snapshots.

    instruction is the original logical instruction, label or metadata object.
    It retains the input reference, including mutable nested payloads; this view
    is not a defensive-copy snapshot.
    p_addr is its zero-based program-memory word offset. Machine instructions
    advance by addr_inc (normally one word, two for WAIT). Labels and metadata
    occupy zero words and may share an offset, including the end of the stream.
    line is the cumulative assembler line count at this record. Labels and
    machine instructions increment it once, even for a two-word WAIT. Metadata
    retains the preceding count, or zero before any label/machine instruction.

    This view hides address widths and line-count rules from consumers without
    requiring QICK dictionary serialization.
    """

    instruction: Instruction
    p_addr: int
    line: int


class IRLinker:
    """Assigns physical addresses to a flat instruction list and resolves labels."""

    def position(
        self, inst_list: Sequence[Instruction]
    ) -> tuple[tuple[PositionedInstruction, ...], IRCursor]:
        """Position flat logical IR without serializing or changing its payloads.

        Accept machine instructions, labels and metadata in source order.
        Return one positioning view per input item, in that same order, and the
        end cursor. Addresses start at zero and account for instruction widths.
        Labels and metadata occupy no words; only metadata adds no line.
        Raise ValueError for duplicate label names, as link() does.
        Returned views retain input references; callers needing detached payloads
        must copy them before changing the input.
        """
        positioned: list[PositionedInstruction] = []
        labels: set[str] = set()
        p_addr = 0
        line = 0
        for inst in inst_list:
            if not isinstance(inst, MetaInst):
                line += 1
            if isinstance(inst, LabelInst):
                name = inst.name.name
                if name in labels:
                    raise ValueError(f"IRLinker.link: duplicate label {name!r}")
                labels.add(name)
            positioned.append(PositionedInstruction(inst, p_addr, line))
            if isinstance(inst, BaseInst):
                p_addr += inst.addr_inc
        return tuple(positioned), IRCursor(final_p_addr=p_addr, final_line=line)

    def link(
        self, inst_list: list[Instruction]
    ) -> tuple[list[dict[str, object]], dict[str, str], list[dict[str, Any]], IRCursor]:
        """Serialize flat logical IR into positioned QICK-compatible dicts.

        Return machine instruction dicts, label addresses, ordered label/meta
        dicts and the end cursor. Positioning follows position() and duplicate
        labels raise ValueError. The input instructions are not changed.
        """
        prog_list: list[dict[str, object]] = []
        labels: dict[str, str] = {}
        meta_infos: list[dict[str, Any]] = []

        positioned, cursor = self.position(inst_list)
        for record in positioned:
            inst = record.instruction
            d = inst.to_dict()
            if isinstance(inst, (MetaInst, LabelInst)):
                d["p_addr"] = record.p_addr
                meta_infos.append(d)
                if isinstance(inst, LabelInst):
                    labels[inst.name.name] = f"&{record.p_addr}"
            else:
                assert isinstance(inst, BaseInst)
                d["P_ADDR"] = record.p_addr
                d["LINE"] = record.line
                prog_list.append(d)

        return prog_list, labels, meta_infos, cursor

    def unlink(
        self,
        prog_list: list[dict[str, Any]],
        labels: dict[str, Any],
        meta_infos: list[dict[str, Any]],
    ) -> list[Instruction]:
        # 1. Strict Validation: Compare labels.keys() with labels tracked in meta_infos
        tracked_label_names = {
            m["name"] for m in meta_infos if m.get("kind") == "label"
        }
        provided_label_names = set(labels.keys())

        if tracked_label_names != provided_label_names:
            only_in_tracked = tracked_label_names - provided_label_names
            only_in_provided = provided_label_names - tracked_label_names
            msg = "Mismatch between tracked labels in meta_infos and provided labels dict."
            if only_in_tracked:
                msg += f" Missing in labels dict: {sorted(list(only_in_tracked))}."
            if only_in_provided:
                msg += f" Not found in meta_infos: {sorted(list(only_in_provided))}."
            raise ValueError(msg)

        # Group tracked markers by p_addr
        markers_by_addr: dict[int, list[dict[str, object]]] = defaultdict(list)
        for m in meta_infos:
            markers_by_addr[m["p_addr"]].append(m)

        logical_insts: list[Instruction] = []

        for d in prog_list:
            p_addr = d["P_ADDR"]

            # 1. Insert tracked markers from meta_infos for this index
            self._append_markers(logical_insts, markers_by_addr.get(p_addr, []))

            # Remove from markers_by_addr so we don't process it again for trailing
            if p_addr in markers_by_addr:
                del markers_by_addr[p_addr]

            # 2. Insert the instruction itself
            logical_insts.append(BaseInst.from_dict(d))

        # Handle trailing markers from meta_infos — markers whose p_addr lies
        # past the last instruction (e.g. a label/meta at end-of-program).
        # Sorting by p_addr keeps cross-address ordering correct; within a
        # single p_addr the original meta_infos append order is preserved
        # because markers_by_addr is a defaultdict(list) populated in link()
        # order, so an interleaved label+meta at the same address keeps its
        # relative order.
        for _p_addr, markers in sorted(markers_by_addr.items()):
            self._append_markers(logical_insts, markers)

        return logical_insts

    @staticmethod
    def _append_markers(
        logical_insts: list[Instruction], markers: list[dict[str, object]]
    ) -> None:
        """Append tracked label/meta markers in their original order."""
        for m in markers:
            if m["kind"] == "label":
                logical_insts.append(
                    LabelInst.from_dict(
                        {
                            "kind": "label",
                            "name": m["name"],
                            "can_remove": m.get("can_remove", False),
                        }
                    )
                )
            elif m["kind"] == "meta":
                logical_insts.append(
                    MetaInst.from_dict(
                        {
                            "kind": "meta",
                            "type": m["type"],
                            "name": m["name"],
                            "info": m.get("info", {}),
                        }
                    )
                )

    @staticmethod
    def parse_label_addr(label_name: str, label_addr: Any) -> int:
        """Return the integer program address represented by label_addr.

        label_name identifies the label in error messages. label_addr may be
        an int or a decimal string with an optional leading '&'; strings use
        Python's int parsing. The value is parsed without changing linker state.

        Raise ValueError for other types or invalid decimal strings. Invalid
        strings retain the int conversion's ValueError as their cause.
        """
        if isinstance(label_addr, int):
            return label_addr

        if isinstance(label_addr, str):
            addr_str = label_addr[1:] if label_addr.startswith("&") else label_addr
            try:
                return int(addr_str)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid label address for {label_name!r}: {label_addr!r}"
                ) from exc

        raise ValueError(
            f"Invalid label address type for {label_name!r}: {label_addr!r}"
        )
