from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union

from .instructions import BaseInst, JumpInst, LabelInst, MetaInst
from .labels import Label
from .operands import Register


class IRNode(ABC):
    """Base class for all IR nodes.

    Every IRNode participates in the IR tree via the children() /
    replace_child() interface for uniform recursive traversal.
    """

    @abstractmethod
    def children(self) -> list[IRNode]:
        """Return the direct IRNode children of this node."""
        ...

    @abstractmethod
    def replace_child(self, old: IRNode, new: IRNode) -> None:
        """Replace a direct child in-place.

        Raises ValueError if `old` is not a direct child.
        Raises TypeError on leaf nodes that have no children.
        """
        ...

    def _into_str(self, indent: int = 0) -> str:
        """Helper for __str__ that takes an indent level."""
        return f"{'    ' * indent}{self.__class__.__name__}()"

    def __str__(self) -> str:
        return self._into_str()


@dataclass
class BasicBlockNode(IRNode):
    """A basic block: a straight-line sequence with an optional terminal jump.

    labels: LabelInst(s) that mark the entry point of this block.
    insts:  Linear instructions (no labels, no jumps except TestInst).
    branch: Optional terminal JumpInst that ends this block.
    disable_opt: When True, the instruction count is frozen (set by jump-table
                  lowering). Post-LIR passes must NOP-pad instead of removing.
    """

    labels: list[LabelInst] = field(default_factory=list)
    insts: list[BaseInst] = field(default_factory=list)
    branch: Optional[JumpInst] = None
    disable_opt: bool = False

    def __post_init__(self) -> None:
        for inst in self.insts:
            if isinstance(inst, MetaInst):
                raise ValueError(
                    f"BasicBlockNode.insts must not contain MetaInst; "
                    f"use standalone MetaInst entries in the chunked stream instead. Got: {inst}"
                )
            if isinstance(inst, LabelInst):
                raise ValueError(
                    f"BasicBlockNode.insts must not contain LabelInst; "
                    f"use BasicBlockNode.labels instead. Got: {inst}"
                )
            if isinstance(inst, JumpInst):
                raise ValueError(
                    f"BasicBlockNode.insts must not contain JumpInst; "
                    f"use BasicBlockNode.branch instead. Got: {inst}"
                )

    def children(self) -> list[IRNode]:
        return []

    def replace_child(self, old: IRNode, new: IRNode) -> None:
        raise TypeError("BasicBlockNode is a leaf node and has no children")

    @property
    def addr_size(self) -> int:
        size = sum(inst.addr_inc for inst in self.insts)
        if self.branch is not None:
            size += self.branch.addr_inc
        return size

    def _into_str(self, indent: int = 0) -> str:
        prefix = "    " * indent
        lines = []
        if self.disable_opt:
            lines.append(f"{prefix}BasicBlockNode(disable_opt={self.addr_size}):")
        for lbl in self.labels:
            lines.append(f"{prefix}{lbl}:")
        for inst in self.insts:
            lines.append(f"{prefix}  {inst}")
        if self.branch is not None:
            lines.append(f"{prefix}  -> {self.branch}")
        return "\n".join(lines) + "\n"


@dataclass
class BlockNode(IRNode):
    """A sequence of IR nodes (structural container)."""

    insts: list[IRNode] = field(default_factory=list)

    def append(self, item: IRNode) -> None:
        self.insts.append(item)

    def children(self) -> list[IRNode]:
        return self.insts

    def replace_child(self, old: IRNode, new: IRNode) -> None:
        idx = self.insts.index(old)
        self.insts[idx] = new

    def _into_str(self, indent: int = 0) -> str:
        prefix = "    " * indent
        return f"{prefix} {self.__class__.__name__}()\n" + "\n".join(
            i._into_str(indent + 1) for i in self.insts
        )


@dataclass
class IRLoop(IRNode):
    """A loop node (pure data — lowering is handled by IRParser.unparse).

    `body` is treated as one full logical iteration, including the loop-carried
    counter update. Later linear passes may merge or reorder that update, so
    callers must not assume it remains the final instruction physically.
    """

    name: str
    counter_reg: Register
    n: Union[int, Register]
    body: IRNode
    range_hint: Optional[tuple[int, int]] = None

    def children(self) -> list[IRNode]:
        return [self.body]

    def replace_child(self, old: IRNode, new: IRNode) -> None:
        if self.body is old:
            self.body = new
        else:
            raise ValueError(
                f"IRLoop.replace_child: {old!r} is not a child of this node"
            )

    def _into_str(self, indent: int = 0) -> str:
        prefix = "    " * indent
        return (
            f"{prefix} IRLoop(name={self.name}, n={self.n}, counter={str(self.counter_reg)}, range_hint={self.range_hint})\n"
            + self.body._into_str(indent + 1)
        )


@dataclass
class IRBranch(IRNode):
    """A branch node (pure data — lowering is handled by IRParser.unparse)."""

    name: str
    compare_reg: Register
    cases: list[IRNode]

    def children(self) -> list[IRNode]:
        return list(self.cases)

    def replace_child(self, old: IRNode, new: IRNode) -> None:
        idx = self.cases.index(old)
        self.cases[idx] = new

    def _into_str(self, indent: int = 0) -> str:
        prefix = "    " * indent
        return (
            f"{prefix} IRBranch(name={self.name}, compare_reg={str(self.compare_reg)})\n"
            + "\n".join(i._into_str(indent + 1) for i in self.cases)
        )


@dataclass
class IRDispatch(IRNode):
    """A dispatch table node (pure data — lowering is handled by IRParser.unparse).

    Represents a value-indexed dispatch: ``value_reg`` selects which target
    label to jump to.  A mandatory out-of-range guard is always emitted: if
    ``value_reg >= len(target_labels)``, control falls through to
    ``target_labels[-1]`` (the last case).  This behaviour is intentional and
    must be documented at call sites that rely on it.

    IRDispatch is a leaf node — case bodies are **not** stored inside it.
    The caller (IRParser._lower_branch) is responsible for emitting the bodies
    after the dispatch table in the chunk stream.
    """

    name: str
    value_reg: Register
    target_labels: list[Label]

    def children(self) -> list[IRNode]:
        return []

    def replace_child(self, old: IRNode, new: IRNode) -> None:
        raise TypeError("IRDispatch is a leaf node and has no children")

    def _into_str(self, indent: int = 0) -> str:
        prefix = "    " * indent
        targets = ", ".join(str(lbl) for lbl in self.target_labels)
        return f"{prefix} IRDispatch(name={self.name}, value_reg={self.value_reg}, targets=[{targets}])\n"


# ---------------------------------------------------------------------------
# Subtree cloning with label / structure-name uniquification
# ---------------------------------------------------------------------------


def _collect_subtree_names(node: IRNode) -> tuple[set[str], set[str]]:
    """Return (defined label names, structure node names) in a subtree.

    Defined label names: every ``LabelInst.name`` inside any BasicBlockNode.
    Structure names: ``name`` of every IRLoop / IRBranch / IRDispatch — these
    drive the labels that unparse() will synthesise (``{name}_start`` etc.),
    so they must also be uniquified when a subtree is duplicated.
    """
    labels: set[str] = set()
    structs: set[str] = set()
    if isinstance(node, BasicBlockNode):
        for lbl in node.labels:
            labels.add(lbl.name.name)
    elif isinstance(node, BlockNode):
        for child in node.insts:
            sub_l, sub_s = _collect_subtree_names(child)
            labels |= sub_l
            structs |= sub_s
    elif isinstance(node, IRLoop):
        structs.add(node.name)
        sub_l, sub_s = _collect_subtree_names(node.body)
        labels |= sub_l
        structs |= sub_s
    elif isinstance(node, IRBranch):
        structs.add(node.name)
        for case in node.cases:
            sub_l, sub_s = _collect_subtree_names(case)
            labels |= sub_l
            structs |= sub_s
    elif isinstance(node, IRDispatch):
        structs.add(node.name)
    return labels, structs


def _clone_node(
    node: IRNode, label_remap: dict[str, Label], name_remap: dict[str, str]
) -> IRNode:
    """Deep-clone a subtree, applying label and structure-name remaps."""
    from .instructions import CallInst, DmemReadInst, RegWriteInst
    from .labels import LabelRef

    def _remap_ref(ref: Optional[LabelRef]) -> Optional[LabelRef]:
        if ref is None or ref.is_pseudo():
            return ref
        new = label_remap.get(ref.as_label().name)
        return LabelRef(new) if new is not None else ref

    def _remap_inst(inst: BaseInst) -> BaseInst:
        import dataclasses

        if isinstance(inst, (JumpInst, RegWriteInst, DmemReadInst, CallInst)):
            new_ref = _remap_ref(inst.label)
            if new_ref is not inst.label:
                return dataclasses.replace(inst, label=new_ref)
        return inst

    def _remap_branch(branch: JumpInst) -> JumpInst:
        import dataclasses

        new_ref = _remap_ref(branch.label)
        if new_ref is not branch.label:
            return dataclasses.replace(branch, label=new_ref)
        return branch

    if isinstance(node, BasicBlockNode):
        return BasicBlockNode(
            labels=[
                LabelInst(
                    name=label_remap.get(lbl.name.name, lbl.name),
                    can_remove=lbl.can_remove,
                )
                for lbl in node.labels
            ],
            insts=[_remap_inst(i) for i in node.insts],
            branch=_remap_branch(node.branch) if node.branch is not None else None,
            disable_opt=node.disable_opt,
        )
    if isinstance(node, BlockNode):
        return BlockNode(
            insts=[_clone_node(c, label_remap, name_remap) for c in node.insts]
        )
    if isinstance(node, IRLoop):
        return IRLoop(
            name=name_remap.get(node.name, node.name),
            counter_reg=node.counter_reg,
            n=node.n,
            body=_clone_node(node.body, label_remap, name_remap),
            range_hint=node.range_hint,
        )
    if isinstance(node, IRBranch):
        return IRBranch(
            name=name_remap.get(node.name, node.name),
            compare_reg=node.compare_reg,
            cases=[_clone_node(c, label_remap, name_remap) for c in node.cases],
        )
    if isinstance(node, IRDispatch):
        return IRDispatch(
            name=name_remap.get(node.name, node.name),
            value_reg=node.value_reg,
            target_labels=[
                label_remap.get(lbl.name, lbl) for lbl in node.target_labels
            ],
        )
    raise TypeError(f"_clone_node: unexpected node type {type(node).__name__}")


def clone_renamed(node: IRNode, allocated: set[str]) -> IRNode:
    """Deep-clone an IR subtree with every internal label and structure name
    replaced by a fresh unique name.

    ``allocated`` is the set of names already in use; the chosen fresh names
    are added to it so successive clones do not collide. This makes it safe to
    duplicate a loop body (including nested IRLoop / IRBranch / IRDispatch)
    k times — each copy gets distinct labels, so the later unparse() step
    synthesises non-colliding ``{name}_start`` etc. for nested structures.
    """
    from .labels import make_label

    label_names, struct_names = _collect_subtree_names(node)
    label_remap = {n: make_label(n, allocated) for n in label_names}
    name_remap = {n: make_label(n, allocated).name for n in struct_names}
    return _clone_node(node, label_remap, name_remap)
