"""IR tree traversal and public pass seams shared by optimization tests."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from zcu_tools.program.v2.ir.factory import IRParser
from zcu_tools.program.v2.ir.instructions import Instruction, NopInst
from zcu_tools.program.v2.ir.node import (
    BasicBlockNode,
    BlockNode,
    IRBranch,
    IRLoop,
    IRNode,
)
from zcu_tools.program.v2.ir.operands import Register
from zcu_tools.program.v2.ir.pipeline import (
    AbsChunkListPass,
    AbsChunkPass,
    AbsIRTreePass,
    PipeLineConfig,
    PipeLineContext,
)


def walk_basic_blocks(node: IRNode) -> Iterator[BasicBlockNode]:
    """Yield basic blocks in tree order, descending into loops and branch cases."""
    if isinstance(node, BasicBlockNode):
        yield node
    elif isinstance(node, BlockNode):
        for child in node.insts:
            yield from walk_basic_blocks(child)
    elif isinstance(node, IRLoop):
        yield from walk_basic_blocks(node.body)
    elif isinstance(node, IRBranch):
        for case in node.cases:
            yield from walk_basic_blocks(case)


def transform_only_child(
    root: BlockNode, pass_: AbsIRTreePass, ctx: PipeLineContext
) -> BlockNode:
    """Exercise a pass's public transform contract on one prepared subtree."""
    assert len(root.insts) == 1
    node = root.insts[0]
    replacement = pass_.transform(node, ctx)
    return BlockNode(insts=[node if replacement is None else replacement])


def run_chunk_passes_on_root(
    root: BlockNode, passes: Sequence[AbsChunkPass | AbsChunkListPass]
) -> BlockNode:
    """Round-trip root through the given chunk passes with a fresh 1024-word context."""
    parser = IRParser()
    chunks = parser.unparse(root)
    ctx = PipeLineContext(config=PipeLineConfig(), pmem_budget=1024)
    for pass_ in passes:
        chunks, _ = pass_.process(chunks, ctx)
    return parser.parse(chunks)


def flatten_root(root: BlockNode) -> list[Instruction]:
    """Lower root to instructions through the parser and lexer, without optimization."""
    from zcu_tools.program.v2.ir.factory import IRLexer, IRParser

    parser = IRParser()
    lexer = IRLexer()
    return lexer.flatten(parser.unparse(root))


def register_loop(name: str, counter: str = "r0", n_reg: str = "n_reg") -> IRLoop:
    """Register-driven IRLoop (n=Register) with a NopInst body."""
    return IRLoop(
        name=name,
        counter_reg=Register(counter),
        n=Register(n_reg),
        body=BlockNode(insts=[BasicBlockNode(insts=[NopInst()])]),
    )
