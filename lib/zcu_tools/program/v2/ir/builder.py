from __future__ import annotations

from typing_extensions import TYPE_CHECKING, Any

from .factory import IRLexer, IRParser
from .labels import Label
from .linker import IRCursor, IRLinker
from .node import RootNode

if TYPE_CHECKING:
    from .base import IRCompileMixin


class IRBuilder:
    def __init__(self, prog: IRCompileMixin):
        self.prog = prog
        self.linker = IRLinker()
        self.lexer = IRLexer()
        self.parser = IRParser(pmem_size=prog.tproccfg["pmem_size"])

    def build(
        self,
        prog_list: list[dict[str, Any]],
        labels: dict[str, Any],
        meta_infos: list[dict[str, Any]],
    ) -> RootNode:
        Label.reset()  # Must precede all Label.make_new() calls; see Label.label_set docstring
        inst_list = self.linker.unlink(prog_list, labels, meta_infos)
        items = self.lexer.lex(inst_list)
        return self.parser.parse(items)

    def unbuild(
        self, ir: RootNode
    ) -> tuple[list[dict], dict[str, str], list[dict[str, Any]], IRCursor]:
        blocks = self.parser.unparse(ir)
        inst_list = self.lexer.flatten(blocks)
        return self.linker.link(inst_list)
