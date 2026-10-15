from zcu_tools.program.v2.ir.base import IRCompileMixin
from zcu_tools.program.v2.mocksoc import make_mock_soccfg


# A minimal concrete subclass to instantiate the mixin
class MockProg(IRCompileMixin):
    def __init__(self):
        soccfg = make_mock_soccfg()
        super().__init__(soccfg)
        self.tproccfg = {"pmem_size": 1024}
        self.prog_list = []
        self.labels = {}
        self.p_addr = 0
        self.line = 0

    def _make_asm(self):
        pass

    def _make_binprog(self):
        pass


def test_ir_compile_mixin_compile():
    prog = MockProg()
    prog.meta_infos.append(
        {"kind": "meta", "type": "dummy", "name": "meta", "p_addr": 0}
    )

    # compile() should clear meta_infos before optimizing
    prog.compile()

    assert prog.meta_infos == []
