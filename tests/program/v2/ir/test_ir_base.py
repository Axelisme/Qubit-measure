import pytest
from qick.asm_v2 import Label
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

    def add_dmem(self, vals):
        self._dmem_buffer = getattr(self, "_dmem_buffer", []) + vals
        return len(self._dmem_buffer) - len(vals)


def test_ir_compile_mixin_labels_and_meta():
    prog = MockProg()
    prog._add_label("my_label")
    prog._add_meta("my_type", "my_name", {"some": "info"})

    assert len(prog.meta_infos) == 2
    assert prog.meta_infos[0] == {"kind": "label", "name": "my_label", "p_addr": 0}
    assert prog.meta_infos[1] == {
        "kind": "meta",
        "type": "my_type",
        "name": "my_name",
        "info": {"some": "info"},
        "p_addr": 0,
    }


def test_ir_compile_mixin_compile():
    prog = MockProg()
    prog._add_meta("dummy", "meta")

    # compile() should clear meta_infos before optimizing
    prog.compile()

    assert prog.meta_infos == []


def test_ir_compile_mixin_materialize_dmem_tables(caplog):
    import logging

    caplog.set_level(logging.DEBUG)

    prog = MockProg()
    prog.prog_list = [{"CMD": "NOP", "P_ADDR": 10}]

    class MockCtx:
        def __init__(self):
            # Mock a table reference: requires actual Label objects
            # And they need to exist in opt_labels to be parsed by IRLinker._parse_label_addr
            self.dmem_tables = [(Label(name="entry_a"), Label(name="entry_b"))]

    opt_labels = {"entry_a": 10, "entry_b": "&20"}
    ctx = MockCtx()

    prog._materialize_dmem_tables(ctx, opt_labels, dmem_base=0)

    # Check that dmem was added
    assert getattr(prog, "_dmem_buffer") == [10, 20]

    # Check debug logs
    log_text = caplog.text
    assert "dmem dispatch: materialized table #0" in log_text
    assert "entry_a -> P_ADDR 10: CMD=NOP" in log_text
    assert "entry_b -> P_ADDR 20: (no inst)" in log_text


def test_ir_compile_mixin_materialize_dmem_tables_mismatch():
    prog = MockProg()
    # If dmem_base is not the current length of dmem_buffer, it should raise RuntimeError
    prog._dmem_buffer = [0, 0]

    class MockCtx:
        def __init__(self):
            self.dmem_tables = [(Label(name="entry"),)]

    opt_labels = {"entry": 0}

    with pytest.raises(RuntimeError, match="dmem dispatch table allocation mismatch"):
        prog._materialize_dmem_tables(
            MockCtx(), opt_labels, dmem_base=0
        )  # dmem_base is 0, but _dmem_buffer has length 2gth 2
