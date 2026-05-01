from qick import QickConfig
from qick.asm_v2 import QickProgramV2


class TestProg(QickProgramV2):
    def _body(self):
        self.nop()
        self.reg_wr("r1", 10)
        self.test("r1 == 10")
        self.jump("HERE")
        self.delay_auto(1.0)
        self.wait(10)
        self.sync_all(100) # Sync exists?
        self.trigger(ros=[0], out_ch=0)

try:
    soccfg = QickConfig()
    prog = TestProg(soccfg)
    for i, inst in enumerate(prog.prog_list):
        print(f"{i}: {inst.get('CMD')} {inst}")
except Exception as e:
    print(f"Error: {e}")
