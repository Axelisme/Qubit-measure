"""Check the installed QICK public timed-macro adaptation seam."""
from qick.asm_v2 import Macro, Pulse, ConfigReadout, Trigger, WriteReg
from zcu_tools.program.v2.base import MyProgramV2
from zcu_tools.program.v2.mocksoc import make_mock_soccfg


class Adapt(Macro):
    def __init__(self, source):
        self.source = source

    def preprocess(self, prog):
        self.source.preprocess(prog)
        self.source.t_regs["t"] = "s_out_time"

    def translate(self, prog):
        WriteReg(dst="s_out_time", src=100).translate(prog)
        self.source.translate(prog)


class Program(MyProgramV2):
    def _initialize(self, cfg):
        self.declare_gen(0, nqz=1)
        self.add_pulse(0, "p", style="const", freq=1000, phase=0, gain=0.5, length=0.1)
        self.declare_readout(0, length=0.1)
        self.add_readoutconfig(0, "ro", freq=1000, gen_ch=0)

    def _body(self, cfg):
        self.append_macro(Adapt(Pulse(ch=0, name="p", t=0, tag=None)))
        self.append_macro(Adapt(ConfigReadout(ch=0, name="ro", t=0, tag=None)))
        self.append_macro(Adapt(Trigger(ros=[0], pins=[], tts=[], t=0, width=0.01, ddr4=False, mr=False, tag=None)))


prog = Program(make_mock_soccfg(1, 1), reps=1, initial_delay=0, relax_delay=0)
insts = [record.instruction.to_dict() for record in prog.compiled_stream.records if record.instruction is not None]
copies = [i for i in insts if i.get("CMD") == "REG_WR" and i.get("DST") == "s14" and i.get("OP") == "s14"]
ports = [i for i in insts if i.get("CMD") in ("WPORT_WR", "DPORT_WR", "TRIG")]
print("self copies:", copies, "ports:", ports)
assert len(copies) == 3
assert len(ports) == 4 and all("TIME" not in i for i in ports)
print("QICK public s_out_time adaptation accepted for pulse, config and trigger")
