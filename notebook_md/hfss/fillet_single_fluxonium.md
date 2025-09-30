<!-- #region vscode={"languageId": "raw"} -->
# Add rounded tee for coupler
<!-- #endregion -->

```python
%reload_ext autoreload
%autoreload 2
import qiskit_metal as metal
from qiskit_metal import designs, draw
from qiskit_metal import MetalGUI, Dict, open_docs
from collections import OrderedDict

from qiskit_metal.qlibrary.sample_shapes.rectangle import Rectangle
from qiskit_metal.qlibrary.sample_shapes.rectangle_hollow import RectangleHollow
from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket
from qiskit_metal.qlibrary.user_components.fillet_q import Fillet_Qubit
from qiskit_metal.qlibrary.user_components.round_tee import Round_Tee

from qiskit_metal.qlibrary.terminations.open_to_ground import OpenToGround
from qiskit_metal.qlibrary.terminations.short_to_ground import ShortToGround

from qiskit_metal.qlibrary.couplers.line_tee import LineTee
from qiskit_metal.qlibrary.couplers.coupled_line_tee import CoupledLineTee
from qiskit_metal.qlibrary.terminations.launchpad_wb import LaunchpadWirebond
from qiskit_metal.qlibrary.terminations.launchpad_wb_coupled import (
    LaunchpadWirebondCoupled,
)
from qiskit_metal.qlibrary.terminations.launchpad_wb_driven import (
    LaunchpadWirebondDriven,
)

from qiskit_metal.qlibrary.tlines.anchored_path import RouteAnchors
from qiskit_metal.qlibrary.tlines.meandered import RouteMeander
from qiskit_metal.qlibrary.tlines.pathfinder import RoutePathfinder
from qiskit_metal.qlibrary.tlines.straight_path import RouteStraight
from qiskit_metal.toolbox_metal.parsing import parse_value

from qiskit_metal.analyses.quantization import EPRanalysis
import pyEPR as epr
from qiskit_metal.analyses.simulation import ScatteringImpedanceSim
from qiskit_metal.analyses.sweep_and_optimize.sweeping import Sweeping

import numpy as np
import matplotlib.pyplot as plt
```

# Circuit design

```python
design = designs.DesignPlanar()
gui = MetalGUI(design)
design.overwrite_enabled = True
design.chips.main.size.size_x = 10.000
design.chips.main.size.size_y = 10.000
design.chips.main.size.size_z = -0.650
design.chips.main.material = "sapphire"
design.variables.cpw_width = 0.015
design.variables.cpw_gap = 0.009
var = design.variables
cpw_width, cpw_gap = var.cpw_width, var.cpw_gap
print(var)
```

```python
Q1 = Fillet_Qubit(
    design,
    "Q1",
    dict(
        pos_x=0,
        pos_y=0,
        orientation=0,
        layer="1",
        hfss_inductance="10nH",
        pad_gap=0.030,
        arm_width=0.008,
        arm_length=0.090,
        arm_fillet=0.020,
        pad_width=0.150,
        pad_height=0.050,
        pad_fillet=0.050,
        pocket_width=0.350,
        pocket_height=0.550,
        pocket_fillet=0.050,
    ),
)
gui.rebuild()
gui.autoscale()
gui.screenshot()
```

```python
x_TL, y_TL = 1.2, 2.0
port_L = LaunchpadWirebondDriven(
    design,
    "port_L",
    options=dict(
        pad_width=0.5,
        pad_height=0.25,
        pad_gap=0.2,
        taper_height=0.5,
        pos_x=-x_TL,
        pos_y=y_TL,
        orientation="  0",
        lead_length=0.05,
        trace_width=0.020,
        trace_gap=0.012,
    ),
)
port_R = LaunchpadWirebondDriven(
    design,
    "port_R",
    options=dict(
        pad_width=0.5,
        pad_height=0.25,
        pad_gap=0.2,
        taper_height=0.5,
        pos_x=x_TL,
        pos_y=y_TL,
        orientation="180",
        lead_length=0.05,
        trace_width=0.020,
        trace_gap=0.012,
    ),
)
TL_width, TL_gap = 0.020, 0.012
TL = RouteStraight(
    design,
    "TL",
    dict(
        pin_inputs=dict(
            start_pin=dict(component="port_L", pin="tie"),
            end_pin=dict(component="port_R", pin="tie"),
        ),
        trace_width=TL_width,
        trace_gap=TL_gap,
        hfss_wire_bonds=True,
    ),
)

R1_len = 4.180
spacing, fillet = 0.300, 0.050  # fillet=1.5*(res_width+res_gap*2)
res_width, res_gap = 0.015, 0.009
l_couple, grd_gap = 0.330, 0.004
R_ext, shift_x, shift_y = 0.200, 0, 0.250
tee_width, tee_length, tee_fillet = 0.020, 0.100, 0.005

Tee = Round_Tee(
    design,
    "Tee",
    dict(
        ref_qc=Q1,
        rel_ori="90",
        shift_x=shift_x,
        shift_y=shift_y,
        cpw_width=res_width,
        cpw_gap=res_gap,
        ext_len=R_ext,
        tee_width=tee_width,
        tee_length=tee_length,
        tee_fillet=tee_fillet,
    ),
)

stg_q1 = ShortToGround(
    design,
    "stg_q1",
    {
        "pos_x": Q1.options.pos_x,
        "pos_y": Tee.options.shift_y + Tee.options.tee_width / 2 + Tee.options.ext_len,
        "orientation": "270",
    },
)
stg_r1 = ShortToGround(
    design,
    "stg_r1",
    {
        "pos_x": Q1.options.pos_x - (l_couple + fillet),
        "pos_y": port_L.options.pos_y
        - TL_gap
        - res_gap
        - (TL_width + res_width) / 2
        - grd_gap,
        "orientation": "180",
    },
)
start_jog = OrderedDict()
start_jog[0] = ["R", spacing * 2]
R1 = RouteMeander(
    design,
    "R1",
    options=dict(
        total_length=R1_len - R_ext,
        fillet=fillet,
        hfss_wire_bonds=True,
        lead=dict(
            start_straight=l_couple + fillet,
            end_straight=0.1,
            start_jogged_extension=start_jog,
        ),
        trace_width=res_width,
        trace_gap=res_gap,
        meander=dict(spacing=spacing, asymmetry=0),
        pin_inputs=Dict(
            start_pin=Dict(component="stg_r1", pin="short"),
            end_pin=Dict(component="stg_q1", pin="short"),
        ),
    ),
)

gui.rebuild()
gui.autoscale()
# gui.zoom_on_components(["Q1"])
gui.screenshot()
```

```python
cl_vac, cl_grd, cl_ext = 0.020, 0.010, 0.150
cl_q1_gap = Rectangle(
    design,
    "cl_q1_gap",
    dict(
        pos_x=Q1.options.pos_x,
        pos_y=Q1.options.pos_y - Q1.options.pocket_height / 2 - cl_grd - cl_ext / 2,
        width=cpw_gap * 2 + cpw_width,
        height=cl_ext,
        subtract=True,
    ),
)
cl_q1_trace = Rectangle(
    design,
    "cl_q1_trace",
    dict(
        pos_x=Q1.options.pos_x,
        pos_y=Q1.options.pos_y - Q1.options.pocket_height / 2 - cl_grd - cl_ext / 2,
        width=cpw_width,
        height=cl_ext - cl_vac * 2,
    ),
)
cl_q1_stg = ShortToGround(
    design,
    "cl_q1_stg",
    dict(
        pos_x=Q1.options.pos_x,
        pos_y=Q1.options.pos_y
        - Q1.options.pocket_height / 2
        - cl_grd
        - cl_ext
        + cl_vac,
        orientation=90,
    ),
)
port_cl = LaunchpadWirebondDriven(
    design,
    "port_cl",
    options=dict(
        pad_width=0.5,
        pad_height=0.25,
        pad_gap=0.2,
        taper_height=0.5,
        pos_x=0,
        pos_y=-1,
        orientation=" 90",
        lead_length=0.05,
    ),
)
cl_q1_route = RoutePathfinder(
    design,
    "cl_q1_route",
    dict(
        trace_width=cpw_width,
        trace_gap=cpw_gap,
        fillet=0.100,
        hfss_wire_bonds=True,
        lead=dict(start_straight=0.100, end_straight=0.100),
        pin_inputs=Dict(
            start_pin=Dict(component="port_cl", pin="tie"),
            end_pin=Dict(component="cl_q1_stg", pin="short"),
        ),
    ),
)
gui.rebuild()
gui.autoscale()
gui.screenshot()
```

```python
q_loop_width, q_loop_height, q_loop_trace = 0.050, 0.010, 0.002
q_loop = RectangleHollow(
    design,
    "q_loop",
    dict(
        pos_x=Q1.options.pos_x
        - (q_loop_trace + q_loop_width + Q1.options.arm_width) / 2,
        pos_y=Q1.options.pos_y,
        width=q_loop_width + q_loop_trace,
        height=q_loop_height + 2 * q_loop_trace,
        inner=dict(
            width=q_loop_width,
            height=q_loop_height,
            offset_x=+q_loop_trace / 2,
            offset_y=0,
            orientation=0,
        ),
    ),
)

loop_H, loop_W, loop_wid = 0.050, 0.090, 0.005
fl_loop_stg_1 = ShortToGround(
    design,
    "fl_loop_stg_1",
    dict(
        pos_x=Q1.options.pos_x - Q1.options.pocket_width / 2,
        pos_y=Q1.options.pos_y - loop_H / 2,
        orientation=180,
    ),
)
fl_loop_stg_2 = ShortToGround(
    design,
    "fl_loop_stg_2",
    dict(
        pos_x=Q1.options.pos_x - Q1.options.pocket_width / 2,
        pos_y=Q1.options.pos_y + loop_H / 2,
        orientation=180,
    ),
)
fl_loop_route = RouteStraight(
    design,
    "fl_loop_route",
    dict(
        trace_width=loop_wid,
        trace_gap=0,
        fillet=0.010,
        hfss_wire_bonds=False,
        lead=dict(start_straight=loop_W, end_straight=loop_W),
        pin_inputs=Dict(
            start_pin=Dict(component="fl_loop_stg_1", pin="short"),
            end_pin=Dict(component="fl_loop_stg_2", pin="short"),
        ),
    ),
)
fl_q1_otg = ShortToGround(
    design,
    "fl_q1_otg",
    dict(
        pos_x=Q1.options.pos_x - Q1.options.pocket_width / 2,
        pos_y=Q1.options.pos_y - loop_H / 2,
        orientation=0,
    ),
)
port_fl = LaunchpadWirebondDriven(
    design,
    "port_fl",
    options=dict(
        pad_width=0.5,
        pad_height=0.25,
        pad_gap=0.2,
        taper_height=0.5,
        pos_x=-x_TL,
        pos_y=fl_loop_stg_1.options.pos_y,
        orientation="  0",
        lead_length=0.05,
    ),
)
fl_q1_route = RouteStraight(
    design,
    "fl_q1_route",
    dict(
        trace_width=cpw_width,
        trace_gap=cpw_gap,
        fillet=0.100,
        hfss_wire_bonds=True,
        lead=dict(start_straight=0.100, end_straight=0.100),
        pin_inputs=Dict(
            start_pin=Dict(component="port_fl", pin="tie"),
            end_pin=Dict(component="fl_q1_otg", pin="short"),
        ),
    ),
)
gui.rebuild()
gui.autoscale()
gui.screenshot()
```

```python
design.components.keys()
```

# Simulation

<!-- #raw -->
1. Cropped circuit near qubit
   a. Q3D, transformed capacitance matrix: EC, coupling C to charge line and resonator
   b. Eigenmode: check fq v.s Lj for EC
2. TL + resonator + qubit
   a. Eigenmode + drivenmodal: TL-resonator for linewidth, omitting qubit
   b. Eigenmode sweep inductance: Qubit-resonator coupling by anticrossing
3. Charge line + qubit with cropped circuit
   Eigenmode + driven modal: qubit reflection linewidth
4. Flux line + qubit with cropped circuit
   Eigenmode + driven modal: qubit reflection (S21) linewidth and mutual inductance (L = Im[Z21])
<!-- #endraw -->

```python
from qiskit_metal.analyses.quantization import EPRanalysis
from qiskit_metal.analyses.simulation import ScatteringImpedanceSim
from qiskit_metal.analyses.quantization import LOManalysis
```

```python
import scipy.constants as c

EC = lambda L, f: f"{c.e**2 / 2 / c.h * (2 * c.pi * f) ** 2 * L * 1e-9:.3f} GHz"
print(EC(40.9e-9, 5.7627e9))
L = lambda EC, f: f"{EC / (c.e**2 / 2 / c.h * (2 * c.pi * f) ** 2) * 1e9:.2f} nH"
print(L(0.85e9, 7.35e9))
f = lambda EC, L: f"{(EC * 2 * c.h / c.e**2 / L) ** 0.5 / 2 / c.pi * 1e-9:.3f} GHz"
print(f(0.212e9, 10e-9))
C = lambda EC: f"{c.e**2 / (2 * EC * c.h) * 1e15:.3f} fF"
print(C(0.855e9))

EL2L = lambda f: 1 / (f * 1e9) * (c.hbar / 2 / c.e) ** 2 * 1e9 / c.h  # GHz->nH
EC2C = lambda EC: c.e**2 / c.h / 2 / EC * 1e-9 * 1e15  # GHz->fF
L2EL = lambda L: 1 / L * (c.hbar / 2 / c.e) ** 2 / c.h  # nH->GHz
print(L2EL(176), "GHz")
C2EC = lambda C: c.e**2 / 2 / C / c.h * 1e6  # fF->GHz
nq = lambda EL, EC: (EL / 32 / EC) ** 0.25
print(nq(1.039, L2EL(40.9)))
```

```python
61.2 / 0.2788692230085067
```

## 1a. Cropped circuit with Q3D

```python
c2 = LOManalysis(design, "q3d")
q3d = c2.sim.renderer
# Manually set solution setups
q3d.start()
c2.sim.default_setup.update(
    dict(
        save_fields=True,
        percent_error=0.1,
        percent_refinement=30,
        max_passes=10,
        min_converged_passes=3,
    )
)
q3d.options.x_buffer_width_mm = 0.25
q3d.options.y_buffer_width_mm = 0.35
q3d.activate_ansys_design("1a_Cropped_qubit_a", "capacitive")
q3d.clean_active_design()
q3d.render_design(
    [
        "Q1",
        "cl_q1_gap",
        "cl_q1_trace",
        "fl_loop_stg_1",
        "fl_loop_stg_2",
        "fl_loop_route",
        "Tee",
    ],
    [],
)
# q3d.modeler.mesh_length(
#     'mesh', ['pad_bot_0_Q1', 'pad_bot_1_Q1', 'pad_top_0_Q1', 'pad_top_1_Q1',
#              'rectangle_R1_ext_trace', 'rectangle_cl_q1_trace',
#              # 'trace_fl_loop_route'
#             ],
#     MaxLength='0.015mm')
# q3d.analyze_setup("Setup")
# c2.sim.close()
```

```python
q3d.save_screenshot()
```

```python
c2.sim.capacitance_matrix, c2.sim.units = q3d.get_capacitance_matrix()
c2.sim.capacitance_all_passes, _ = q3d.get_capacitance_all_passes()
# c2.sim.close()
c2.sim.capacitance_matrix
```

```python
C_matrix = c2.sim.capacitance_matrix.to_numpy()
M = np.eye(C_matrix.shape[0])
M[1, 2], M[2, 1], M[2, 2] = 1, 1, -1
C_matrix = np.dot(np.dot(np.linalg.inv(M), C_matrix), np.linalg.inv(M))
np.set_printoptions(precision=4, suppress=True)
print(C_matrix)
```

```python
C_matrix.diagonal()
```

```python
import scipy.constants as c

# (Cg1+Cg2)/2 + C12
# f"EC = {c.e**2/(2*C_matrix[2,2]*1e-15) / c.h * 1e-9:.4f} GHz"
print(
    "Result:",
    f"\nEC = {c.e**2 / (2 * C_matrix[2, 2] * 1e-15) / c.h * 1e-9:.4f} GHz, Cq = {C_matrix[2, 2]:.3f} fF, Cc = {C_matrix[2, 4]:.3f} fF",
    f"\nCc/Cq\t = {C_matrix[2, 4] / C_matrix[2, 2]:.4f}",
)
```

```python
# f"EC = {c.e**2/(((31.476+32.521)/2+2.8177 + 1.5135 + 0.0674)*1e-15) / c.h * 1e-9:.4f} GHz"
f"EC = {c.e**2 / ((41.77557 + 42.83942) / 2 + 3.90581 + 3.32508 - 0.14850) / 1e-15 / c.h * 1e-9:.4f} GHz"
```

```python
func_C = lambda EC: c.e**2 / (2 * EC * c.h) * 1e15 * 1e-9
EC_list = np.linspace(0.8, 1.6, 101)
C_list = func_C(EC_list)

plt.plot(EC_list, C_list, ".")
plt.grid()
plt.show()
```

```python
from scipy.optimize import curve_fit

eps, CQ = [9.4, 10, 11.58], [17.9259, 18.9842, 21.6984]
y = lambda x, a, b: a * x + b
coeff, _ = curve_fit(y, eps, CQ)
print(coeff)
```

```python
# x = (y-b)/a
(20.717 - 1.6924201) / 1.72792898
```

```python
# plt.plot([9.4, 10, 11.58], [17.9259, 18.9842, 21.6984], '.-')
plt.plot([9.4, 10, 11.58], [1.0806, 1.0203, 0.8927], ".-")
plt.xlabel(r"$\epsilon$")
plt.ylabel("$C_q$ (fF)")
plt.show()
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## 1b. Cropped circuit with eigenmode
<!-- #endregion -->

```python
eig_cropped = EPRanalysis(design, "hfss")
cropped_hfss = eig_cropped.sim.renderer
cropped_hfss.options["x_buffer_width_mm"] = 0.4
cropped_hfss.options["y_buffer_width_mm"] = 0.4
cropped_hfss.start()
cropped_hfss.activate_ansys_design("1b_Cropped_qubit_eig_08", "eigenmode")
cropped_hfss.clean_active_design()
cropped_hfss.render_design(
    [
        "Q1",
        "cl_q1_gap",
        "cl_q1_trace",
        # 'fl_loop_stg_1', 'fl_loop_stg_2', 'fl_loop_route',
        "R1_ext_gap",
        "R1_ext_trace",
    ],
    [],
)
cropped_hfss.modeler.mesh_length(
    "mesh",
    [
        "pad_bot_0_Q1",
        "pad_bot_1_Q1",
        "pad_top_0_Q1",
        "pad_top_1_Q1",
        "rectangle_R1_ext_trace",
        "rectangle_cl_q1_trace",
        # 'trace_fl_loop_route'
    ],
    MaxLength="0.025mm",
)
```

```python
cropped_hfss.save_screenshot()
eig_cropped.sim.close()
```

```python
EC = lambda L, f: f"{c.e**2 / 2 / c.h * (2 * c.pi * f) ** 2 * L * 1e-9:.3f} GHz"
print(EC(70e-9, 10.0e9))
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## 2a. and 2b. Resonator linewidth with eigenmode and driven modal
<!-- #endregion -->

```python
R1_len
```

```python
# Better run simulation on large-RAM PC
dm = ScatteringImpedanceSim(design, "hfss")
dm.renderer.start()
dm.renderer.activate_ansys_design(
    f"2a_resonator_linewidth_L={R1_len:.3f}mm_Lc={(l_couple - 1.5 * (res_width + res_gap * 2)):.3f}mm",
    "drivenmodal",
)
dm.renderer.options["x_buffer_width_mm"] = 0.35
dm.renderer.options["y_buffer_width_mm"] = 0.35
# dm.setup.update(
#     name = "setup", freq_ghz = 6.4, max_delta_s = 0.005, max_passes = 10, min_converged = 3, basis_order = 1
# )
# dm.setup.sweep_setup.update(
#     name = "sweep_setup", start_ghz = 6.3, stop_ghz = 6.5, count = 401, type = "Fast"
# )
dm.renderer.clean_active_design()
port_list = [("port_L", "in", 50), ("port_R", "in", 50)]
dm.renderer.render_design(
    [
        "Q1",
        "port_L",
        "port_R",
        "TL",
        "R1",
        "Tee",
        "stg_q1",
        "stg_r1",
        "cl_q1_gap",
        "cl_q1_trace",
        "fl_loop_stg_1",
        "fl_loop_stg_2",
        "fl_loop_route",
    ],
    [],
    port_list=port_list,
)
# dm.renderer.modeler.mesh_length('trace', ['trace_R1', 'trace_TL_L', 'trace_TL_R', 'rectangle_R1_ext_trace',
#                                           'second_cpw_couple_tee', 'prime_cpw_couple_tee'], MaxLength='0.007mm')
# dm.renderer.modeler.mesh_length('port', ['Port_port_L_in', 'Port_port_R_in', 'launch_pad_port_L', 'launch_pad_port_R'], MaxLength='0.020mm')
dm.close()
```

## 2c. Qubit-resonator coupling

<!-- #raw -->
Using same model as metioned above.
<!-- #endraw -->

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## 3. Charge line decay rate
<!-- #endregion -->

```python
dm_cl = ScatteringImpedanceSim(design, "hfss")
dm_cl.renderer.start()
dm_cl.renderer.activate_ansys_design("3a_charge_line_eig_4", "drivenmodal")
dm_cl.renderer.options["x_buffer_width_mm"] = 0.4
dm_cl.renderer.options["y_buffer_width_mm"] = 0.4
# dm_cl.setup.update(
#     name = "setup", freq_ghz = 6.4, max_delta_s = 0.005, max_passes = 10, min_converged = 3, basis_order = 1
# )
# dm_cl.setup.sweep_setup.update(
#     name = "sweep_setup", start_ghz = 6.3, stop_ghz = 6.5, count = 401, type = "Fast"
# )
dm_cl.renderer.clean_active_design()
port_list = [("port_cl", "in", 50)]
dm_cl.renderer.render_design(
    [
        "Q1",
        "R1_ext_gap",
        "R1_ext_trace",
        "cl_q1_stg",
        "port_cl",
        "cl_q1_route",
        "cl_q1_gap",
        "cl_q1_trace",
        # 'fl_loop_stg_1', 'fl_loop_stg_2', 'fl_loop_route'
    ],
    [],
    port_list=port_list,
)
# dm_cl.renderer.modeler.mesh_length('cpw_mesh', ['trace_R1', 'trace_TL'], MaxLength='0.007mm')
dm_cl.close()
```

## 4a. Flux line mutual inductance

```python
dm_fl = ScatteringImpedanceSim(design, "hfss")
dm_fl.renderer.start()
dm_fl.renderer.activate_ansys_design("4a_fl_mutual_dm", "drivenmodal")
dm_fl.renderer.options["x_buffer_width_mm"] = 0.4
dm_fl.renderer.options["y_buffer_width_mm"] = 0.4
# dm_fl.setup.update(
#     name = "setup", freq_ghz = 6.4, max_delta_s = 0.005, max_passes = 10, min_converged = 3, basis_order = 1
# )
# dm_fl.setup.sweep_setup.update(
#     name = "sweep_setup", start_ghz = 6.3, stop_ghz = 6.5, count = 401, type = "Fast"
# )
dm_fl.renderer.clean_active_design()
port_list = [("port_fl", "in", 50)]
dm_fl.renderer.render_design(
    [
        "Q1",
        "R1_ext_gap",
        "R1_ext_trace",
        "cl_q1_gap",
        "cl_q1_trace",
        "fl_q1_otg",
        "port_fl",
        "fl_q1_route",
        "fl_loop_stg_1",
        "fl_loop_stg_2",
        "fl_loop_route",
        "q_loop",
    ],
    [],
    port_list=port_list,
)
# dm.renderer.modeler.mesh_length('cpw_mesh', ['trace_R1', 'trace_TL'], MaxLength='0.007mm')
dm_fl.close()
```

<!-- #region jp-MarkdownHeadingCollapsed=true -->
## 4b. q3d mutual inductance
<!-- #endregion -->

```python
M = LOManalysis(design, "q3d")
q3d = M.sim.renderer
# M.sim.default_setup.update(dict(save_fields=True, percent_error=0.1, percent_refinement=30, max_passes=25, min_converged_passes=5))
# Manually set up solution setups
q3d.start()
q3d.options.x_buffer_width_mm = 0.25
q3d.options.y_buffer_width_mm = 0.25
q3d.activate_ansys_design("4b_Cropped_qubit_L", "capacitive")
q3d.clean_active_design()
q3d.render_design(
    [
        "Q1",
        "Q1_top_pad",
        "Q1_bot_pad",
        "cl_q1_gap",
        "cl_q1_trace",
        "fl_loop_stg_1",
        "fl_loop_stg_2",
        "fl_loop_route",
        "R1_ext_gap",
        "R1_ext_trace",
        "q_loop",
    ],
    [],
)
# q3d.analyze_setup("Setup")
M.sim.close()
```

```python
# d,w,L: distance from flux line, loop_width, loop_length
# Ip: current applied for one sc flux quanta in loop
import scipy.constants as c
import numpy as np

d, w, L = 10e-6, 10e-6, 10e-6
Ip = (
    lambda d,
    w,
    L: f"{c.pi * c.h / (c.mu_0 * c.e * w) / (np.log((d + L) / d)) * 1e6:.2f} uA"
)
M = lambda d, w, L: f"{c.mu_0 / c.pi * w * np.log((d + L) / d) * 1e12:.3f} pH"
print(Ip(d, w, L))
print(M(d, w, L))
```
