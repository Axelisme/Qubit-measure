"""Fixed-coordinate joint FTF circuits, multilevel baths and parent readout.

Pure numeric GHz/ns/K inputs only. Environment owns YAML/device/channel mapping;
Evolution owns propagation and ADC owns the shared-line conditional field solve.
"""

from .model import FluxDynamics, JointModel
from .parameters import FluxoniumCircuit, JointParameters, Resonator, SquidCircuit
