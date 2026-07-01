"""predictor — a Service: a Builder whose Node produces by pure computation.

The predictor is the worked example that validates "a Service goes through the
same Builder path as a measurement". It is NOT in the user's Node list and it
touches no hardware: its Builder curries only ``tools`` into the Node (no soc /
Result / round_hook / Plotter), and the Node's ``produce`` computes
``predict_freq`` / ``cur_m`` from the flux point alone. To the orchestrator it
is indistinguishable from a measurement provider — same ``provides`` /
``requires`` / ``produce`` three-interface surface, zero ``isinstance``.

The predictor stays out of the orchestrator: it is loaded only because some Node
(qubit_freq) requires ``predict_freq``, and it participates purely through
requirement resolution.

The actual prediction is delegated to the sweep-lived ``tools.predictor`` (a
``Predictor``; either a ``SimplePredictor`` fallback or a real
``FluxoniumPredictor`` adapter) — its ``bias`` is the cross-flux-point state a
short-lived Node cannot hold, which is precisely why it lives in Tools. The
*calibration* face (qubit_freq handing its measured freq back to adjust the
prediction) is a Tools method a Node triggers.
"""

from __future__ import annotations

from zcu_tools.gui.app.autofluxdep.nodes.builder import Builder, Node, RunEnv
from zcu_tools.gui.app.autofluxdep.nodes.io import Patch, Snapshot


class PredictorNode(Node):
    """Pure-compute Node: predicts this flux point's qubit freq + matrix element.

    Curries in only the flux point and ``tools`` (no soc / Result / round_hook).
    ``produce`` ignores the snapshot (it depends on nothing) and reads the
    sweep-lived ``tools.predictor`` to compute the predictions for this flux.
    """

    def __init__(self, env: RunEnv) -> None:
        self._env = env

    def produce(self, snapshot: Snapshot) -> Patch:
        del snapshot  # the predictor depends on nothing — flux is in the env
        env = self._env
        predictor = env.tools.predictor if env.tools is not None else None
        if predictor is None:
            return Patch()  # no predictor bound → produce nothing (downstream skips)

        patch = Patch()
        patch.set("predict_freq", float(predictor.predict_freq(env.flux)))
        patch.set("cur_m", float(predictor.predict_matrix_element(env.flux)))
        return patch


class PredictorBuilder(Builder):
    """The predictor Service — a stateless Builder producing pure-compute Nodes.

    Declares only ``provides``; it requires nothing (it predicts from flux). Its
    Node curries no measurement environment, so ``make_init_result`` /
    ``make_plotter`` stay the Builder no-ops (a Service draws and stores
    nothing).
    """

    name = "predictor"
    provides = ("predict_freq", "cur_m")

    def build_node(self, env: RunEnv) -> PredictorNode:
        return PredictorNode(env)
