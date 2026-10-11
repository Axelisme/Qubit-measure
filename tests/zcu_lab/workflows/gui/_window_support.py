"""Window RPC call types and explicit Start batches."""

from collections.abc import Callable

from tests.zcu_lab.workflows._sim_inputs import Inputs
from zcu_lab.workflows.gui.remote import FormWrite

RpcRequest = Callable[[str, dict[str, object]], dict[str, object]]


def start_plan_batch(_inputs: Inputs) -> list[FormWrite]:
    """Select the fixture's native modules through captured public library roots.

    Waveform's reference/inline union has no writable discriminator child path;
    reference selections use existing supported leaves without a tree encoder.
    Omit points so Start must use defaults rather than the old GUI draft.
    """
    return [
        {"path": ["flux_dev"], "value": "flux"},
        {"path": ["flux_unit"], "value": "native"},
        {"path": ["fluxes", "kind"], "value": "list"},
        {"path": ["fluxes", "values"], "value": [1.0, 1.1]},
        {"path": ["pi_length_us"], "value": {"expr": "0.06 * 2"}},
        {"path": ["qub_pulse", "kind"], "value": "ref"},
        {"path": ["qub_pulse", "name"], "value": "Q1.pulses.pi"},
        {"path": ["readout", "kind"], "value": "ref"},
        {"path": ["readout", "name"], "value": "Q1.readout.ro"},
    ]
