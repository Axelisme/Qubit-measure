"""Run-local drive-gain feedback used by calibrated flux sweeps."""

import math
from dataclasses import dataclass, field

from zcu_lab.workflows.common.feedback import LogStepController


@dataclass
class RabiDriveFeedback:
    """Blend pi-product prediction with a query-aged log-step gain proposal.

    pi_product is the gap-smoothed accepted pi duration times acquisition gain,
    or None before the first accepted fit. last_index is its zero-based flux
    index. controller uses step_gain=.5 and confidence decay over three queries.
    Quality-trusted too-fast rejections may propose a lower gain without learning
    an accepted pi product. Other failures only age confidence on selection.
    """

    pi_product: float | None = None
    last_index: int | None = None
    controller: LogStepController = field(
        default_factory=lambda: LogStepController(step_gain=0.5, decay_points=3.0)
    )

    def select_gain(self, seed_product: float, target_pi_us: float) -> float:
        """Return a gain in (0, 1] from a positive seed product and target in us.

        The open-loop product/(1.2*target) blends geometrically with the latest
        controller proposal. Each call ages its confidence, even after failure.
        Invalid positive inputs raise ValueError before selecting a gain.
        """
        if not all(math.isfinite(x) and x > 0 for x in (seed_product, target_pi_us)):
            raise ValueError("Rabi seed product and target must be positive and finite")
        product = seed_product if self.pi_product is None else self.pi_product
        base = min(1.0, product / (1.2 * target_pi_us))
        latest = self.controller.latest()
        if latest is not None:
            base *= math.exp(latest.confidence * math.log(latest.value / base))
        # Keep a strictly positive representable gain without imposing a new .01 floor.
        return min(1.0, max(float.fromhex("0x1p-1022"), base))

    def observe_too_fast(self, pi_us: float, gain: float, target_pi_us: float) -> None:
        """Propose a lower gain from a quality-trusted rejected fast fit.

        pi_us and target_pi_us are positive finite durations in us, with target
        strictly longer than pi_us. gain is the actual acquisition gain in (0, 1].
        Invalid inputs raise ValueError. Only the controller proposal changes;
        the accepted pi product and its index remain untouched.
        """
        if not all(math.isfinite(x) and x > 0 for x in (pi_us, gain, target_pi_us)):
            raise ValueError("Fast Rabi observation requires positive finite inputs")
        if gain > 1 or pi_us >= target_pi_us:
            raise ValueError("Fast Rabi requires gain <= 1 and target longer than pi")
        self.controller.propose(gain, math.log(pi_us / target_pi_us))

    def observe(
        self,
        index: int,
        pi_us: float,
        gain: float,
        target_pi_us: float,
        *,
        control: bool = True,
    ) -> None:
        """Record an accepted fit and propose the next gain toward target_pi_us.

        index is the current flux index. pi_us and gain are the measured pulse
        duration and actual acquisition gain. Product smoothing uses .7**gap;
        the first accepted observation replaces the seed without smoothing.
        control=False learns the product without changing the gain proposal, so
        fixed-gain observations remain available if auto mode is enabled later.
        """
        product = pi_us * gain
        if self.pi_product is None or self.last_index is None:
            self.pi_product = product
        else:
            weight = 0.7 ** max(1, index - self.last_index)
            self.pi_product = (1 - weight) * product + weight * self.pi_product
        self.last_index = index
        if control:
            self.controller.propose(gain, math.log(pi_us / target_pi_us))
