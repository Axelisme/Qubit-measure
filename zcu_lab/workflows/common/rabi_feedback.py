"""Run-local drive-gain feedback used by calibrated flux sweeps."""

import math
from dataclasses import dataclass, field
from typing import Literal

from zcu_lab.workflows.common.feedback import LogStepController
from zcu_lab.workflows.common.gain import GainClamp, clamp_gain


@dataclass
class RabiDriveFeedback:
    """Blend pi-product prediction with a query-aged log-step gain proposal.

    pi_product is the gap-smoothed accepted pi duration times acquisition gain,
    or None before the first accepted fit. last_index is its zero-based flux
    index. controller uses step_gain=.5 and confidence decay over three queries.
    Quality-trusted too-fast/too-slow rejections propose lower/higher gain without
    learning an accepted pi product. Until an accepted observation clears the
    direction, confidence decay cannot move gain opposite the latest rejection.
    If the open-loop base lies in that opposite direction, selection holds the
    proposal instead of decaying toward base. Other failures only age confidence.
    """

    pi_product: float | None = None
    last_index: int | None = None
    controller: LogStepController = field(
        default_factory=lambda: LogStepController(step_gain=0.5, decay_points=3.0)
    )

    _rejected_direction: Literal["fast", "slow"] | None = field(
        default=None, init=False
    )

    def select_gain(
        self,
        seed_product: float,
        target_pi_us: float,
        *,
        m_ratio: float | None = None,
        gain_clamped: list[GainClamp] | None = None,
    ) -> float:
        """Return a gain in (0, 1] from a positive seed product and target in us.

        The open-loop product/(1.2*target) is divided by m_ratio before clipping
        to 1. m_ratio is the current charge matrix element / the seed calibration
        matrix element; None disables feed-forward. A provided ratio must be
        positive and finite. The closed-loop proposal is unchanged and blends
        geometrically with the corrected base. Each call ages the proposal's
        confidence, even after failure.
        A too-slow proposal is a lower bound for base; a too-fast proposal is an
        upper bound. This prevents aging from reversing trusted rejection feedback.
        Invalid positive inputs raise ValueError before selecting a gain.
        gain_clamped optionally collects any before/after DAC limiting.
        """
        if not all(math.isfinite(x) and x > 0 for x in (seed_product, target_pi_us)):
            raise ValueError("Rabi seed product and target must be positive and finite")
        product = seed_product if self.pi_product is None else self.pi_product
        if m_ratio is not None and (not math.isfinite(m_ratio) or m_ratio <= 0):
            raise ValueError("Rabi matrix-element ratio must be positive and finite")
        base = clamp_gain(
            product / (1.2 * target_pi_us) / (1.0 if m_ratio is None else m_ratio),
            "rabi.base",
            gain_clamped,
            lower=float.fromhex("0x1p-1022"),
        )
        latest = self.controller.latest()
        if latest is not None:
            if self._rejected_direction == "slow":
                base = max(base, latest.value)
            elif self._rejected_direction == "fast":
                base = min(base, latest.value)
            base *= math.exp(latest.confidence * math.log(latest.value / base))
        # Keep a strictly positive representable gain without imposing a new .01 floor.
        return clamp_gain(
            base, "rabi.drive", gain_clamped, lower=float.fromhex("0x1p-1022")
        )

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
        self._rejected_direction = "fast"

    def observe_too_slow(self, pi_us: float, gain: float, target_pi_us: float) -> None:
        """Propose higher gain from a quality-trusted rejected slow fit.

        Durations are positive finite us, with pi_us strictly above target_pi_us.
        gain is the actual acquisition gain in (0, 1]. Invalid inputs raise
        ValueError. Accepted product/index remain unchanged. Until acceptance or
        another trusted rejection, confidence decay cannot lower this proposal.
        """
        if not all(math.isfinite(x) and x > 0 for x in (pi_us, gain, target_pi_us)):
            raise ValueError("Slow Rabi observation requires positive finite inputs")
        if gain > 1 or pi_us <= target_pi_us:
            raise ValueError("Slow Rabi requires gain <= 1 and target shorter than pi")
        self.controller.propose(gain, math.log(pi_us / target_pi_us))
        self._rejected_direction = "slow"

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
        self._rejected_direction = None
        if control:
            self.controller.propose(gain, math.log(pi_us / target_pi_us))
