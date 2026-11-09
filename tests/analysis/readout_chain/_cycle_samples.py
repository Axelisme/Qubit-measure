"""Independent cycle generator used by public-model acceptance tests."""

import numpy as np
from zcu_tools.analysis.readout_chain import CycleModel, CycleNuisances, CycleSums


def draw_rows(rng: np.random.Generator, probabilities: np.ndarray) -> np.ndarray:
    return (
        rng.random(len(probabilities))[:, None]
        > np.cumsum(probabilities, axis=1)[:, :2]
    ).sum(axis=1)


def cycle_samples(
    model: CycleModel,
    x: float,
    parameters: dict[str, float],
    nuisance: CycleNuisances,
    count: int,
    seed: int,
) -> np.ndarray:
    """Simulate the Markov state path and exact continuous read-jump time."""
    rng = np.random.default_rng(seed)
    fx = model.curve(np.array([x]), parameters)[0]
    drive = (1 - fx[:, None]) * nuisance.drive_endpoints[:, 0] + fx[
        :, None
    ] * nuisance.drive_endpoints[:, 1]
    read = np.diag(nuisance.read_retention) + np.outer(
        1 - nuisance.read_retention, nuisance.read_landing
    )
    state = rng.choice(3, size=count, p=nuisance.thermal)
    iq = np.empty((count, 16, 2))
    for k in range(16):
        if k % 2 == 0:
            state = draw_rows(rng, drive[state])
        after = draw_rows(rng, read[state])
        probability = read[state, after]
        changed = state != after
        u = np.ones(count)
        rate = -np.log1p(-probability[changed])
        u[changed] = -np.log1p(-rng.random(changed.sum()) * probability[changed]) / rate
        iq[:, k] = nuisance.centers[after] + u[:, None] * (
            nuisance.centers[state] - nuisance.centers[after]
        )
        iq[:, k] += rng.normal(scale=np.sqrt(nuisance.variance), size=(count, 2))
        state = after
    return features(iq)


def features(iq: np.ndarray) -> np.ndarray:
    """Apply the saved layout directly to shots, independently of the moment engine."""
    result = np.zeros((len(iq), 65))
    result[:, :32] = iq.reshape(len(iq), 32)
    for reads, offset in [
        ([0], 32),
        (list(range(2, 16, 2)), 35),
        (list(range(1, 16, 2)), 38),
    ]:
        for c, (a, b) in enumerate(((0, 0), (0, 1), (1, 1))):
            result[:, offset + c] = (iq[:, reads, a] * iq[:, reads, b]).sum(axis=1)
    for i in range(16):
        for j in range(i + 1, 16):
            d = sum(i < 2 * k <= j for k in range(8))
            gap = j - i
            if d == 0:
                group = 0
            elif d == 1:
                group = 1 if gap == 1 else 2
            elif d == 2:
                group = 3 if gap <= 3 else 4
            else:
                group = 5
            for a in range(2):
                for b in range(2):
                    result[:, 41 + 4 * group + 2 * a + b] += iq[:, i, a] * iq[:, j, b]
    return result


def sampled_sums(
    model: CycleModel,
    nuisance: CycleNuisances,
    axis: np.ndarray,
    parameters: dict[str, float],
    count: int,
    seed: int,
) -> CycleSums:
    sums = np.empty((len(axis), 65))
    products = np.zeros((65, 65))
    for i, x in enumerate(axis):
        shots = cycle_samples(model, x, parameters, nuisance, count, seed + 1000 * i)
        sums[i] = shots.sum(axis=0)
        products += shots.T @ shots
    return CycleSums(axis, np.full(len(axis), float(count)), sums, products)
