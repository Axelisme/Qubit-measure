"""Event tables for the fixed 65 cycle features."""

from functools import lru_cache
from itertools import combinations_with_replacement

import numpy as np

from zcu_tools.core.chain_cycle_groups import pair_group, same_read_group

ORDERS = tuple((a, b) for a in range(5) for b in range(5 - a))


def _events(factors: tuple[tuple[int, int], ...]) -> tuple[list[int], list[int]]:
    powers: dict[int, list[int]] = {}
    for read, component in factors:
        powers.setdefault(read, [0, 0])[component] += 1
    positions = sorted(powers)
    orders = [ORDERS.index(tuple(powers[k])) for k in positions]
    return positions, orders


@lru_cache(maxsize=1)
def tables() -> tuple[np.ndarray, np.ndarray]:
    """Mean rows (group,n,pos*4,orders*4); second rows (g1,g2,n,pos*4,orders*4).

    Primitive products are accumulated directly into stored groups, avoiding
    an intermediate 560-by-560 covariance for every sweep point.
    """
    features: list[tuple[int, tuple[tuple[int, int], ...]]] = [
        (2 * k + a, ((k, a),)) for k in range(16) for a in (0, 1)
    ]
    for k in range(16):
        group = same_read_group(k)
        for c, (a, b) in enumerate(((0, 0), (0, 1), (1, 1))):
            features.append((32 + 3 * group + c, ((k, a), (k, b))))
    for i in range(16):
        for j in range(i + 1, 16):
            for a in (0, 1):
                for b in (0, 1):
                    features.append(
                        (41 + 4 * pair_group(i, j) + 2 * a + b, ((i, a), (j, b)))
                    )
    means, seconds = [], []
    for group, factors in features:
        pos, order = _events(factors)
        means.append(
            [
                group,
                len(pos),
                *pos,
                *([0] * (4 - len(pos))),
                *order,
                *([0] * (4 - len(pos))),
            ]
        )
    for (g, f), (h, v) in combinations_with_replacement(features, 2):
        pos, order = _events(f + v)
        # Distinct primitive terms in the same stored sum contribute twice.
        factor = 2 if g == h and f != v else 1
        seconds.append(
            [
                g,
                h,
                factor,
                len(pos),
                *pos,
                *([0] * (4 - len(pos))),
                *order,
                *([0] * (4 - len(pos))),
            ]
        )
    return np.asarray(means, dtype=np.int64), np.asarray(seconds, dtype=np.int64)
