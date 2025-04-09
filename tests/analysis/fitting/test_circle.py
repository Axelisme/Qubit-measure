import unittest

import numpy as np

from lib.zcu_tools.analysis.fitting.circle import circle_fit


class TestCircle(unittest.TestCase):
    def test_circle_fit(self):
        # Create a circular pattern of points with known center and radius
        num_points = 100
        true_center = np.array([3.0, 2.0])
        true_radius = 5.0

        # Generate angles around the circle
        theta = np.linspace(0, 2 * np.pi, num_points)

        # Generate points on the circle with some noise
        noise = np.random.normal(0, 0.1, num_points) + 1j * np.random.normal(
            0, 0.1, num_points
        )
        signals = (
            true_radius * np.exp(1j * theta)
            + true_center[0]
            + 1j * true_center[1]
            + noise
        )

        # Fit the circle
        center, radius = circle_fit(signals)

        # Check if the fitted parameters are close to the true values
        self.assertTrue(np.allclose(center, true_center, atol=0.2))
        self.assertAlmostEqual(radius, true_radius, delta=0.2)

    def test_circle_fit_partial_arc(self):
        # Test with points forming only a partial arc
        num_points = 50
        true_center = np.array([-2.0, 1.0])
        true_radius = 3.0

        # Generate angles for a partial arc (not a full circle)
        theta = np.linspace(0, np.pi, num_points)

        # Generate points on the partial arc with some noise
        noise = np.random.normal(0, 0.05, num_points) + 1j * np.random.normal(
            0, 0.05, num_points
        )
        signals = (
            true_radius * np.exp(1j * theta)
            + true_center[0]
            + 1j * true_center[1]
            + noise
        )

        # Fit the circle
        center, radius = circle_fit(signals)

        # Check if the fitted parameters are reasonably close to the true values
        # For partial arcs, we expect less accuracy, so we use larger tolerances
        self.assertTrue(np.allclose(center, true_center, atol=0.5))
        self.assertAlmostEqual(radius, true_radius, delta=0.5)


if __name__ == "__main__":
    unittest.main()
