"""差速/全向运动学速度与转向约束测试。"""
import unittest

import numpy as np

from wanda_nav.motion import build_linear_segment, trapezoidal_speed_profile, yaw_steps


class MotionKinematicsTest(unittest.TestCase):
    def test_speed_profile_respects_limits_and_exact_distance(self):
        dt, distance, vmax, accel, decel = 0.1, 8.7, 3.0, 1.0, 1.5
        speeds = trapezoidal_speed_profile(distance, vmax, accel, decel, dt)
        self.assertGreaterEqual(float(speeds.min()), 0.0)
        self.assertLessEqual(float(speeds.max()), vmax + 1e-9)
        self.assertLessEqual(float(np.diff(speeds).max()), accel * dt + 1e-9)
        self.assertLessEqual(float(-np.diff(speeds).min()), decel * dt + 1e-9)
        traveled = float(np.sum(speeds * dt))
        self.assertAlmostEqual(traveled, distance, places=8)

    def test_short_segment_uses_triangle_profile_without_speed_spike(self):
        speeds = trapezoidal_speed_profile(0.12, 3.0, 1.0, 1.5, 0.1)
        self.assertLess(float(speeds.max()), 1.0)
        self.assertGreaterEqual(float(speeds.min()), 0.0)

    def test_linear_segment_obeys_kinematic_limits_and_reaches_target(self):
        def try_step(start, end):
            return end

        def is_navigable(point, margin):
            return True

        frames, final_position, _ = build_linear_segment(
            np.zeros(3), np.array([5., 0., 0.]), 0.0, 0.0, 10,
            3.0, 1.0, 1.5, try_step, is_navigable)
        positions = np.asarray([np.zeros(3), *[frame[0] for frame in frames]])
        dt = 0.1
        velocity = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt
        self.assertLessEqual(float(velocity.max()), 3.0 + 1e-6)
        self.assertLessEqual(float(np.diff(velocity).max()), 1.0 * dt + 1e-6)
        self.assertLessEqual(float(-np.diff(velocity).min()), 1.5 * dt + 1e-6)
        self.assertTrue(np.allclose(final_position, [5., 0., 0.]))

    def test_yaw_steps_obey_angular_speed_limit(self):
        yaws = np.asarray([0.0, *yaw_steps(0.0, 2.7, 1.2, 0.1)])
        delta = (np.diff(yaws) + np.pi) % (2 * np.pi) - np.pi
        self.assertLessEqual(float(np.abs(delta).max()), 1.2 * 0.1 + 1e-9)
        self.assertAlmostEqual(float(yaws[-1]), 2.7)


if __name__ == "__main__":
    unittest.main()
