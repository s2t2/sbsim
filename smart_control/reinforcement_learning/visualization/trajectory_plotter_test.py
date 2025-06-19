import os
import shutil
import tempfile
import unittest
import numpy as np # For sample data

# Attempt to import TrajectoryPlotter, handling potential import issues for plotting libraries
try:
    from smart_control.reinforcement_learning.visualization.trajectory_plotter import TrajectoryPlotter
    # Try to import matplotlib.pyplot to see if it's available and set backend
    import matplotlib
    matplotlib.use('Agg') # Use a non-interactive backend for tests
    import matplotlib.pyplot as plt
    PLOTTER_AVAILABLE = True
except ImportError:
    PLOTTER_AVAILABLE = False

@unittest.skipIf(not PLOTTER_AVAILABLE, "TrajectoryPlotter or Matplotlib not available, skipping tests.")
class TestTrajectoryPlotter(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # Ensure the directory exists, mkdtemp should do this but double check
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    # Tests for plot_actions
    def test_plot_actions_creates_file_basic(self):
        actions = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        save_path = os.path.join(self.test_dir, "actions_plot_basic.png")
        TrajectoryPlotter.plot_actions(actions, save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_plot_actions_creates_file_with_timestamps(self):
        actions = [[0.1], [0.3], [0.5]]
        timestamps = np.array(["2023-01-01T00:00:00", "2023-01-01T00:05:00", "2023-01-01T00:10:00"], dtype='datetime64[s]')
        save_path = os.path.join(self.test_dir, "actions_plot_timestamps.png")
        TrajectoryPlotter.plot_actions(actions, save_path, timestamps=timestamps)
        self.assertTrue(os.path.exists(save_path))

    def test_plot_actions_empty_data(self):
        actions = []
        save_path = os.path.join(self.test_dir, "actions_plot_empty.png")
        TrajectoryPlotter.plot_actions(actions, save_path)
        self.assertTrue(os.path.exists(save_path)) # Should still create an empty plot

    # Tests for plot_rewards
    def test_plot_rewards_creates_file_basic(self):
        rewards = [1.0, 1.5, 0.5, 2.0]
        save_path = os.path.join(self.test_dir, "rewards_plot_basic.png")
        TrajectoryPlotter.plot_rewards(rewards, save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_plot_rewards_creates_file_with_timestamps(self):
        rewards = [10, 12, 11]
        timestamps = np.array(["2023-01-02T10:00:00", "2023-01-02T10:05:00", "2023-01-02T10:10:00"], dtype='datetime64[s]')
        save_path = os.path.join(self.test_dir, "rewards_plot_timestamps.png")
        TrajectoryPlotter.plot_rewards(rewards, save_path, timestamps=timestamps)
        self.assertTrue(os.path.exists(save_path))

    def test_plot_rewards_empty_data(self):
        rewards = []
        save_path = os.path.join(self.test_dir, "rewards_plot_empty.png")
        TrajectoryPlotter.plot_rewards(rewards, save_path)
        self.assertTrue(os.path.exists(save_path))

    # Tests for plot_cumulative_reward
    def test_plot_cumulative_reward_creates_file_basic(self):
        rewards = [0.1, -0.05, 0.2, 0.15]
        save_path = os.path.join(self.test_dir, "cumulative_reward_plot_basic.png")
        TrajectoryPlotter.plot_cumulative_reward(rewards, save_path)
        self.assertTrue(os.path.exists(save_path))

    def test_plot_cumulative_reward_creates_file_with_timestamps(self):
        rewards = [5, 2, -3, 6]
        timestamps = np.array(["2023-01-03T12:00:00", "2023-01-03T12:05:00", "2023-01-03T12:10:00", "2023-01-03T12:15:00"], dtype='datetime64[s]')
        save_path = os.path.join(self.test_dir, "cumulative_reward_plot_timestamps.png")
        TrajectoryPlotter.plot_cumulative_reward(rewards, save_path, timestamps=timestamps)
        self.assertTrue(os.path.exists(save_path))

    def test_plot_cumulative_reward_empty_data(self):
        rewards = []
        save_path = os.path.join(self.test_dir, "cumulative_reward_plot_empty.png")
        TrajectoryPlotter.plot_cumulative_reward(rewards, save_path)
        self.assertTrue(os.path.exists(save_path))

if __name__ == "__main__":
    # This allows running the tests directly
    unittest.main()
