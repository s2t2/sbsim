import unittest
import tensorflow as tf # For tf.test.TestCase

from smart_control.reinforcement_learning.observers import base_observer
from tf_agents.trajectories import trajectory
from unittest.mock import MagicMock # Though not explicitly used yet, good to have for potential future mocks.


# Concrete subclass for testing BaseObserver
class ConcreteObserver(base_observer.BaseObserver):
    def __init__(self, priority=0):
        super().__init__(priority)
        self.called_with = None
        self.call_count = 0

    def __call__(self, traj):
        self.called_with = traj
        self.call_count += 1


class BaseObserverTest(tf.test.TestCase):

    def test_init_priority(self):
        observer_default = ConcreteObserver()
        self.assertEqual(observer_default.priority, 0)

        observer_custom = ConcreteObserver(priority=10)
        self.assertEqual(observer_custom.priority, 10)

    def test_call_method(self):
        observer = ConcreteObserver()
        # Create a dummy trajectory.
        # For simplicity, we can use MagicMock if the trajectory's internal structure isn't critical for this test.
        # Or, create a minimal trajectory if specific fields are accessed.
        dummy_trajectory = trajectory.Trajectory(
            step_type=MagicMock(),
            observation=MagicMock(),
            action=MagicMock(),
            policy_info=MagicMock(),
            next_step_type=MagicMock(),
            reward=MagicMock(),
            discount=MagicMock()
        )

        self.assertIsNone(observer.called_with)
        self.assertEqual(observer.call_count, 0)

        observer(dummy_trajectory)

        self.assertIs(observer.called_with, dummy_trajectory)
        self.assertEqual(observer.call_count, 1)

    def test_less_than_operator(self):
        observer1_p0 = ConcreteObserver(priority=0)
        observer2_p1 = ConcreteObserver(priority=1)
        observer3_p1_again = ConcreteObserver(priority=1) # Same priority as observer2
        observer4_p_minus_1 = ConcreteObserver(priority=-1)

        self.assertTrue(observer1_p0 < observer2_p1)
        self.assertFalse(observer2_p1 < observer1_p0)

        self.assertFalse(observer2_p1 < observer3_p1_again) # Equal priorities
        self.assertFalse(observer3_p1_again < observer2_p1) # Equal priorities

        self.assertTrue(observer4_p_minus_1 < observer1_p0)
        self.assertFalse(observer1_p0 < observer4_p_minus_1)

        self.assertTrue(observer4_p_minus_1 < observer2_p1)
        self.assertFalse(observer2_p1 < observer4_p_minus_1)


if __name__ == '__main__':
    # Using tf.test.main() if in a TF environment, otherwise unittest.main()
    # For consistency with TF Agent's style, tf.test.main() is often preferred.
    tf.test.main()
