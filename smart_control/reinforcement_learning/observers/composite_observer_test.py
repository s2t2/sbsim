import unittest
import tensorflow as tf
from unittest.mock import MagicMock, call

from smart_control.reinforcement_learning.observers import composite_observer
from smart_control.reinforcement_learning.observers import base_observer as sc_base_observer
from tf_agents.trajectories import trajectory


# Helper for Mock Observers
def create_mock_observer(priority, name="MockObserver"): # Added name for easier debugging
    observer = MagicMock(spec=sc_base_observer.BaseObserver, name=name)
    observer.priority = priority
    # The __lt__ method needs to correctly compare with another observer's priority
    observer.__lt__ = lambda self_mock, other_mock: self_mock.priority < other_mock.priority
    observer.__gt__ = lambda self_mock, other_mock: self_mock.priority > other_mock.priority
    observer.__eq__ = lambda self_mock, other_mock: self_mock.priority == other_mock.priority
    # For sorting to work correctly, __le__ and __ge__ might also be needed by some sort implementations,
    # or a total_ordering decorator. Python's list.sort() or sorted() primarily use __lt__.
    return observer


class CompositeObserverTest(tf.test.TestCase):

    def test_init_with_observers_list(self):
        mock_observer_p1 = create_mock_observer(priority=1, name="ObsP1")
        mock_observer_p0 = create_mock_observer(priority=0, name="ObsP0")
        mock_observer_p2 = create_mock_observer(priority=2, name="ObsP2")

        observers_list = [mock_observer_p1, mock_observer_p0, mock_observer_p2]
        comp_observer = composite_observer.CompositeObserver(observers_list)

        self.assertEqual(len(comp_observer._observers), 3)
        # Verify sorting by priority (ascending)
        self.assertEqual(comp_observer._observers[0].priority, 0)
        self.assertEqual(comp_observer._observers[1].priority, 1)
        self.assertEqual(comp_observer._observers[2].priority, 2)
        self.assertIs(comp_observer._observers[0], mock_observer_p0)
        self.assertIs(comp_observer._observers[1], mock_observer_p1)
        self.assertIs(comp_observer._observers[2], mock_observer_p2)

    def test_init_with_empty_list_or_none(self):
        comp_observer_empty_list = composite_observer.CompositeObserver([])
        self.assertEqual(len(comp_observer_empty_list._observers), 0)

        comp_observer_none = composite_observer.CompositeObserver(None)
        self.assertEqual(len(comp_observer_none._observers), 0)

    def test_add_observer(self):
        comp_observer = composite_observer.CompositeObserver()
        self.assertEqual(len(comp_observer._observers), 0)

        mock_observer_p1 = create_mock_observer(priority=1, name="ObsP1_add")
        comp_observer.add_observer(mock_observer_p1)
        self.assertEqual(len(comp_observer._observers), 1)
        self.assertIs(comp_observer._observers[0], mock_observer_p1)

        mock_observer_p0 = create_mock_observer(priority=0, name="ObsP0_add")
        comp_observer.add_observer(mock_observer_p0)
        self.assertEqual(len(comp_observer._observers), 2)
        self.assertIs(comp_observer._observers[0], mock_observer_p0) # Should be sorted
        self.assertIs(comp_observer._observers[1], mock_observer_p1)

        mock_observer_p2 = create_mock_observer(priority=2, name="ObsP2_add")
        comp_observer.add_observer(mock_observer_p2)
        self.assertEqual(len(comp_observer._observers), 3)
        self.assertIs(comp_observer._observers[0], mock_observer_p0)
        self.assertIs(comp_observer._observers[1], mock_observer_p1)
        self.assertIs(comp_observer._observers[2], mock_observer_p2)

    def test_call_with_observers(self):
        # Store calls in a list to verify order
        call_order_tracker = []

        def side_effect_for_call(priority_value):
            def actual_side_effect(traj):
                call_order_tracker.append(priority_value)
            return actual_side_effect

        mock_observer_p1 = create_mock_observer(priority=1, name="CallObsP1")
        mock_observer_p1.__call__.side_effect = side_effect_for_call(1)

        mock_observer_p0 = create_mock_observer(priority=0, name="CallObsP0")
        mock_observer_p0.__call__.side_effect = side_effect_for_call(0)
        
        mock_observer_p2 = create_mock_observer(priority=2, name="CallObsP2")
        mock_observer_p2.__call__.side_effect = side_effect_for_call(2)

        observers_list = [mock_observer_p1, mock_observer_p0, mock_observer_p2] # Intentionally unsorted
        comp_observer = composite_observer.CompositeObserver(observers_list)

        dummy_trajectory = trajectory.Trajectory(
            step_type=MagicMock(), observation=MagicMock(), action=MagicMock(),
            policy_info=MagicMock(), next_step_type=MagicMock(), reward=MagicMock(),
            discount=MagicMock()
        )

        comp_observer(dummy_trajectory)

        # Verify __call__ was called on each mock
        mock_observer_p0.__call__.assert_called_once_with(dummy_trajectory)
        mock_observer_p1.__call__.assert_called_once_with(dummy_trajectory)
        mock_observer_p2.__call__.assert_called_once_with(dummy_trajectory)

        # Verify call order based on priority
        self.assertEqual(call_order_tracker, [0, 1, 2])

    def test_call_with_no_observers(self):
        comp_observer = composite_observer.CompositeObserver()
        dummy_trajectory = trajectory.Trajectory(
            step_type=MagicMock(), observation=MagicMock(), action=MagicMock(),
            policy_info=MagicMock(), next_step_type=MagicMock(), reward=MagicMock(),
            discount=MagicMock()
        )
        try:
            comp_observer(dummy_trajectory)
        except Exception as e:
            self.fail(f"CompositeObserver call failed with no observers: {e}")


if __name__ == '__main__':
    tf.test.main()
