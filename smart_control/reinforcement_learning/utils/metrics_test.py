import unittest
import tensorflow as tf # Using tf.test.TestCase as per typical TF Agents style

from smart_control.reinforcement_learning.utils import metrics as metrics_utils
from tf_agents.metrics import tf_metrics


class MetricsUtilsTest(tf.test.TestCase):

    def test_get_average_reward_metric_default_args(self):
        metric = metrics_utils.get_average_reward_metric()
        self.assertIsInstance(metric, tf_metrics.AverageReturnMetric)
        self.assertEqual(metric.name, 'AverageReturn')
        # TF-Agents AverageReturnMetric default batch_size is 1.
        # The function get_average_reward_metric also defaults batch_size to 1.
        self.assertEqual(metric._batch_size, 1) # _batch_size is how it's stored internally

    def test_get_average_reward_metric_custom_args(self):
        custom_name = 'MyCustomAverageReturn'
        custom_batch_size = 10
        metric = metrics_utils.get_average_reward_metric(name=custom_name, batch_size=custom_batch_size)
        self.assertIsInstance(metric, tf_metrics.AverageReturnMetric)
        self.assertEqual(metric.name, custom_name)
        self.assertEqual(metric._batch_size, custom_batch_size)

    def test_get_eval_metrics_custom_batch_size(self):
        custom_batch_size = 5
        eval_metrics_list = metrics_utils.get_eval_metrics(batch_size=custom_batch_size)

        self.assertIsInstance(eval_metrics_list, list)
        self.assertEqual(len(eval_metrics_list), 2)

        # Check for AverageReturnMetric
        avg_return_metric = None
        for m in eval_metrics_list:
            if isinstance(m, tf_metrics.AverageReturnMetric):
                avg_return_metric = m
                break
        self.assertIsNotNone(avg_return_metric, "AverageReturnMetric not found in eval_metrics_list")
        self.assertEqual(avg_return_metric.name, 'AverageReturn')
        self.assertEqual(avg_return_metric._batch_size, custom_batch_size)

        # Check for AverageEpisodeLengthMetric
        avg_ep_len_metric = None
        for m in eval_metrics_list:
            if isinstance(m, tf_metrics.AverageEpisodeLengthMetric):
                avg_ep_len_metric = m
                break
        self.assertIsNotNone(avg_ep_len_metric, "AverageEpisodeLengthMetric not found in eval_metrics_list")
        self.assertEqual(avg_ep_len_metric.name, 'AverageEpisodeLength')
        self.assertEqual(avg_ep_len_metric._batch_size, custom_batch_size)

    def test_get_eval_metrics_default_batch_size(self):
        # Test with default batch_size (assuming it's 1 as per function signature)
        eval_metrics_list = metrics_utils.get_eval_metrics()

        self.assertIsInstance(eval_metrics_list, list)
        self.assertEqual(len(eval_metrics_list), 2)
        
        default_batch_size = 1 # As per function's default

        # Check for AverageReturnMetric
        avg_return_metric = next(m for m in eval_metrics_list if isinstance(m, tf_metrics.AverageReturnMetric))
        self.assertEqual(avg_return_metric.name, 'AverageReturn')
        self.assertEqual(avg_return_metric._batch_size, default_batch_size)

        # Check for AverageEpisodeLengthMetric
        avg_ep_len_metric = next(m for m in eval_metrics_list if isinstance(m, tf_metrics.AverageEpisodeLengthMetric))
        self.assertEqual(avg_ep_len_metric.name, 'AverageEpisodeLength')
        self.assertEqual(avg_ep_len_metric._batch_size, default_batch_size)


if __name__ == '__main__':
    tf.test.main()
