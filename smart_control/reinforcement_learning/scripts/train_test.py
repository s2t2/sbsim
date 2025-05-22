import unittest
from unittest.mock import patch, MagicMock, PropertyMock, call

import tensorflow as tf
# from absl import flags # Not strictly needed if we mock FLAGS access or it's not used by train_eval_fn directly

from smart_control.reinforcement_learning.scripts import train as sc_train # aliased
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy, tf_policy # tf_policy for spec
from tf_agents.agents.sac import sac_agent # For spec of SACAgent mock
from smart_control.reinforcement_learning.replay_buffer import replay_buffer as sc_replay_buffer
from tf_agents.utils import common as tf_agents_common # For Checkpointer
from tf_agents.metrics import tf_metrics as tfa_tf_metrics
from tf_agents.eval import metric_utils
from smart_control.reinforcement_learning import utils as sc_utils # For metrics, environment
from smart_control.reinforcement_learning import observers as sc_observers

from tf_agents.specs import tensor_spec, array_spec
from tf_agents.trajectories import time_step as ts_lib
from tf_agents.trajectories import trajectory
from tf_agents.policies import policy_step


# It's a very long list of patches.
@patch('smart_control.reinforcement_learning.scripts.train.gin.query_parameter', MagicMock(return_value=None))
@patch('smart_control.reinforcement_learning.scripts.train.gin.bind_parameter', MagicMock())
@patch('smart_control.reinforcement_learning.scripts.train.gin.operative_config_str', MagicMock(return_value=""))
@patch('tf.compat.v1.train.get_or_create_global_step', autospec=True)
@patch('smart_control.reinforcement_learning.utils.environment.create_tf_environment', autospec=True)
@patch('smart_control.reinforcement_learning.agents.sac_agent.SACAgent', autospec=True)
@patch('smart_control.reinforcement_learning.replay_buffer.replay_buffer.ReplayBuffer', autospec=True)
@patch('tf.keras.optimizers.Adam', autospec=True) # Assuming Adam is used directly, adjust if it's different (e.g. tf.compat.v1.train.AdamOptimizer)
@patch('tf_agents.policies.random_tf_policy.RandomTFPolicy', autospec=True)
@patch('tf.summary.create_file_writer', autospec=True)
@patch('smart_control.reinforcement_learning.utils.metrics.get_eval_metrics', autospec=True)
@patch('smart_control.reinforcement_learning.utils.metrics.get_average_reward_metric', autospec=True)
@patch('tf_agents.metrics.tf_metrics.TFStepMetric', autospec=True) # Renamed to avoid clash
@patch('smart_control.reinforcement_learning.scripts.train.eval_policy', autospec=True) # Mock the local eval_policy
@patch('tf_agents.utils.common.Checkpointer', autospec=True)
@patch('tf.summary.scalar', autospec=True)
@patch('tf.summary.text', autospec=True)
# Mocking the observers module directly if specific observer classes are instantiated
@patch('smart_control.reinforcement_learning.observers.CompositeObserver', autospec=True)
@patch('smart_control.reinforcement_learning.observers.PrintStatusObserver', autospec=True)
@patch('smart_control.reinforcement_learning.observers.RenderingObserver', autospec=True)
class TrainEvalFnTest(tf.test.TestCase):

    def setUp(self,
              MockRenderingObserver, MockPrintStatusObserver, MockCompositeObserver, # Observer mocks
              MockSummaryText, MockSummaryScalar, # tf.summary mocks
              MockCheckpointer, MockEvalPolicy, # train.py dependencies
              MockTFStepMetric, MockGetAverageRewardMetric, MockGetEvalMetrics, # metrics mocks
              MockSummaryWriter, MockRandomTFPolicy, MockAdamOptimizer, # Agent/Policy/Optimizer/Writer
              MockReplayBuffer, MockSACAgent, MockCreateTFEnvironment, MockGetGlobalStep # Core components
              ):
        super().setUp()

        # Store mocks if needed for assertions on constructor calls within tests
        self.mock_rendering_observer_class = MockRenderingObserver
        self.mock_print_status_observer_class = MockPrintStatusObserver
        self.mock_composite_observer_class = MockCompositeObserver
        self.mock_summary_text = MockSummaryText
        self.mock_summary_scalar = MockSummaryScalar
        self.mock_checkpointer_class = MockCheckpointer
        self.mock_eval_policy_fn = MockEvalPolicy
        self.mock_tf_step_metric_class = MockTFStepMetric
        self.mock_get_average_reward_metric_fn = MockGetAverageRewardMetric
        self.mock_get_eval_metrics_fn = MockGetEvalMetrics
        self.mock_summary_writer_class = MockSummaryWriter
        self.mock_random_tf_policy_class = MockRandomTFPolicy
        self.mock_adam_optimizer_class = MockAdamOptimizer
        self.mock_replay_buffer_class = MockReplayBuffer
        self.mock_sac_agent_class = MockSACAgent
        self.mock_create_tf_environment_fn = MockCreateTFEnvironment
        self.mock_get_global_step_fn = MockGetGlobalStep

        # --- Configure common mock return values ---
        self.mock_global_step = MagicMock(spec=tf.Variable)
        self.mock_global_step.numpy.return_value = 0 # Initial global step
        self.mock_get_global_step_fn.return_value = self.mock_global_step

        self.mock_env = MagicMock(spec=tf_py_environment.TFPyEnvironment)
        self.observation_spec = tensor_spec.TensorSpec((2,), tf.float32, 'obs')
        self.action_spec = tensor_spec.BoundedTensorSpec((1,), tf.float32, minimum=0, maximum=1, name='act')
        self.time_step_spec = ts_lib.time_step_spec(self.observation_spec)
        self.mock_env.time_step_spec.return_value = self.time_step_spec
        self.mock_env.action_spec.return_value = self.action_spec
        self.mock_env.batch_size = 1 # Typically 1 for non-batched env
        self.mock_create_tf_environment_fn.return_value = self.mock_env

        self.mock_agent = self.mock_sac_agent_class.return_value
        # Agent needs a policy, train_step_counter, action_spec, time_step_spec
        self.mock_agent.policy = MagicMock(spec=tf_policy.TFPolicy)
        self.mock_agent.collect_policy = MagicMock(spec=tf_policy.TFPolicy)
        self.mock_agent.train_step_counter = tf.Variable(0, dtype=tf.int64) # Give it a real variable
        self.mock_agent.action_spec = self.action_spec
        self.mock_agent.time_step_spec = self.time_step_spec


        self.mock_replay_buffer = self.mock_replay_buffer_class.return_value
        # Replay buffer might need data_spec
        # sample_traj = trajectory.from_transition(ts_lib.restart(np.zeros(self.observation_spec.shape, dtype=self.observation_spec.dtype.as_numpy_dtype), batch_size=1),
        #                                         policy_step.PolicyStep(action=tf.zeros(self.action_spec.shape, dtype=self.action_spec.dtype)),
        #                                         ts_lib.termination(np.zeros(self.observation_spec.shape, dtype=self.observation_spec.dtype.as_numpy_dtype), reward=0.0))
        # self.mock_replay_buffer.data_spec = tensor_spec.from_spec(array_spec.ArraySpec.from_tensor_spec(sample_traj.data_spec))


        self.mock_optimizer = self.mock_adam_optimizer_class.return_value
        
        self.mock_random_policy = self.mock_random_tf_policy_class.return_value

        self.mock_summary_writer_instance = self.mock_summary_writer_class.return_value
        self.mock_summary_writer_instance.as_default = MagicMock().__enter__ # For 'with ... as ...'

        self.mock_eval_metrics = [MagicMock(spec=tfa_tf_metrics.AverageReturnMetric)]
        self.mock_get_eval_metrics_fn.return_value = self.mock_eval_metrics
        self.mock_avg_reward_metric = self.mock_get_average_reward_metric_fn.return_value
        
        self.mock_checkpointer = self.mock_checkpointer_class.return_value
        self.mock_checkpointer.initialize_or_restore.return_value = True # Assume restore is successful or init

        self.mock_eval_policy_fn.return_value = {'AverageReturn': 100.0} # Dummy eval result

        # Default parameters for train_eval_fn
        self.root_dir = 'test_root_dir'
        self.agent_name = 'SACAgent'
        self.env_name = 'TestEnv-v0'
        self.num_iterations = 1 # Keep small for testing
        self.train_steps_per_iteration = 1
        self.collect_steps_per_iteration = 1
        self.initial_collect_policy = self.mock_random_policy
        self.num_initial_collect_steps = 0 # Default to 0, can override in specific tests
        self.replay_buffer_capacity = 10000
        self.train_batch_size = 64
        self.learning_rate = 1e-3
        self.eval_interval = 1
        self.num_eval_episodes = 1
        self.save_interval = 1
        self.summary_interval = 1
        self.use_tf_functions = False # Keep False for easier mocking/debugging
        self.seed = 42

    def test_initialization_phase(self):
        # This test focuses on verifying the setup before the main loop
        sc_train.train_eval_fn(
            root_dir=self.root_dir,
            agent_name=self.agent_name,
            env_name=self.env_name,
            num_iterations=0, # Set to 0 to only test initialization
            # Pass other necessary params...
            train_steps_per_iteration=self.train_steps_per_iteration,
            collect_steps_per_iteration=self.collect_steps_per_iteration,
            initial_collect_policy=self.initial_collect_policy,
            num_initial_collect_steps=0, # No initial collection for this test
            replay_buffer_capacity=self.replay_buffer_capacity,
            train_batch_size=self.train_batch_size,
            learning_rate=self.learning_rate,
            eval_interval=self.eval_interval,
            num_eval_episodes=self.num_eval_episodes,
            save_interval=self.save_interval,
            summary_interval=self.summary_interval,
            use_tf_functions=self.use_tf_functions,
            seed=self.seed,
            # observers are often gin-configured, assuming None for direct call unless specified
            env_observers=None, 
            train_observers=None,
            eval_observers=None,
        )

        self.mock_create_tf_environment_fn.assert_called() # Should be called for train_env and eval_env
        self.assertEqual(self.mock_create_tf_environment_fn.call_count, 2)

        self.mock_get_global_step_fn.assert_called_once()
        self.mock_adam_optimizer_class.assert_called_once_with(learning_rate=self.learning_rate)
        
        self.mock_sac_agent_class.assert_called_once_with(
            time_step_spec=self.time_step_spec,
            action_spec=self.action_spec,
            actor_network=unittest.mock.ANY, # Gin configured, hard to match exactly without deep gin mock
            critic_network=unittest.mock.ANY,
            actor_optimizer=self.mock_optimizer,
            critic_optimizer=self.mock_optimizer, # Assuming same optimizer for actor and critic in this setup
            alpha_optimizer=self.mock_optimizer,  # Assuming same optimizer
            train_step_counter=self.mock_global_step # Agent uses the global step
        )
        
        self.mock_replay_buffer_class.assert_called_once_with(
            data_spec=self.mock_agent.collect_data_spec, # Agent's collect_data_spec
            batch_size=self.mock_env.batch_size, # Env's batch_size
            max_length=self.replay_buffer_capacity
        )

        self.mock_summary_writer_class.assert_called() # train and eval writers
        self.assertEqual(self.mock_summary_writer_class.call_count, 2)
        
        self.mock_get_eval_metrics_fn.assert_called_once()
        self.mock_get_average_reward_metric_fn.assert_called_once_with(batch_size=self.num_eval_episodes) # Corrected batch_size

        self.mock_checkpointer_class.assert_called_once()
        # Check some key args for Checkpointer
        args, kwargs = self.mock_checkpointer_class.call_args
        self.assertTrue(kwargs['ckpt_dir'].startswith(self.root_dir))
        self.assertIs(kwargs['agent'], self.mock_agent)
        self.assertIs(kwargs['global_step'], self.mock_global_step)
        self.assertIs(kwargs['replay_buffer'], self.mock_replay_buffer)

        # Check if observers are initialized (if they were passed or gin-configured)
        # self.mock_composite_observer_class.assert_called() # Example

    def test_initial_data_collection_phase(self):
        num_initial_steps = 5
        # --- Configure Mocks for Initial Collection ---
        # Environment behavior
        initial_obs = np.zeros(self.observation_spec.shape, dtype=self.observation_spec.dtype.as_numpy_dtype)
        current_time_step = ts_lib.restart(initial_obs, batch_size=self.mock_env.batch_size)
        
        # Simulate a sequence of time steps for the collection
        time_step_sequence = [current_time_step]
        for i in range(num_initial_steps):
            if i < num_initial_steps -1:
                next_ts = ts_lib.transition(
                    observation=np.full_like(initial_obs, i + 1.0),
                    reward=tf.constant([1.0] * self.mock_env.batch_size, dtype=tf.float32),
                    discount=tf.constant([1.0] * self.mock_env.batch_size, dtype=tf.float32)
                )
            else: # Last step
                next_ts = ts_lib.termination(
                    observation=np.full_like(initial_obs, num_initial_steps + 1.0),
                    reward=tf.constant([1.0] * self.mock_env.batch_size, dtype=tf.float32)
                )
            time_step_sequence.append(next_ts)

        # current_time_step is called once at the start of collection, then subsequent time_steps come from env.step
        self.mock_env.current_time_step.return_value = time_step_sequence[0] 
        self.mock_env.step.side_effect = time_step_sequence[1:]


        # Policy (RandomTFPolicy for initial collection) action
        # self.mock_random_policy is already configured in setUp
        dummy_action = tf.zeros(self.action_spec.shape, dtype=self.action_spec.dtype)
        self.mock_random_policy.action.return_value = policy_step.PolicyStep(action=dummy_action)
        
        # Replay Buffer: data_spec should be compatible.
        # We assume it's correctly set up based on agent's collect_data_spec during ReplayBuffer init.
        # The add_batch mock is on self.mock_replay_buffer.

        sc_train.train_eval_fn(
            root_dir=self.root_dir,
            agent_name=self.agent_name,
            env_name=self.env_name,
            num_iterations=0, # Focus only on initial collection
            train_steps_per_iteration=self.train_steps_per_iteration,
            collect_steps_per_iteration=self.collect_steps_per_iteration,
            initial_collect_policy=self.mock_random_policy, # Use the mocked random policy
            num_initial_collect_steps=num_initial_steps,
            replay_buffer_capacity=self.replay_buffer_capacity,
            train_batch_size=self.train_batch_size,
            learning_rate=self.learning_rate,
            eval_interval=100, # Don't run eval
            num_eval_episodes=self.num_eval_episodes,
            save_interval=100, # Don't run save
            summary_interval=self.summary_interval,
            use_tf_functions=self.use_tf_functions,
            seed=self.seed,
            env_observers=None, train_observers=None, eval_observers=None
        )

        # --- Assertions for Initial Collection ---
        # current_time_step is called once before the loop
        self.mock_env.current_time_step.assert_called_once()
        
        # action and step are called num_initial_steps times
        self.assertEqual(self.mock_random_policy.action.call_count, num_initial_steps)
        self.assertEqual(self.mock_env.step.call_count, num_initial_steps)
        
        # replay_buffer.add_batch is called num_initial_steps times
        self.assertEqual(self.mock_replay_buffer.add_batch.call_count, num_initial_steps)

        # Verify the trajectories added to the replay buffer
        current_ts_mock = time_step_sequence[0]
        for i in range(num_initial_steps):
            policy_action_mock = self.mock_random_policy.action.return_value # It returns the same dummy action
            next_ts_mock = time_step_sequence[i+1]
            
            expected_traj = trajectory.from_transition(current_ts_mock, policy_action_mock, next_ts_mock)
            actual_call_args = self.mock_replay_buffer.add_batch.call_args_list[i][0][0] # first arg of ith call
            
            self.assertAllEqual(actual_call_args.observation, expected_traj.observation)
            self.assertAllEqual(actual_call_args.action, expected_traj.action)
            self.assertAllEqual(actual_call_args.reward, expected_traj.reward)
            current_ts_mock = next_ts_mock

    def test_main_training_collection_loop_eval_checkpoint(self):
        # Override some defaults for this specific test
        num_iterations = 1
        train_steps_per_iteration = 2
        collect_steps_per_iteration = 3
        eval_interval = 1
        save_interval = 1
        summary_interval = 1 # Ensure summaries are written

        # --- Configure Mocks for the Main Loop ---
        # Environment and Collection Policy
        initial_obs_loop = np.zeros(self.observation_spec.shape, dtype=self.observation_spec.dtype.as_numpy_dtype)
        current_time_step_loop = ts_lib.restart(initial_obs_loop, batch_size=self.mock_env.batch_size)
        
        time_step_sequence_loop = [current_time_step_loop]
        for i in range(collect_steps_per_iteration * num_iterations): # Total collect steps
            time_step_sequence_loop.append(ts_lib.transition(
                np.full_like(initial_obs_loop, float(i + 1)), tf.constant([1.0]*self.mock_env.batch_size), tf.constant([1.0]*self.mock_env.batch_size)
            ))
        
        self.mock_env.current_time_step.return_value = time_step_sequence_loop[0]
        self.mock_env.step.side_effect = time_step_sequence_loop[1:]

        dummy_collect_action = tf.zeros(self.action_spec.shape, dtype=self.action_spec.dtype)
        self.mock_agent.collect_policy.action.return_value = policy_step.PolicyStep(action=dummy_collect_action)

        # Replay Buffer for training
        # Create a dummy trajectory that as_dataset will return
        # This trajectory should match agent.train_data_spec
        # For simplicity, we'll use a trajectory based on time_step_spec and action_spec
        sample_obs_train = np.zeros(self.observation_spec.shape, dtype=self.observation_spec.dtype.as_numpy_dtype)
        sample_action_train = np.zeros(self.action_spec.shape, dtype=self.action_spec.dtype.as_numpy_dtype)

        # This is a simplified trajectory for training. TF-Agents training typically expects
        # trajectories with num_steps dimension (e.g., from n-step returns).
        # The ReplayBuffer.as_dataset usually handles this.
        # We mock the output of the iterator.
        # train_data_spec = self.mock_agent.train_data_spec (if available, else construct one)
        # For SAC, train_data_spec is typically a Trajectory.
        # Let's assume train_data_spec is the same as data_spec for simplicity here.
        # And the dataset returns items one by one (batch_size from train_batch_size is handled by dataset batching)
        
        # Mocking as_dataset().as_numpy_iterator()
        mock_dataset = MagicMock()
        # This should yield (experience, sample_info) tuples
        # Experience should be a trajectory matching agent.train_data_spec
        # Let's make a sample experience (trajectory)
        dummy_train_experience = trajectory.Trajectory(
            step_type=tf.constant([ts_lib.StepType.FIRST] * self.train_batch_size, dtype=tf.int32),
            observation=tf.stack([tf.constant(sample_obs_train, dtype=tf.float32)] * self.train_batch_size),
            action=tf.stack([tf.constant(sample_action_train, dtype=tf.float32)] * self.train_batch_size),
            policy_info=(), # Assuming empty policy_info for simplicity
            next_step_type=tf.constant([ts_lib.StepType.MID] * self.train_batch_size, dtype=tf.int32),
            reward=tf.constant([0.0] * self.train_batch_size, dtype=tf.float32),
            discount=tf.constant([1.0] * self.train_batch_size, dtype=tf.float32)
        )
        mock_iterator = MagicMock()
        mock_iterator.__iter__.return_value = iter([(dummy_train_experience, MagicMock())] * train_steps_per_iteration * num_iterations)
        mock_dataset.as_numpy_iterator.return_value = mock_iterator
        self.mock_replay_buffer.as_dataset.return_value = mock_dataset
        
        # Agent training
        self.mock_agent.train.return_value = sac_agent.LossInfo(loss=tf.constant(0.123), extra=()) # Dummy loss

        # Observers (assuming they are passed as a list of mocks)
        mock_train_observer_instance = MagicMock(spec=sc_observers.BaseObserver)
        mock_env_observer_instance = MagicMock(spec=sc_observers.BaseObserver)
        
        # --- Call train_eval_fn ---
        returned_eval_metrics = sc_train.train_eval_fn(
            root_dir=self.root_dir,
            agent_name=self.agent_name,
            env_name=self.env_name,
            num_iterations=num_iterations,
            train_steps_per_iteration=train_steps_per_iteration,
            collect_steps_per_iteration=collect_steps_per_iteration,
            initial_collect_policy=self.mock_random_policy,
            num_initial_collect_steps=0, # No initial collection for this test
            replay_buffer_capacity=self.replay_buffer_capacity,
            train_batch_size=self.train_batch_size,
            learning_rate=self.learning_rate,
            eval_interval=eval_interval,
            num_eval_episodes=self.num_eval_episodes,
            save_interval=save_interval,
            summary_interval=summary_interval,
            use_tf_functions=self.use_tf_functions,
            seed=self.seed,
            env_observers=[mock_env_observer_instance], # Pass mock observer
            train_observers=[mock_train_observer_instance] # Pass mock observer
        )

        # --- Assertions for Main Loop ---
        # Collection
        total_collect_calls = collect_steps_per_iteration * num_iterations
        self.assertEqual(self.mock_agent.collect_policy.action.call_count, total_collect_calls)
        self.assertEqual(self.mock_env.step.call_count, total_collect_calls)
        self.assertEqual(self.mock_replay_buffer.add_batch.call_count, total_collect_calls)
        self.assertEqual(mock_env_observer_instance.call_count, total_collect_calls) # Env observer called for each collection step


        # Training
        self.mock_replay_buffer.as_dataset.assert_called() # Called once to create the dataset iterator
        total_train_calls = train_steps_per_iteration * num_iterations
        self.assertEqual(self.mock_agent.train.call_count, total_train_calls)
        # Verify train was called with the experience from the mocked iterator
        for i in range(total_train_calls):
             train_call_arg = self.mock_agent.train.call_args_list[i][0][0] # First arg of train call
             self.assertAllEqual(train_call_arg.observation, dummy_train_experience.observation)
        self.assertEqual(mock_train_observer_instance.call_count, total_train_calls) # Train observer called for each train step


        # Global step update
        self.assertEqual(self.mock_global_step.assign_add.call_count, total_train_calls)

        # Evaluation
        self.mock_eval_policy_fn.assert_called() # Should be called based on eval_interval
        self.assertEqual(self.mock_eval_policy_fn.call_count, num_iterations // eval_interval)
        # Verify summary scalar was called for AverageReturn
        self.mock_summary_scalar.assert_any_call('Metrics/AverageReturn', 100.0, step=self.mock_global_step)


        # Checkpointing
        self.mock_checkpointer.save.assert_called() # Should be called based on save_interval
        self.assertEqual(self.mock_checkpointer.save.call_count, num_iterations // save_interval)
        self.mock_checkpointer.save.assert_called_with(global_step=self.mock_global_step)
        
        # Return value
        self.assertEqual(returned_eval_metrics, {'AverageReturn': 100.0})


if __name__ == '__main__':
    tf.test.main()
