"""Script for training a reinforcement learning agent.

This script orchestrates the training of an RL agent (e.g., SAC) for smart
building control. It leverages a pre-populated replay buffer (created by, for
example, `populate_starter_buffer.py`) to initialize the agent's learning
process.

The training loop involves:
- Setting up training and evaluation environments.
- Creating the RL agent, policies, and necessary metrics.
- Loading the initial replay buffer.
- Running a data collection actor to gather new experiences.
- Running a learner to update the agent's policy based on sampled experiences.
- Periodically evaluating the agent's performance and saving checkpoints.
- Logging metrics and summaries for TensorBoard.
"""

import os

# Set WRAPT_DISABLE_EXTENSIONS to true before importing tensorflow to avoid
# potential issues with certain TensorFlow versions or environments.
# See: https://github.com/tensorflow/tensorflow/issues/63548#issuecomment-2008941537
os.environ["WRAPT_DISABLE_EXTENSIONS"] = "true"

# Standard library imports should generally come after environment variable settings
# but before other third-party or project-specific imports if they might be affected.
import argparse # pylint: disable=wrong-import-position
import datetime # pylint: disable=wrong-import-position
import logging # pylint: disable=wrong-import-position

import tensorflow as tf # pylint: disable=wrong-import-position
from tf_agents.agents.tf_agent import TFAgent # pylint: disable=wrong-import-position
from tf_agents.environments import tf_py_environment # pylint: disable=wrong-import-position
from tf_agents.metrics import tf_metrics # pylint: disable=wrong-import-position
from tf_agents.policies import greedy_policy # pylint: disable=wrong-import-position
from tf_agents.policies import py_tf_eager_policy # pylint: disable=wrong-import-position
from tf_agents.train import actor as tf_agents_actor # pylint: disable=wrong-import-position
from tf_agents.train import learner as tf_agents_learner # pylint: disable=wrong-import-position
from tf_agents.train import triggers # pylint: disable=wrong-import-position
from tf_agents.train.utils import spec_utils # pylint: disable=wrong-import-position

# Project-specific imports
from smart_control.reinforcement_learning.agents.sac_agent import create_sac_agent
from smart_control.reinforcement_learning.observers.composite_observer import CompositeObserver
from smart_control.reinforcement_learning.observers.print_status_observer import PrintStatusObserver
from smart_control.reinforcement_learning.replay_buffer.replay_buffer import ReplayBufferManager
from smart_control.reinforcement_learning.utils.config import CONFIG_PATH
from smart_control.reinforcement_learning.utils.config import EXPERIMENT_RESULTS_PATH
from smart_control.reinforcement_learning.utils.environment import create_and_setup_environment

# Configure logging for the script
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] [%(filename)s:%(lineno)d] [%(message)s]",
)
logger = logging.getLogger(__name__)


def train_agent(
    starter_buffer_path: str,
    experiment_name: str,
    agent_type: str = "sac",
    train_iterations: int = 100000,
    collect_steps_per_iteration: int = 1,
    batch_size: int = 256,
    log_interval: int = 100,
    eval_interval: int = 1000,
    num_eval_episodes: int = 5,
    checkpoint_interval: int = 1000,
    learner_iterations: int = 1, # Changed default from 200 to 1
) -> TFAgent:
  """Trains an RL agent using data from a pre-populated replay buffer.

  Args:
    starter_buffer_path (str): Path to the directory containing the checkpoint
      of the pre-populated Reverb replay buffer.
    experiment_name (str): A unique name for this training experiment. Results
      (summaries, checkpoints) will be saved in a subdirectory under
      `EXPERIMENT_RESULTS_PATH` named after this.
    agent_type (str): The type of RL agent to train. Currently, only 'sac'
      (Soft Actor-Critic) is supported.
    train_iterations (int): The total number of training iterations to run.
      Each iteration typically involves data collection and policy updates.
    collect_steps_per_iteration (int): The number of new environment steps to
      collect and add to the replay buffer in each training iteration.
    batch_size (int): The number of experiences (trajectory sequences) to
      sample from the replay buffer for each gradient update.
    log_interval (int): The interval (in training steps) at which to log
      training progress and metrics.
    eval_interval (int): The interval (in training iterations) at which to
      evaluate the agent's performance on the evaluation environment.
    num_eval_episodes (int): The number of episodes to run for each
      evaluation.
    checkpoint_interval (int): The interval (in training iterations) at which
      to checkpoint the replay buffer's state.
    learner_iterations (int): The number of gradient updates (learner runs)
      to perform per training iteration.

  Returns:
    TFAgent: The trained TF-Agents agent instance.

  Raises:
    FileExistsError: If the `summary_dir` for the experiment already exists.
    ValueError: If an unsupported `agent_type` is specified.
  """
  # Define path for the environment's Gin configuration
  scenario_config_path = os.path.join(CONFIG_PATH, "sim_config_1_day.gin")

  # Create a unique directory for this experiment's results and summaries
  current_time_str = datetime.datetime.now().strftime("%Y_%m_%d-%H%M%S")
  summary_dir = os.path.join(
      EXPERIMENT_RESULTS_PATH, f"{experiment_name}_{current_time_str}"
  )
  logger.info("Experiment results will be saved to: %s", summary_dir)

  try:
    os.makedirs(summary_dir) # exist_ok=False by default
  except FileExistsError as e:
    logger.error("Experiment directory '%s' already exists. Exiting.", summary_dir)
    raise FileExistsError(
        f"Directory {summary_dir} already exists. Please use a unique "
        "experiment_name or remove the existing directory."
    ) from e

  # Create training and evaluation environments
  logger.info("Creating training and evaluation environments...")
  # Metrics for the training environment will be saved within summary_dir
  train_py_env = create_and_setup_environment(
      gin_config_file=scenario_config_path,
      metrics_path=os.path.join(summary_dir, "metrics", "train"),
  )
  eval_py_env = create_and_setup_environment(
      gin_config_file=scenario_config_path,
      metrics_path=os.path.join(summary_dir, "metrics", "eval"), # Separate eval metrics
  )

  # Wrap Python environments in TensorFlow environments for TF-Agents
  train_tf_env = tf_py_environment.TFPyEnvironment(train_py_env)
  eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

  # Global step counter for training
  train_step_counter = tf.Variable(0, trainable=False, dtype=tf.int64, name="train_step")

  # Get observation, action, and time_step specifications from the environment
  _, action_spec, time_step_spec = spec_utils.get_tensor_specs(train_tf_env)

  # Create the RL agent
  logger.info("Creating %s agent...", agent_type.upper())
  if agent_type.lower() == "sac":
    agent = create_sac_agent(
        time_step_spec=time_step_spec, action_spec=action_spec
        # Other SAC parameters can be configured here if needed
    )
  else:
    raise ValueError(
        f"Unsupported agent type: {agent_type}. Currently, only 'sac' is "
        "supported."
    )
  logger.info("Agent %s created successfully.", agent.name)

  # Define policies for collection and evaluation
  collect_policy = agent.collect_policy
  eval_policy = greedy_policy.GreedyPolicy(agent.policy) # Greedy for evaluation

  # Define training metrics
  train_metrics = [
      tf_metrics.NumberOfEpisodes(),
      tf_metrics.EnvironmentSteps(),
      tf_metrics.AverageReturnMetric(batch_size=num_eval_episodes), # Consistent batch_size
      tf_metrics.AverageEpisodeLengthMetric(batch_size=num_eval_episodes),
  ]

  # Define evaluation metrics
  eval_metrics = [
      tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
      tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
  ]

  # Initialize ReplayBufferManager and load the pre-populated buffer
  logger.info("Initializing ReplayBufferManager...")
  replay_manager = ReplayBufferManager(
      data_spec=agent.collect_data_spec, # Spec of data agent expects
      capacity=50000,  # Default capacity, can be made configurable
      checkpoint_dir=starter_buffer_path, # Path to load from
      sequence_length=agent.collect_data_spec.action.shape[0] if len(agent.collect_data_spec.action.shape) > 0 else 2, # Sensible default
  )
  logger.info(
      "Attempting to load starter replay buffer from: %s", starter_buffer_path
  )
  rb_instance, rb_observer = replay_manager.load_replay_buffer()
  logger.info(
      "Replay buffer loaded. Current size: %d frames.",
      replay_manager.num_frames(),
  )

  # Create a TensorFlow dataset from the replay buffer for training
  logger.info("Creating TF dataset from replay buffer...")
  dataset = rb_instance.as_dataset(
      sample_batch_size=batch_size,
      num_steps=agent.collect_data_spec.action.shape[0] if len(agent.collect_data_spec.action.shape) > 0 else 2, # Match sequence length or default
      num_parallel_calls=tf.data.AUTOTUNE,
  ).prefetch(tf.data.AUTOTUNE)

  # Create observers for data collection
  collect_status_observer = PrintStatusObserver(
      status_interval_steps=log_interval // 10 or 1, # Log more frequently during collect
      environment=train_tf_env,
      replay_buffer_instance=rb_instance,
  )
  # rb_observer adds trajectories to the Reverb replay buffer.
  collect_combined_observers = CompositeObserver(
      [collect_status_observer, rb_observer]
  )

  # Create data collection actor (interacts with the training environment)
  logger.info("Creating data collection actor...")
  collect_actor_instance = tf_agents_actor.Actor(
      env=train_tf_env, # TF environment for the actor
      policy=collect_policy, # Agent's data collection policy
      train_step=train_step_counter,
      steps_per_run=collect_steps_per_iteration,
      metrics=tf_agents_actor.collect_metrics(10), # Aggregate over 10 episodes
      summary_dir=os.path.join(summary_dir, learner.TRAIN_DIR, "collect"),
      observers=[collect_combined_observers],
  )

  # Create evaluation actor (interacts with the evaluation environment)
  logger.info("Creating evaluation actor...")
  eval_status_observer = PrintStatusObserver(
      status_interval_steps=1, environment=eval_tf_env # Log every step in eval
  )
  eval_actor_instance = tf_agents_actor.Actor(
      env=eval_tf_env, # TF environment for evaluation
      policy=eval_policy, # Agent's greedy evaluation policy
      train_step=train_step_counter, # Use same global step for reference
      episodes_per_run=num_eval_episodes,
      metrics=tf_agents_actor.eval_metrics(num_eval_episodes),
      summary_dir=os.path.join(summary_dir, "eval"),
      observers=[eval_status_observer], # Only print status for eval
  )

  # Create the learner, responsible for updating the agent's policy
  logger.info("Creating agent learner...")
  policy_save_path = os.path.join(summary_dir, learner.POLICY_SAVED_MODEL_DIR)
  agent_learner = tf_agents_learner.Learner(
      root_dir=summary_dir, # Root for all learner outputs (checkpoints, etc.)
      train_step=train_step_counter,
      agent=agent,
      experience_dataset_fn=lambda: dataset, # Function that returns the dataset
      # Define triggers for actions like saving models, logging
      triggers=[
          triggers.PolicySavedModelTrigger(
              saved_model_dir=policy_save_path,
              agent=agent,
              train_step=train_step_counter,
              interval=eval_interval, # Save policy when evaluation happens
              save_greedy_policy=True # Save greedy policy for deployment
          ),
          triggers.StepPerSecondLogTrigger(
              train_step=train_step_counter, interval=log_interval
          ),
      ],
      # Checkpoint interval for the learner (agent's network weights)
      checkpoint_interval=checkpoint_interval,
      summary_interval=log_interval, # How often to write training summaries
  )

  # Main training loop
  logger.info(
      "Starting training for %d iterations (outer loops).", train_iterations
  )
  # Reset training metrics at the start
  for metric in train_metrics:
    metric.reset()

  for i in range(train_iterations):
    current_iter_step = train_step_counter.numpy() # Global step at start of iter
    logger.info(
        "Training iteration %d/%d (Global step: %d)",
        i + 1, train_iterations, current_iter_step
    )

    # Evaluation phase
    if i % eval_interval == 0:
      logger.info(
          "Evaluation phase at iteration %d (step %d)...", i + 1, current_iter_step
      )
      eval_actor_instance.run()
      # Log evaluation metrics
      with eval_actor_instance.summary_writer.as_default(): # type: ignore[union-attr]
        for metric in eval_metrics:
          metric.log_result() # TF-Agents metrics log themselves
          logger.info("Eval Metric: %s = %s", metric.name, metric.result())
        tf.summary.scalar("eval_average_return", eval_metrics[0].result(), step=current_iter_step)
        tf.summary.scalar("eval_average_episode_length", eval_metrics[1].result(), step=current_iter_step)

    # Data collection phase
    logger.info(
        "Collection phase at iteration %d (step %d)...", i + 1, current_iter_step
    )
    collect_actor_instance.run()
    # Log collection metrics
    with collect_actor_instance.summary_writer.as_default(): # type: ignore[union-attr]
        for metric in train_metrics: # Assuming train_metrics are for collection
            metric.log_result()
            logger.info("Collect Metric: %s = %s", metric.name, metric.result())
        # Specific logging for average return and length from collection
        tf.summary.scalar("collect_average_return", train_metrics[2].result(), step=current_iter_step)
        tf.summary.scalar("collect_average_episode_length", train_metrics[3].result(), step=current_iter_step)


    # Training phase (agent learning)
    logger.info(
        "Learner phase at iteration %d (step %d) for %d learner steps...",
        i + 1, current_iter_step, learner_iterations
    )
    # The learner internally increments train_step_counter `learner_iterations` times.
    agent_learner.run(iterations=learner_iterations)

    # Replay buffer checkpointing (Reverb server handles its own checkpointing)
    # If manual checkpoint of Reverb data is needed via client:
    if i % checkpoint_interval == 0 and rb_instance:
      logger.info("Requesting Reverb replay buffer checkpoint...")
      rb_instance.py_client.checkpoint()

  # Final evaluation after all training iterations
  logger.info("Training complete. Performing final evaluation...")
  eval_actor_instance.run()
  final_step = train_step_counter.numpy()
  with eval_actor_instance.summary_writer.as_default(): # type: ignore[union-attr]
    for metric in eval_metrics:
      metric.log_result()
      logger.info("Final Eval Metric: %s = %s", metric.name, metric.result())
    tf.summary.scalar("eval_average_return", eval_metrics[0].result(), step=final_step)
    tf.summary.scalar("eval_average_episode_length", eval_metrics[1].result(), step=final_step)


  # Ensure learner also checkpoints at the end
  agent_learner.save_checkpoint()
  logger.info(
      "Agent training finished. Models and summaries saved in: %s", summary_dir
  )
  return agent


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Train a reinforcement learning agent for smart building "
      "control using a pre-populated Reverb replay buffer."
  )
  parser.add_argument(
      "--starter_buffer_path", # Consistent snake_case
      type=str,
      required=True,
      help="Path to the checkpoint directory of the starter Reverb replay buffer.",
  )
  parser.add_argument(
      "--experiment_name", # Consistent snake_case
      type=str,
      required=True,
      help="Unique name for the experiment. Results will be saved in a "
           "subdirectory named after this.",
  )
  parser.add_argument(
      "--agent_type", # Consistent snake_case
      type=str,
      default="sac",
      choices=["sac"], # Currently only SAC is fully implemented here
      help="Type of RL agent to train. Default: 'sac'.",
  )
  parser.add_argument(
      "--train_iterations", # Consistent snake_case
      type=int,
      default=10000, # Increased default
      help="Total number of outer training iterations. Default: 10000.",
  )
  parser.add_argument(
      "--collect_steps_per_iteration", # Consistent snake_case
      type=int,
      default=1, # As per original, might be low for some setups
      help="Number of new environment steps to collect per training "
           "iteration. Default: 1.",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=256,
      help="Batch size for sampling from the replay buffer during training. "
           "Default: 256.",
  )
  parser.add_argument(
      "--log_interval", # Consistent snake_case
      type=int,
      default=100, # Original was 1, increased for less verbose logs
      help="Interval (in training steps) for logging training metrics and "
           "summaries. Default: 100.",
  )
  parser.add_argument(
      "--eval_interval", # Consistent snake_case
      type=int,
      default=1000, # Original was 10
      help="Interval (in training iterations) for evaluating the agent's "
           "performance. Default: 1000.",
  )
  parser.add_argument(
      "--num_eval_episodes", # Consistent snake_case
      type=int,
      default=5, # Original was 1
      help="Number of episodes to run for each evaluation. Default: 5.",
  )
  parser.add_argument(
      "--checkpoint_interval", # Consistent snake_case
      type=int,
      default=1000, # Original was 10
      help="Interval (in training iterations) for checkpointing the agent "
           "and replay buffer. Default: 1000.",
  )
  parser.add_argument(
      "--learner_iterations", # Consistent snake_case
      type=int,
      default=1, # Original was 200, TF-Agents Learner typically runs 1 iter per .run()
      help="Number of gradient updates (learner runs) to perform per outer "
           "training iteration. Default: 1.",
  )
  args = parser.parse_args()

  train_agent(
      starter_buffer_path=args.starter_buffer_path,
      experiment_name=args.experiment_name,
      agent_type=args.agent_type,
      train_iterations=args.train_iterations,
      collect_steps_per_iteration=args.collect_steps_per_iteration,
      batch_size=args.batch_size,
      log_interval=args.log_interval,
      eval_interval=args.eval_interval,
      num_eval_episodes=args.num_eval_episodes,
      checkpoint_interval=args.checkpoint_interval,
      learner_iterations=args.learner_iterations,
  )
  logger.info("Agent training script finished.")
