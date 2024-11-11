import os
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from typing import Callable
from thesis.environments.vizdoom_env import VizDoomEnv

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def make_env(vizdoom_config: str, seed: int = None) -> Callable:
    def _init():
        env = VizDoomEnv(config_path=vizdoom_config, visibility=False)
        env = Monitor(env)
        if seed is not None:
            env.seed(seed)
        return env
    return _init

def train(logdir: str, modeldir: str, cycles: int, length: int, cfg: str, params: str):
    with open(params, 'r') as f:
        params_config = yaml.safe_load(f)
    dqn_params = params_config.get('dqn', {})

    env_callable = make_env(cfg)
    env = env_callable()

    # Initialize DQN with default params overwritten by config
    model = DQN(
        policy=dqn_params.get('policy', 'CnnPolicy'),
        env=env,
        learning_rate=dqn_params.get('learning_rate', 1e-4),
        buffer_size=dqn_params.get('buffer_size', 1000000),
        learning_starts=dqn_params.get('learning_starts', 100),
        batch_size=dqn_params.get('batch_size', 32),
        tau=dqn_params.get('tau', 1.0),
        gamma=dqn_params.get('gamma', 0.99),
        train_freq=dqn_params.get('train_freq', 4),
        gradient_steps=dqn_params.get('gradient_steps', 1),
        target_update_interval=dqn_params.get('target_update_interval', 10000),
        exploration_initial_eps=dqn_params.get('exploration_initial_eps', 1.0),
        exploration_final_eps=dqn_params.get('exploration_final_eps', 0.05),
        exploration_fraction=dqn_params.get('exploration_fraction', 0.1),
        max_grad_norm=dqn_params.get('max_grad_norm', 10),
        verbose=dqn_params.get('verbose', 1),
        tensorboard_log=os.path.join(logdir, "tensorboard")
    )

    # Training loop with saving at the end of each cycle
    for cycle in range(1, cycles + 1):
        print(f"Starting DQN training cycle {cycle}/{cycles}...")
        model.learn(
            total_timesteps=length,
            reset_num_timesteps=False
        )
        
        cycle_model_path = os.path.join(modeldir, f'model_steps_{cycle * length}.zip')
        model.save(cycle_model_path)
        print(f"Completed DQN training cycle {cycle}/{cycles}. Model saved to {cycle_model_path}.")

    env.close()
