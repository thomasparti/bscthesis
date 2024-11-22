import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecTransposeImage, VecFrameStack
from typing import Callable
from thesis.environments.vizdoom_env import VizDoomEnv

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def make_env(vizdoom_config: str, rank: int = 0) -> Callable:
    def _init():
        env = VizDoomEnv(config_path=vizdoom_config, visibility=False)
        return env
    return _init

def train(logdir: str, modeldir: str, cycles: int, length: int, cfg: str, params: str):
    with open(params, 'r') as f:
        params_config = yaml.safe_load(f)
    ppo_params = params_config.get('ppo', {})

    num_envs = ppo_params.get('num_envs', 4)

    env_fns = [make_env(cfg, rank=i) for i in range(num_envs)]
    if num_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecMonitor(vec_env)

    vec_env = VecTransposeImage(vec_env)

#    frame_stack = 4
#    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)

    # Default params
    model = PPO(
        policy=ppo_params.get('policy', 'CnnPolicy'),
        env=vec_env,
        learning_rate=ppo_params.get('learning_rate', 2.5e-4),
        n_steps=ppo_params.get('n_steps', 2048),
        batch_size=ppo_params.get('batch_size', 64),
        n_epochs=ppo_params.get('n_epochs', 10),
        gamma=ppo_params.get('gamma', 0.99),
        gae_lambda=ppo_params.get('gae_lambda', 0.95),
        clip_range=ppo_params.get('clip_range', 0.2),
        ent_coef=ppo_params.get('ent_coef', 0.0),
        vf_coef=ppo_params.get('vf_coef', 0.5),
        max_grad_norm=ppo_params.get('max_grad_norm', 0.5),
        verbose=ppo_params.get('verbose', 1),
        tensorboard_log=os.path.join(logdir, "tensorboard")
    )

    for cycle in range(1, cycles + 1):
        print(f"Starting PPO training cycle {cycle}/{cycles}...")
        model.learn(
            total_timesteps=length,
            reset_num_timesteps=False
        )
        
        cycle_model_path = os.path.join(modeldir, f'model_steps_{cycle * length}.zip')
        model.save(cycle_model_path)
        print(f"Completed PPO training cycle {cycle}/{cycles}. Model saved to {cycle_model_path}.")

    vec_env.close()
