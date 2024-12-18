import os
import sys
import time
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from thesis.environments.vizdoom_env import VizDoomEnv
import cv2
import vizdoom as vzd

def agent(
    cfg: str,
    model: str,
    logdir: str,
    video: Optional[str],
    episodes: int = 10,
    delay: float = 0.0,
    render: bool = False,
    framestack: int = 1,
    obsx: int = 42,
    obsy: int = 42,
    buffers: str = 'rd',
    frameskip: int = 4,
    use_depth: bool = False
):
    scaling = 3.0
    fps = 15

    if not os.path.exists(model):
        print(f"Error: Model file '{model}' does not exist.")
        sys.exit(1)

    os.makedirs(logdir, exist_ok=True)

    base_env = VizDoomEnv(
        config_path=cfg,
        obsx=obsx,
        obsy=obsy,
        buffers=buffers,
        frame_skip=frameskip,
    )
    monitored_env = Monitor(base_env, filename=os.path.join(logdir, "monitor.csv"))

    if framestack > 1:
        env = DummyVecEnv([lambda: monitored_env])
        env = VecFrameStack(env, n_stack=framestack)
    else:
        env = monitored_env

    video_writer = None
    if video:
        video_dir = os.path.dirname(video)
        os.makedirs(video_dir, exist_ok=True)
        temp_obs, _ = env.reset()
        
        if use_depth:
            temp_frame = env.render_depth(scaling=scaling, show=render)
        else:
            temp_frame = env.render(scaling=scaling, show=render)
        
        if temp_frame is not None:
            if len(temp_frame.shape) == 2:
                height, width = temp_frame.shape
                is_color = False
            else:
                height, width, layers = temp_frame.shape
                is_color = True
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video, fourcc, fps, (width, height), isColor=is_color)
            if not video_writer.isOpened():
                print(f"Error: Cannot open video file for writing: {video}")
                sys.exit(1)
            print(f"Video recording enabled. Videos will be saved to '{video}'.")
        else:
            print("Error: Unable to retrieve frame for video initialization.")
            sys.exit(1)
    else:
        print("Video recording not enabled. Main episode execution will be skipped.")

    try:
        model = PPO.load(model, env=env, tensorboard_log=logdir)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    print(f"PPO model loaded from '{model}'.")

    if video:
        print(f"Running {episodes} episodes with the PPO agent...")
        for episode in range(1, episodes + 1):
            obs, info = env.reset()
            done = False
            truncated = False
            total_reward = 0.0
            step = 0
            print(f"Starting Episode {episode}...")
            while not done and not truncated:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                step += 1

                if use_depth:
                    frame = env.render_depth(scaling=scaling, show=render)
                else:
                    frame = env.render(scaling=scaling, show=render)
                
                if frame is not None:
                    video_writer.write(frame)
                else:
                    print(f"Warning: No frame captured at step {step} for episode {episode}")

                if delay > 0.0:
                    time.sleep(delay)
            print(f"Episode {episode}: Total Reward = {total_reward}, Steps = {step}")
    else:
        print("Skipping main episode execution since video recording is not enabled.")

    if video and video_writer:
        video_writer.release()
        print(f"Video saved to {video}")

    if episodes > 0:
        print("Starting evaluation of the agent...")
        eval_env = Monitor(
            VizDoomEnv(
                config_path=cfg,
                obsx=obsx,
                obsy=obsy,
                buffers=buffers,
                frame_skip=frameskip
            ),
            filename=os.path.join(logdir, "evaluation_monitor.csv")
        )
        try:
            mean_reward, std_reward = evaluate_policy(
                model, eval_env, n_eval_episodes=episodes, deterministic=True
            )
            print(f"\nEvaluation over {episodes} episodes: Mean Reward = {mean_reward}, Std Reward = {std_reward}")
        except Exception as e:
            print(f"Error during evaluation: {e}")
        finally:
            eval_env.close()
    
    env.close()
    print("PPO agent run completed.")
