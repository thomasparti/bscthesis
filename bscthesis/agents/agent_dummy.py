import os
import sys
import time
from typing import Optional, List
from thesis.environments.vizdoom_env import VizDoomEnv
import vizdoom as vzd
import cv2


def get_default_config_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs')


def get_default_video_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'videos')


def agent(cfg: str, video: Optional[str], episodes: int = 10, delay: float = 0.0, render: bool = False):
    scaling = 1.0
    grayscale = False
    fps = 15
    
    resolution = vzd.ScreenResolution.RES_1920X1080
    env = VizDoomEnv(config_path=cfg, internalres=resolution)
    

    video_writer = None
    if video:
        video_dir = os.path.dirname(video)
        os.makedirs(video_dir, exist_ok=True)
        video_path = video

        temp_obs, _ = env.reset()
        temp_frame = env.render(scaling=scaling, grayscale=grayscale, show=render)
        if temp_frame is not None:
            if len(temp_frame.shape) == 2:
                height, width = temp_frame.shape
                is_color = False
            else:
                height, width, layers = temp_frame.shape
                is_color = True

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            if is_color:
                video_writer = cv2.VideoWriter(video, fourcc, fps, (width, height))
            else:
                video_writer = cv2.VideoWriter(video, fourcc, fps, (width, height), isColor=False)

            if not video_writer.isOpened():
                print(f"Error: Cannot open video file for writing: {video}")
                sys.exit(1)
        else:
            print("Error: Unable to retrieve frame for video initialization.")
            sys.exit(1)

        print(f"Video recording enabled. Videos will be saved to '{video}'.")
    else:
        if render:
            print("Rendering enabled.")
        else:
            print("Video recording and rendering disabled.")


    print(f"Running {episodes} episodes with the random agent...")
    for episode in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0
        step = 0

        print(f"Starting Episode {episode}...")

        while not done and not truncated:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if video:
                frame = env.render(scaling=scaling, grayscale=grayscale, show=render)
                if frame is not None:
                        video_writer.write(frame)
            elif render:
                env.render(scaling=scaling, grayscale=grayscale, show=render)

            if delay > 0.0:
                time.sleep(delay)

        print(f"Episode {episode}: Total Reward = {total_reward}, Steps = {step}")

    if video and video_writer:
        video_writer.release()
        print(f"Video saved to {video}")

    env.close()
    print("Random agent run completed.")


if __name__ == '__main__':
    cfg = 'test.cfg'
    cfg_path = os.path.join(get_default_config_dir(), cfg)
    video = 'test.mp4'
    video_path = os.path.join(get_default_video_dir(), video)
    agent(cfg_path, video_path, episodes=5)
