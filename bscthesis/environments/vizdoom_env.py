import os
from typing import Any, Tuple, Optional, List
import gymnasium as gym
from gymnasium import spaces
import vizdoom as vzd
import numpy as np
import cv2
import time
import random


class VizDoomEnv(gym.Env):
    def __init__(
        self, 
        config_path: str, 
        number_maps: int = 10,
        reward_threshold: float = 0.6,
        visibility: bool = False, 
        internalres: Optional[vzd.ScreenResolution] = None,
        obsx: int = 42,
        obsy: int = 42,
        frame_skip: int = 4,
        buffers: str = 'rd',
        blind: bool = False,
        clip: Optional[List[float]] = (-1, 1),
        living_reward: float = -0.01
    ):
        super(VizDoomEnv, self).__init__()

        self.blind = blind
        self.game = vzd.DoomGame()
        self.game.load_config(config_path)
        self.living_reward = living_reward
        self.game.set_living_reward(self.living_reward)
        # Generated mazes start from map00 instead of map01
        self.game.set_doom_map('map00')
        if internalres is not None:
            self.game.set_screen_resolution(internalres)
        self.game.set_window_visible(visibility)
        self.game.set_depth_buffer_enabled(False)
        self.game.set_labels_buffer_enabled(False)
        self.enable_rgb = 'r' in buffers
        self.enable_depth = 'd' in buffers
        if self.enable_depth:
            self.game.set_depth_buffer_enabled(True)
        self.game.init()

        self.screen_height = self.game.get_screen_height()
        self.screen_width = self.game.get_screen_width()
        self.screen_channels = 0
        if self.enable_rgb:
            self.screen_channels += 3
        if self.enable_depth:
            self.screen_channels += 1

        self.obsx = obsx
        self.obsy = obsy
        self.frame_skip = frame_skip
        self.clip = clip
        self.number_maps = number_maps
        self.reward_threshold = reward_threshold
        self.episode_reward = 0.0
        self.map_pool = 1
        self.current_map_index = 0

        self._setup_action_space()
        self._setup_observation_space()

        self.cv_window_name = "VizDoom Environment"

    def _setup_action_space(self):
        num_actions = self.game.get_available_buttons_size()
        self.action_space = spaces.Discrete(num_actions)
        buttons = self.game.get_available_buttons()
        self.actions = np.identity(len(buttons), dtype=int)

    def _setup_observation_space(self):
        self.observation_space = spaces.Box(
            low=0, 
            high=255,
            shape=(self.obsy, self.obsx, self.screen_channels),
            dtype=np.uint8
        )

    def process_observation(self, state):
        if self.blind:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
        if state is None:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
        buffers = []
        if self.enable_rgb:
            rgb = np.array(state.screen_buffer)
            rgb = np.transpose(rgb, (1, 2, 0))
            rgb = cv2.resize(rgb, (self.obsx, self.obsy), interpolation=cv2.INTER_AREA)
            buffers.append(rgb)
        if self.enable_depth:
            depth = np.array(state.depth_buffer)
            depth = cv2.resize(depth, (self.obsx, self.obsy), interpolation=cv2.INTER_AREA)
            depth = np.expand_dims(depth, axis=2)
            buffers.append(depth)
        obs = np.concatenate(buffers, axis=2)
        return obs

    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict]:
        action_tuple = self.actions[action]
        reward = self.game.make_action(action_tuple, self.frame_skip)
        if self.clip:
            reward = np.clip(reward, self.clip[0], self.clip[1])

        self.episode_reward += reward

        done = self.game.is_episode_finished()
        truncated = False
        info = {}
        if done:
            state = self.game.get_state()
            if state is not None and state.screen_buffer is not None:
                obs = self.process_observation(state)
            else:
                obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
            
            if self.episode_reward >= self.reward_threshold and self.current_map_index == self.map_pool - 1:
                if self.map_pool < self.number_maps:
                    self.map_pool += 1
                    print(f"Reward threshold met! Increasing number of maps to {self.map_pool}.")
            #print(f"Episode reward is {self.episode_reward}.")
        else:
            state = self.game.get_state()
            if state is not None and state.screen_buffer is not None:
                obs = self.process_observation(state)
            else:
                obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return obs, reward, done, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, dict]:
        self.current_map_index = random.randint(0, self.map_pool - 1)
        random_map_name = f"map{self.current_map_index:02}"  # Zero-padded map name (e.g., map00, map01)
        self.game.set_doom_map(random_map_name)
        #print(f"Randomly selected map: {random_map_name}")

        if seed is not None:
            self.game.set_seed(seed)

        self.game.new_episode()
        state = self.game.get_state()

        self.episode_reward = 0.0

        if state is not None and state.screen_buffer is not None:
            obs = self.process_observation(state)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)

        info = {}
        return obs, info

    def render(self, scaling: float = 1.0, grayscale: bool = False, show: bool = False) -> np.ndarray:
        state = self.game.get_state()
        if state is not None and state.screen_buffer is not None:
            frame = np.array(state.screen_buffer, dtype=np.uint8)
            frame = np.transpose(frame, (1, 2, 0))
        else:
            frame = np.zeros((self.screen_height, self.screen_width, self.screen_channels), dtype=np.uint8)
        if grayscale:
            if self.screen_channels >= 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                frame = frame.squeeze()
        else:
            if self.screen_channels >= 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if scaling != 1.0:
            width = int(frame.shape[1] * scaling)
            height = int(frame.shape[0] * scaling)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        if show:
            cv2.imshow(self.cv_window_name, frame)
            cv2.waitKey(1)
        return frame

    def render_depth(self, scaling: float = 1.0, show: bool = False) -> np.ndarray:
        state = self.game.get_state()
        if state is not None and state.depth_buffer is not None:
            frame = np.array(state.depth_buffer, dtype=np.uint8)
        else:
            frame = np.zeros((self.screen_height, self.screen_width), dtype=np.uint8)
        if scaling != 1.0:
            width = int(frame.shape[1] * scaling)
            height = int(frame.shape[0] * scaling)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        if show:
            cv2.imshow("Depth Buffer", frame)
            cv2.waitKey(1)
        return frame

    def close(self):
        self.game.close()
        cv2.destroyAllWindows()
