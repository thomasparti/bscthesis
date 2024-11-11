import os
from typing import Any, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces
import vizdoom as vzd
import numpy as np
import cv2
import time

class VizDoomEnv(gym.Env):

    def __init__(self, config_path: str, visibility: bool = True):
        super(VizDoomEnv, self).__init__()

        self.game = vzd.DoomGame()
        self.game.load_config(config_path)
        self.game.set_window_visible(visibility)
        self.game.init()

        self.screen_height = self.game.get_screen_height()
        self.screen_width = self.game.get_screen_width()
        self.screen_channels = self.game.get_screen_channels()

        self._setup_action_space()
        self._setup_observation_space()

        self.cv_window_name = "VizDoom Environment"

    def _setup_action_space(self):
        num_actions = self.game.get_available_buttons_size()
        self.action_space = spaces.Discrete(num_actions)
        buttons = self.game.get_available_buttons()
        self.actions = np.identity(len(buttons), dtype=int)

    def _setup_observation_space(self):
        state = self.game.get_state()
        if state is not None and state.screen_buffer is not None:
            screen = state.screen_buffer
        else:
            screen = np.zeros((self.screen_height, self.screen_width, self.screen_channels), dtype=np.uint8)

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.screen_height, self.screen_width, self.screen_channels),
                                            dtype=np.uint8)

    def process_observation(self, state):
        if state is None:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
        
        state = np.array(state.screen_buffer)
        state = np.transpose(state, (1, 2, 0))
        return state

    def step(self, action: int) -> Tuple[Any, float, bool, bool, dict]:
        action_tuple = self.actions[action]

        reward = self.game.make_action(action_tuple)
        done = self.game.is_episode_finished()
        truncated = False
        info = {}

        if done:
            state = self.game.get_state()
            if state is not None and state.screen_buffer is not None:
                obs = self.process_observation(state)
            else:
                obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        else:
            state = self.game.get_state()
            if state is not None and state.screen_buffer is not None:
                obs = self.process_observation(state)
            else:
                obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
    
        return obs, reward, done, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Any, dict]:
        if seed is not None:
            self.game.set_seed(seed)

        self.game.new_episode()
        state = self.game.get_state()
        if state is not None and state.screen_buffer is not None:
            obs = self.process_observation(state)
        else:
            obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        info = {}
        return obs, info

    def render(self, scaling: float = 1.0, grayscale: bool = False) -> None:
        state = self.game.get_state()
        if state is not None and state.screen_buffer is not None:
            frame = np.array(state.screen_buffer, dtype=np.uint8)
            frame = np.transpose(frame, (1, 2, 0))
        else:
            frame = np.zeros((self.screen_height, self.screen_width, self.screen_channels), dtype=np.uint8)

        if grayscale:
            if self.screen_channels == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                pass
        else:
            if self.screen_channels == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if scaling != 1.0:
            width = int(frame.shape[1] * scaling)
            height = int(frame.shape[0] * scaling)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        cv2.imshow(self.cv_window_name, frame)
        cv2.waitKey(1)

    def close(self):
        self.game.close()
        cv2.destroyAllWindows()

