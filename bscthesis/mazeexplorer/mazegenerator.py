import datetime
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .maze import generate_mazes
from .script_manipulator import write_config, write_acs
from .wad import generate_wads
from .compile_acs import compile_acs

dir_path = os.path.dirname(os.path.realpath(__file__))

import os
import shutil
import tempfile
from pathlib import Path

class MazeGenerator:
    def __init__(
        self,
        unique_maps=False,
        number_maps=1,
        keys=9,
        size=(10, 10),
        random_spawn=False,
        random_textures=False,
        random_key_positions=False,
        floor_texture="CEIL5_2",
        ceilling_texture="CEIL5_1",
        wall_texture="STONE2",
        actions="MOVE_FORWARD TURN_LEFT TURN_RIGHT",
        episode_timeout=1500,
        complexity=0.7,
        density=0.7,
        mazes_path=None
    ):
        """
        Initializes the MazeGenerator with specified parameters.

        :param unique_maps: If set, every map will only be generated once.
        :param number_maps: Number of maps to generate.
        :param keys: Number of keys to collect in each maze.
        :param size: Tuple indicating the size of the maze (width, height).
        :param random_spawn: Whether to randomize the spawn position.
        :param random_textures: Whether to randomize textures.
        :param random_key_positions: Whether to randomize key positions.
        :param floor_texture: Texture for the floor.
        :param ceilling_texture: Texture for the ceilling.
        :param wall_texture: Texture for the walls.
        :param actions: Available actions in the maze.
        :param episode_timeout: Timeout for each maze generation.
        :param complexity: Complexity of the maze generation (0 to 1).
        :param density: Density of the maze (0 to 1).
        :param mazes_path: Path to save the generated mazes. Creates a temporary directory if None.
        """
        self.unique_maps = unique_maps
        self.number_maps = number_maps
        self.keys = keys
        self.size = size
        self.random_spawn = random_spawn
        self.random_textures = random_textures
        self.random_key_positions = random_key_positions
        self.floor_texture = floor_texture
        self.ceilling_texture = ceilling_texture
        self.wall_texture = wall_texture
        self.actions = actions
        self.episode_timeout = episode_timeout
        self.complexity = complexity
        self.density = density

        self.mazes_path = mazes_path if mazes_path is not None else tempfile.mkdtemp()
        self._setup_mazes_directory()

    def _setup_mazes_directory(self):
        shutil.rmtree(self.mazes_path, ignore_errors=True)
        os.makedirs(self.mazes_path, exist_ok=True)
        print(f"Mazes directory initialized at: {self.mazes_path}")

    def generate_mazes(self):
        write_acs(
            keys=self.keys,
            random_spawn=self.random_spawn,
            random_textures=self.random_textures,
            random_key_positions=self.random_key_positions,
            map_size=self.size,
            number_maps=self.number_maps,
            floor_texture=self.floor_texture,
            ceilling_texture=self.ceilling_texture,
            wall_texture=self.wall_texture
        )
        print("ACS scripts written.")

        compile_acs(self.mazes_path)
        print("ACS scripts compiled.")

        maze_dir = os.path.join(
            self.mazes_path, f"{self.size[0]}x{self.size[1]}"
        )
        mazes = generate_mazes(
            maze_dir,
            self.number_maps,
            self.size[0],
            self.size[1],
            complexity=self.complexity,
            density=self.density
        )
        print(f"{self.number_maps} mazes generated at {maze_dir}.")

        outputs_dir = os.path.join(self.mazes_path, "outputs")
        os.makedirs(outputs_dir, exist_ok=True)
        wad_path = os.path.join(self.mazes_path, f"{self.size[0]}x{self.size[1]}.wad")
        try:
            generate_wads(
                maze_dir,
                wad_path,
                os.path.join(outputs_dir, "maze.o")
            )
            print(f"WAD file created at {wad_path}.")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{e.strerror}\n"
                "Ensure all required submodules are pulled.\n"
                "If not, run:\n\n\tgit submodule update --init --recursive"
            )

        cfg_path = write_config(
            maze_dir,
            self.actions,
            episode_timeout=self.episode_timeout
        )
        print(f"Configuration file written at {cfg_path}.")

        return cfg_path

    def save_mazes(self, destination_dir):
        if not os.path.exists(self.mazes_path):
            raise FileNotFoundError(f"Mazes path '{self.mazes_path}' does not exist.")
        
        if os.path.exists(destination_dir):
            raise FileExistsError(f"Destination directory '{destination_dir}' already exists.")
        
        shutil.copytree(self.mazes_path, destination_dir)
        print(f"Mazes have been successfully saved to {destination_dir}.")

    def generate_and_save_mazes(self, destination_dir):
        cfg_path = self.generate_mazes()
        self.save_mazes(destination_dir)
        return cfg_path
