import os
import sys
import argparse
from mazeexplorer import MazeGenerator

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate and save mazes.")
    parser.add_argument('-s', type=int, default=10, help="Size of the maze (s x s). Defaults to 10.")
    parser.add_argument('-m', type=int, default=10, help="Number of maps to generate. Defaults to 10.")
    parser.add_argument('-r', action='store_true', help="Enable texture randomization.")
    parser.add_argument('-n', '--name', type=str, required=True, help="Name of the output file.")
    parser.add_argument('-k', action='store_true', help="Disable random key positions.")
    parser.add_argument('-w', action='store_true', help="Disable random spawn.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    size = (args.s, args.s)
    number_maps = args.m
    random_textures = args.r
    name = args.name
    keys = 0 if args.k else 1
    random_spawn = not args.w
    maze_generator = MazeGenerator(
        unique_maps=True,
        number_maps=number_maps,
        keys=keys,
        size=size,
        random_spawn=random_spawn,
        random_textures=random_textures,
        random_key_positions=(keys == 1),
        complexity=0.8,
        density=0.6
    )
    destination_directory = os.path.join("./maps", name)
    try:
        config_file_path = maze_generator.generate_and_save_mazes(destination_directory)
        print(f"Maze configuration generated at: {config_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
