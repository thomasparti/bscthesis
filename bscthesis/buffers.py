import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from train.train_ppo import train as train_ppo
from agents.agent_ppo import agent as agent_ppo

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train or run PPO agent in the ViZDoom environment."
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-t', '--train',
        action='store_true',
        help="Training mode."
    )
    group.add_argument(
        '-a', '--agent',
        action='store_true',
        help="Agent inference mode."
    )
    
    parser.add_argument(
        '--logdir',
        type=str,
        help="Directory for logs."
    )
    parser.add_argument(
        '--modeldir',
        type=str,
        help="Directory to save trained models."
    )
    parser.add_argument(
        '-c', '--cycles',
        type=int,
        help="Number of training cycles."
    )
    parser.add_argument(
        '-l', '--length',
        type=int,
        help="Length (number of timesteps) of one training cycle."
    )
    parser.add_argument(
        '--cfg',
        type=str,
        help="Path to the ViZDoom environment configuration file."
    )
    parser.add_argument(
        '-p', '--params',
        type=str,
        help="Path to the YAML parameters configuration file."
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help="Path to the trained model to be used."
    )
    parser.add_argument(
        '-v', '--video',
        type=str,
        metavar='VIDEO_PATH',
        help="Path and filename to save recorded footage as video (e.g., ./logs/ppo/video.mp4)."
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help="Enable rendering of the environment."
    )
    parser.add_argument(
        '-e', '--episodes',
        type=int,
        default=10,
        help="Number of episodes to run the inference for. Defaults to 10."
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.0,
        help="Delay (in seconds) between each step. Defaults to 0.0."
    )
    parser.add_argument(
        '-f', '--framestack',
        type=int,
        default=1,
        help="Number of frames to stack."
    )
    parser.add_argument(
        '-x', '--obsx',
        type=int,
        default=42,
        help="Width of the observation (obsx)."
    )
    parser.add_argument(
        '-y', '--obsy',
        type=int,
        default=42,
        help="Height of the observation (obsy)."
    )
    parser.add_argument(
        '-b', '--buffers',
        type=str,
        default='rd',
        help="Buffers to use for observation (e.g., 'r', 'd', 'rd')."
    )
    parser.add_argument(
        '-s', '--frameskip',
        type=int,
        default=4,
        help="Frame skipping value."
    )
    parser.add_argument(
        '-d', '--depth',
        action='store_true',
        help="Use render_depth method for video recording instead of the default render method."
    )
    
    args = parser.parse_args()
    
    if args.train:
        required_train_args = ['logdir', 'modeldir', 'cycles', 'length', 'cfg', 'params']
        missing_args = [arg for arg in required_train_args if getattr(args, arg) is None]
        if missing_args:
            parser.error(
                f"The following arguments are required for training mode: {', '.join('--' + arg for arg in missing_args)}"
            )
    elif args.agent:
        required_agent_args = ['cfg', 'model', 'logdir']
        missing_args = [arg for arg in required_agent_args if getattr(args, arg) is None]
        if missing_args:
            parser.error(
                f"The following arguments are required for agent mode: {', '.join('--' + arg for arg in missing_args)}"
            )
    
    return args

def main():
    args = parse_arguments()
    
    if args.train:
        logdir = args.logdir
        modeldir = args.modeldir
        cycles = args.cycles
        length = args.length
        cfg = args.cfg
        params = args.params
        framestack = args.framestack
        obsx = args.obsx
        obsy = args.obsy
        buffers = args.buffers
        frameskip = args.frameskip
        
        if not os.path.exists(cfg):
            print(f"Error: ViZDoom config file '{cfg}' does not exist.")
            sys.exit(1)
        if not os.path.exists(params):
            print(f"Error: Parameters config file '{params}' does not exist.")
            sys.exit(1)
        if not os.path.isdir(logdir):
            try:
                os.makedirs(logdir, exist_ok=True)
            except Exception as e:
                print(f"Error creating log directory '{logdir}': {e}")
                sys.exit(1)
        if not os.path.isdir(modeldir):
            try:
                os.makedirs(modeldir, exist_ok=True)
            except Exception as e:
                print(f"Error creating model directory '{modeldir}': {e}")
                sys.exit(1)
        
        print("Starting PPO training...")
        train_ppo(
            logdir=logdir,
            modeldir=modeldir,
            cycles=cycles,
            length=length,
            cfg=cfg,
            params=params,
            framestack=framestack,
            obsx=obsx,
            obsy=obsy,
            buffers=buffers,
            frameskip=frameskip
        )
        print("PPO training completed.")
        
    elif args.agent:
        cfg = args.cfg
        model_path = args.model
        logdir = args.logdir
        video = args.video
        episodes = args.episodes
        delay = args.delay
        render = args.render
        framestack = args.framestack
        obsx = args.obsx
        obsy = args.obsy
        buffers = args.buffers
        frameskip = args.frameskip
        use_depth = args.depth
        
        if not os.path.exists(cfg):
            print(f"Error: ViZDoom config file '{cfg}' does not exist.")
            sys.exit(1)
        if not os.path.exists(model_path):
            print(f"Error: Model file '{model_path}' does not exist.")
            sys.exit(1)
        if not os.path.isdir(logdir):
            try:
                os.makedirs(logdir, exist_ok=True)
            except Exception as e:
                print(f"Error creating log directory '{logdir}': {e}")
                sys.exit(1)
        
        print("Running PPO agent...")
        agent_ppo(
            cfg=cfg,
            model=model_path,
            logdir=logdir,
            video=video,
            episodes=episodes,
            delay=delay,
            render=render,
            framestack=framestack,
            obsx=obsx,
            obsy=obsy,
            buffers=buffers,
            frameskip=frameskip,
            use_depth=use_depth
        )
        print("PPO agent run completed.")
        
    else:
        print("Either training (-t) or agent (-a) mode must be specified.")
        sys.exit(1)

if __name__ == "__main__":
    main()
