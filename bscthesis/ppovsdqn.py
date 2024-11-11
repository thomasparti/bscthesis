import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from train.train_ppo import train as train_ppo
from train.train_dqn import train as train_dqn
from agents.agent_ppo import agent as agent_ppo
from agents.agent_dqn import agent as agent_dqn

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train or run PPO/DQN agents in the ViZDoom environment."
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-t', '--train',
        choices=['ppo', 'dqn'],
        help="Training mode: choose between 'ppo' or 'dqn'."
    )
    group.add_argument(
        '-a', '--agent',
        choices=['ppo', 'dqn'],
        help="Agent mode: choose between 'ppo' or 'dqn'."
    )
    
    # Training mode arguments
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
    
    # Agent mode arguments
    parser.add_argument(
        '-v', '--video',
        action='store_true',
        help="Flag to save recorded footage as video."
    )
    parser.add_argument(
        '--model',
        type=str,
        help="Path to the trained model to be used."
    )
    
    args = parser.parse_args()
    
    # Validate arguments based on selected mode
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
        agent_type = args.train
        logdir = args.logdir
        modeldir = args.modeldir
        cycles = args.cycles
        length = args.length
        cfg = args.cfg
        params = args.params
        
        # Validate paths
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
        
        # Call the appropriate training function
        if agent_type == 'ppo':
            print("Starting PPO training...")
            train_ppo(
                logdir=logdir,
                modeldir=modeldir,
                cycles=cycles,
                length=length,
                cfg=cfg,
                params=params
            )
            print("PPO training completed.")
        elif agent_type == 'dqn':
            print("Starting DQN training...")
            train_dqn(
                logdir=logdir,
                modeldir=modeldir,
                cycles=cycles,
                length=length,
                cfg=cfg,
                params=params
            )
            print("DQN training completed.")
        else:
            print("Invalid training agent type. Choose 'ppo' or 'dqn'.")
            sys.exit(1)
    
    elif args.agent:
        agent_type = args.agent
        cfg = args.cfg
        model_path = args.model
        logdir = args.logdir
        video = args.video
        
        # Validate paths
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
        
        # Call the appropriate agent function
        if agent_type == 'ppo':
            print("Running PPO agent...")
            agent_ppo(
                cfg=cfg,
                model=model_path,
                logdir=logdir,
                video=video
            )
            print("PPO agent run completed.")
        elif agent_type == 'dqn':
            print("Running DQN agent...")
            agent_dqn(
                cfg=cfg,
                model=model_path,
                logdir=logdir,
                video=video
            )
            print("DQN agent run completed.")
        else:
            print("Invalid agent type. Choose 'ppo' or 'dqn'.")
            sys.exit(1)
    
    else:
        print("Either training (-t) or agent (-a) mode must be specified.")
        sys.exit(1)

if __name__ == "__main__":
    main()
