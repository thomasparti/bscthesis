import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from thesis.plots.dqn_vs_dqn_opt_plots import generate_dqn_vs_dqn_opt_time_plot
from thesis.plots.hyperparams_plot import generate_hyperparams_plot
from thesis.plots.performance_plot import generate_perf_plot, generate_bar_plot
from thesis.plots.maze_plot import generate_maze_plot


def get_default_data_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')


def load_data(filename, data_dir=None):
    if data_dir is None:
        data_dir = get_default_data_dir()
    file_path = os.path.join(data_dir, filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{filename}' not found in directory '{data_dir}'")
    return pd.read_csv(file_path)


def load_data_manual(manual_dict):
    df = pd.DataFrame(list(manual_dict.items()), columns=['Step', 'Value'])
    df['Wall time'] = 0
    df = df[['Wall time', 'Step', 'Value']]
    return df


def create_hyperparams_plots(data_dir=get_default_data_dir(), save_dir='images'):
    generate_hyperparams_plot(data_dir=data_dir, save_dir=save_dir)


def create_dqn_vs_dqn_opt_plots():
    data_dict = {
        'DQN Default': load_data('ppovsdqn/dqn_vs_dqn_opt/bs/bs_dqn.csv'),
        'DQN Optimized': load_data('ppovsdqn/dqn_vs_dqn_opt/bs/bs_dqn_opt.csv'),
    }
    generate_dqn_vs_dqn_opt_time_plot(data_dict['DQN Default'], data_dict['DQN Optimized'])
    generate_perf_plot(data_dict, 'ppovsdqn/dqn_vs_dqn_opt_bs_perf.png', smoothing=0.6)


def create_ppo_vs_ppo_opt_plots():
    data_dict = {
        'PPO Default': load_data('ppovsdqn/ppo_vs_ppo_opt/dtc/dtc_ppo.csv'),
        'PPO Optimized': load_data('ppovsdqn/ppo_vs_ppo_opt/dtc/dtc_ppo_opt.csv')
    }
    generate_perf_plot(data_dict, 'ppovsdqn/ppo_vs_ppo_opt_dtc_perf.png')
    
    data_dict = {
        'PPO Default': load_data('ppovsdqn/ppo_vs_ppo_opt/smz/smz_ppo.csv'),
        'PPO Optimized': load_data('ppovsdqn/ppo_vs_ppo_opt/smz/smz_ppo_opt.csv')
    }
    generate_perf_plot(data_dict, 'ppovsdqn/ppo_vs_ppo_opt_smz_perf.png')


def create_ppo_vs_dqn_plots():
    data_dict = {
        'PPO Default': load_data('ppovsdqn/ppo_vs_dqn/dtc/dtc_dqn.csv'),
        'DQN Default': load_data('ppovsdqn/ppo_vs_dqn/dtc/dtc_ppo.csv'),
    }
    generate_perf_plot(data_dict, 'ppovsdqn/ppo_vs_dqn_dtc_perf.png')

    data_dict = {
        'PPO Default': load_data('ppovsdqn/ppo_vs_dqn/smz/smz_ppo.csv'),
        'DQN Default': load_data('ppovsdqn/ppo_vs_dqn/smz/smz_dqn.csv'),
    }
    generate_perf_plot(data_dict, 'ppovsdqn/ppo_vs_dqn_smz_perf.png', smoothing=0.6)

def create_resolution_plots():
    data_dict = {
        '42x42': load_data('display/res/dtc_42x42.csv'),
        '64x64': load_data('display/res/dtc_64x64.csv'),
        '128x128': load_data('display/res/dtc_128x128.csv'),
        '256x256': load_data('display/res/dtc_256x256.csv'),
    }
    generate_perf_plot(data_dict, 'resolution/res_perf_dtc.png', hue_name='Resolution')

    data_dict = {
        '42x42': load_data('display/res/smz_42x42.csv'),
        '64x64': load_data('display/res/smz_64x64.csv'),
        '128x128': load_data('display/res/smz_128x128.csv'),
        '256x256': load_data('display/res/smz_256x256.csv'),
    }
    generate_perf_plot(data_dict, 'resolution/res_perf_smz.png', hue_name='Resolution', smoothing=0.95)

    data_dict_manual = {
        'DTC and SMZ': load_data_manual({
         5292: 1300.1336085796356,
         12288: 1392.3371500968933,
         49152: 2073.02507519722,
         196608: 5975.747797727585
        }),
    }
    generate_perf_plot(data_dict_manual, 'resolution/res_params_time.png', hue_name=None, xlabel='Number input of parameters', ylabel='Training time (seconds)', smoothing=0)

    data_dict_manual = {
        'DTC and SMZ': load_data_manual({
            5292: 1.4,
            12288: 7.1,
            49152: 56,
            196608: 298
        }),
    }
    generate_perf_plot(data_dict_manual, 'resolution/res_params_size.png', hue_name=None, xlabel='Number input of parameters', ylabel='Model size (megabytes)', smoothing=0)


def create_frameskip_plots():
    data_dict = {
        '1': load_data('display/sk/smz_sk_1.csv'),
        '2': load_data('display/sk/smz_sk_2.csv'),
        '4': load_data('display/sk/smz_sk_4.csv'),
        '6': load_data('display/sk/smz_sk_6.csv'),
        '8': load_data('display/sk/smz_sk_8.csv'),
        '10': load_data('display/sk/smz_sk_10.csv'),
    }
    generate_perf_plot(data_dict, 'frameskip/fs_smz_perf.png', hue_name='Frameskip values', smoothing=0.95)

    # Values are directly taken from the paper
    data_dict_manual = {
        'BS': load_data_manual({
        1: 67.1,
        2: 68.5,
        3: 77.7,
        4: 77.6,
        5: 75,
        6: 74.8,
        7: 84.2,
        8: 74.1,
        9: 83.1,
        10: 74.1,
        11: 80.3,
        15: 61.9,
        20: 70.7,
        25: 66,
        30: 73.6,
        35: 40.8,
        40: 61.4,
        45: 45.8,
        50: 43.4
        }),
    }
    generate_perf_plot(data_dict_manual, 'frameskip/paper_fs.png', hue_name=None, xlabel='Number of frames skipped', ylabel='Mean performance', smoothing=0.6)

    # Manually extracted mean values
    data_dict_manual = {
        'BS': load_data_manual({
         '1': 8.141251184542973,
         '2': 29.918685862753126,
         '4': 56.98409713506699,
         '6': 43.7772452990214,
         '8': 31.0837036702368,
         '10': 19.719458728366426
        }),
    }
    generate_perf_plot(data_dict_manual, 'frameskip/fs_perf_values.png', hue_name=None, xlabel='Number of frames skipped', ylabel='Mean performance', smoothing=0)


def create_framestack_plots():
    data_dict = {
        '1 (no frame-stacking)': load_data('display/fs/pp.csv'),
        '4': load_data('display/fs/pp_fs.csv'),
    }
    generate_perf_plot(data_dict, 'framestack/pp_fs.png', hue_name='Frame-stack values', smoothing=0.9)

    data_dict = {
        '1 (no frame-stacking)': load_data('display/fs/pp_sk.csv'),
        '4': load_data('display/fs/pp_fs_sk.csv'),
    }
    generate_perf_plot(data_dict, 'framestack/pp_fs_sk.png', hue_name='Frame-stack values', smoothing=0.9)


def create_buffers_plots():
    data_dict = {
        'Grayscale': load_data('display/buffers/dtc/dtc_gs.csv'),
        'Depth': load_data('display/buffers/dtc/dtc_d.csv'),
        'RGB': load_data('display/buffers/dtc/dtc_rgb.csv'),
    }
    generate_perf_plot(data_dict, 'buffers/buffers_dtc.png', hue_name='Buffer', smoothing=0.8)

    data_dict = {
        'Grayscale': load_data('display/buffers/smz/smz_gs.csv'),
        'Depth': load_data('display/buffers/smz/smz_d.csv'),
        'RGB': load_data('display/buffers/smz/smz_rgb.csv'),
    }
    generate_perf_plot(data_dict, 'buffers/buffers_smz.png', hue_name='Buffer', smoothing=0.9)

    data_dict = {
        'Grayscale': load_data('display/buffers/mwh/mwh_gs.csv'),
        'Depth': load_data('display/buffers/mwh/mwh_d.csv'),
        'RGB': load_data('display/buffers/mwh/mwh_rgb.csv'),
    }
    generate_perf_plot(data_dict, 'buffers/buffers_mwh.png', hue_name='Buffer', smoothing=0.9)

    data_dict = {
        'Depth + RGB': load_data('display/buffers/mwh/mwh_rgbd.csv'),
        'Depth': load_data('display/buffers/mwh/mwh_d.csv'),
        'RGB': load_data('display/buffers/mwh/mwh_rgb.csv'),
    }
    generate_perf_plot(data_dict, 'buffers/rgbd_mwh.png', hue_name='Buffer', smoothing=0.9)

    data_dict = {
        'Depth + RGB': load_data('display/buffers/dtc/dtc_rgbd.csv'),
        'Depth': load_data('display/buffers/dtc/dtc_d.csv'),
        'RGB': load_data('display/buffers/dtc/dtc_rgb.csv'),
    }
    generate_perf_plot(data_dict, 'buffers/rgbd_dtc.png', hue_name='Buffer', smoothing=0.8)

    data_dict = {
        'Depth + RGB': load_data('display/buffers/smz/smz_rgbd.csv'),
        'Depth': load_data('display/buffers/smz/smz_d.csv'),
        'RGB': load_data('display/buffers/smz/smz_rgb.csv'),
    }
    generate_perf_plot(data_dict, 'buffers/rgbd_smz.png', hue_name='Buffer', smoothing=0.9)

def create_reward_shaping_plots():
    data_dict = {
        '-0.0001': load_data('rewshape/mwh_lr_00001.csv'),
        '-0.001': load_data('rewshape/mwh_lr_0001.csv'),
        '-0.01': load_data('rewshape/mwh_lr_001.csv'),
        '-0.1': load_data('rewshape/mwh_lr_01.csv'),
        '0': load_data('rewshape/mwh_nolr.csv'),
    }
    generate_perf_plot(data_dict, 'rewshape/mwh_rew_len.png', hue_name='Living reward', ylabel='Mean episode length', smoothing=0.8)

def create_number_maps_plots():
    data_dict_manual = {
        'Map numbers': load_data_manual({
         1: -7.8680699999999995,
         3: -6.169309999999999,
         5: -7.28041,
         10: -7.878979999999999,
         20: -9.031,
        }),
    }
    generate_perf_plot(data_dict_manual, 'mapnumber/map_number_comp.png', hue_name=None, xlabel='Number of maps', ylabel='Mean performance', smoothing=0)

def create_curr_learning_plots():

    data_dict = {
        'Default': load_data('texture/10x10x3.csv'),
        'Randomized': load_data('texture/10x10x3_randomtextures.csv'),
        'Adaptive': load_data('texture/10x10x3_adaptive.csv'),
    }
    generate_perf_plot(data_dict, 'texture/texture_perf.png', hue_name='Textures', smoothing=0.9)

    data_dict_manual = {
         'Default': -10.67477,
         'Randomized': -8.56893,
         'Adaptive': -7.68742,
    }
    generate_bar_plot(data_dict_manual, 'texture/texture_perf_bar.png', xlabel='Textures', ylabel='Mean performance')

    data_dict = {
        '10x10': load_data('size/10x10x3.csv'),
        '20x20': load_data('size/20x20x3.csv'),
        'Adaptive': load_data('size/adaptive_size.csv'),
    }
    generate_perf_plot(data_dict, 'size/size_perf.png', hue_name='Map sizes', smoothing=0.9)

    data_dict_manual = {
         '10x10': -11.902370000000001,
         '20x20': -12.65466,
         'Adaptive': -10.61442,
    }
    generate_bar_plot(data_dict_manual, 'size/size_perf_bar.png', xlabel='Map sizes', ylabel='Mean performance')

def create_maze_plots():
    maze1 = os.path.join(get_default_data_dir(), 'mazes/maze1.txt')
    maze2 = os.path.join(get_default_data_dir(), 'mazes/maze2.txt')
    maze3 = os.path.join(get_default_data_dir(), 'mazes/maze3.txt')
    maze4 = os.path.join(get_default_data_dir(), 'mazes/maze4.txt')
    maze5 = os.path.join(get_default_data_dir(), 'mazes/maze5.txt')
    maze6 = os.path.join(get_default_data_dir(), 'mazes/maze6.txt')
    generate_maze_plot(maze1, 'mazes/maze1.png')
    generate_maze_plot(maze2, 'mazes/maze2.png')
    generate_maze_plot(maze3, 'mazes/maze3.png')
    generate_maze_plot(maze4, 'mazes/maze4.png')
    generate_maze_plot(maze5, 'mazes/maze5.png')
    generate_maze_plot(maze6, 'mazes/maze6.png')

def main():
    create_dqn_vs_dqn_opt_plots()
    create_hyperparams_plots()
    create_ppo_vs_ppo_opt_plots()
    create_ppo_vs_dqn_plots()
    create_resolution_plots()
    create_frameskip_plots()
    create_framestack_plots()
    create_buffers_plots()
    create_reward_shaping_plots()
    create_number_maps_plots()
    create_curr_learning_plots()
    create_maze_plots()

if __name__ == '__main__':
    main()

