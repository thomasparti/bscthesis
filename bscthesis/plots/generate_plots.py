import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from thesis.plots.dqn_vs_dqn_opt_plots import generate_dqn_vs_dqn_opt_time_plot
from thesis.plots.hyperparams_plot import generate_hyperparams_plot
from thesis.plots.performance_plot import generate_perf_plot


def get_default_data_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data')


def load_data(filename, data_dir=None):
    if data_dir is None:
        data_dir = get_default_data_dir()
    file_path = os.path.join(data_dir, filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File '{filename}' not found in directory '{data_dir}'")
    return pd.read_csv(file_path)


def create_hyperparams_plots(data_dir=get_default_data_dir(), save_dir='images'):
    generate_hyperparams_plot(data_dir=data_dir, save_dir=save_dir)


def create_dqn_vs_dqn_opt_plots(data_dqn_filename='ppovsdqn/dqn_vs_dqn_opt/bs/bs_dqn.csv', data_dqn_opt_filename='ppovsdqn/dqn_vs_dqn_opt/bs/bs_dqn_opt.csv', save_dir='images'):
    data_dqn = load_data(data_dqn_filename)
    data_dqn_opt = load_data(data_dqn_opt_filename)
    generate_dqn_vs_dqn_opt_time_plot(data_dqn, data_dqn_opt)
    generate_perf_plot(data_dqn, 'DQN Default', data_dqn_opt, 'DQN Optimized', 'dqn_vs_dqn_opt_bs_perf.png')


def create_ppo_vs_ppo_opt_plots():
    data_ppo = load_data('ppovsdqn/ppo_vs_ppo_opt/dtc/dtc_ppo.csv')
    data_ppo_opt = load_data('ppovsdqn/ppo_vs_ppo_opt/dtc/dtc_ppo_opt.csv')
    generate_perf_plot(data_ppo, 'PPO Default', data_ppo_opt, 'PPO Optimized', 'ppo_vs_ppo_opt_dtc_perf.png')

    data_ppo = load_data('ppovsdqn/ppo_vs_ppo_opt/smz/smz_ppo.csv')
    data_ppo_opt = load_data('ppovsdqn/ppo_vs_ppo_opt/smz/smz_ppo_opt.csv')
    generate_perf_plot(data_ppo, 'PPO Default', data_ppo_opt, 'PPO Optimized', 'ppo_vs_ppo_opt_smz_perf.png')


def create_ppo_vs_dqn_plots():
    data_dqn = load_data('ppovsdqn/ppo_vs_dqn/dtc/dtc_dqn.csv')
    data_ppo = load_data('ppovsdqn/ppo_vs_dqn/dtc/dtc_ppo.csv')
    generate_perf_plot(data_ppo, 'PPO Default', data_dqn, 'DQN Default', 'ppo_vs_dqn_dtc_perf.png')

    data_dqn = load_data('ppovsdqn/ppo_vs_dqn/smz/smz_dqn.csv')
    data_ppo = load_data('ppovsdqn/ppo_vs_dqn/smz/smz_ppo.csv')
    generate_perf_plot(data_ppo, 'PPO Default', data_dqn, 'DQN Default', 'ppo_vs_dqn_smz_perf.png')


def main():
    create_dqn_vs_dqn_opt_plots()
    create_hyperparams_plots()
    create_ppo_vs_ppo_opt_plots()
    create_ppo_vs_dqn_plots()


if __name__ == '__main__':
    main()

