import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_dqn_vs_dqn_opt_time_plot(data_dqn, data_dqn_opt, save_dir='images', save_name='ppovsdqn/dqn_vs_dqn_opt_bs_time.png'):
    exact_performance_target = data_dqn_opt['Value'].max()
    initial_time_opt = data_dqn_opt['Wall time'].min()
    target_time_opt_exact = data_dqn_opt.loc[data_dqn_opt['Value'] == exact_performance_target, 'Wall time'].values[0]
    time_to_exact_target_opt = target_time_opt_exact - initial_time_opt

    closest_performance_dqn_exact = data_dqn.iloc[(data_dqn['Value'] - exact_performance_target).abs().argmin()]
    initial_time_dqn = data_dqn['Wall time'].min()
    target_time_dqn_exact = closest_performance_dqn_exact['Wall time']
    time_to_exact_target_dqn = target_time_dqn_exact - initial_time_dqn

    training_times_exact = pd.DataFrame({
        'Algorithm': ['DQN Default', 'DQN Optimized'],
        'Training Time': [time_to_exact_target_dqn, time_to_exact_target_opt]
    })

    plt.figure(figsize=(8, 5), dpi=300)
    sns.barplot(data=training_times_exact, x='Algorithm', y='Training Time', hue='Algorithm', 
                palette=['#FF8C00', '#00C0FF'], edgecolor="black", dodge=False, legend=False)
    plt.xlabel("Algorithm", fontsize=14)
    plt.ylabel("Training Time (seconds)", fontsize=14)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    plt.close()
