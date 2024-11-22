import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_hyperparams_plot(data_dir, save_dir='images', save_name='ppovsdqn/hyperparams_comparison.png'):
    a2c_default = pd.read_csv(os.path.join(data_dir, 'hyperparams/A2C_default.csv'), sep=';')
    a2c_optimized = pd.read_csv(os.path.join(data_dir, 'hyperparams/A2C_optimized.csv'), sep=';')
    dqn_default = pd.read_csv(os.path.join(data_dir, 'hyperparams/DQN_default.csv'), sep=';')
    dqn_optimized = pd.read_csv(os.path.join(data_dir, 'hyperparams/DQN_optimized.csv'), sep=';')
    ppo_default = pd.read_csv(os.path.join(data_dir, 'hyperparams/PPO_default.csv'), sep=';')
    ppo_optimized = pd.read_csv(os.path.join(data_dir, 'hyperparams/PPO_optimized.csv'), sep=';')

    for df in [a2c_default, a2c_optimized, dqn_default, dqn_optimized, ppo_default, ppo_optimized]:
        if 'Reward;Algorithm;Hyperparameter' in df.columns:
            df[['Reward', 'Algorithm', 'Hyperparameter']] = df['Reward;Algorithm;Hyperparameter'].str.split(';', expand=True)
            df.drop(columns=['Reward;Algorithm;Hyperparameter'], inplace=True)
        df['Performance'] = pd.to_numeric(df['Reward'], errors='coerce')

    df_comparison_merged = pd.concat([a2c_default, a2c_optimized, dqn_default, dqn_optimized, ppo_default, ppo_optimized], ignore_index=True)

    average_performance = df_comparison_merged.groupby(['Algorithm', 'Hyperparameter'])['Performance'].mean().reset_index()

    color_map = {'Default': '#FF8C00', 'Optimized': '#00C0FF'}

    plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(data=average_performance, x='Algorithm', y='Performance', hue='Hyperparameter', 
                palette=color_map, edgecolor="black")
    plt.xlabel("Algorithm", fontsize=14)
    plt.ylabel("Performance", fontsize=14)
    plt.legend(title='Hyperparameter', loc='upper left')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    plt.close()
