import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_perf_plot(data1, algoname1, data2,algoname2, save_name, save_dir='images'):
    data1['Algorithm'] = algoname1
    data2['Algorithm'] = algoname2
    combined_data = pd.concat([data1, data2], ignore_index=True)

    plt.figure(figsize=(8, 5), dpi=300)
    sns.lineplot(data=combined_data, x='Step', y='Value', hue='Algorithm', 
                 palette={algoname1: '#FF8C00', algoname2: '#00C0FF'}, linewidth=2.5)
    plt.xlabel("Training Steps", fontsize=14)
    plt.ylabel("Performance", fontsize=14)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    plt.close()
