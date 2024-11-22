import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def generate_perf_plot(data_dict, save_name, hue_name='Algorithms',
                       xlabel='Training Steps', ylabel='Performance', smoothing=0.8, save_dir='images'):
    color_list = ['#FF8C00', '#00C0FF', '#DC143C', '#228B22', '#FFD700', '#800080']

    plt.figure(figsize=(8, 5), dpi=300)

    if hue_name is not None:
        labels = list(data_dict.keys())
        combined_data = pd.DataFrame()
        for label, data in data_dict.items():
            data = data.copy()
            data[hue_name] = label
            combined_data = pd.concat([combined_data, data], ignore_index=True)
        combined_data = combined_data.sort_values([hue_name, 'Step']).reset_index(drop=True)
        if smoothing > 0.0:
            def smooth_data(series, smoothing_factor):
                smoothed = []
                last = series.iloc[0]
                for val in series:
                    last = last * smoothing_factor + val * (1 - smoothing_factor)
                    smoothed.append(last)
                return pd.Series(smoothed, index=series.index)
            combined_data['Value'] = combined_data.groupby(hue_name)['Value'].transform(lambda x: smooth_data(x, smoothing))
        sns.lineplot(data=combined_data, x='Step', y='Value', hue=hue_name,
                     palette=color_list[:len(labels)], linewidth=2.5, hue_order=labels)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.legend(title=hue_name, fontsize=10)
    else:
        for idx, (label, data) in enumerate(data_dict.items()):
            data = data.copy()
            if smoothing > 0.0:
                def smooth_data(series, smoothing_factor):
                    smoothed = []
                    last = series.iloc[0]
                    for val in series:
                        last = last * smoothing_factor + val * (1 - smoothing_factor)
                        smoothed.append(last)
                    return pd.Series(smoothed, index=series.index)
                data['Value'] = smooth_data(data['Value'], smoothing)
            sns.lineplot(data=data, x='Step', y='Value',
                         color=color_list[idx % len(color_list)], linewidth=2.5, label=None)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    plt.close()
