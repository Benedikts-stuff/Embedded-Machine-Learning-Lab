import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_inference_time_benchmark_results(stats, title, path):
    names = list(stats.keys())
    sums = [np.sum(stats[name]) for name in names]
    print("SUMS, ", sums)
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(names, sums, color=['#3498db', '#e67e22', '#2ecc71'])
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f} ms', 
                 ha='center', va='bottom', fontweight='bold')

    plt.title(title)
    plt.ylabel('Time (ms)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(path)
    print(f"Plots saved as {path}")


def plot_ap_benchmark_results(stats, title, path):
    names = list(stats.keys())
    values = [stats[name] for name in names]
    
    plt.figure(figsize=(10, 6))
    
    colors = ['#9b59b6', '#3498db', '#1abc9c', '#f1c40f', '#e67e22']
    bars = plt.bar(names, values, color=colors[:len(names)])
    
    for bar in bars:
        yval = bar.get_height()
        label_text = f'{yval*100:.2f}%' if yval <= 1.0 else f'{yval:.2f}'
        
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, label_text, 
                 ha='center', va='bottom', fontweight='bold')

    plt.title(title)
    plt.ylabel('Average Precision (AP)')
    
    if max(values) <= 1.0:
        plt.ylim(0, 1.1) 
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(path)
    print(f"AP-Plot saved under '{path}'")


def plot_person_only_training_history(history):
    epochs = range(1, len(history["loss"]) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epoche')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, history["loss"], color=color, marker='o', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx() 
    color = 'tab:blue'
    ax2.set_ylabel('Average Precision (AP)', color=color)
    ax2.plot(epochs, history["ap"], color=color, marker='s', label='Validation AP')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.1)

    plt.title('Training Progress: Loss vs. Average Precision')
    fig.tight_layout()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('./Plots/training_history_person_only.png')
    plt.show()


def plot_ablation_comparison(results):
    plt.figure(figsize=(12, 7))
    
    for name, df in results.items():
        plt.plot(df['epoch'], df['ap'], label=f"{name} (Best: {df['ap'].max():.4f})", linewidth=2)
    
    plt.axhline(y=0.67, color='red', linestyle='--', label='Baseline Multi-Class (0.67)')
    plt.title('Ablation Study: Tuning Depth vs. Average Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Average Precision (AP)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("./Plots/ablation_study_comparison.png")
    plt.show()



def plot_pareto_frontier(pareto_df, baseline_ap=0.67):
    plt.figure(figsize=(12, 7))
    
    plt.axhline(y=baseline_ap, color='gray', linestyle='--', linewidth=1.5, label=f'Baseline AP ({baseline_ap:.2f})')
    
    for i, row in pareto_df.iterrows():
        color = 'forestgreen' if row['Peak AP'] >= baseline_ap else 'crimson'
        
        scatter = plt.scatter(row['95% Conv. Epoch'], row['Peak AP'], 
                            s=1500 * (1 - row['Stability (std)'] * 10), 
                            color=color,
                            label=row['Scenario'], 
                            alpha=0.6,
                            edgecolors='black')
        
        plt.text(row['95% Conv. Epoch'] + 0.1, row['Peak AP'] + 0.002, 
                 f"{row['Scenario']}\n(AP: {row['Peak AP']:.3f})", 
                 fontsize=9, fontweight='bold', va='bottom')

    plt.xlabel('Convergence Speed (Epochs within 95% of Peak AP)', fontsize=11)
    plt.ylabel('maximum AP ($AP$)', fontsize=11)
    plt.title('Pareto-Analysis: Performance vs. Stability vs. Speed', fontsize=14, pad=20)
    
    plt.annotate('Sweet Spot: Fast & Precise', xy=(pareto_df['95% Conv. Epoch'].min(), pareto_df['Peak AP'].max()),
                 xytext=(pareto_df['95% Conv. Epoch'].min() + 1, pareto_df['Peak AP'].max() + 0.01),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.savefig('./Plots/pareto_frontier_analysis.png', dpi=300)
    plt.show()


def plot_pruning_tradeoff(pruning_results, save_path='./Plots/pruning_tradeoff.png'):
    df = pd.DataFrame(pruning_results)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(df['inf_time'], df['ap'], marker='o', linestyle='-', linewidth=2, markersize=8, color='royalblue', label='Pruning Kurve')
    
    for i, row in df.iterrows():
        plt.annotate(f"{row['ratio']*100:.0f}%", 
                     (row['inf_time'], row['ap']),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center', fontsize=9, fontweight='bold')

    plt.xlabel('Inference time (ms/Image)', fontsize=12)
    plt.ylabel('mAP (Average Precision)', fontsize=12)
    plt.title('Pruning Trade-off: Performance vs. Speed', fontsize=14)
    
    plt.scatter(df.iloc[0]['inf_time'], df.iloc[0]['ap'], color='red', s=150, edgecolors='black', label='Basis Modell (0%)', zorder=5)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()




def plot_complexity_vs_ap(results, save_path='./Plots/flops_vs_ap.png'):
    df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['mflops'], df['ap'], marker='s', color='forestgreen', label='Model Efficiency')
    
    plt.gca().invert_xaxis() 
    plt.xlabel('Komplexität (MFLOPs) - [Weniger ist effizienter]')
    plt.ylabel('Average Precision (AP)')
    plt.title('Genauigkeit im Verhältnis zum Rechenaufwand')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)


def plot_quantization_results(fp32_stats, int8_stats, path='./Plots/quantization_comparison.png'):
    labels = ['Average Precision', 'Inference Time (ms)', 'Model Size (MB)']
    fp32_vals = [fp32_stats['ap'], fp32_stats['time'], fp32_stats['size']]
    int8_vals = [int8_stats['ap'], int8_stats['time'], int8_stats['size']]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, fp32_vals, width, label='FP32 (Pruned)', color='skyblue')
    rects2 = ax.bar(x + width/2, int8_vals, width, label='INT8 (Quantized)', color='salmon')

    ax.set_ylabel('Scores')
    ax.set_title('Quantization Impact: FP32 vs. INT8')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig(path)
    print(f"Quantization plot saved to {path}")