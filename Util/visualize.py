import matplotlib.pyplot as plt
import numpy as np

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