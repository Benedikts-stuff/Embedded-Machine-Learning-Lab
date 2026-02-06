import torch
import os
import re
import gc
import random
import numpy as np
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Deine spezifischen Imports
from Networks.tinyyolov2_default import TinyYoloV2
from Networks.tinyyolov2_fused_weights import TinyYoloV2Fused
from Networks.tinyyolov2_pruned_person_only import TinyYoloV2FusedDynamic
from Networks.tinyyolov2_quantized_fused import QTinyYoloV2
from Util.evaluate import evaluate_person_accuracy
from Util.dataloader import VOCDataLoaderPerson

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def extract_cfg_from_state_dict(sd):
    cfg = []
    for key in sorted(sd.keys()):
        if 'weight' in key and len(sd[key].shape) == 4:
            cfg.append(sd[key].shape[0])
    return cfg

def run_full_benchmark():
    # Wir setzen die Engine für PC (x86)
    torch.backends.quantized.engine = 'fbgemm'
    set_seed()
    
    checkpoint_folder = "./StateDicts/"
    output_csv = "Logs/final_evaluation_report.csv"
    os.makedirs("Logs", exist_ok=True)
    
    # DataLoader (PC: is_baseline=True für saubere Validierung ohne Augmentierung)
    val_loader = VOCDataLoaderPerson(batch_size=1, train=False, is_baseline=True) 

    results = []
    checkpoints = [f for f in os.listdir(checkpoint_folder) if f.endswith('.pt')]

    for ckpt in checkpoints:
        print(f"\n--- Testing: {ckpt} ---")
        try:
            ckpt_path = os.path.join(checkpoint_folder, ckpt)
            is_quantized = "int8" in ckpt or "quantized" in ckpt
            current_device = "cpu" if is_quantized else ("cuda" if torch.cuda.is_available() else "cpu")

            checkpoint = torch.load(ckpt_path, map_location=current_device)
            sd = checkpoint['sd'] if 'sd' in checkpoint else checkpoint
            
            if "voc_pretrained.pt" in ckpt:
                num_classes = 20
                model = TinyYoloV2(num_classes=20).to(current_device)
                model.load_state_dict(sd)
            elif "voc_fused.pt" in ckpt:
                num_classes = 20
                model = TinyYoloV2Fused(num_classes=num_classes).to(current_device)
                model.load_state_dict(torch.load("./StateDicts/voc_fused.pt", map_location=current_device, weights_only=False))

            elif is_quantized:
                num_classes = 1
                cfg = checkpoint.get('cfg', extract_cfg_from_state_dict(sd))
                if len(cfg) > 8: cfg = cfg[:8] # Nur die 8 Conv-Layer
                
                model = QTinyYoloV2(channels=cfg, num_classes=1).to(current_device)
                
                for name, loaded_param in sd.items():
                    try:
                        parts = name.split('.')
                        submod = model
                        for part in parts[:-1]:
                            submod = getattr(submod, part)
                        
                        delattr(submod, parts[-1])
                        setattr(submod, parts[-1], loaded_param)
                    except Exception:
                        setattr(model, name, loaded_param)
                
                model.repack_all() 
            else:
                num_classes = 1
                cfg = checkpoint.get('cfg', extract_cfg_from_state_dict(sd))
                if len(cfg) > 8: cfg = cfg[:8]
                
                model = TinyYoloV2FusedDynamic(num_classes=1, channels=cfg).to(current_device)
                model.load_state_dict(sd)

            model.eval()

            # --- ACCURACY TEST ---
            num_test_images = len(val_loader.dataset)
            # Zum schnellen Testen num_samples=100, für finalen Report das ganze Set
            ap_val = evaluate_person_accuracy(
                model=model, 
                num_classes_model=num_classes, 
                device=current_device, 
                test_loader=val_loader, 
                num_samples=num_test_images
            )

            # --- METADATEN ---
            model_type = "Baseline"
            pruning_val = 0
            if "pruned" in ckpt or "pct" in ckpt:
                model_type = "Pruned"
                match = re.search(r'(\d+)pct', ckpt)
                pruning_val = int(match.group(1)) if match else 0
            elif "abalation" in ckpt:
                model_type = "Ablation"
            elif is_quantized:
                model_type = "Quantized"
                # Falls bei int8 auch ein Prozentwert im Namen steht:
                match = re.search(r'(\d+)pct', ckpt)
                pruning_val = int(match.group(1)) if match else 0

            results.append({
                "Filename": ckpt,
                "Type": model_type,
                "Pruning_Pct": pruning_val,
                "AP_Person": ap_val,
                "Is_Quantized": is_quantized
            })
            
            print(f"-> Success: AP = {ap_val:.4f}")
            del model
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"!! Error evaluating {ckpt}: {e}")

    # --- CSV & PLOT ---
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    # --- VISUALISIERUNG ---
    plt.figure(figsize=(14, 8))
    sns.set_context("talk")
    
    # Plot 1: Pruning Progress (nur Float Modelle)
    prune_plot_data = df[(df['Type'] == "Pruned") & (df['Is_Quantized'] == False)].sort_values('Pruning_Pct')
    if not prune_plot_data.empty:
        plt.plot(prune_plot_data['Pruning_Pct'], prune_plot_data['AP_Person'], 
                 marker='o', markersize=10, label='Pruned (Float)', linewidth=3)

    # Plot 2: Quantized Points
    quant_data = df[df['Is_Quantized'] == True]
    if not quant_data.empty:
        plt.scatter(quant_data['Pruning_Pct'], quant_data['AP_Person'], 
                    color='red', s=150, label='INT8 Quantized', zorder=5)

    # Plot 3: Baseline (VOC Pretrained)
    baseline = df[df['Filename'] == "voc_pretrained.pt"]
    if not baseline.empty:
        plt.axhline(y=baseline['AP_Person'].values[0], color='gray', linestyle='--', label='VOC Baseline')

    # Plot 4: Ablation Study
    ablation_data = df[df['Type'] == "Ablation"]
    for i, row in ablation_data.iterrows():
        plt.scatter(0, row['AP_Person'], marker='x', s=100, label=f"Ablation: {row['Filename']}")

    plt.title("Summary: Person Detection Accuracy across Model Pipeline")
    plt.xlabel("Pruning Percentage (%)")
    plt.ylabel("Average Precision (AP) - Person Class")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pipeline_performance_summary.png")
    plt.show()

if __name__ == "__main__":
    run_full_benchmark()
