import torch
torch.backends.quantized.engine = 'fbgemm'
import gc
import os
import copy
import random
import numpy as np
from Training.fusing_weights import fuse_weights
from Training.person_only_training import get_person_only_sd_experimental
from Training.quantization import tensor_scale_asym,tensor_scale, calibrate_model
from Networks.tinyyolov2_default import TinyYoloV2
from Networks.tinyyolov2_fused_weights import TinyYoloV2Fused
from Networks.tinyyolov2_pruned_person_only import TinyYoloV2FusedDynamic
from Networks.tinyyolov2_quantized_fused import QTinyYoloV2
from Util.evaluate import run_comparison_benchmark, run_pareto_analysis,get_model_complexity, run_comparison_benchmark_person, evaluate_model_accuracy, evaluate_person_accuracy
from Util.visualize import plot_quantization_results,plot_inference_time_benchmark_results, plot_pareto_frontier, plot_ap_benchmark_results, plot_person_only_training_history, plot_ablation_comparison, plot_pruning_tradeoff
from Util.loss import YoloLoss
from Util.dataloader import VOCDataLoader, VOCDataLoaderPerson
import pandas as pd
import tqdm
from Training.pruning import l1_structured_pruning_yolo, densify_yolo_state_dict, fine_tune_smart_pruned


def cleanup(*args):
    for model in args:
        if model is not None:
            model.to('cpu')

    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    print("VRAM cleared")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed {seed} set")

def get_device():
    device = ''
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Use CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Use MPS")
    else:
        device = torch.device("cpu")
        print("Use CPU")
    return device



def get_fused_sd(default_net):
    fused_sd = fuse_weights(default_net)
    torch.save(fused_sd, "./StateDicts/voc_fused.pt")
    return fused_sd


def run_ablation_study(fused_sd, device, criterion, lr_mode, lr_threshold, use_ap):
    results = {}
    num_epochs = 15
    eval_samples = 1000

    print(f">>> Phase 1: Warming up the Head (first {num_epochs} epochs) on persons only <<<")
    warmup_model = TinyYoloV2Fused(num_classes=1).to(device)
    warmup_model.load_state_dict(fused_sd, strict=False)

    best_warmup_sd, warmup_df = get_person_only_sd_experimental(
        num_epochs=num_epochs, eval_samples=eval_samples, model=warmup_model, device=device, 
        criterion=criterion, unfreeze_epoch=99, use_ap=True, start_epoch=0
    )
    warmup_sd_path = f"./StateDicts/warmup_checkpoint_person_only_{num_epochs}.pt"
    torch.save(best_warmup_sd, warmup_sd_path)
    
    scenarios = {
        "Only Head": {"layers_to_freeze_forever": [f'conv{i}' for i in range(1, 9)]},
        "Unfreeze (8)": {"layers_to_freeze_forever": [f'conv{i}' for i in range(1, 8)]},
        "Unfreeze (7-8)": {"layers_to_freeze_forever": [f'conv{i}' for i in range(1, 7)]},
        "Unfreeze (6-7-8)": {"layers_to_freeze_forever": [f'conv{i}' for i in range(1, 6)]},
        "Unfreeze (5-6-7-8)": {"layers_to_freeze_forever": [f'conv{i}' for i in range(1, 5)]},
        "Unfreeze (4-5-6-7-8)": {"layers_to_freeze_forever": [f'conv{i}' for i in range(1, 4)]},
        "Unfreeze (3-4-5-6-7-8)": {"layers_to_freeze_forever": [f'conv{i}' for i in range(1, 3)]},
        "Unfreeze (2-3-4-5-6-7-8)": {"layers_to_freeze_forever": [f'conv{i}' for i in range(1, 2)]},
        "Full Unfreeze (1-8)": {"layers_to_freeze_forever": []}
    }

    for name, config in scenarios.items():
        try:
            print(f"\n\n>>> RUNNING SCENARIO: {name} <<<")
            model = TinyYoloV2Fused(num_classes=1).to(device)
            model.load_state_dict(best_warmup_sd)
            
            best_sd, df_history = get_person_only_sd_experimental(
                num_epochs=30, #30
                eval_samples=eval_samples, 
                model=model, 
                device=device, 
                criterion=criterion,
                frozen_backbone_layers=config["layers_to_freeze_forever"],
                lr_mode=lr_mode,
                lr_threshold=lr_threshold,
                use_ap=use_ap,
                start_epoch=15,#15
                unfreeze_epoch=0

            )

            plot_person_only_training_history(history=df_history)
            full_df = pd.concat([warmup_df, df_history], ignore_index=True)
            results[name] = full_df
            torch.save(best_sd, f"./StateDicts/ablation_{name.replace(' ', '_')}.pt")

            torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"!!! ERROR in Scenario {name}: {e} !!!")
            print("Skipping to next scenario...")
            continue

    baseline_val = 0.6
    pareto_results = run_pareto_analysis(results, baseline_ap=baseline_val)
    plot_pareto_frontier(pareto_results, baseline_ap=baseline_val)    

    return results


def run_pruning_study(save_path_plot, device, criterion, loader_new, train_loader):
    targets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = []
    
    best_sd_path = "./StateDicts/voc_fused_person_only_best_pruned.pt"
    best_overall_ap = 0
    best_overall_sd = None
    best_overall_cfg = None

    print("\n--- Starting Iterative Smart Pruning Study ---")
    current_sd = torch.load("./StateDicts/voc_fused_person_only.pt")
    current_cfg = [16, 32, 64, 128, 256, 512, 1024, 1024]
    current_remaining_pct = 1.0 

    base_model = TinyYoloV2FusedDynamic(num_classes=1, channels=current_cfg).to(device)
    base_model.load_state_dict(current_sd)
    base_flops, base_macs = get_model_complexity(base_model)
    
    print("Evaluating Baseline (0% Pruning, 2000 samples)...")
    ap_base = evaluate_person_accuracy(base_model, 1, device, loader_new, num_samples=1000)
    inf_time_base = np.mean(run_comparison_benchmark_person({"base": base_model}, device, num_samples=1000)["base"])
    
    results.append({
        "ratio": 0.0, "channels": current_cfg, "inf_time": inf_time_base,
        "ap": ap_base, "mflops": base_flops, "reduction_pct": 0.0, "macs": base_macs
    })

    for target_pruned_pct in targets:
        target_remaining_pct = 1.0 - target_pruned_pct
        
        step_ratio = 1.0 - (target_remaining_pct / current_remaining_pct)
        
        print(f"\n>>> ITERATION: Target {target_pruned_pct*100:.0f}% Total Pruning")
        print(f"    Step Pruning Rate: {step_ratio*100:.1f}%")

        masked_sd = l1_structured_pruning_yolo(current_sd, step_ratio)
        dense_sd, new_cfg = densify_yolo_state_dict(masked_sd)
        
        pruned_model = TinyYoloV2FusedDynamic(num_classes=1, channels=new_cfg).to(device)
        pruned_model.load_state_dict(dense_sd)
        
        print(f"    Fine-tuning with Early Stopping (Max 20 epochs)...")
        current_ap = fine_tune_smart_pruned(pruned_model, train_loader, loader_new, device, criterion)
        
        stats = run_comparison_benchmark_person({"pruned": pruned_model}, device, num_samples=1000)
        inf_time = np.mean(stats["pruned"])
        current_flops, current_macs = get_model_complexity(pruned_model)
        reduction = (1 - (current_flops / base_flops)) * 100

        results.append({
            "ratio": target_pruned_pct,
            "channels": new_cfg,
            "inf_time": inf_time,
            "ap": current_ap,
            "mflops": current_flops,
            "reduction_pct": reduction,
            "macs": current_macs
        })

        if current_ap > best_overall_ap:
            best_overall_ap = current_ap
            best_overall_sd = copy.deepcopy(pruned_model.state_dict())
            best_overall_cfg = new_cfg
            print(f"    *** New Overall Best Model found at {target_pruned_pct*100:.0f}% Pruning! ***")
        
        current_sd = pruned_model.state_dict()
        current_remaining_pct = target_remaining_pct
        
        torch.save({'sd': current_sd, 'cfg': new_cfg}, f"./StateDicts/pruned_checkpoint_{int(target_pruned_pct*100)}pct.pt")

    torch.save(best_overall_sd, best_sd_path)
    torch.save(best_overall_cfg, "./StateDicts/best_pruned_cfg.pt")
    
    print(f"\nPruning Study Finished. Best AP: {best_overall_ap:.4f}")
    plot_pruning_tradeoff(results, save_path=save_path_plot)
    
    return pd.DataFrame(results)


def run_quantization_pipeline(float_model, device, loader_new):
    print("\n---Static Quantization (INT8) ---")
    
    torch.save(float_model.state_dict(), "temp_fp32.pt")
    size_fp32 = os.path.getsize("temp_fp32.pt") / (1024 * 1024)
    
    calib_dict = calibrate_model(float_model, loader_new, device, num_batches=12)
    q_model = transfer_to_quantized_model(float_model, calib_dict)
    q_model.to('cpu')
    
    stats_q = run_comparison_benchmark_person({"Quantized (INT8)": q_model}, 'cpu', num_samples=2000)
    inf_time_q = np.mean(stats_q["Quantized (INT8)"])
    ap_q = evaluate_person_accuracy(q_model, 1, 'cpu', loader_new, num_samples=2000)
    
    torch.save(q_model.state_dict(), "temp_int8.pt")
    size_int8 = os.path.getsize("temp_int8.pt") / (1024 * 1024)
    
    os.remove("temp_fp32.pt")
    os.remove("temp_int8.pt")
    
    return q_model, ap_q, inf_time_q, size_int8, size_fp32


def transfer_to_quantized_model(float_model, calib_dict):
    q_model = QTinyYoloV2(channels=float_model.channels)
    
    q_model.in_scale = torch.tensor(calib_dict["input_scale"])
    q_model.in_zp = torch.tensor(calib_dict["input_zp"])
    
    for i in range(1, 10):
        float_layer = getattr(float_model, f"conv{i}")
        q_layer = getattr(q_model, f"qconv{i}")
        weight = float_layer.weight.data
        
        num_out_channels = weight.shape[0]
        scales = torch.zeros(num_out_channels)
        for c in range(num_out_channels):
            max_val = weight[c].abs().max().item()
            scales[c] = max_val / 127.0 if max_val > 0 else 1.0
        
        zero_points = torch.zeros(num_out_channels, dtype=torch.int32)
        q_weight = torch.quantize_per_channel(weight, scales, zero_points, 0, torch.qint8)
        
        q_layer.weight = torch.nn.Parameter(q_weight, requires_grad=False)
        q_layer.bias = torch.nn.Parameter(float_layer.bias.data, requires_grad=False)
        
        q_layer.conv_scale = torch.tensor(calib_dict[f"conv{i}_raw_scale"])
        q_layer.conv_zp = torch.tensor(calib_dict[f"conv{i}_raw_zp"])

        if i < 9:
            q_layer.relu_scale = torch.tensor(calib_dict[f"conv{i}_relu_scale"])
            q_layer.relu_zp = torch.tensor(calib_dict[f"conv{i}_relu_zp"])
        
    q_model.repack_all()
    
    return q_model


if __name__ == "__main__":
    set_seed()
    device = get_device()

    default_net = TinyYoloV2(num_classes=20).to(device)
    default_net.load_state_dict(torch.load("./StateDicts/voc_pretrained.pt", map_location=device, weights_only=False))

    fused_net_multi_class = TinyYoloV2Fused(num_classes=20).to(device)
    fused_net_multi_class.load_state_dict(get_fused_sd(default_net))

    models_to_test = {
        "Default (20c)": default_net,
        "Fused (20c)": fused_net_multi_class,
    }

    #--------------Plot inference time of default net vs inference time of fused net------------------------------------------------
    inference_stats = run_comparison_benchmark(models_to_test, device, num_samples=2000)
    plot_inference_time_benchmark_results(inference_stats, title="Total imference time for 2000 datapoints (RTX 2070 Super)", path='./Plots/inference_comparison_bars.png' )

    #--------------Fine tune for person only detection------------------------------------------------------------------------------
    fused_sd = torch.load("./StateDicts/voc_fused.pt")
    if 'conv9.weight' in fused_sd:
        print("Cleaning conv9 from state dict for transfer learning...")
        del fused_sd['conv9.weight']
        del fused_sd['conv9.bias']

    criterion = YoloLoss(anchors=TinyYoloV2Fused(num_classes=1).anchors) 

    study_results = run_ablation_study(fused_sd, device, criterion, lr_mode='max', lr_threshold=1e-4, use_ap=True)

    for name, df in study_results.items():
        csv_path = f"./Logs/history_{name.replace(' ', '_').lower()}.csv"
        df.to_csv(csv_path, index=False)
        print(f"History for {name} saved to {csv_path}")

    plot_ablation_comparison(study_results)

    best_overall_ap = -1.0
    best_scenario_name = ""
    for name, df in study_results.items():
        if df['ap'].max() > best_overall_ap:
            best_overall_ap = df['ap'].max()
            best_scenario_name = name

    print(f"\nWINNER: Scenario '{best_scenario_name}' with AP: {best_overall_ap:.4f}")
    
    # Winner of the comparison saved
    best_sd_path = f"./StateDicts/ablation_{best_scenario_name.replace(' ', '_')}.pt"
    final_person_only_sd = torch.load(best_sd_path)
    torch.save(final_person_only_sd, "./StateDicts/voc_fused_person_only.pt")

    # benchmark and Validation of person only finetuning
    fused_net_person_only = TinyYoloV2Fused(num_classes=1).to(device)
    fused_net_person_only.load_state_dict(torch.load("./StateDicts/voc_fused_person_only.pt"))

    models_to_test = {
        "Default (20c)": default_net,
        "Fused (20c)": fused_net_multi_class,
        "Best Person Only (1c)": fused_net_person_only
    }

    inference_stats = run_comparison_benchmark(models_to_test, device, num_samples=2000)
    plot_inference_time_benchmark_results(
        inference_stats, 
        title=f"Inference Time (2000 samples) - Winner: {best_scenario_name}", 
        path='./Plots/inference_comparison_on_person_only.png'
    )

    loader_baseline = VOCDataLoaderPerson(train=False, batch_size=1, is_baseline=True)

    loader_new = VOCDataLoaderPerson(train=False,batch_size=1, is_baseline=False)

    ap_baseline = evaluate_person_accuracy(default_net, num_classes_model=20, device=device, test_loader=loader_baseline, num_samples=2000)
    ap_finetuned = evaluate_person_accuracy(fused_net_person_only, num_classes_model=1, device=device, test_loader=loader_new ,num_samples=2000)

    plot_ap_benchmark_results(
        {"Default (20c)": ap_baseline, "Best Fine-Tuned (1c)": ap_finetuned}, 
        "Final AP Comparison: Default vs. Optimized Head", 
        path='./Plots/ap_comparison_bars_final.png'
    )   


    #--------------Prune the Person Only fine tunes net-----------------------------------------------------------------------------
    train_loader = VOCDataLoaderPerson(train=True, batch_size=32, shuffle=True, is_baseline=False)
    pruning_results = run_pruning_study('./Plots/pruning_comparison.png', device, criterion, loader_new, train_loader) # resuls is DataFrame
    csv_path = f"./Logs/history_pruning.csv"
    pruning_results.to_csv(csv_path, index=False)
    print(f"History for Pruning saved to {csv_path}")

    best_row = pruning_results.loc[pruning_results['ap'].idxmax()]
    best_cfg = best_row['channels']
    ap_finetuned = best_row['ap']
    inf_time_finetuned = best_row['inf_time']

    #--------------Quantization-----------------------------------------------------------------------------------------------------
    device_quantized = 'cpu'
    percent_steps = [10, 20, 30, 40, 50, 60, 70, 80]
    sweep_data = []

    for pct in percent_steps:
        print(f"\n" + "="*60)
        print(f"PROCESSING: {pct}% Pruned Model")
        print("="*60)
        
        checkpoint_path = f"./StateDicts/pruned_checkpoint_{pct}pct.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device_quantized)

        best_cfg = checkpoint['cfg']
        state_dict = checkpoint['sd']
        
        model_fp32 = TinyYoloV2FusedDynamic(num_classes=1, channels=best_cfg).to(device_quantized)
        model_fp32.load_state_dict(state_dict)
        
        print(f"Evaluating FP32 {pct}%...")
        ap_fp32 = evaluate_person_accuracy(model_fp32, 1, device_quantized, loader_new, num_samples=2000)
        stats_f = run_comparison_benchmark_person({"FP32": model_fp32}, device_quantized, num_samples=2000)
        time_fp32 = np.mean(stats_f["FP32"])
        
        q_model, ap_int8, time_int8, size_int8, size_fp32 = run_quantization_pipeline(model_fp32, device_quantized, loader_new)

        sweep_data.append({
            "pruning_pct": pct,
            "ap_fp32": ap_fp32,
            "time_fp32_ms": time_fp32,
            "fps_fp32": 1000.0 / time_fp32,
            "size_fp32_mb": size_fp32,
            "ap_int8": ap_int8,
            "time_int8_ms": time_int8,
            "fps_int8": 1000.0 / time_int8,
            "size_int8_mb": size_int8,
            "speedup_factor": time_fp32 / time_int8
        })
        
        fp32_stats = {'ap': ap_fp32, 'time': time_fp32, 'size': size_fp32}
        int8_stats = {'ap': ap_int8, 'time': time_int8, 'size': size_int8}
        plot_quantization_results(fp32_stats, int8_stats, path=f'./Plots/quantization_comparison_{pct}pct.png')
        
        torch.save(q_model.state_dict(), f"./StateDicts/voc_person_only_int8_{pct}pct.pt")

    df_sweep = pd.DataFrame(sweep_data)
    df_sweep.to_csv("./Logs/final_optimization_sweep_results.csv", index=False)
    
    print("\n" + "#"*60)
    print("FINAL SWEEP RESULTS")
    print("#"*60)
    print(df_sweep[['pruning_pct', 'ap_fp32', 'ap_int8', 'fps_fp32', 'fps_int8', 'speedup_factor']])


    #--------------CLEANUP-----------------------------------------------------------------------------------------------------------
    cleanup(fused_net_multi_class)
    del fused_net_multi_class

    cleanup(default_net)
    del default_net


