import torch
import gc
from Training.fusing_weights import fuse_weights
from Training.person_only_training import get_person_only_sd_experimental
from Networks.tinyyolov2_default import TinyYoloV2
from Networks.tinyyolov2_fused_weights import TinyYoloV2Fused
from Util.evaluate import run_comparison_benchmark, run_pareto_analysis, run_comparison_benchmark_person, evaluate_model_accuracy, evaluate_person_accuracy
from Util.visualize import plot_inference_time_benchmark_results, plot_pareto_frontier, plot_ap_benchmark_results, plot_person_only_training_history, plot_ablation_comparison
from Util.loss import YoloLoss
import pandas as pd


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
    eval_samples = 250

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
                num_epochs=30, 
                eval_samples=eval_samples, 
                model=model, 
                device=device, 
                criterion=criterion,
                frozen_backbone_layers=config["layers_to_freeze_forever"],
                lr_mode=lr_mode,
                lr_threshold=lr_threshold,
                use_ap=use_ap,
                start_epoch=15,
                unfreeze_epoch=0

            )

            full_df = pd.concat([warmup_df, df_history], ignore_index=True)
            results[name] = full_df
            torch.save(best_sd, f"./StateDicts/ablation_{name.replace(' ', '_')}.pt")

            torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"!!! ERROR in Scenario {name}: {e} !!!")
            print("Skipping to next scenario...")
            continue

    baseline_val = 0.67
    pareto_results = run_pareto_analysis(results, baseline_ap=baseline_val)
    plot_pareto_frontier(pareto_results, baseline_ap=baseline_val)    

    return results


if __name__ == "__main__":
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

    ap_baseline = evaluate_person_accuracy(default_net, num_classes_model=20, device=device, num_samples=2000)
    ap_finetuned = evaluate_person_accuracy(fused_net_person_only, num_classes_model=1, device=device, num_samples=2000)

    plot_ap_benchmark_results(
        {"Default (20c)": ap_baseline, "Best Fine-Tuned (1c)": ap_finetuned}, 
        "Final AP Comparison: Default vs. Optimized Head", 
        path='./Plots/ap_comparison_bars_final.png'
    )


    #--------------Prune the Person Only fine tunes net-----------------------------------------------------------------------------










    #--------------CLEANUP-----------------------------------------------------------------------------------------------------------
    cleanup(fused_net_multi_class)
    del fused_net_multi_class

    cleanup(default_net)
    del default_net


