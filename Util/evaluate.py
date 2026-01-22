import time 
import torch
import os
from torchinfo import summary
from Util.ap import precision_recall_levels, ap
from Util.yolo import nms, filter_boxes
import tqdm
from Util.dataloader import VOCDataLoader
from Util.dataloader import VOCDataLoaderPerson
import numpy as np
import pandas as pd

def run_comparison_benchmark(models, device, num_samples=500):
    loader = VOCDataLoader(train=False, batch_size=1)
    stats = {name: [] for name in models.keys()}
    
    print(f"Start Time Benchmark for {num_samples} Images")
    with torch.no_grad():
        for i, (img, _) in tqdm.tqdm(enumerate(loader)):
            if i >= num_samples: break
            img = img.to(device)
            
            for name, net in models.items():
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = net(img)
                torch.cuda.synchronize()
                stats[name].append((time.perf_counter() - start) * 1000)

    return stats

def run_comparison_benchmark_person(models, device, num_samples=500):
    loader = VOCDataLoaderPerson(train=False, batch_size=1)
    stats = {name: [] for name in models.keys()}
    
    print(f"Start Time Benchmark for {num_samples} Images")
    with torch.no_grad():
        for i, (img, _) in tqdm.tqdm(enumerate(loader)):
            if i >= num_samples: break
            img = img.to(device)
            
            for name, net in models.items():
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = net(img)
                torch.cuda.synchronize()
                stats[name].append((time.perf_counter() - start) * 1000)

    return stats


def evaluate_person_accuracy(model, num_classes_model, device, test_loader, num_samples=500):
    model.eval()
    test_precision = []
    test_recall = []

    PERSON_CLASS_ID = 14 if num_classes_model == 20 else 0 


    print(f"Evaluate {'Original tinyyolo' if num_classes_model==20 else 'Fine-Tuned'} on Person Only testset...")
    with torch.no_grad():
        for idx, (input, target) in tqdm.tqdm(enumerate(test_loader), total=num_samples):
            input, target = input.to(device), target.to(device)
            output = model(input, yolo=True) #
            
            output_boxes = filter_boxes(output, 0.0) 
            
            filtered_boxes = []
            for box in output_boxes[0]:
                if int(box[-1]) == PERSON_CLASS_ID:
                    box_copy = box.clone()
                    box_copy[-1] = 0.0
                    filtered_boxes.append(box_copy)
            
            if len(filtered_boxes) > 0:
                output_boxes = [torch.stack(filtered_boxes)]
            else:
                output_boxes = [torch.zeros((0, 7))]
            
            output_final = nms(output_boxes, 0.5)
            
            precision, recall = precision_recall_levels(target[0], output_final[0])
            test_precision.append(precision)
            test_recall.append(recall)
            
            if idx + 1 == num_samples: break
                
    return ap(test_precision, test_recall)



def evaluate_model_accuracy(model, test_loader, device,  num_samples=500):
    model.eval()
    test_precision = []
    test_recall = []
    
    print(f"Compute Average Precision for {num_samples} Images")
    with torch.no_grad():
        for idx, (input, target) in tqdm.tqdm(enumerate(test_loader), total=num_samples):
            input, target = input.to(device), target.to(device)
            output = model(input, yolo=True)
            output = filter_boxes(output, 0.0)
            output = nms(output, 0.5)
            
            precision, recall = precision_recall_levels(target[0], output[0])
            test_precision.append(precision)
            test_recall.append(recall)
            
            if idx + 1 == num_samples:
                break
    return ap(test_precision, test_recall)


def count_parameters(model, path):
    size_bytes = os.path.getsize(path)
    return size_bytes / (1024 * 1024), sum(p.numel() for p in model.parameters() if p.requires_grad)



def run_pareto_analysis(study_results, baseline_ap):
    summary = []

    for name, df in study_results.items():
        peak_ap = df['ap'].max()
        delta_to_baseline = peak_ap - baseline_ap
        stability = df['ap'].tail(5).std()
        threshold = peak_ap * 0.95
        convergence_epoch = df[df['ap'] >= threshold]['epoch'].iloc[0]
        
        score = peak_ap - (stability * 2) 

        summary.append({
            "Scenario": name,
            "Peak AP": peak_ap,
            "Delta to Baseline": f"{delta_to_baseline:+.4f}",
            "Stability (std)": stability,
            "95% Conv. Epoch": convergence_epoch,
            "Score": score
        })

    pareto_df = pd.DataFrame(summary).sort_values(by="Score", ascending=False)
    
    print(f"\n--- PARETO OPTIMALITY ANALYSIS - (Baseline AP: {baseline_ap:.4f}) ---")
    print(pareto_df.to_string(index=False))
    
    best_scenario = pareto_df.iloc[0]['Scenario']
    print(f"\nRecommendation: Use '{best_scenario}' for Pruning.")
    return pareto_df



def get_model_complexity(model, input_res=320):
    stats = summary(model, input_size=(1, 3, input_res, input_res), verbose=0)
    macs = stats.total_mult_adds
    flops = 2 * macs
    
    return flops / 1e6, macs