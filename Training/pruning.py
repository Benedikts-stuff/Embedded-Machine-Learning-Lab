import torch
import torch.nn as nn
import numpy as np
import copy
import tqdm
from Util.evaluate import evaluate_person_accuracy

def l1_structured_pruning_yolo(state_dict, prune_ratio):
    state_dict = copy.deepcopy(state_dict)
    for i in range(1, 9):
        w = state_dict[f"conv{i}.weight"]
        out_channels = w.size(0)
        k = int(prune_ratio * out_channels)
        if k == 0: continue
        
        l1_norm = torch.sum(torch.abs(w), dim=(1, 2, 3))
        _, idx = torch.topk(l1_norm, k, largest=False)

        w[idx] = 0
        state_dict[f"conv{i}.weight"] = w
        state_dict[f"conv{i}.bias"][idx] = 0
    return state_dict


def densify_yolo_state_dict(state_dict):
    state_dict = copy.deepcopy(state_dict)
    new_channels = []
    keep_indices = {}

    for i in range(1, 9):
        w = state_dict[f"conv{i}.weight"]
        b = state_dict[f"conv{i}.bias"]
        
        keep_idx = [ch for ch in range(w.size(0)) if w[ch].abs().sum() != 0]
        if not keep_idx: keep_idx = [0]
        
        state_dict[f"conv{i}.weight"] = w[keep_idx]
        state_dict[f"conv{i}.bias"] = b[keep_idx]
        
        new_channels.append(len(keep_idx))
        keep_indices[i] = keep_idx

    for i in range(1, 9):
        next_layer = i + 1
        current_keep_idx = keep_indices[i]
        
        w_next = state_dict[f"conv{next_layer}.weight"]
        state_dict[f"conv{next_layer}.weight"] = w_next[:, current_keep_idx, :, :]

    return state_dict, new_channels


def fine_tune_pruned_model(model, train_loader, device, criterion, epochs=5):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    model.train()
    for _ in tqdm.tqdm(range(epochs)):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, yolo=False)
            loss, _ = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


def fine_tune_smart_pruned(model, train_loader, val_loader, device, criterion, max_epochs=20, patience=4):
    """
    Optimiertes Retraining mit Early Stopping und LR-Scheduling.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_ap_local = -1.0
    best_state_local = None
    epochs_no_improve = 0
    
    for epoch in range(max_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss, _ = criterion(model(inputs, yolo=False), targets)
            loss.backward()
            optimizer.step()
        
        current_ap = evaluate_person_accuracy(model, 1, device, val_loader, num_samples=500)
        scheduler.step(current_ap)
        
        print(f"      [Epoch {epoch+1:02d}] Current AP: {current_ap:.4f} (Best: {max(best_ap_local, current_ap):.4f})")

        if current_ap > best_ap_local + 0.005:
            best_ap_local = current_ap
            best_state_local = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"      >>> Early Stopping nach {epoch+1} Epochen.")
            break
            
    model.load_state_dict(best_state_local)
    final_logging_ap = evaluate_person_accuracy(model, 1, device, val_loader, num_samples=2000)
    return final_logging_ap