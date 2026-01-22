import torch.optim.lr_scheduler as lr_scheduler
from Util.dataloader import VOCDataLoaderPerson
from Util.evaluate import evaluate_person_accuracy
import torch.optim as optim
import tqdm
import copy
import pandas as pd
import numpy as np


def get_person_only_sd_experimental(num_epochs, eval_samples, model, device, criterion, 
                                   unfreeze_epoch=15, start_epoch=0,
                                   frozen_backbone_layers=None, lr_mode = 'min', lr_threshold=1e-4, use_ap=False):
    if frozen_backbone_layers is None:
        frozen_backbone_layers = []

    best_ap = -1.0
    best_model_sd = None
    history = {"epoch": [], "loss": [], "ap": [], "lr_backbone": [], "lr_head": []}

    train_loader = VOCDataLoaderPerson(train=True, batch_size=64, shuffle=True)
    val_loader = VOCDataLoaderPerson(train=False, batch_size=1, is_baseline=False, shuffle=True)

    if unfreeze_epoch > 0:
        for name, param in model.named_parameters():
            param.requires_grad = ('conv9' in name)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=lr_mode, factor=0.5, patience=4, threshold=lr_threshold) # modus muss min sein wenn ich den loss benutze statt AP

    for epoch in range(num_epochs):
        model.train()
        global_epoch = start_epoch + epoch + 1
        
        if epoch == unfreeze_epoch:
            print(f"--- Global Epoch {global_epoch}: Unfreezing Stage (Exceptions: {frozen_backbone_layers}) ---")
            
            backbone_params = []
            for name, param in model.named_parameters():
                is_exception = any(layer_name in name for layer_name in frozen_backbone_layers)
                
                if 'conv9' in name:
                    param.requires_grad = True 
                elif is_exception:
                    param.requires_grad = False
                else:
                    param.requires_grad = True 
                
                if param.requires_grad and 'conv9' not in name:
                    backbone_params.append(param)

            optimizer = optim.Adam([
                {'params': backbone_params, 'lr': 1e-6},
                {'params': model.conv9.parameters(), 'lr': 1e-4}
            ])
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode=lr_mode, factor=0.5, patience=3, threshold=lr_threshold) # hier auch wieder zu min wenn loss benutzt

        running_loss = 0.0
        for inputs, targets in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, yolo=False)
            loss, _ = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)

        if np.isnan(avg_loss):
            print("!!! Loss explodedto NaN so we break this scenario")
            break

        current_ap = evaluate_person_accuracy(model, 1, device, val_loader ,eval_samples)

        history["epoch"].append(global_epoch)
        history["loss"].append(avg_loss)
        history["ap"].append(current_ap)
        
        lrs = [group['lr'] for group in optimizer.param_groups]
        history["lr_backbone"].append(lrs[0] if len(lrs) > 1 else lrs[0])
        history["lr_head"].append(lrs[1] if len(lrs) > 1 else lrs[0])

        if current_ap > best_ap:
            best_ap = current_ap
            best_model_sd = copy.deepcopy(model.state_dict())
            print(f"New best AP: {best_ap:.4f}")

        if not use_ap:
            scheduler.step(avg_loss) #scheduler basierend auf loss
        else: 
            scheduler.step(current_ap)
    
    return best_model_sd, pd.DataFrame(history)
