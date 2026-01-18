import torch.optim.lr_scheduler as lr_scheduler
from Util.dataloader import VOCDataLoaderPerson
from Util.evaluate import evaluate_person_accuracy
import torch.optim as optim
import tqdm
import copy


def get_person_only_sd(num_epochs, eval_samples, model, device, criterion):
    best_ap = -1.0
    best_model_sd = None
    history = {"loss": [], "ap": []}

    train_loader = VOCDataLoaderPerson(train=True, batch_size=64, shuffle=True)

    for name, param in model.named_parameters():
        if 'conv9' not in name:
            param.requires_grad = False

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=1e-3)


    print("Start Person-only Fine-Tuning...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        if epoch == 15:
            print("--- Switching to Global Fine-Tuning (Unfreezing all layers) ---")
            for name, param in model.named_parameters():
                param.requires_grad = True
                #optimizer.param_groups[0]['lr'] = 0.000001
            
            backbone_params = [
                model.conv1.parameters(), model.conv2.parameters(), 
                model.conv3.parameters(), model.conv4.parameters(),
                model.conv5.parameters(), model.conv6.parameters(), 
                model.conv7.parameters(), model.conv8.parameters()
            ]

            backbone_params_list = []
            for p_gen in backbone_params:
                backbone_params_list.extend(list(p_gen))

            optimizer = optim.Adam([
                {'params': backbone_params_list, 'lr': 1e-6}, # Backbone: Sehr vorsichtig
                {'params': model.conv9.parameters(), 'lr': 1e-4} # Kopf: Darf mehr lernen
            ])

            #for param_group in optimizer.param_groups:
             #   param_group['lr'] = 0.000001
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        for idx, (inputs, targets) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs, yolo=False)
            loss, _ = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)

        current_ap = evaluate_person_accuracy(
            model, 
            num_classes_model=1,
            device=device, 
            num_samples=eval_samples
        )

        history["loss"].append(avg_loss)
        history["ap"].append(current_ap)

        print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f} and Avg precision: {current_ap:.4f}")
        
        if current_ap > best_ap:
            best_ap = current_ap
            best_model_sd = copy.deepcopy(model.state_dict())
            print(f"New best AP: {best_ap:.4f}")
        else:
            print(f"AP: {current_ap:.4f} (Best so far: {best_ap:.4f})")

        scheduler.step(avg_loss)

        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"--- Epoch {epoch+1} finished | Avg Loss: {avg_loss:.4f} | Current LR: {current_lr:.6f} ---")
    
    return best_model_sd, history
