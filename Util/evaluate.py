import time 
import torch
from Util.ap import precision_recall_levels, ap
from Util.yolo import nms, filter_boxes
import tqdm
from Util.dataloader import VOCDataLoader
from Util.dataloader import VOCDataLoaderPerson

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


def evaluate_person_accuracy(model, num_classes_model, device, num_samples=500):
    model.eval()
    test_precision = []
    test_recall = []
    test_loader = VOCDataLoaderPerson(train=False, batch_size=1)

    PERSON_CLASS_ID = 14 if num_classes_model == 20 else 0 


    print(f"Evaluate{'Original tinyyolo' if num_classes_model==20 else 'Fine-Tuned'} on Person Only testset...")
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