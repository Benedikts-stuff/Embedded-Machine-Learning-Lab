import torch
import csv
import time
import numpy as np
import os
import gc  
from Util.dataloader import VOCDataLoaderPerson
from Networks.tinyyolov2_pruned_person_only import TinyYoloV2FusedDynamic
from Networks.tinyyolov2_quantized_fused import QTinyYoloV2
from Networks.tinyyolov2_fused_weights import TinyYoloV2Fused
from Networks.tinyyolov2_default import TinyYoloV2

def benchmark_jetson_nano():
    if torch.backends.quantized.engine != 'qnnpack':
        torch.backends.quantized.engine = 'qnnpack'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking started on: {device}")

    pretrained_sd = "voc_pretrained.pt"
    model_files = [
        "voc_pretrained.pt",
        "voc_fused.pt",
        "voc_fused_person_only.pt",
        "pruned_checkpoint_10pct.pt",
        "pruned_checkpoint_20pct.pt",
        "pruned_checkpoint_30pct.pt",
        "pruned_checkpoint_40pct.pt",
        "pruned_checkpoint_50pct.pt",
        "pruned_checkpoint_60pct.pt",
        "pruned_checkpoint_70pct.pt",
        "pruned_checkpoint_80pct.pt",
        "voc_person_only_int8_10pct.pt",
        "voc_person_only_int8_20pct.pt",
        "voc_person_only_int8_30pct.pt",
        "voc_person_only_int8_40pct.pt",
        "voc_person_only_int8_50pct.pt",
        "voc_person_only_int8_60pct.pt",
        "voc_person_only_int8_70pct.pt",
        "voc_person_only_int8_80pct.pt",
    ]

    try:
        loader = VOCDataLoaderPerson(train=False, batch_size=1, is_baseline=False)
        img, _ = next(iter(loader))
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    results = []
    num_runs = 100 
    warmup = 15 

    base_dir = os.path.dirname(os.path.abspath(__file__))

    for file_name in model_files:
        path = os.path.join(base_dir, "StateDicts", file_name)
        if not os.path.exists(path):
            print(f"Skipping {file_name} - Not found.")
            continue

        print(f"\nTesting: {file_name}")
        
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            if isinstance(checkpoint, dict) and 'cfg' in checkpoint:
                cfg = checkpoint['cfg']
                sd = checkpoint['sd']
            else:
                sd = checkpoint
                try:
                    prefix = "qconv" if "qconv1.weight" in sd else "conv"

                    cfg = []
                    for i in range(1,9):
                        channels = sd[f"{prefix}{i}.weight"].shape[0]
                        cfg.append(channels)
                    print(f"detected structure: {cfg}")
                except KeyError:
                    cfg = [16,32,64,128,256,512,1024,1024]
            

            if "voc_pretrained.pt" in file_name.lower():
                model = TinyYoloV2(num_classes=20)
                model.load_state_dict(sd)
                inf_device = device
            
            elif "voc_fused.pt" in file_name.lower():
                model = TinyYoloV2Fused(num_classes=20)
                model.load_state_dict(sd)
                inf_device = device

            elif "voc_fused_person_only.pt" in file_name.lower():
                model = TinyYoloV2Fused(num_classes=1)
                model.load_state_dict(sd)
                inf_device = device

            elif "int8" in file_name.lower():
                model = QTinyYoloV2(channels=cfg)
                for name, param in sd.items():
                    parts = name.split('.')
                    target = model 
                    for part in parts[:-1]:
                        target = getattr(target, part)

                    setattr(target, parts[-1], torch.nn.Parameter(param, requires_grad=False))
                model.repack_all()
                inf_device = torch.device("cpu")
            else:
                model = TinyYoloV2FusedDynamic(num_classes=1, channels=cfg)
                model.load_state_dict(sd)
                inf_device = device

            del sd
            del checkpoint
            gc.collect()

            model.to(inf_device)
            model.eval()
            test_img = img.to(inf_device)

            with torch.no_grad():
                for _ in range(warmup):
                    _ = model(test_img)

            latencies = []
            with torch.no_grad():
                for _ in range(num_runs):
                    if inf_device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    start = time.perf_counter()
                    _ = model(test_img)
                    
                    if inf_device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    latencies.append((time.perf_counter() - start) * 1000)

            avg_lat = np.mean(latencies)
            fps = 1000.0 / avg_lat
            
            print(f"Result: {fps:.2f} FPS ({avg_lat:.2f} ms)")

            results.append({
                "Model": file_name,
                "Device": inf_device.type,
                "Latency_ms": avg_lat,
                "FPS": fps,
                "Std_ms": np.std(latencies)
            })

            del model
            del test_img
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Critical error testing {file_name}: {e}")
            continue


    if not results: 
        print("No resultd")
        return
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'jetson_nano_benchmark_results.csv')

    keys = results[0].keys()
    with open(csv_path, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
    
    print("\n All tests finished")


if __name__ == "__main__":
    benchmark_jetson_nano()
