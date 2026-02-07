import torch
import csv
import time
import numpy as np
import os
import gc  
import onnxruntime as ort
from Util.dataloader import VOCDataLoaderPerson
from Networks.tinyyolov2_pruned_person_only import TinyYoloV2FusedDynamic
from Networks.tinyyolov2_quantized_fused import QTinyYoloV2
from Networks.tinyyolov2_fused_weights import TinyYoloV2Fused
from Networks.tinyyolov2_default import TinyYoloV2

def get_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def benchmark_jetson_nano():
    if torch.backends.quantized.engine != 'qnnpack':
        torch.backends.quantized.engine = 'qnnpack'
    
    cuda_available = torch.cuda.is_available()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pt_dir = os.path.join(base_dir, "StateDicts")
    onnx_dir = os.path.join(base_dir, "StateDicts/ONNX_FINAL_FP32")

    try:
        loader = VOCDataLoaderPerson(train=False, batch_size=1, is_baseline=False)
        img, _ = next(iter(loader))
        img_np = img.numpy()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    results = []
    num_runs = 100 
    warmup = 15 

    test_tasks = []
    
    if os.path.exists(pt_dir):
        for f in os.listdir(pt_dir):
            if f.endswith('.pt'):
                if "int8" in f.lower():
                    test_tasks.append({'folder': pt_dir, 'file': f, 'type': 'pt', 'dev': 'cpu'})
                else:
                    test_tasks.append({'folder': pt_dir, 'file': f, 'type': 'pt', 'dev': 'cpu'})
                    if cuda_available:
                        test_tasks.append({'folder': pt_dir, 'file': f, 'type': 'pt', 'dev': 'cuda'})

    if os.path.exists(onnx_dir):
        for f in os.listdir(onnx_dir):
            if f.endswith('.onnx'):
                test_tasks.append({'folder': onnx_dir, 'file': f, 'type': 'onnx', 'dev': 'cpu'})
                if cuda_available:
                    test_tasks.append({'folder': onnx_dir, 'file': f, 'type': 'onnx', 'dev': 'cuda'})

    for task in test_tasks:
        folder, file_name, m_type, dev_goal = task['folder'], task['file'], task['type'], task['dev']
        path = os.path.join(folder, file_name)
        print(f"\n>>> Running: {file_name} on {dev_goal.upper()} ({m_type.upper()})")
        
        try:
            params = "N/A"
            latencies = []

            if m_type == 'onnx':
                provider = ['CUDAExecutionProvider'] if dev_goal == 'cuda' else ['CPUExecutionProvider']
                session = ort.InferenceSession(path, providers=provider)
                input_name = session.get_inputs()[0].name
                
                for _ in range(warmup): _ = session.run(None, {input_name: img_np})
                for _ in range(num_runs):
                    start = time.perf_counter()
                    _ = session.run(None, {input_name: img_np})
                    latencies.append((time.perf_counter() - start) * 1000)
                del session

            else:
                checkpoint = torch.load(path, map_location='cpu')
                sd = checkpoint['sd'] if isinstance(checkpoint, dict) and 'sd' in checkpoint else checkpoint
                
                if isinstance(checkpoint, dict) and 'cfg' in checkpoint:
                    cfg = checkpoint['cfg']
                else:
                    try:
                        prefix = "qconv" if "qconv1.weight" in sd else "conv"
                        cfg = [sd[f"{prefix}{i}.weight"].shape[0] for i in range(1, 9)]
                    except: cfg = [16, 32, 64, 128, 256, 512, 1024, 1024]

                if "voc_pretrained" in file_name.lower(): model = TinyYoloV2(num_classes=20)
                elif "voc_fused.pt" == file_name.lower(): model = TinyYoloV2Fused(num_classes=20)
                elif "int8" in file_name.lower():
                    model = QTinyYoloV2(channels=cfg)
                    for name, param in sd.items():
                        parts = name.split('.'); target = model
                        for part in parts[:-1]: target = getattr(target, part)
                        setattr(target, parts[-1], torch.nn.Parameter(param, requires_grad=False))
                    model.repack_all()
                else: model = TinyYoloV2FusedDynamic(num_classes=1, channels=cfg)

                if "int8" not in file_name.lower(): model.load_state_dict(sd)

                params = get_model_parameters(model)
                inf_device = torch.device(dev_goal)
                model.to(inf_device).eval()
                test_img = img.to(inf_device)

                with torch.no_grad():
                    for _ in range(warmup): _ = model(test_img)
                    for _ in range(num_runs):
                        if dev_goal == 'cuda': torch.cuda.synchronize()
                        start = time.perf_counter()
                        _ = model(test_img)
                        if dev_goal == 'cuda': torch.cuda.synchronize()
                        latencies.append((time.perf_counter() - start) * 1000)
                del model

            avg_lat = np.mean(latencies)
            jitter = np.std(latencies)
            fps = 1000.0 / avg_lat
            print(f"    FPS: {fps:.2f} | Jitter: {jitter:.4f}ms")

            results.append({
                "Model": file_name,
                "Format": m_type.upper(),
                "Device": dev_goal.upper(),
                "Size_MB": round(get_file_size_mb(path), 2),
                "Params": params,
                "Avg_Latency_ms": round(avg_lat, 2),
                "Jitter_ms": round(jitter, 4),
                "FPS": round(fps, 2)
            })
            gc.collect()
            if cuda_available: torch.cuda.empty_cache()

        except Exception as e:
            print(f"    Error: {e}")

    if results:
        df_path = os.path.join(base_dir, 'final_hardware_benchmark.csv')
        keys = results[0].keys()
        with open(df_path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)
        print(f"\nBenchmark abgeschlossen! Datei: {df_path}")

if __name__ == "__main__":
    benchmark_jetson_nano()