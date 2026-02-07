import torch
import os

# Deine spezifischen Imports
from Networks.tinyyolov2_default import TinyYoloV2
from Networks.tinyyolov2_fused_weights import TinyYoloV2Fused
from Networks.tinyyolov2_pruned_person_only import TinyYoloV2FusedDynamic

def extract_cfg_from_state_dict(sd):
    cfg = []
    for key in sorted(sd.keys()):
        if 'weight' in key and len(sd[key].shape) == 4:
            cfg.append(sd[key].shape[0])
    return cfg

def run_simple_onnx_export():
    device = torch.device("cpu")
    checkpoint_folder = "./StateDicts/"
    output_folder = "./StateDicts/ONNX_FINAL_FP32"
    os.makedirs(output_folder, exist_ok=True)

    # Wir nehmen alle .pt Dateien (außer die int8, da die eh nicht direkt gehen)
    checkpoints = [f for f in os.listdir(checkpoint_folder) 
                   if f.endswith('.pt') and "int8" not in f.lower()]

    dummy_input = torch.randn(1, 3, 320, 320)

    for ckpt in checkpoints:
        print(f"--- Exporting: {ckpt} ---")
        try:
            path = os.path.join(checkpoint_folder, ckpt)
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            sd = checkpoint['sd'] if isinstance(checkpoint, dict) and 'sd' in checkpoint else checkpoint
            
            # Modell-Logik wie gehabt
            if "voc_pretrained" in ckpt:
                model = TinyYoloV2(num_classes=20)
            elif "voc_fused.pt" == ckpt:
                model = TinyYoloV2Fused(num_classes=20)
            else:
                cfg = checkpoint.get('cfg', extract_cfg_from_state_dict(sd)) if isinstance(checkpoint, dict) else extract_cfg_from_state_dict(sd)
                if len(cfg) > 8: cfg = cfg[:8]
                model = TinyYoloV2FusedDynamic(num_classes=1, channels=cfg)

            model.load_state_dict(sd)
            model.eval()

            onnx_path = os.path.join(output_folder, ckpt.replace(".pt", ".onnx"))

            # Der entscheidende Export: Opset 11 + Kein Dynamo
            torch.onnx.export(
                model, 
                dummy_input, 
                onnx_path,
                export_params=True,
                opset_version=11, 
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
            )
            print(f"✅ Erstellt: {onnx_path}")

        except Exception as e:
            print(f"❌ Fehler bei {ckpt}: {e}")

if __name__ == "__main__":
    run_simple_onnx_export()