import torch
import gc
from Training.fusing_weights import fuse_weights
from Training.person_only_training import get_person_only_sd
from Networks.tinyyolov2_default import TinyYoloV2
from Networks.tinyyolov2_fused_weights import TinyYoloV2Fused
from Util.evaluate import run_comparison_benchmark,run_comparison_benchmark_person, evaluate_model_accuracy, evaluate_person_accuracy
from Util.visualize import plot_inference_time_benchmark_results, plot_ap_benchmark_results, plot_person_only_training_history
from Util.loss import YoloLoss


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

def get_person_only_tuned_sd(num_epochs, eval_samples,model, device, criterion):
    person_only_sd, history = get_person_only_sd(num_epochs=num_epochs, eval_samples=eval_samples,model=model, device=device,
                                        criterion=criterion)
    torch.save(person_only_sd, "./StateDicts/voc_fused_person_only.pt")

    return history




if __name__ == "__main__":
    device = get_device()

    default_net = TinyYoloV2(num_classes=20).to(device)
    default_net.load_state_dict(torch.load("../voc_pretrained.pt", map_location=device, weights_only=False))

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
    fused_net_single_class = TinyYoloV2Fused(num_classes=1).to(device)
    fused_sd = torch.load("./StateDicts/voc_fused.pt")
    if 'conv9.weight' in fused_sd:
        print("Remove conv9 from state dict bacause I dont want to generate a new fused state dict for the 1 class Tinyyolo")
        del fused_sd['conv9.weight']
        del fused_sd['conv9.bias']
    fused_net_single_class.load_state_dict(fused_sd, strict=False)

    criterion = YoloLoss(anchors=fused_net_single_class.anchors)

    history = get_person_only_tuned_sd(num_epochs=40, eval_samples=250, model=fused_net_single_class, device=device, criterion=criterion)
    plot_person_only_training_history(history=history)

    cleanup(fused_net_single_class)
    del fused_net_single_class

    fused_net_person_only = TinyYoloV2Fused(num_classes=1).to(device)
    fused_net_person_only.load_state_dict(torch.load("./StateDicts/voc_fused_person_only.pt"))

    models_to_test = {
        "Default (20c)": default_net,
        "Fused (20c)": fused_net_multi_class,
        "Fused Person Only Net (1c)": fused_net_person_only
    }

    inference_stats_person_only = run_comparison_benchmark(models_to_test, device, num_samples=2000)
    plot_inference_time_benchmark_results(inference_stats_person_only, title="Total inference time for 2000 datapoints on Persons Only (RTX 2070 Super)", path='./Plots/inference_comparison_on_person_only.png')

    ap_baseline_person = evaluate_person_accuracy(default_net,  num_classes_model=20, device=device, num_samples=2000)
    ap_finetuned_person = evaluate_person_accuracy(fused_net_person_only,  num_classes_model=1, device=device, num_samples=2000)

    plot_ap_benchmark_results({"Default Model AP": ap_baseline_person, "Person Only Model AP" :ap_finetuned_person}, "Average Prcision on persons only: Fine Tuned vs Default", path='./Plots/ap_comparison_bars.png')


    #--------------Prune the Person Only fine tunes net-----------------------------------------------------------------------------










    #--------------CLEANUP-----------------------------------------------------------------------------------------------------------
    cleanup(fused_net_multi_class)
    del fused_net_multi_class

    cleanup(default_net)
    del default_net


