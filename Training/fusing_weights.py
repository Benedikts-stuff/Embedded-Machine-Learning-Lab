import torch

def fuse_weights(net_with_batch_norm):
    fused_sd = {}
    orig_net = net_with_batch_norm.eval()


    def get_fused_params(conv, bn):
        w = conv.weight
        mean = bn.running_mean
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight
        beta = bn.bias
        w_f = w * (gamma / var_sqrt).reshape([-1, 1, 1, 1])
        b_f = beta - (gamma * mean / var_sqrt)
        return w_f, b_f
    
    for i in range(1, 9):
        w_f, b_f = get_fused_params(getattr(orig_net, f'conv{i}'), getattr(orig_net, f'bn{i}'))
        fused_sd[f'conv{i}.weight'] = w_f
        fused_sd[f'conv{i}.bias'] = b_f

    fused_sd['conv9.weight'] = orig_net.conv9.weight
    fused_sd['conv9.bias'] = orig_net.conv9.bias
    fused_sd["anchors"] = orig_net.anchors
    return fused_sd