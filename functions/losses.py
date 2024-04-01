import torch
from torchvision import transforms
import torch.nn.functional as F

def reduce(x, size):
    return transforms.Resize([size, size])(x)

def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          t: torch.LongTensor,
                          b: torch.Tensor, bin, keepdim=False):
    size = x0.shape[-1] // (2 ** bin)
    x0_down = reduce(x0, size)
    x0_up = reduce(x0, size*2)
    
    e = torch.randn_like(x0_up)
    
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    e_down = F.avg_pool2d(e, 2) * 2
    x_down = x0_down * a.sqrt() + e_down * (1.0 - a).sqrt()
    
    x_up = x0_up * a.sqrt() + e * (1.0 - a).sqrt()
    output, output_up = model(x_down, t.float())
    if keepdim:
        return (e_down - output).square().sum(dim=(1, 2, 3)) + 0.05 * (x_up - output_up).square().sum(dim=(1, 2, 3))
    else:
        return (e_down - output).square().sum(dim=(1, 2, 3)).mean(dim=0) + 0.05 * (x_up - output_up).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}
