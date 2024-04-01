import torch


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et, _ = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

    return xs, x0_preds

# def generalized_steps(x, seq, model, b, **kwargs):
#     n = x.size(0)
#     seq_next = [-1] + list(seq[:-1])
    
#     for i, j in zip(reversed(seq), reversed(seq_next)):
#         t = torch.tensor([i]).to(x.device)
#         next_t = torch.tensor([j]).to(x.device)
#         at = compute_alpha(b, t.long()).squeeze()
#         at_next = compute_alpha(b, next_t.long()).squeeze()
#         xt = x
#         et = et = model(xt.unsqueeze(0), t).squeeze()

#         x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

#         c2 = ((1 - at_next)).sqrt()
#         xt_next = at_next.sqrt() * x0_t + c2 * et
#         x = xt_next
#     return x

def reverse_generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_prev = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(seq, seq_prev):
            t = (torch.ones(n) * i).to(x.device)
            prev_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_prev = compute_alpha(b, prev_t.long())
            xt_prev = xs[-1].to('cuda')
            et = model(xt_prev, prev_t)
            
            #x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            #x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_prev) * (1 - at_prev) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_prev) - c1 ** 2).sqrt()
            
            xt = (at/at_prev).sqrt() * (xt_prev - c1 * torch.randn_like(x) - c2 * et) + (1 - at).sqrt() * et
            xs.append(xt.to('cpu'))

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
