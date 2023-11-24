from __future__ import absolute_import, division

import os
import torch
import numpy as np

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a


def generalized_steps(x, input_feat, radar, limb_len_gt, src_mask, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).cuda()
            next_t = (torch.ones(n) * j).cuda()
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1]
            et, limb_len_pred = model(xt, src_mask, t.float(), input_feat, radar, None)
            et = et + 0*limb_cond(xt, et, at, limb_len_pred) # 200 for gt, 50 for pred
            # print(et[0,:,-3:], limb_cond(xt, et, at, limb_len_gt)[0,:,-3:])

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt() 
            x0_preds.append(x0_t)
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next)
    return xs, x0_preds, limb_len_pred


def limb_cond(x, et, at, limb_len):
    with torch.enable_grad():
        # calculate the limb length based on prediction
       
        limb_len = torch.autograd.Variable(limb_len, requires_grad = True)
        x = torch.autograd.Variable(x, requires_grad = True)
        # out_gra = (x - et * (1 - at).sqrt()) / at.sqrt()
        limb_list_pr = []
        limb_pairs = [[0, 1], [1, 2], [2, 3],
                            [0, 4], [4, 5], [5, 6],
                            [0, 7], [7, 8], [8, 9], [9,10],
                            [8, 11], [11, 12], [12, 13],
                            [8, 14], [14, 15], [15, 16]]
        for i in range(len(limb_pairs)):
            limb_src = limb_pairs[i][0]
            limb_end = limb_pairs[i][1]
            limb_vector_pr = x[:, limb_src, -3:] - x[:, limb_end, -3:]
            limb_list_pr.append(limb_vector_pr)

        
        out_limb = torch.stack(limb_list_pr, dim = 1)
        out_limb_len = torch.sqrt(torch.sum(torch.pow(out_limb, 2), dim=-1))

        F_limb = torch.abs((out_limb_len - limb_len)).sum(dim=-1).mean(dim=0)

        out_delta = torch.autograd.grad(F_limb, x)[0]

        return out_delta