import os
import torch
import torch.optim as optim
import numpy as np

import copy 
from matplotlib import pyplot as plt


def print_error(action_error_sum, is_train):
    mean_error_p1, mean_error_p2 = print_error_action(action_error_sum, is_train)

    return mean_error_p1, mean_error_p2

def print_error_action(action_error_sum, is_train):
    mean_error_each = {'p1': 0.0, 'p2': 0.0}
    mean_error_all = {'p1': AccumLoss(), 'p2': AccumLoss()}

    if is_train == 0:
        print("{0:=^12} {1:=^10} {2:=^8}".format("Action", "p#1 mm", "p#2 mm"))

    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")
            
        mean_error_each['p1'] = action_error_sum[action]['p1'].avg * 1000.0
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg * 1000.0
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        if is_train == 0:
            print("{0:>6.2f} {1:>10.2f}".format(mean_error_each['p1'], mean_error_each['p2']))

    if is_train == 0:
        print("{0:<12} {1:>6.2f} {2:>10.2f}".format("Average", mean_error_all['p1'].avg, \
            mean_error_all['p2'].avg))
    
    return mean_error_all['p1'].avg, mean_error_all['p2'].avg


def test_calculation(predicted, target, action, error_sum, MAE=False):
    error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum)
    if not MAE:
        error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum)
    return error_sum

def mpjpe_by_action_p1(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    batch_num = predicted.size(0)
    frame_num = predicted.size(1)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name]['p1'].update(torch.mean(dist).item()*batch_num*frame_num, batch_num*frame_num)
    else:
        for i in range(batch_num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name]['p1'].update(torch.mean(dist[i]).item()*frame_num, frame_num)
            
    return action_error_sum


def mpjpe_by_action_p2(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred, gt)
    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]
        action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]
            action_error_sum[action_name]['p2'].update(np.mean(dist), 1)
            
    return action_error_sum

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



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: {'p1':AccumLoss(), 'p2':AccumLoss()} for i in range(len(actions))})
    return error_sum


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def get_optimizer(config, parameters):
    if config.optim.optimizer == 'Adam':
        return optim.Adam(parameters, lr=config.optim.lr, weight_decay=0.000,
                          betas=(0.9, 0.999), amsgrad=config.optim.amsgrad,
                          eps=config.optim.eps)
    elif config.optim.optimizer == 'RMSProp':
        return optim.RMSprop(parameters, lr=config.optim.lr, weight_decay=0.000)
    elif config.optim.optimizer == 'SGD':
        return optim.SGD(parameters, lr=config.optim.lr, momentum=0.9)
    else:
        raise NotImplementedError(
            'Optimizer {} not understood.'.format(config.optim.optimizer))

def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step / decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a


def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)