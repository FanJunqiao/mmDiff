import torch
import torch.nn as nn
from tqdm import tqdm
import os
import copy
import numpy as np


@torch.inference_mode()
def phase1_save(net, dataloader, device, amp = False, save_flag = '', root_path=""):
    net.eval()
    num_val_batches = len(dataloader)
    num = 0

    criterion = nn.MSELoss()
    val_loss = 0
    mean_pjpe = 0

    joint_gt_array_train = []
    joint_pred_array_train = []
    radar_feat_array_train = []
    radar_input_array_train = []


    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Saving round', unit='batch', leave=False):
            num += 1
            radar, joints = batch
            radar = radar.to(device=device, dtype=torch.float32)  # B, N, X
            joints = joints.to(device=device, dtype=torch.float32)


            # normalize joints with the center of body
            joints_original = copy.deepcopy(joints)
            joints_centers = joints_original[:, [0], :]
            joints = torch.sub(joints, joints_centers)
           


            # predict the mask
            joints_predict, joint_emb = net(radar)
            loss = criterion(joints_predict, joints.float())
            val_loss += loss.item()
            mean_pjpe += MPJPE_mean(joints_predict, joints)

            #append array
            if save_flag[:4] == "test" or save_flag == "train":
                joint_gt_array_train.append(joints.cpu().detach().numpy())
                joint_pred_array_train.append(joints_predict.cpu().detach().numpy())
                radar_feat_array_train.append(joint_emb.cpu().detach().numpy())
                radar_input_array_train.append(radar[:,0,:,:].cpu().detach().numpy())
        print(val_loss/num, mean_pjpe/num)

            


    if save_flag[:4] == "test":
        print("Saving Test.")
        with open(os.path.join(root_path, f"joint_gt_array_{save_flag}.npy"), "wb") as f:
            nparray = np.concatenate(joint_gt_array_train)
            print(nparray.shape)
            np.save(f, nparray)
        
        # np transform and save to npz
        with open(os.path.join(root_path, f"joint_pred_array_{save_flag}.npy"), "wb") as f:
            nparray = np.concatenate(joint_pred_array_train)
            print(nparray.shape)
            np.save(f, nparray)
        
        # np transform and save to npz
        with open(os.path.join(root_path, f"radar_feat_array_{save_flag}.npy"), "wb") as f:
            nparray = np.concatenate(radar_feat_array_train)
            print(nparray.shape)
            np.save(f, nparray)


        
        nparray = np.concatenate(radar_input_array_train)
        del radar_input_array_train
        bin = nparray.shape[0] // 1 + 1
        for i in range(1):
            with open(os.path.join(root_path, f"radar_input_array_{save_flag}.npy"), "wb") as f:
                end = min(nparray.shape[0], bin)
                print(nparray[:end].shape)
                np.save(f, nparray[:end])
                nparray = nparray[end:]
        del nparray

        
            
    elif save_flag == "train":
        print("Saving Train.")
        # np transform and save to npz
        with open(os.path.join(root_path, "joint_gt_array_train.npy"), "wb") as f:
            nparray = np.concatenate(joint_gt_array_train)
            print(nparray.shape)
            np.save(f, nparray)
        # np transform and save to npz
        with open(os.path.join(root_path, "joint_pred_array_train.npy"), "wb") as f:
            nparray = np.concatenate(joint_pred_array_train)
            print(nparray.shape)
            np.save(f, nparray)
        # np transform and save to npz
        with open(os.path.join(root_path, "radar_feat_array_train.npy"), "wb") as f:
            nparray = np.concatenate(radar_feat_array_train)
            print(nparray.shape)
            np.save(f, nparray)

        # np transform and save to npz
        nparray = np.concatenate(radar_input_array_train)
        del radar_input_array_train
        bin = nparray.shape[0] // 5 + 1
        for i in range(5):
            with open(os.path.join(root_path, f"radar_input_array_train{i}.npy"), "wb") as f:
                end = min(nparray.shape[0], bin)
                print(nparray[:end].shape)
                np.save(f, nparray[:end])
                nparray = nparray[end:]
    return 


def MPJPE_mean(pred, true, type = None, p=False):
    assert pred.shape == true.shape


    pred_np = pred.cpu().detach().numpy()
    true_np = true.cpu().detach().numpy()
    dists = calc_dists(pred_np, true_np)

    if p == True:
        print("pred:", pred_np[:,5:7,:])
        print("True", true_np[:,5:7,:])
        print(calc_dists(pred_np, true_np))
    mean = dist_mean(dists)


    return mean

def calc_dists(preds, target):
    preds = preds.astype(np.float32)
    target = target.astype(np.float32)
    dists = np.zeros((preds.shape[0], preds.shape[1])) # batch size and joints size
    for b in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            # if target[n, c, 0] > 1 and target[n, c, 1] > 1:
            #     normed_preds = preds[n, c, :] / normalize[n]
            #     normed_targets = target[n, c, :] / normalize[n]
            dists[b, j] = np.linalg.norm(preds[b,j,:] - target[b,j,:])
    return dists

def dist_mean(dists):
    ''' Return mean distance while ignoring values with a -1 '''
    dist_cal = np.not_equal(dists, -1)
    num_dist_cal = dist_cal.sum()
    if num_dist_cal > 0:
        return np.mean(dists[dist_cal])
    else:
        return -1

