import os
import shutil
import pickle

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import Sampler
from termcolor import colored, cprint

import utils.func as func
import copy
import cv2

def print_args(args):
    opts = vars(args)
    cprint("{:>30}  Options  {}".format("=" * 15, "=" * 15), 'yellow')
    for k, v in sorted(opts.items()):
        print("{:>30}  :  {}".format(k, v))
    cprint("{:>30}  Options  {}".format("=" * 15, "=" * 15), 'yellow')


def param_count(net):
    return sum(p.numel() for p in net.parameters()) / 1e6

def out_loss_auc(
        loss_all_, auc_all_, acc_hm_all_, outpath
):
    loss_all = copy.deepcopy(loss_all_)
    if acc_hm_all_ is not None:
        acc_hm_all = copy.deepcopy(acc_hm_all_)
    auc_all = copy.deepcopy(auc_all_)

    for k, l in zip(loss_all.keys(), loss_all.values()):
        np.save(os.path.join(outpath, "{}.npy".format(k)), np.vstack((np.arange(1, len(l) + 1), np.array(l))).T)

    if acc_hm_all_ is not None and len(acc_hm_all):
        for key ,value in acc_hm_all.items():
            acc_hm_all[key]=np.array(value)
        np.save(os.path.join(outpath, "acc_hm_all.npy"), acc_hm_all)


    if len(auc_all):
        for key ,value in auc_all.items():
            auc_all[key]=np.array(value)
        np.save(os.path.join(outpath, "auc_all.npy"), np.array(auc_all))


def saveloss(d):
    for k, v in zip(d.keys(), d.values()):
        mat = np.array(v)
        np.save(os.path.join("losses", "{}.npy".format(k)), mat)


def save_checkpoint(
        state,
        checkpoint='checkpoint',
        filename='checkpoint.pth',
        snapshot=None,
        # is_best=False
        is_best=None
):
    # preds = to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    fileprefix = filename.split('.')[0]
    # torch.save(state, filepath)
    torch.save(state['model'].state_dict(), filepath)

    if snapshot and state['epoch'] % snapshot == 0:
        shutil.copyfile(
            filepath,
            os.path.join(
                checkpoint,
                '{}_{}.pth'.format(fileprefix, state['epoch'])
            )
        )

    [auc, best_acc] = is_best

    for key in auc.keys():
        if auc[key] > best_acc[key]:
            shutil.copyfile(
                filepath,
                os.path.join(
                    checkpoint,
                    '{}_{}best.pth'.format(fileprefix, key)
                )
            )

def load_checkpoint(model, checkpoint):
    name = checkpoint
    checkpoint = torch.load(name)
    pretrain_dict = clean_state_dict(checkpoint['state_dict'])
    model_state = model.state_dict()
    state = {}
    for k, v in pretrain_dict.items():
        if k in model_state:
            state[k] = v
        else:
            print(k, ' is NOT in current model')
    model_state.update(state)
    model.load_state_dict(model_state)
    print(colored('loaded {}'.format(name), 'cyan'))


def clean_state_dict(state_dict):
    """save a cleaned version of model without dict and DataParallel

    Arguments:
        state_dict {collections.OrderedDict} -- [description]

    Returns:
        clean_model {collections.OrderedDict} -- [description]
    """

    clean_model = state_dict
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    clean_model = OrderedDict()
    if any(key.startswith('module') for key in state_dict):
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            clean_model[name] = v
    else:
        return state_dict

    return clean_model


def save_pred(preds, checkpoint='checkpoint', filename='preds_valid.mat'):
    preds = func.to_numpy(preds)
    filepath = os.path.join(checkpoint, filename)
    scipy.io.savemat(filepath, mdict={'preds': preds})

def square_xyxy(xyxy, imgh, imgw):
    if torch.is_tensor(xyxy):
        square = xyxy.new_zeros(xyxy.shape)
        cx = (xyxy[:, 2] + xyxy[:, 0]) / 2
        cy = (xyxy[:, 3] + xyxy[:, 1]) / 2
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        width = torch.max(w, h) / 2 # b * 1
                
        square[:, 0] = cx - width
        square[:, 1] = cy - width
        square[:, 2] = cx + width
        square[:, 3] = cy + width

        # boundary check
        left_offset = (0 - square[:, 0])[:, None]
        left_offset[left_offset<0] = 0
        square[:, [0, 2]] += left_offset.repeat(1, 2)

        right_offset = (square[:, 2] - imgw)[:, None]
        right_offset[right_offset<0] = 0
        square[:, [0, 2]] -= right_offset.repeat(1, 2)

        top_offset = (0 - square[:, 1])[:, None]
        top_offset[top_offset<0] = 0
        square[:, [1, 3]] += top_offset.repeat(1, 2)

        bottom_offset = (square[:, 3] - imgh)[:, None]
        bottom_offset[bottom_offset<0] = 0
        square[:, [1, 3]] -= bottom_offset.repeat(1, 2)
    else: # numpy
        square = xyxy.copy()
        cx = (xyxy[2] + xyxy[0]) / 2
        cy = (xyxy[3] + xyxy[1]) / 2
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        width = max(w, h) / 2 # b * 1

        square[0] = cx - width
        square[1] = cy - width
        square[2] = cx + width
        square[3] = cy + width

        # boundary check
        left_offset = 0 - square[0]
        if left_offset < 0:
            left_offset = 0
        square[[0, 2]] += left_offset

        right_offset = square[2] - imgw
        if right_offset < 0:
            right_offset = 0
        square[[0, 2]] -= right_offset

        top_offset = 0 - square[1]
        if top_offset < 0:
            top_offset = 0
        square[[1, 3]] += top_offset

        bottom_offset = square[3] - imgh
        if bottom_offset < 0:
            bottom_offset = 0
        square[[1, 3]] -= bottom_offset
    return square


# https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def load_pkl(path):
    """
    Load pickle data.
    Parameter
    ---------
    path: Path to pickle file.
    Return
    ------
    Data in pickle file.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def initialize_weights(m):
    if isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

class BatchSampler:
    # source: https://github.com/CaoWGG/multi-scale-training, for multi-scale training
    def __init__(self, sampler, batch_size, drop_last, multiscale_step=None, img_sizes=None):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        if multiscale_step is not None and multiscale_step < 1 :
            raise ValueError("multiscale_step should be > 0, but got "
                             "multiscale_step={}".format(multiscale_step))
        if multiscale_step is not None and img_sizes is None:
            raise ValueError("img_sizes must a list, but got img_sizes={} ".format(img_sizes))

        self.multiscale_step = multiscale_step
        self.img_sizes = img_sizes

    def __iter__(self):
        num_batch = 0
        batch = []
        size = 320
        for idx in self.sampler:
            batch.append([idx, size])
            if len(batch) == self.batch_size:
                yield batch
                num_batch+=1
                batch = []
                if self.multiscale_step and num_batch % self.multiscale_step == 0 :
                    size = np.random.choice(self.img_sizes)
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size