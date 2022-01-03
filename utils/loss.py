import torch
import torch.nn.functional as F
from pytorch3d.transforms import matrix_to_quaternion, quaternion_invert, quaternion_multiply
from einops import rearrange
import numpy as np
from manopth.rot6d import compute_rotation_matrix_from_ortho6d

from utils.misc import quaternion_to_axis_angle

class DetLoss:
    def __init__(
            self,
            lambda_hm=100,
            lambda_pafm=1.0,
            lambda_lm=1.0,
    ):
        self.lambda_hm = lambda_hm
        self.lambda_pafm = lambda_pafm
        self.lambda_lm = lambda_lm
        self.dip_tip = np.ones(20)
        self.dip_tip[[2, 3, 6, 7, 10, 11, 14, 15, 18, 19]] = 10

    def compute_loss(self, preds, targs, infos, kp_idx):        
        batch_kp_size = torch.sum(kp_idx)
        hm_mask = infos['hm_mask'][kp_idx]
        paf_mask = infos['paf_mask'][kp_idx]
        batch_size = infos['batch_size']

        final_loss = torch.Tensor([0]).cuda()
        det_losses = {}

        pred_hm = preds['h_map'][kp_idx]
        pred_paf = preds['paf_map'][kp_idx]
        pred_lm = preds['l_map'][kp_idx]

        targ_hm = targs['hmap'][kp_idx]  # B*21*32*32
        targ_hm_tile = targ_hm.unsqueeze(2).expand(targ_hm.size(0), targ_hm.size(1), 3, targ_hm.size(2), targ_hm.size(3)) # B*21*3*32*32
        targ_paf = targs['paf_maps'][kp_idx]
        targ_lm = targs['lm'][kp_idx]

        # compute hmloss anyway
        hm_loss = torch.Tensor([0]).cuda()
        if self.lambda_hm:
            hm_mask = hm_mask.unsqueeze(-1)
            njoints = pred_hm.size(1)
            pred_hm = pred_hm.reshape((batch_kp_size, njoints, -1)).split(1, 1)
            targ_hm = targ_hm.reshape((batch_kp_size, njoints, -1)).split(1, 1)
            for idx in range(njoints):
                pred_hmapi = pred_hm[idx].squeeze()  # (B, 1, 1024)->(B, 1024)
                targ_hmi = targ_hm[idx].squeeze()
                hm_loss += 0.5 * F.mse_loss(
                    pred_hmapi.mul(hm_mask[:, idx]),  # (B, 1024) mul (B, 1)
                    targ_hmi.mul(hm_mask[:, idx])
                )  # mse calculate the loss of every sample  (in fact it calculate minbacth_loss/32*32 )
            final_loss += self.lambda_hm * hm_loss
        det_losses["det_hm"] = hm_loss

        # compute paf loss
        loss_paf = torch.Tensor([0]).cuda()        
        if self.lambda_pafm:
            paf_mask = paf_mask.unsqueeze(-1)
            pred_paf = rearrange(pred_paf, 'b (j c) h w -> b j c h w', c=2)
            targ_paf = rearrange(targ_paf, 'b (j c) h w -> b j c h w', c=2)
            njoints = pred_paf.size(1)    
            pred_paf = pred_paf.reshape((batch_kp_size, njoints, -1)).split(1, 1)
            targ_paf = targ_paf.reshape((batch_kp_size, njoints, -1)).split(1, 1)
            for idx in range(njoints):
                pred_pafi = pred_paf[idx].squeeze()  # (B, 1, 1024)->(B, 1024)
                targ_pafi = targ_paf[idx].squeeze()
                loss_paf += 0.5 * F.mse_loss(
                    pred_pafi.mul(paf_mask[:, idx//2]),  # (B, 1024) mul (B, 1)
                    targ_pafi.mul(paf_mask[:, idx//2])
                ) * self.dip_tip[idx] # mse calculate the loss of every sample  (in fact it calculate minbacth_loss/32*32 )
            
            final_loss += self.lambda_pafm * loss_paf
        det_losses["det_pafm"] = loss_paf
        
        # compute lm loss
        loss_lm = torch.Tensor([0]).cuda()
        if self.lambda_lm:
            loss_lm = torch.norm((pred_lm - targ_lm) * targ_hm_tile) / batch_kp_size  # loss of every sample
            final_loss += self.lambda_lm * loss_lm
        det_losses["det_lm"] = loss_lm
    
        det_losses["det_total"] = final_loss
        return final_loss, det_losses, batch_kp_size

class IKLoss:
    def __init__(
            self,
            lambda_rot=10,
            lambda_cos=1,
            lambda_shape=0.5,
            lambda_norm=5e-3
    ):
        self.lambda_rot = lambda_rot
        self.lambda_cos = lambda_cos
        self.lambda_shape = lambda_shape
        self.lambda_norm = lambda_norm

    def compute_loss(self, preds, targs, batch_size):
        ik_losses = {}
        infer_rot6d = preds['rot6d']
        infer_rot6d_ = rearrange(infer_rot6d, 'b j c -> (b j) c')
        infer_rotmat = compute_rotation_matrix_from_ortho6d(infer_rot6d_)
        infer_shape = preds['shape']

        targ_rotmat = targs['rotmat']
        targ_shape = targs['shape']

        final_loss = torch.Tensor([0]).cuda()
        # compute rotation loss
        infer_pose_ = rearrange(infer_rotmat, '(b j) r c -> b j r c', j=16)
        loss_rot = F.mse_loss(infer_pose_, targ_rotmat) * self.lambda_rot
        final_loss += loss_rot
        ik_losses['rot'] = loss_rot
        
        # compute cosine loss
        infer_quat = matrix_to_quaternion(infer_rotmat)
        infer_quat = rearrange(infer_quat, '(b j) c -> b j c', j=16)
        infer_quat_inv = quaternion_invert(rearrange(infer_quat, 'b j c -> (b j) c'))
        gt_quat = matrix_to_quaternion(rearrange(targ_rotmat, 'b j r c -> (b j) r c'))
        loss_cos = (1 - quaternion_multiply(gt_quat, infer_quat_inv)[:, 0]).sum() / batch_size * self.lambda_cos
        final_loss += loss_cos
        ik_losses['cos'] = loss_cos

        # compute shape loss
        loss_shape = F.mse_loss(infer_shape, targ_shape) * self.lambda_shape
        final_loss += loss_shape
        ik_losses['shape'] = loss_shape

        # compute regularization loss
        theta_norm = torch.linalg.norm(infer_rot6d, dim=(1, 2)).sum() / batch_size * self.lambda_norm
        final_loss += theta_norm
        ik_losses['norm'] = theta_norm

        return final_loss, ik_losses
