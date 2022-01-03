import sys

sys.path.append("./")
import torch
from torch import nn
from einops import rearrange, repeat
from manopth.rot6d import compute_rotation_matrix_from_ortho6d
from manopth.manolayer import ManoLayer

from model.yolo import YOLOHand
from model.iknet import IKNet
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Hand(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.yolohand = YOLOHand(args)
        self.iknet = IKNet(inc=21*3, depth=6, width=1024)
        self.model_init(args.detnet, args.iknet)
        self.mano_layer = ManoLayer(mano_root='mano/models', use_pca=False, flat_hand_mean=False,
                                    root_rot_mode='rotmat', joint_rot_mode='rotmat').to(device)
        self.shape_mean = torch.Tensor(config.SHAPE_MEAN)[None, :].to(device)
        self.shape_std = torch.Tensor(config.SHAPE_STD)[None, :].to(device)
        
    def forward(self, x, targets=None):
        # YOLOv3 and DetNet forwarding
        results = self.yolohand(x, targets)
        if results is None:
            return None
        xyz = results['xyz'].clone()
        ref_bone_length = torch.linalg.norm(xyz[:, 0] - xyz[:, 9], dim=1, keepdim=True)[:, None, :].repeat(1, 21, 3)
        xyz = (xyz - xyz[:, 9][:, None, :].repeat(1, 21, 1)) / ref_bone_length

        # IKNet forwarding
        rot6d, shape = self.iknet(xyz)

        # MANO forwarding
        shape = self.shape_mean.repeat(len(shape), 1) + shape[:, None].repeat(1, 10) * self.shape_std.repeat(len(shape), 1)
        rot6d = rearrange(rot6d, 'b j c -> (b j) c')
        rotmat = compute_rotation_matrix_from_ortho6d(rot6d)
        rotmat = rearrange(rotmat, '(b j) h w -> b j h w', j=16)
        verts, xyz_ik = self.mano_layer(rotmat, shape)
        results['xyz_ik'] = xyz_ik
        results['verts'] = verts
        
        return results

    def model_init(self, detnet, iknet):
        pretrain_dict = torch.load(detnet, map_location=device)
        model_state = self.yolohand.state_dict()
        state = {}
        for k, v in pretrain_dict.items():
            if k in model_state:
                state[k] = v
            else:
                print(k, ' is NOT in current model')
        model_state.update(state)
        self.yolohand.load_state_dict(model_state)
        if iknet is not None:
            self.iknet.load_state_dict(torch.load(iknet))
