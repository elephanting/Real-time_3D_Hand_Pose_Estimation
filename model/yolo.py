import sys
import argparse
import numbers
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
import numpy as np
from einops import rearrange, repeat
from yolov3.models import Darknet
from yolov3.utils.utils import non_max_suppression

sys.path.append("./")
from utils.misc import clean_state_dict, square_xyxy
import config as cfg
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class net_2d(nn.Module):
    def __init__(self, input_features, output_features, stride, joints=21):
        super().__init__()
        self.project =nn.Sequential(conv3x3(input_features, output_features, stride), nn.BatchNorm2d(output_features),
                                     nn.ReLU())
        self.prediction = nn.Conv2d(output_features, joints, 1, 1, 0)

    def forward(self, x):
        x = self.project(x)
        x = self.prediction(x)
        x[:, cfg.KP_INDEX] = x[:, cfg.KP_INDEX].sigmoid()
        return x

class net_3d(nn.Module):
    def __init__(self, input_features, output_features, stride, joints=21):
        super().__init__()
        self.project = nn.Sequential(conv3x3(input_features, output_features, stride), nn.BatchNorm2d(output_features),
                                     nn.ReLU())
        self.prediction = nn.Conv2d(output_features, joints*3, 1, 1, 0)

    def forward(self, x):
        x = self.prediction(self.project(x))
        x = rearrange(x, 'b (j l) h w -> b j l h w', l=3)
        return x

class YOLOHand(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = Darknet(args.cfg, args.img_size)
        self.pyramid_size = [] # list[(c, h, w)]
        self._init_pyramid()
        self.fpn = FeaturePyramidNetwork([size[0] for size in self.pyramid_size], 64)
        self.net2d = net_2d(320, 320, stride=1, joints=21+40)
        self.location = net_3d(input_features=320+40+21, output_features=320+40+21, stride=1)

    def forward(self, x, target=None):
        result = {}
        layer_dict = OrderedDict()
        batch, c, h, w = x.size()

        # update pyramid size
        for i in range(self.pyramid_height):
            self.pyramid_size[i][1] = h / 2 ** (i+1)
            self.pyramid_size[i][2] = w / 2 ** (i+1)

        # backbone forwarding
        # A workaround to get intermediate layer feature maps
        # see: https://github.com/pytorch/vision/issues/1895
        outs = []
        hook1 = self.backbone.outlayer1.register_forward_hook(lambda self, input, output: outs.append(output))
        hook2 = self.backbone.outlayer2.register_forward_hook(lambda self, input, output: outs.append(output))
        hook3 = self.backbone.outlayer3.register_forward_hook(lambda self, input, output: outs.append(output))
        hook4 = self.backbone.outlayer4.register_forward_hook(lambda self, input, output: outs.append(output))
        hook5 = self.backbone.outlayer5.register_forward_hook(lambda self, input, output: outs.append(output))
        
        # pred: predicted bbox, yolo_out: yolo-head feature maps
        if self.training:
            yolo_out = self.backbone(x)
        else:
            pred, yolo_out = self.backbone(x) 
        hook1.remove()
        hook2.remove()
        hook3.remove()
        hook4.remove()
        hook5.remove()
        layer_dict['level0'] = outs[0]
        layer_dict['level1'] = outs[1]
        layer_dict['level2'] = outs[2]
        layer_dict['level3'] = outs[3]
        layer_dict['level4'] = outs[4]

        # training or evaluation
        if target is not None:
            # feature pyramid network forwarding
            features = self.fpn(layer_dict)
            features = [v for k, v in features.items()]
            rois = target['kp2d_box'].clone()

            # ROI align the intermediate layers
            cropped_feature = []
            for i, roi in enumerate(rois):
                tmp_roi = roi.clone()

                # collect features from intermediate layers
                feature = []
                for j in range(self.pyramid_height):                   
                    tmp_roi[[0, 2]] = roi[[0, 2]] * self.pyramid_size[j][2]
                    tmp_roi[[1, 3]] = roi[[1, 3]] * self.pyramid_size[j][1]
                    feature.append(torchvision.ops.roi_align(features[j][i][None, ...], [tmp_roi[None, ...]], output_size=self.args.hm_size, aligned=True))
                cropped_feature.append(torch.cat(feature, dim=1)) # batch * 320 * h * w
            cropped_feature = torch.cat(cropped_feature, dim=0)

            # predict heatmaps and PAF maps
            paf_heatmaps = self.net2d(cropped_feature)
            infer_hmap = paf_heatmaps[:, cfg.KP_INDEX]
            infer_paf = paf_heatmaps[:, cfg.PAF_INDEX]
            
            # predict location maps
            x = torch.cat([cropped_feature, paf_heatmaps], dim=1)                     
            infer_lmap = self.location(x)

            # estimate 2D keypoint
            uv, argmax = self.map_to_uv(infer_hmap)
            if self.args.paf:
                heatmap_peaks = self.find_peak(infer_hmap, topk=3, joint_lst=cfg.DIP_TIP, uv=uv) # b*21*topk*2
                uv, argmax = self.find_connected_joints(infer_paf, heatmap_peaks, uv)
            
            # estimate 3D keypoint
            xyz = self.lmap_to_xyz(infer_lmap, argmax)

            # get real uv, heatmap coordinates to image coordinates
            rois[:, [0, 2]] *= w
            rois[:, [1, 3]] *= h
            uv = self.heatmaps_to_keypoints(infer_hmap, rois, argmax)

            result = {
                'yolo': yolo_out,
                'h_map': infer_hmap,
                'paf_map': infer_paf,
                'l_map': infer_lmap,
                'uv': uv[:, :, :2],
                'xyz': xyz
            }
            return result
        # demo
        else:
            # feature pyramid network forwarding    
            features_lst = self.fpn(layer_dict)
            features = [v[0][None, ...] for k, v in features_lst.items()]
            features_mr = [v[1][None, ...] for k, v in features_lst.items()]

            # flip bounding box if label is left hand
            # pred size: 2(input and mirror image) * size_of_feature * 6(x,y,w,h,obj_conf,right_hand_conf)
            pred[1, :, 0] = w - pred[1, :, 0]
            pred_b, pred_h, pred_w = pred.shape

            # combine input image and mirror image prediction
            # expand_pred size: 1 * (2*size_of_feature) * 7(x,y,w,h,obj_conf,right,left_hand_conf)
            expand_pred = torch.zeros((1, pred_h*2, pred_w+1)).to(pred.device)
            expand_pred[:, :pred_h, :pred_w] = pred[0]
            expand_pred[:, pred_h:, :pred_w-1] = pred[1, :, :pred_w-1]
            expand_pred[:, pred_h:, -1] = pred[1, :, -1]
            pred = non_max_suppression(expand_pred, self.args.conf_thres, self.args.iou_thres,
                                    multi_label=False, agnostic=True)[0]
            if pred is not None:
                pred_box = []
                pred_box.append(pred[pred[:, 5] == 0].clone())
                pred_box.append(pred[pred[:, 5] == 1].clone())
                pred_box = torch.cat(pred_box, dim=0)
            else:
                return None

            # ROI align the intermediate layers
            cropped_feature = []
            hand_box = []
            is_left = []
            for i, det in enumerate([pred]): # detections for image i
                if len(det) > self.args.max:
                    det = det[torch.argsort(det[:, 4], descending=True)[:self.args.max]]
                right_hand = det[det[:, 5] == 0]
                left_hand = det[det[:, 5] == 1]
                right_hand = square_xyxy(right_hand[:, :4], imgh=h, imgw=w)
                left_hand = square_xyxy(left_hand[:, :4], imgh=h, imgw=w)

                # flip left hand bounding box
                left_hand_mr = left_hand.clone()
                left_hand_mr[:, [0, 2]] = w - left_hand_mr[:, [0, 2]]
                left_hand_mr[:, [0, 2]] = left_hand_mr[:, [2, 0]]
                
                if len(right_hand):
                    hand_box.append(right_hand.clone())
                    for _ in range(len(right_hand)):
                        is_left.append(False)

                    # collect feature maps from intermediate layers      
                    feature = []
                    tmp_roi = right_hand.clone()
                    for j in range(self.pyramid_height):                    
                        tmp_roi[:, [0, 2]] = right_hand[:, [0, 2]] * (self.pyramid_size[j][2] / w)
                        tmp_roi[:, [1, 3]] = right_hand[:, [1, 3]] * (self.pyramid_size[j][1] / h)
                        feature.append(torchvision.ops.roi_align(features[j][i][None, ...], [tmp_roi], output_size=self.args.hm_size, aligned=True))
                    cropped_feature.append(torch.cat(feature, dim=1)) # num_right * 320 * 32 * 32
            
                if len(left_hand_mr):
                    hand_box.append(left_hand_mr.clone())
                    for _ in range(len(left_hand_mr)):
                        is_left.append(True)

                    # collect feature maps from intermediate layers
                    feature = []
                    tmp_roi = left_hand_mr.clone()
                    for j in range(self.pyramid_height):                    
                        tmp_roi[:, [0, 2]] = left_hand_mr[:, [0, 2]] * (self.pyramid_size[j][2] / w)
                        tmp_roi[:, [1, 3]] = left_hand_mr[:, [1, 3]] * (self.pyramid_size[j][1] / h)
                        feature.append(torchvision.ops.roi_align(features_mr[j][i][None, ...], [tmp_roi], output_size=self.args.hm_size, aligned=True))
                    cropped_feature.append(torch.cat(feature, dim=1)) # num_left * 320 * 32 * 32
            cropped_feature = torch.cat(cropped_feature, dim=0)
           
            # predict heatmaps and PAF maps
            paf_heatmaps = self.net2d(cropped_feature)
            infer_hmap = paf_heatmaps[:, cfg.KP_INDEX]
            infer_paf = paf_heatmaps[:, cfg.PAF_INDEX]

            # predict location maps
            x = torch.cat([cropped_feature, paf_heatmaps], dim=1)
            infer_lmap = self.location(x)

            # estimate 2D keypoint
            uv, argmax = self.map_to_uv(infer_hmap)
            if self.args.paf:
                heatmap_peaks = self.find_peak(infer_hmap, topk=3, joint_lst=cfg.DIP_TIP, uv=uv) # b*21*topk*2
                uv, argmax = self.find_connected_joints(infer_paf, heatmap_peaks, uv)

            # estimate 3D keypoint
            xyz = self.lmap_to_xyz(infer_lmap, argmax)
            
            # get real uv, heatmap coordinates to image coordinates
            hand_box = torch.cat(hand_box, dim=0)
            uv = self.heatmaps_to_keypoints(infer_hmap, hand_box, argmax)
            uv[is_left, :, 0] = w - uv[is_left, :, 0]

            result = {
                'box': pred_box,
                'uv': uv[:, :, :2],
                'xyz': xyz
            }
            return result

    def _init_pyramid(self):
        if not hasattr(self.args, 'h') or not hasattr(self.args, 'w'):
            self.args.h = self.args.img_size
            self.args.w = self.args.img_size
        img = torch.zeros((1, 3, self.args.h, self.args.w)).to(device)  # init img
        self.backbone.to(device)

        outs = []
        hook1 = self.backbone.outlayer1.register_forward_hook(lambda self, input, output: outs.append(output))
        hook2 = self.backbone.outlayer2.register_forward_hook(lambda self, input, output: outs.append(output))
        hook3 = self.backbone.outlayer3.register_forward_hook(lambda self, input, output: outs.append(output))
        hook4 = self.backbone.outlayer4.register_forward_hook(lambda self, input, output: outs.append(output))
        hook5 = self.backbone.outlayer5.register_forward_hook(lambda self, input, output: outs.append(output))
        output = self.backbone(img) # run once
        hook1.remove()
        hook2.remove()
        hook3.remove()
        hook4.remove()
        hook5.remove()
        
        self.pyramid_size = [list(feature.shape[1:]) for feature in outs] # c * h * w
        self.pyramid_height = len(self.pyramid_size)

    @staticmethod
    def map_to_uv(hmap):
        b, j, h, w = hmap.shape
        hmap = rearrange(hmap, 'b j h w -> b j (h w)')
        argmax = torch.argmax(hmap, -1, keepdim=True)
        u = argmax % w
        v = argmax // w
        uv = torch.cat([u, v], dim=-1)

        return uv, argmax
    
    @staticmethod
    def lmap_to_xyz(lmap, argmax):
        lmap = rearrange(lmap, 'b j l h w -> b j (h w) l')
        index = repeat(argmax, 'b j i -> b j i c', c=3)
        xyz = torch.gather(lmap, dim=2, index=index).squeeze(2)
        return xyz

    # https://github.com/pytorch/vision/blob/e8dded4c05ee403633529cef2e09bf94b07f6170/torchvision/models/detection/roi_heads.py#L226
    @staticmethod
    def heatmaps_to_keypoints(maps, rois, argmax):
        """Extract predicted keypoint locations from heatmaps. Output has shape
        (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
        for each keypoint.
        """
        # This function converts a discrete image coordinate in a HEATMAP_SIZE x
        # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
        # consistency with keypoints_to_heatmap_labels by using the conversion from
        # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
        # continuous coordinate.
        offset_x = rois[:, 0]
        offset_y = rois[:, 1]

        widths = rois[:, 2] - rois[:, 0]
        heights = rois[:, 3] - rois[:, 1]
        widths = widths.clamp(min=1)
        heights = heights.clamp(min=1)
        widths_ceil = widths.ceil()
        heights_ceil = heights.ceil()

        num_keypoints = maps.shape[1]

        xy_preds = torch.zeros((len(rois), 3, num_keypoints), dtype=torch.float32, device=maps.device)
        end_scores = torch.zeros((len(rois), num_keypoints), dtype=torch.float32, device=maps.device)
        for i in range(len(rois)):
            roi_map_width = int(widths_ceil[i].item())
            roi_map_height = int(heights_ceil[i].item())
            width_correction = widths[i] / roi_map_width
            height_correction = heights[i] / roi_map_height

            _, h, w = maps[i].shape
            scale_x = roi_map_width / w
            scale_y = roi_map_height / h
            pos = argmax[i, :, 0]
            
            x_int = (pos % w).float()
            y_int = ((pos - x_int) // w).float()
            x_int *= scale_x
            y_int *= scale_y
            
            x = (x_int + 0.5) * width_correction
            y = (y_int + 0.5) * height_correction
            xy_preds[i, 0, :] = x + offset_x[i]
            xy_preds[i, 1, :] = y + offset_y[i]
            xy_preds[i, 2, :] = 1

        return xy_preds.permute(0, 2, 1)
    
    def line_integral(self, parent, child, paf):
        '''
        parent: batch*n*2
        child: batch*n*2
        paf: batch*10*32*32
        '''
        #print(paf.shape)
        batch_size, _, h, w = paf.shape
        pafX = paf[:, [0, 2, 4, 6, 8]]
        pafY = paf[:, [1, 3, 5, 7, 9]]
        n = parent.shape[1]
        device = paf.device

        # building the vectors
        parent = parent.float()
        child = child.float()
        dx, dy = child[:, :, 0] - parent[:, :, 0], child[:, :, 1] - parent[:, :, 1]
        normVec = torch.linalg.norm(child - parent, dim=2)
        vx, vy = dx / normVec, dy / normVec # b*n*1

        # sampling
        thre = 0  # limb width
        min_x = torch.max(torch.round(torch.min(parent[:, :, 0], child[:, :, 0]) - thre).long(), torch.zeros_like(parent[:, :, 0]).long())
        max_x = torch.min(torch.round(torch.max(parent[:, :, 0], child[:, :, 0]) + thre).long(), torch.full_like(parent[:, :, 0], w-1).long())
        min_y = torch.max(torch.round(torch.min(parent[:, :, 1], child[:, :, 1]) - thre).long(), torch.zeros_like(parent[:, :, 1]).long())
        max_y = torch.min(torch.round(torch.max(parent[:, :, 1], child[:, :, 1]) + thre).long(), torch.full_like(parent[:, :, 1], h-1).long())
        
        scores = torch.zeros(batch_size, n).to(device)
        xs = torch.zeros(batch_size, n, cfg.NUM_SAMPLE, dtype=torch.long).to(device)
        ys = torch.zeros(batch_size, n, cfg.NUM_SAMPLE, dtype=torch.long).to(device)
        for i in range(batch_size):
            for j in range(n):
                step_x = float(max_x[i, j]-min_x[i, j]) / cfg.NUM_SAMPLE
                step_y = float(max_y[i, j]-min_y[i, j]) / cfg.NUM_SAMPLE

                if step_x:
                    xs[i, j] = torch.arange(min_x[i, j], max_x[i, j], step=step_x, device=device).long()
                else:
                    xs[i, j] = min_x[i, j].long()

                if step_y:
                    ys[i, j] = torch.arange(min_y[i, j], max_y[i, j], step=step_y, device=device).long()
                else:
                    ys[i, j] = min_y[i, j].long()
        ys_xs = torch.stack([ys, xs], dim=-1) # b*n*NUM_SAMPLE*2

        # evaluating on the field
        pafX = pafX.repeat(1, n // 5, 1, 1) # b*n*32*32
        pafY = pafY.repeat(1, n // 5, 1, 1) # b*n*32*32

        pafXs = pafX[torch.arange(batch_size)[:, None, None], torch.arange(n)[:, None], ys_xs[:, :, :, 0], ys_xs[:, :, :, 1]] # b*n*NUM_SAMPLE
        pafYs = pafY[torch.arange(batch_size)[:, None, None], torch.arange(n)[:, None], ys_xs[:, :, :, 0], ys_xs[:, :, :, 1]] # b*n*NUM_SAMPLE

        # integral
        score = pafXs * vx[..., None].repeat(1, 1, cfg.NUM_SAMPLE) + pafYs * vy[..., None].repeat(1, 1, cfg.NUM_SAMPLE) # b*n*NUM_SAMPLE
        score[score < cfg.THRESH_VECTOR_SCORE] = -1
        score = torch.sum(score, dim=2)
        return score

    def find_peak(self, orig_heatmap, topk=3, joint_lst=range(21)):
        batch_size, num_channel, h, w = orig_heatmap.shape
        
        # remove low confidence value
        heatmap = orig_heatmap.clone()
        heatmap[heatmap < cfg.THRESH_HEAT] = 0.

        heatmap_maxima = F.max_pool2d_with_indices(heatmap, kernel_size=3, stride=1, padding=1)[1] # b*21*32*32
        batch_maxima = []
        for b in range(batch_size):
            channel_maxima = []
            for c in joint_lst:
                heatmap_maxima_b_c = heatmap_maxima[b, c].reshape(-1)
                candidates = heatmap_maxima_b_c.unique()
                nice_peaks = candidates[(heatmap_maxima_b_c[candidates]==candidates).nonzero()].squeeze(-1)
                nice_peaks = nice_peaks[nice_peaks != 0] # remove index (0,0) local maxima
                
                # sorting
                heatmap_b_c = heatmap[b, c].reshape(-1)
                confidences = heatmap_b_c[nice_peaks]
                value, idx = torch.sort(confidences, dim=-1, descending=True)
                nice_peaks = nice_peaks[idx]

                # choose topk
                nice_peaks = nice_peaks[:topk]
                
                # check if num of peak is less than topk
                if len(nice_peaks) == 0:
                    nice_peaks = torch.argmax(orig_heatmap[b, c].reshape(-1), dim=-1).repeat(topk)
                elif len(nice_peaks) < topk:
                    padding = topk - len(nice_peaks)
                    padding = torch.full(size=[padding], fill_value=nice_peaks[0]).to(nice_peaks.device)
                    nice_peaks = torch.cat([padding, nice_peaks], dim=0)
                
                # convert index from 1D to 2D
                nice_peaks_x = nice_peaks % w
                nice_peaks_y = nice_peaks // w
                nice_peaks = torch.cat((nice_peaks_x[:, None], nice_peaks_y[:, None]), dim=1)
                
                channel_maxima.append(nice_peaks[None, ...])
            channel_maxima = torch.cat(channel_maxima, dim=0)
            batch_maxima.append(channel_maxima[None, ...])
        batch_maxima = torch.cat(batch_maxima, dim=0)

        return batch_maxima

    def find_connected_joints(self, paf, heatmap_peaks, uv):
        '''
        paf: b*40*32*32 || b*30*32*32
        conf_map: b*20*32*32 || b*15*32*32
        peak: b*21*topk*2
        uv: b*21*2
        '''
        batch_size, _, h, w = paf.shape
        device = paf.device
        index = [0, 1, 2, 1, 2, 0, 2, 0, 1]
        paf_DIP_index = [4, 5, 12, 13, 20, 21, 28, 29, 36, 37]
        paf_TIP_index = [6, 7, 14, 15, 22, 23, 30, 31, 38, 39]

        MCP_idx = [1, 5, 9, 13, 17]
        PIP_idx = [2, 6, 10, 14, 18]
        DIP_idx = [3, 7, 11, 15, 19]
        TIP_idx = [4, 8, 12, 16, 20]

        PIPs = uv[:, PIP_idx] # b*5*2
        PIPs = PIPs.repeat(1, 3, 1) # b*15*2
        DIPs = torch.zeros(batch_size, 15, 2).to(device) # b*15*2
        DIPs[:, :5] = heatmap_peaks[:, DIP_idx, 0]
        DIPs[:, 5:10] = heatmap_peaks[:, DIP_idx, 1]
        DIPs[:, 10:] = heatmap_peaks[:, DIP_idx, 2]

        scores = torch.zeros(batch_size, 15)
        DIP_score = self.line_integral(PIPs, DIPs, paf[:, paf_DIP_index], verbose=False) # b
        DIP_score = DIP_score.repeat(1, 3)
        DIPs = DIPs.repeat(1, 3, 1)
        
        TIPs = torch.zeros(batch_size, 45, 2).to(device) # b*45*2
        TIPs[:, :5] = heatmap_peaks[:, TIP_idx, 0]
        TIPs[:, 5:10] = heatmap_peaks[:, TIP_idx, 1]
        TIPs[:, 10:15] = heatmap_peaks[:, TIP_idx, 2]

        TIPs[:, 15:20] = heatmap_peaks[:, TIP_idx, 1]
        TIPs[:, 20:25] = heatmap_peaks[:, TIP_idx, 2]
        TIPs[:, 25:30] = heatmap_peaks[:, TIP_idx, 0]

        TIPs[:, 30:35] = heatmap_peaks[:, TIP_idx, 2]
        TIPs[:, 35:40] = heatmap_peaks[:, TIP_idx, 0]
        TIPs[:, 40:] = heatmap_peaks[:, TIP_idx, 1]

        TIP_score = self.line_integral(DIPs, TIPs, paf[:, paf_TIP_index], verbose=False) # b

        final_score = DIP_score + TIP_score
        Thumb_score = torch.argmax(final_score[:, [0, 5, 10, 15, 20, 25, 30, 35, 40]], dim=1)

        Index_score =  torch.argmax(final_score[:, [1, 6, 11, 16, 21, 26, 31, 36, 41]], dim=1)
        Middle_score =  torch.argmax(final_score[:, [2, 7, 12, 17, 22, 27, 32, 37, 42]], dim=1)
        Ring_score =  torch.argmax(final_score[:, [3, 8, 13, 18, 23, 28, 33, 38, 43]], dim=1)
        Little_score =  torch.argmax(final_score[:, [4, 9, 14, 19, 24, 29, 34, 39, 44]], dim=1)
        
        uv[:, DIP_idx[0]] = heatmap_peaks[torch.arange(batch_size), DIP_idx[0], Thumb_score % 3]
        uv[:, DIP_idx[1]] = heatmap_peaks[torch.arange(batch_size), DIP_idx[1], Index_score % 3]
        uv[:, DIP_idx[2]] = heatmap_peaks[torch.arange(batch_size), DIP_idx[2], Middle_score % 3]
        uv[:, DIP_idx[3]] = heatmap_peaks[torch.arange(batch_size), DIP_idx[3], Ring_score % 3]
        uv[:, DIP_idx[4]] = heatmap_peaks[torch.arange(batch_size), DIP_idx[4], Little_score % 3]

        tip_ranking = torch.Tensor([0, 1, 2, 1, 2, 0, 2, 0, 1]).to(device)[None, ...].repeat(batch_size, 1).long()

        uv[:, TIP_idx[0]] = heatmap_peaks[torch.arange(batch_size), TIP_idx[0], tip_ranking[torch.arange(batch_size), Thumb_score]]
        uv[:, TIP_idx[1]] = heatmap_peaks[torch.arange(batch_size), TIP_idx[1], tip_ranking[torch.arange(batch_size), Index_score]]
        uv[:, TIP_idx[2]] = heatmap_peaks[torch.arange(batch_size), TIP_idx[2], tip_ranking[torch.arange(batch_size), Middle_score]]
        uv[:, TIP_idx[3]] = heatmap_peaks[torch.arange(batch_size), TIP_idx[3], tip_ranking[torch.arange(batch_size), Ring_score]]
        uv[:, TIP_idx[4]] = heatmap_peaks[torch.arange(batch_size), TIP_idx[4], tip_ranking[torch.arange(batch_size), Little_score]]
        
        argmax = (uv[:, :, 1] * w + uv[:, :, 0])[..., None]
        return uv.to(device), argmax.to(device)