# Copyright (c) Lixin YANG, Jiasen Li. All Rights Reserved.
import torch
import numpy as np


def gen_heatmap(img, pt, sigma):
    """generate heatmap based on pt coord.

    :param img: original heatmap, zeros
    :type img: np (H,W) float32
    :param pt: keypoint coord.
    :type pt: np (2,) int32
    :param sigma: guassian sigma
    :type sigma: float
    :return
    - generated heatmap, np (H, W) each pixel values id a probability
    - flag 0 or 1: indicate wheather this heatmap is valid(1)

    """

    pt = pt.astype(np.int32)
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (
            ul[0] >= img.shape[1]
            or ul[1] >= img.shape[0]
            or br[0] < 0
            or br[1] < 0
    ):
        # If not, just return the image as is
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print(pt)
        return img, 0

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img, 1


def get_heatmap_pred(heatmaps):
    """ get predictions from heatmaps in torch Tensor
        return type: torch.LongTensor
    """
    assert heatmaps.dim() == 4, 'Score maps should be 4-dim (B, nJoints, H, W)'
    maxval, idx = torch.max(heatmaps.view(heatmaps.size(0), heatmaps.size(1), -1), 2)

    maxval = maxval.view(heatmaps.size(0), heatmaps.size(1), 1)
    idx = idx.view(heatmaps.size(0), heatmaps.size(1), 1)

    preds = idx.repeat(1, 1, 2).float()  # (B, njoint, 2)

    preds[:, :, 0] = (preds[:, :, 0]) % heatmaps.size(3)  # + 1
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / heatmaps.size(3))  # + 1

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds

def putGaussianMaps(center, sigma, grid_y=32, grid_x=32, stride=1):
    start = stride / 2.0 - 0.5
    y_range = [i for i in range(int(grid_y))]
    x_range = [i for i in range(int(grid_x))]
    xx, yy = np.meshgrid(x_range, y_range)
    xx = xx * stride + start
    yy = yy * stride + start
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    exponent = d2 / 2.0 / sigma / sigma
    mask = exponent <= 4.6052
    cofid_map = np.exp(-exponent)
    cofid_map = np.multiply(mask, cofid_map)
    accumulate_confid_map = cofid_map
    accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0
    
    return accumulate_confid_map

def batch_keypoints_to_heatmap(keypoints, rois, heatmap_size):
    # B*21*2, B*4 
    # type: (Tensor, Tensor, int) -> Tuple[Tensor, Tensor]
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    valid = (valid_loc).long()
    #vis = keypoints[..., 2] > 0
    #valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid
    # 1D -> 2D (num_kp->num_kp*2)
    u = (heatmaps % heatmap_size)[..., None]
    v = (heatmaps // heatmap_size)[..., None]
    heatmaps = torch.cat((u, v), dim=-1)#.permute(0, 2, 1)        

    return heatmaps, valid

def keypoints_to_heatmap(keypoints, box, heatmap_size):
    # box: xyxy form
    #outside_img_idx1 = keypoints < 0
    #gg = keypoints[outside_img_idx1]
    ##if len(gg) > 0:
    #   print(keypoints)
    offset_x = box[0, None]
    offset_y = box[1, None]
    scale_x = heatmap_size / (box[2, None] - box[0, None])
    scale_y = heatmap_size / (box[3, None] - box[1, None])

    #offset_x = offset_x[:, None]
    #offset_y = offset_y[:, None]
    #scale_x = scale_x[:, None]
    #scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    #x_boundary_inds = x > box[2, None]#[:, None]
    #y_boundary_inds = y > box[3, None]#[:, None]

    x = (x - offset_x) * scale_x
    x = np.floor(x).astype(np.long)
    y = (y - offset_y) * scale_y
    y = np.floor(y).astype(np.long)

    #x[x_boundary_inds] = heatmap_size - 1
    #y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    valid = valid_loc.astype(np.long)
    #vis = keypoints[..., 2] > 0
    #valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind# * valid
    # 1D -> 2D (num_kp->num_kp*2)
    u = (heatmaps % heatmap_size)[..., None]
    v = (heatmaps // heatmap_size)[..., None]
    heatmaps = np.concatenate((u, v), axis=-1)#.permute(0, 2, 1)

    invalid_idx = heatmaps < 0
    heatmaps[invalid_idx] = 0

    invalid_idx = heatmaps > heatmap_size - 1
    heatmaps[invalid_idx] = heatmap_size - 1     

    return heatmaps, valid

# a(parent) -> b(child)
def putVecMaps(centerA, centerB, grid_y, grid_x, stride=1):
    centerA = centerA.astype(float)
    centerB = centerB.astype(float)
    count = np.zeros((grid_y, grid_x))
    vec_map = np.zeros((grid_y, grid_x, 2))

    thre = 1  # limb width
    centerB = centerB / stride
    centerA = centerA / stride

    limb_vec = centerB - centerA
    norm = np.linalg.norm(limb_vec)
    if (norm < 1e-12):
        # print 'limb is too short, ignore it...'
        return np.transpose(vec_map, (2, 0, 1)), count[None, ...]
    limb_vec_unit = limb_vec / norm
    # print 'limb unit vector: {}'.format(limb_vec_unit)

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # mask is 2D

    #print(vec_map[yy, xx].shape)
    try:
        vec_map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
    except:
        print(min_x, max_x, min_y, max_y)
        print(centerA, centerB)
        raise RuntimeError("vecmap")
    #print(vec_map[yy, xx])
    vec_map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :]

    mask = np.logical_or.reduce(
        (np.abs(vec_map[:, :, 0]) > 0, np.abs(vec_map[:, :, 1]) > 0))

    count[mask == True] += 1

    mask = count == 0

    count[mask == True] = 1

    vec_map = np.divide(vec_map, count[:, :, np.newaxis])
    count[mask == True] = 0

    return np.transpose(vec_map, (2, 0, 1)), count[None, ...]

if __name__ == '__main__':
    keypoints = np.array([[5, 7], [7, 10], [10, 13]])
    batch_keypoints = torch.Tensor([[[5, 7], [7, 10]]])
    batch_box = torch.Tensor([[3, 4, 9, 12]])
    box = np.array([3, 4, 9, 12])
    res, _ = keypoints_to_heatmap(keypoints, box, 32)
    res2, _ = batch_keypoints_to_heatmap(batch_keypoints, batch_box, 32)
    print(res)
    print(res2)


