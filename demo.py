import os
import sys
import argparse
import time
import cv2
import numpy as np
import torch
from einops import rearrange
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from manopth.manolayer import ManoLayer
from yolov3.utils.datasets import letterbox

import config
from utils.smoother import OneEuroFilter
from utils.vis import plot3d, visualize_2d, CVplot2D
from utils import func
from model.hand import Hand
from render.o3d_render import o3d_render

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_result(img, results, scale=(1, 1), padding=(0, 0)):
    uv = func.to_numpy(results['uv'])
    xyz = func.to_numpy(results['xyz'][0])
    box = results['box']
    
    '''
    fig = plt.figure(figsize=(6, 3))
    ax1 = fig.add_subplot(121)
    plt.imshow(img)
    ax1 = visualize_2d(ax1, uv, box)

    ax2 = fig.add_subplot(122, projection='3d')
    #print(xyz.shape)
    ax2 = plot3d(xyz, ax2)

    #render_result = results['render']
    #ax3 = fig.add_subplot(133)
    #plt.imshow(render_result)

    width, height = fig.get_size_inches() * fig.get_dpi()
    canvas = FigureCanvas(fig)
    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(int(height), int(width), 3)
    image = cv2.resize(image, (640, 320))
    
    plt.close()
    '''
    h, w, _ = img.shape
    uv[:, :, 0] = uv[:, :, 0] - padding[0]
    uv[:, :, 1] = uv[:, :, 1] - padding[1]

    box[:, [0, 2]] = box[:, [0, 2]] - padding[0]
    box[:, [1, 3]] = box[:, [1, 3]] - padding[1]

    uv[:, :, 0] = (np.clip(uv[:, :, 0], 0, w-1)) / scale[0]
    uv[:, :, 1] = (np.clip(uv[:, :, 1], 0, h-1)) / scale[1]

    box[:, [0, 2]] = (np.clip(box[:, [0, 2]], 0, w-1)) / scale[0]
    box[:, [1, 3]] = (np.clip(box[:, [1, 3]], 0, w-1)) / scale[1]

    img2d = CVplot2D(img, uv, box).astype(np.uint8)
 
    return cv2.cvtColor(img2d, cv2.COLOR_RGB2BGR)

def demo(capture, args):
    ret, frame_large = capture.read()
    if ret:
        img, ratio, (dw, dh) = letterbox(frame_large, new_shape=args.img_size)
        h, w, _ = img.shape
        args.h = h
        args.w = w
        center = np.array([w/2, h/2])
        orig_shape = frame_large.shape

        render = o3d_render(orig_shape, args.max)
        extrinsic = render.extrinsic
        extrinsic[0:4, 3] = 0
        
        render.extrinsic = extrinsic
        render.intrinsic = [config.CAM_FX,config.CAM_FY]
        render.updata_params()
        render.environments('render/render_option.json', 1000)

        mesh_x = orig_shape[1] * 1.
        mesh_y = orig_shape[0] * 1.
        mesh_img_size = np.array([mesh_x, mesh_y])

        if args.max == 2:
            right_mesh_smoother = OneEuroFilter(4.0, 0.0)
            left_mesh_smoother = OneEuroFilter(4.0, 0.0)
        elif args.max == 1:
            mesh_smoother = OneEuroFilter(4.0, 0.0)

        if args.video:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output.avi', fourcc, 30.0, (orig_shape[1]*2, orig_shape[0]))
    else:
        return
    
    model = Hand(args).to(device).eval()
    left_box = []
    right_box = []
    while True:
        ret, frame = capture.read()
        if ret is False:
            break
        
        hands = [False] * args.max
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_large = frame.copy()
        
        img, ratio, (dw, dh) = letterbox(frame_large, new_shape=args.img_size)
        img_mr = cv2.flip(img, 1)
        img_mr = img_mr / 255.

        img = img / 255.
        img = torch.from_numpy(img).permute(2, 0, 1).to(device, dtype=torch.float)[None, ...]
        img_mr = torch.from_numpy(img_mr).permute(2, 0, 1).to(device, dtype=torch.float)[None, ...]
        img = torch.cat([img, img_mr], dim=0)

        results = model(img)

        if results is None:
            ret = frame_large[:, :, ::-1]
            for ii in range(args.max):
                render.rendering(None, config.HAND_COLOR, ii)
            render_img = render.capture_img()
            render_img = render_img[:ret.shape[0], :ret.shape[1]]
            ret = np.hstack((ret, render_img))
            cv2.imshow('aa', ret)
            if cv2.waitKey(1) == ord('q'):
                break
            continue
        
        verts = results['verts']        
        verts_mean = torch.mean(verts, dim=1)
        verts_mean = verts_mean[:, None, :].repeat(1, 778, 1)
        verts -= verts_mean

        # for better visualization
        verts *= 2
        verts = func.to_numpy(verts)

        # offset and scale
        init_pos = np.array([0., 0., 900.])
        box = results['box']
        box = func.to_numpy(box)
        results['box'] = box
        
        right_hand_index = box[:, 5] == 0
        left_hand_index = box[:, 5] == 1
        
        box_center = (box[:, [0, 1]] + box[:, [2, 3]]) / 2
        box_offset = center - box_center
        box_offset *= 2
       
        for ii, vert in enumerate(verts):
            vert_offset = init_pos.copy()
            vert_offset[:2] -= box_offset[ii]
            vert += vert_offset

            # reverse mesh x axis if left hand
            if left_hand_index[ii]:
                wrist = vert[0, 0]
                vert[:, 0] = -vert[:, 0]
                offset = wrist - vert[0, 0]
                vert[:, 0] += offset
            
            # smoothing
            if args.max == 1:
                vert = mesh_smoother.process(vert)
            elif args.max == 2 and left_hand_index[ii]:
                vert = left_mesh_smoother.process(vert)
            elif args.max == 2 and not left_hand_index[ii]:
                vert = right_mesh_smoother.process(vert)

            render.rendering(vert, config.HAND_COLOR, ii)
            hands[ii] = True

        # rendering
        for ii in range(args.max):
            if not hands[ii]:
                render.rendering(None, config.HAND_COLOR, ii)
        render_img = render.capture_img()

        ret = plot_result(frame_large, results, scale=ratio, padding=(dw, dh))
        render_img = render_img[:ret.shape[0], :ret.shape[1]]
        ret = np.hstack((ret, render_img))
        cv2.imshow('aa', ret)
        
        if cv2.waitKey(1) == ord('q'):
            break
        if args.video:
            out.write(ret)

    # Release everything if job is finished
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov3/yolov3/cfg/yolo-fpn.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='yolov3/yolov3/data/hand.names', help='*.names path')
    parser.add_argument('--img-size', type=int, default=320, help='inference size (pixels)')
    parser.add_argument('--hm-size', type=int, default=32)
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--max', type=int, default=2, help='maximum number of hand to detect')
    parser.add_argument('--detnet', type=str, default='checkpoints/detnet_demo.pth', help='yolo hand path')
    parser.add_argument('--iknet', type=str, default='checkpoints/iknet_demo.pth', help='iknet path')
    parser.add_argument('--video', type=str, help='video path')
    parser.add_argument('--paf', action='store_true', help='use heatmaps and PAF maps to estimate 2D keypoints')
    args = parser.parse_args()
    
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        cap = cv2.VideoCapture(0)
    demo(cap, args)
    
    
    

