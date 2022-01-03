import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import cv2

def plot3d(joints_, ax, title=None):
    if len(joints_) != 21:
        return ax
    joints = joints_.copy()
    ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], 'yo', label='keypoint')

    ax.plot(joints[:5, 0], joints[:5, 1],
             joints[:5, 2],
             'r',
             label='thumb')

    ax.plot(joints[[0, 5, 6, 7, 8, ], 0], joints[[0, 5, 6, 7, 8, ], 1],
             joints[[0, 5, 6, 7, 8, ], 2],
             'm',
             label='index')
    ax.plot(joints[[0, 9, 10, 11, 12, ], 0], joints[[0, 9, 10, 11, 12], 1],
             joints[[0, 9, 10, 11, 12], 2],
             'b',
             label='middle')
    ax.plot(joints[[0, 13, 14, 15, 16], 0], joints[[0, 13, 14, 15, 16], 1],
             joints[[0, 13, 14, 15, 16], 2],
             'c',
             label='ring')
    ax.plot(joints[[0, 17, 18, 19, 20], 0], joints[[0, 17, 18, 19, 20], 1],
             joints[[0, 17, 18, 19, 20], 2],
             'g',
             label='pinky')
    # snap convention
    ax.plot(joints[4][0], joints[4][1], joints[4][2], 'rD', label='thumb')
    ax.plot(joints[8][0], joints[8][1], joints[8][2], 'ro', label='index')
    ax.plot(joints[12][0], joints[12][1], joints[12][2], 'ro', label='middle')
    ax.plot(joints[16][0], joints[16][1], joints[16][2], 'ro', label='ring')
    ax.plot(joints[20][0], joints[20][1], joints[20][2], 'ro', label='pinky')
    # plt.plot(joints [1:, 0], joints [1:, 1], joints [1:, 2], 'o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.set_xlim(xmin=-1.0,xmax=1.0)
    #ax.set_ylim(ymin=-1.0,ymax=1.0)
    #ax.set_zlim(zmin=-1.0,zmax=1.0)
    # plt.legend()
    # ax.view_init(330, 110)
    ax.view_init(-90, -90)
    return ax

def visualize_2d(ax,
                 hand_joints=None,
                 box=None,
                 links=[(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                        (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]):
    # box : xyxy form
    ax.axis('off')
    if hand_joints is not None:
        visualize_joints_2d(ax, hand_joints, joint_idxs=False, links=links)
    if box is not None:
        if box.shape == (4,):
            top = np.zeros((2, 2))
            top[0] = np.array([box[0], box[1]])
            top[1] = np.array([box[2], box[1]])
            plt.plot(top[:, 0].astype(int), top[:, 1].astype(int), 'b')

            left = np.zeros((2, 2))
            left[0] = np.array([box[0], box[1]])
            left[1] = np.array([box[0], box[3]])
            plt.plot(left[:, 0].astype(int), left[:, 1].astype(int), 'b')

            right = np.zeros((2, 2))
            right[0] = np.array([box[2], box[1]])
            right[1] = np.array([box[2], box[3]])
            plt.plot(right[:, 0].astype(int), right[:, 1].astype(int), 'b')

            bottom = np.zeros((2, 2))
            bottom[0] = np.array([box[0], box[3]])
            bottom[1] = np.array([box[2], box[3]])
            plt.plot(bottom[:, 0].astype(int), bottom[:, 1].astype(int), 'b')
        else: # shape == (5,)
            if box[0] == 0:
                c = 'r' # right hand
            else:
                c = 'b' # left hand
            top = np.zeros((2, 2))
            top[0] = np.array([box[1], box[2]])
            top[1] = np.array([box[3], box[2]])
            ax.plot(top[:, 0].astype(int), top[:, 1].astype(int), c)

            left = np.zeros((2, 2))
            left[0] = np.array([box[1], box[2]])
            left[1] = np.array([box[1], box[4]])
            ax.plot(left[:, 0].astype(int), left[:, 1].astype(int), c)

            right = np.zeros((2, 2))
            right[0] = np.array([box[3], box[2]])
            right[1] = np.array([box[3], box[4]])
            ax.plot(right[:, 0].astype(int), right[:, 1].astype(int), c)

            bottom = np.zeros((2, 2))
            bottom[0] = np.array([box[1], box[4]])
            bottom[1] = np.array([box[3], box[4]])
            ax.plot(bottom[:, 0].astype(int), bottom[:, 1].astype(int), c)
    return ax

def visualize_joints_2d(ax,
                        joints,
                        joint_idxs=True,
                        links=None,
                        alpha=1,
                        scatter=True,
                        linewidth=2):
    if links is None:
        links = [(0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                 (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)]
    # Scatter hand joints on image
    x = joints[:, 0]
    y = joints[:, 1]
    if scatter:
        ax.scatter(x, y, 1, 'r')

    # Add idx labels to joints
    for row_idx, row in enumerate(joints):
        if joint_idxs:
            plt.annotate(str(row_idx), (row[0], row[1]))
    _draw2djoints(ax, joints, links, alpha=alpha, linewidth=linewidth)
    ax.axis('equal')


def _draw2djoints(ax, annots, links, alpha=1, linewidth=1):
    colors = ['r', 'm', 'b', 'c', 'g']

    for finger_idx, finger_links in enumerate(links):
        for idx in range(len(finger_links) - 1):
            _draw2dseg(
                ax,
                annots,
                finger_links[idx],
                finger_links[idx + 1],
                c=colors[finger_idx],
                alpha=alpha,
                linewidth=linewidth)


def _draw2dseg(ax, annot, idx1, idx2, c='r', alpha=1, linewidth=1):
    ax.plot([annot[idx1, 0], annot[idx2, 0]], [annot[idx1, 1], annot[idx2, 1]],
            c=c,
            alpha=alpha,
            linewidth=linewidth)

def CVplot2D(img, kp2d, box=None):
    color = [(255, 0, 0), (199,21,133), (0, 0, 255), (0, 255, 255), (0, 255, 0)]
    kp2d = kp2d.astype(int)
    for kp in kp2d:
        kp = tuple(map(tuple, kp))

        # plot joints
        for i in range(21):
            cv2.circle(img, kp[i], 1, (0, 0, 255), -1)

        # plot skeletons
        for i in range(5):
            cv2.line(img, kp[0], kp[i*4+1], color[i], 1)
            cv2.line(img, kp[i*4+1], kp[i*4+2], color[i], 1)
            cv2.line(img, kp[i*4+2], kp[i*4+3], color[i], 1)
            cv2.line(img, kp[i*4+3], kp[i*4+4], color[i], 1)

    if box is not None:
        for b in box:
            left = b[5]
            b = b.astype(int)
            b = tuple(map(tuple, b[:4].reshape(2, 2)))
            if left:
                cv2.rectangle(img, b[0], b[1], (0, 0, 255), 1)
            else:
                cv2.rectangle(img, b[0], b[1], (255, 0, 0), 1)
    return img
