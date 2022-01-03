stb_joints = [
    'loc_bn_palm_L',
    'loc_bn_pinky_L_01',
    'loc_bn_pinky_L_02',
    'loc_bn_pinky_L_03',
    'loc_bn_pinky_L_04',
    'loc_bn_ring_L_01',
    'loc_bn_ring_L_02',
    'loc_bn_ring_L_03',
    'loc_bn_ring_L_04',
    'loc_bn_mid_L_01',
    'loc_bn_mid_L_02',
    'loc_bn_mid_L_03',
    'loc_bn_mid_L_04',
    'loc_bn_index_L_01',
    'loc_bn_index_L_02',
    'loc_bn_index_L_03',
    'loc_bn_index_L_04',
    'loc_bn_thumb_L_01',
    'loc_bn_thumb_L_02',
    'loc_bn_thumb_L_03',
    'loc_bn_thumb_L_04',
]

rhd_joints = [
    'loc_bn_palm_L',
    'loc_bn_thumb_L_04',
    'loc_bn_thumb_L_03',
    'loc_bn_thumb_L_02',
    'loc_bn_thumb_L_01',
    'loc_bn_index_L_04',
    'loc_bn_index_L_03',
    'loc_bn_index_L_02',
    'loc_bn_index_L_01',
    'loc_bn_mid_L_04',
    'loc_bn_mid_L_03',
    'loc_bn_mid_L_02',
    'loc_bn_mid_L_01',
    'loc_bn_ring_L_04',
    'loc_bn_ring_L_03',
    'loc_bn_ring_L_02',
    'loc_bn_ring_L_01',
    'loc_bn_pinky_L_04',
    'loc_bn_pinky_L_03',
    'loc_bn_pinky_L_02',
    'loc_bn_pinky_L_01'
]

snap_joint_names = [
    'loc_bn_palm_L',
    'loc_bn_thumb_L_01',
    'loc_bn_thumb_L_02',
    'loc_bn_thumb_L_03',
    'loc_bn_thumb_L_04',
    'loc_bn_index_L_01',
    'loc_bn_index_L_02',
    'loc_bn_index_L_03',
    'loc_bn_index_L_04',
    'loc_bn_mid_L_01',
    'loc_bn_mid_L_02',
    'loc_bn_mid_L_03',
    'loc_bn_mid_L_04',
    'loc_bn_ring_L_01',
    'loc_bn_ring_L_02',
    'loc_bn_ring_L_03',
    'loc_bn_ring_L_04',
    'loc_bn_pinky_L_01',
    'loc_bn_pinky_L_02',
    'loc_bn_pinky_L_03',
    'loc_bn_pinky_L_04'
]

SNAP_BONES = [
    (0, 1, 2, 3, 4),
    (0, 5, 6, 7, 8),
    (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16),
    (0, 17, 18, 19, 20)
]

SNAP_PARENT = [
    0,  # 0's parent
    0,  # 1's parent
    1,
    2,
    3,
    0,  # 5's parent
    5,
    6,
    7,
    0,  # 9's parent
    9,
    10,
    11,
    0,  # 13's parent
    13,
    14,
    15,
    0,  # 17's parent
    17,
    18,
    19,
]

JOINT_COLORS = (
    (216, 31, 53),
    (214, 208, 0),
    (136, 72, 152),
    (126, 199, 216),
    (0, 0, 230),
)

DEFAULT_CACHE_DIR = 'datasets/data/.cache'

USEFUL_BONE = [1, 2, 3,
               5, 6, 7,
               9, 10, 11,
               13, 14, 15,
               17, 18, 19]

kinematic_tree = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]

ID2ROT = {
        2: 13, 3: 14, 4: 15,
        6: 1, 7: 2, 8: 3,
        10: 4, 11: 5, 12: 6,
        14: 10, 15: 11, 16: 12,
        18: 7, 19: 8, 20: 9,
    }

HAND_MESH_MODEL_PATH = 'hand_mesh/hand_mesh_model.pkl'
# use left hand
OFFICIAL_MANO_PATH = './mano/models/MANO_LEFT.pkl'
IK_UNIT_LENGTH = 0.09473151311686484 # in meter

HAND_COLOR = [228/255, 178/255, 148/255]

# only for rendering
CAM_FX = 620.744
CAM_FY = 621.151

KP_INDEX = list(range(0, 61, 3))
PAF_INDEX = [x for x in range(61) if x not in KP_INDEX]

THRESH_VECTOR_SCORE = 0.01
THRESH_HEAT = 0.01
NUM_SAMPLE = 5
THRESH_PAF_CNT = NUM_SAMPLE * 0.4

JOINT_PAIR = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

SHAPE_MEAN = [ 0.15910966,  1.24914071, -0.64187749,  0.5400079,  -3.29426494, -0.83807857,
  0.31873315, -0.22016137,  1.33633995,  0.83711511]

SHAPE_STD = [0.99800293, 1.66176237, 1.40137928, 1.46973533, 2.23193049, 1.57620704,
 1.62400539, 2.8165403,  1.76102397, 1.30528381]

yolo_hyper = {
    'giou': 3.54,  # giou loss gain
    'cls': 37.4,  # cls loss gain
    'cls_pw': 1.0,  # cls BCELoss positive_weight
    'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
    'obj_pw': 1.0,  # obj BCELoss positive_weight
    'iou_t': 0.20,  # iou training threshold
    'lr0': 0.001,  # initial learning rate (SGD=5E-3, Adam=5E-4)
    'lrf': 0.0005,  # final learning rate (with cos scheduler)
    'momentum': 0.937,  # SGD momentum
    'weight_decay': 0.0005,  # optimizer weight decay
    'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
}