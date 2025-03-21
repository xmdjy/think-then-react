"""
Throw anything here that is a constant,
or when you want to get rid of thinking about "where should I put this?"
"""

from itertools import pairwise

SMPLX_KEY_SHAPE = {
    'transl': (3), 'global_orient': (1, 3, 3), 'body_pose': (21, 3, 3), 'betas': (10), 'left_hand_pose': (15, 3, 3), 'right_hand_pose': (15, 3, 3), 'jaw_pose': (1, 3, 3), 'leye_pose': (1, 3, 3), 'reye_pose': (1, 3, 3), 'expression': (10)
}

SMPLX_ROTATION_KEYS = ('global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose')

JOINTS3D_22_KINEMATIC_CHAIN = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20]
]

EDGE22_INDICES_UNDIRCTIONAL = []
for chain in JOINTS3D_22_KINEMATIC_CHAIN:
    chain_edges = []
    for i, j in pairwise(chain):
        EDGE22_INDICES_UNDIRCTIONAL.append([i, j])

EDGE22_INDICES = []
for chain in JOINTS3D_22_KINEMATIC_CHAIN:
    chain_edges = []
    for i, j in pairwise(chain):
        chain_edges.extend([(i, j), (j, i)])
    EDGE22_INDICES.extend(chain_edges)
    for i in range(22):
        EDGE22_INDICES.append((i, i))

EDGE_INDEX_INFO = {
    'joints3d_22': EDGE22_INDICES,
    'joints12d_22': EDGE22_INDICES
}

MOTION_REPRESENTATION_INFO = {
    'intergen_262': {
        'feature_size': 262,
        'key_to_range': {
            'pos': [0, 66],
            'vel': [66, 132],
            'rot': [132, 258],
            'foot': [258, 262]
        }
    },

    'joints3d_22': {
        'feature_size': [22, 3]
    },

    'joints12d_22': {
        'feature_size': [22, 12],
        'key_to_range': {
            'pos': [0, 3],
            'vel': [3, 6],
            'rot': [6, 12],
        },
    },

    'foot_indices' : {
        'left': [7, 10],
        'right': [8, 11]
    },

    'tokens': {
        'feature_size': 512
    },

    'tokens_512': {
        'feature_size': 512
    }
}

TEXT_FEATURE_INFO = {
    'google-bert/bert-base-uncased': {
        'feature_size': 768
    },
    'openai/clip-vit-base-patch32': {
        'feature_size': 512
    },
    'openai/clip-vit-large-patch14': {
        'feature_size': 768
    },
}

VALUE_RANGES = {
    'x': [-4.268601281738281, 4.268601281738281],
    'z': [-3.7807260585784914, 4.461718423461914],
    'r': [-3.1416857620286556, 3.1416867933963655]
}

INTERX_LABEL_MAPPING = {
    0: 'Hug', 1: 'Handshake', 2: 'Wave', 3: 'Grab', 4: 'Hit',
    5: 'Kick', 6: 'Posing', 7: 'Push', 8: 'Pull', 9: 'Sit-on-leg',
    10: 'Slap', 11: 'Pat-on-back', 12: 'Point-finger-at', 13: 'Walk-towards', 14: 'Knock-over',
    15: 'Step-on-foot', 16: 'High-five', 17: 'Chase', 18: 'Whisper-in-ear', 19: 'Support-with-hand',
    20: 'Finger-guessing', 21: 'Dance', 22: 'Link-arms', 23: 'Shoulder-to-shoulder', 24: 'Bend',
    25: 'Carry-on-back', 26: 'Massage-shoulder', 27: 'Massage-leg', 28: 'Hand-wrestling', 29: 'Chat',
    30: 'Pat-on-cheek', 31: 'Thumb-up', 32: 'Touch-head', 33: 'Imitate', 34: 'Kiss-on-cheek',
    35: 'Help-up', 36: 'Cover-mouth', 37: 'Look-back', 38: 'Block', 39: 'Fly-kiss'
}

# just suppose
INTERX_FAMILIARITY_MAPPING = {
    1: 'stranger',
    2: 'schoolmate',
    3: 'friend',
    4: 'lover',
}

INTERX_GROUP_TO_FAMILIARITY = {
    1: 1, 2: 4, 3: 1, 4: 4, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1,
    11: 1, 12: 4, 13: 4, 14: 1, 15: 1, 16: 2, 17: 1, 18: 1, 19: 3, 20: 3,
    21: 3, 22: 1, 23: 1, 24: 1, 25: 1, 26: 1, 27: 4, 28: 1, 29: 4, 30: 1,
    31: 2, 32: 4, 33: 4, 34: 3, 35: 4, 36: 2, 37: 1, 38: 3, 39: 1, 40: 1,
    41: 1, 42: 1, 43: 1, 44: 4, 45: 2, 46: 1, 47: 2, 48: 1, 49: 3, 50: 1,
    51: 4, 52: 3, 53: 3, 54: 2, 55: 2, 56: 1, 57: 3, 58: 1, 59: 4
}
