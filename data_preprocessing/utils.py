#%%  joints3d_22 to intergen_262
import sys
import os
import pickle
import copy
import numpy as np
from pathlib import Path
import torch

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/../')
import third_party.HumanML3D.common.quaternion as quat
from src.utils.constants import JOINTS3D_22_KINEMATIC_CHAIN


face_joint_indx = [2, 1, 17, 16]


def normalize_single_joints3d_22(motion):
    motion = copy.copy(motion)
    # put on floor
    floor_height = motion.min(axis=0).min(axis=0)[1]
    motion[:, :, 1] -= floor_height

    # reactor xz at origin
    re_root_pose_init = motion[0]
    re_root_xz_init = re_root_pose_init[0] * np.array([1, 0, 1])
    motion -= re_root_xz_init

    # reactor face Z+
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across = re_root_pose_init[r_hip] - re_root_pose_init[l_hip]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0, 0, 1]])
    root_quat_init = quat.qbetween_np(forward_init, target)
    root_quat_init_for_all = np.ones(motion.shape[:-1] + (4,)) * root_quat_init
    motion = quat.qrot_np(root_quat_init_for_all, motion)

    return motion, (re_root_xz_init[0], re_root_xz_init[2], np.arctan2(forward_init[0, 2], forward_init[0, 0]))


def denormalize_single_joints3d_22(motion, x, z, r):
    motion = copy.copy(motion)
    # recover rotation
    forward_init = np.array([[0, 0, 1]])
    target = np.array([[np.cos(r), 0, np.sin(r)]])
    root_quat_init = quat.qbetween_np(forward_init, target)
    root_quat_init_for_all = np.ones(motion.shape[:-1] + (4,)) * root_quat_init
    motion = quat.qrot_np(root_quat_init_for_all, motion)
    motion += np.array([x, 0, z])
    return motion


def normalize_dual_joints3d_22(action, reaction):
    action = copy.copy(action)
    reaction = copy.copy(reaction)
    # put on floor
    re_floor_height = reaction.min(axis=0).min(axis=0)[1]
    a_floor_height = action.min(axis=0).min(axis=0)[1]

    reaction[:, :, 1] -= re_floor_height
    action[:, :, 1] -= a_floor_height

    # reactor xz at origin
    re_root_pose_init = reaction[0]
    re_root_xz_init = re_root_pose_init[0] * np.array([1, 0, 1])
    reaction -= re_root_xz_init
    action -= re_root_xz_init

    # reactor face Z+
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across = re_root_pose_init[r_hip] - re_root_pose_init[l_hip]
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
    target = np.array([[0, 0, 1]])
    root_quat_init = quat.qbetween_np(forward_init, target)
    root_quat_init_for_all = np.ones(reaction.shape[:-1] + (4,)) * root_quat_init
    reaction = quat.qrot_np(root_quat_init_for_all, reaction)
    action = quat.qrot_np(root_quat_init_for_all, action)

    return action, reaction


def mirror_joints3d_22(motion):
    assert len(motion.shape) == 3
    mirrored = copy.copy(motion)
    mirrored[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    tmp = mirrored[:, right_chain]
    mirrored[:, right_chain] = mirrored[:, left_chain]
    mirrored[:, left_chain] = tmp
    return mirrored


def mirror_text(text: str):
    return text.replace(
        "left", "tmp").replace("right", "left").replace("tmp", "right").replace(
        "clockwise", "tmp").replace("counterclockwise", "clockwise").replace("tmp", "counterclockwise")
