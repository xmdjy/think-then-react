"""
Refer to [HumanML3D](https://github.com/EricGuo5513/HumanML3D)
and [InterGen](https://github.com/tr3e/InterGen)
"""

import copy
from typing import Union
import argparse
import numpy as np
import torch

import third_party.HumanML3D.common.quaternion as quat
import third_party.HumanML3D.paramUtil as hparam
from third_party.HumanML3D.common.skeleton import Skeleton


class MotionRepresentationConverter:
    # Actually, you only need to focus on joints3d_22 and intergen_262, we do not use humanml3d_263
    def __init__(self):
        self.tgt_offsets = None
        self.n_raw_offsets = None
        self.face_joint_indx = [2, 1, 17, 16]
    
    def tokenize_value(self, value_range, num_bins, value):
        if value < value_range[0] or value > value_range[1]:
            raise ValueError(f'{value} is not in the range of {value_range}')
        bin_width = (value_range[1] - value_range[0]) / num_bins
        bin_index = int((value - value_range[0]) / bin_width)
        if value == value_range[1]:
            bin_index = num_bins - 1
        return bin_index
    
    def detokenize_value(self, value_range, num_bins, bin_index):
        bin_width = (value_range[1] - value_range[0]) / num_bins

        # Calculate the center of the bin
        bin_center = value_range[0] + (bin_index + 0.5) * bin_width

        return bin_center

    def norm_joint3d_22(self, x):
        joints3d_22 = copy.copy(x)
        floor_height = joints3d_22.min(axis=0).min(axis=0)[1]
        joints3d_22[:, :, 1] -= floor_height

        # reactor xz at origin
        re_root_pose_init = joints3d_22[0]
        re_root_xz_init = re_root_pose_init[0] * np.array([1, 0, 1])
        joints3d_22 -= re_root_xz_init

        # reactor face Z+
        r_hip, l_hip, sdr_r, sdr_l = self.face_joint_indx
        across = re_root_pose_init[r_hip] - re_root_pose_init[l_hip]
        across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
        forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
        forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
        target = np.array([[0, 0, 1]])
        root_quat_init = quat.qbetween_np(forward_init, target)
        root_quat_init_for_all = np.ones(joints3d_22.shape[:-1] + (4,)) * root_quat_init
        joints3d_22 = quat.qrot_np(root_quat_init_for_all, joints3d_22)
        return joints3d_22, (re_root_xz_init[0], re_root_xz_init[2], np.arctan2(forward_init[0, 0], forward_init[0, 2]))

    def norm_dual_joints3d_22(self, action, reaction):
        action = copy.copy(action)
        reaction = copy.copy(reaction)
        # put on floor
        floor_height = reaction.min(axis=0).min(axis=0)[1]
        reaction[:, :, 1] -= floor_height
        action[:, :, 1] -= floor_height

        # reactor xz at origin
        re_root_pose_init = reaction[0]
        re_root_xz_init = re_root_pose_init[0] * np.array([1, 0, 1])
        reaction -= re_root_xz_init
        action -= re_root_xz_init

        # reactor face Z+
        r_hip, l_hip, sdr_r, sdr_l = self.face_joint_indx
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

    def unnorm_joints3d_22(self, motion, x, z, r):
        motion = copy.copy(motion)
        # recover rotation
        forward_init = np.array([[0, 0, 1]])
        target = np.array([[np.cos(r), 0, np.sin(r)]])
        root_quat_init = quat.qbetween_np(forward_init, target)
        root_quat_init_for_all = np.ones(motion.shape[:-1] + (4,)) * root_quat_init
        motion = quat.qrot_np(root_quat_init_for_all, motion)
        motion += np.array([x, 0, z])
        return motion

    def init_h263_data(self):
        n_raw_offsets = torch.from_numpy(hparam.t2m_raw_offsets)
        self.n_raw_offsets = n_raw_offsets
        example_data = np.load('data/tgt_offsets.npy')
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        tgt_skel = Skeleton(n_raw_offsets, hparam.t2m_kinematic_chain, 'cpu')
        # (joints_num, 3)
        tgt_offsets = tgt_skel.get_offsets_joints(example_data[:, 0, :])
        self.tgt_offsets = tgt_offsets
    
    def get_convert_func(self, src, tgt):
        match (src, tgt):
            case ('j3d', 'i262'):
                return self.joints3d_22_to_intergen_262
            case ('i262', 'j3d'):
                return self.intergen_262_to_joints3d_22

            case ('j3d', 'j12d'):
                return self.joints3d_22_to_joints12d_22
            case ('j12d', 'j3d'):
                return self.joints12d_22_to_joints3d_22
            
            case ('j3d', 'h263'):
                return self.joints3d_22_to_humanml3d_263
            case ('h263', 'j3d'):
                return self.humanm3d_263_to_joints3d_22
            
            case ('j12d', 'i262'):
                return self.joints12d_22_to_intergen_262
            
            case _:
                raise ValueError(f'Conversion from {src} to {tgt} is not supported now.')
        
    @torch.no_grad()
    def convert(self, src: str, tgt: str, motion: Union[np.ndarray, torch.Tensor], **kwargs):
        func = self.get_convert_func(src, tgt)
        if 'h263' in src + tgt and self.tgt_offsets is None:
            self.init_h263_data()
        
        return func(motion, **kwargs)
    
    def __call__(self, src: str, tgt: str, motion: Union[np.ndarray, torch.Tensor], **kwargs):
        return self.convert(src=src, tgt=tgt, motion=motion, **kwargs)

    def joints3d_22_to_intergen_262(self, joints3d_22, feet_thre=0.002, norm=False):
        is_input_torch = False
        raw_device = 'cpu'
        if isinstance(joints3d_22, torch.Tensor):
            is_input_torch = True
            raw_device = joints3d_22.device
            joints3d_22 = joints3d_22.detach().cpu().numpy()

        l_idx1, l_idx2 = 5, 8
        fid_r, fid_l = [8, 11], [7, 10]
        r_hip, l_hip = 2, 1
        n_raw_offsets = torch.from_numpy(hparam.t2m_raw_offsets)
        kinematic_chain = hparam.t2m_kinematic_chain
        if norm:
            # put on floor
            floor_height = joints3d_22.min(axis=0).min(axis=0)[1]
            joints3d_22[:, :, 1] -= floor_height

            # reactor xz at origin
            re_root_pose_init = joints3d_22[0]
            re_root_xz_init = re_root_pose_init[0] * np.array([1, 0, 1])
            joints3d_22 -= re_root_xz_init

            # reactor face Z+
            r_hip, l_hip, sdr_r, sdr_l = self.face_joint_indx
            across = re_root_pose_init[r_hip] - re_root_pose_init[l_hip]
            across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
            forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
            forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]
            target = np.array([[0, 0, 1]])
            root_quat_init = quat.qbetween_np(forward_init, target)
            root_quat_init_for_all = np.ones(joints3d_22.shape[:-1] + (4,)) * root_quat_init
            joints3d_22 = quat.qrot_np(root_quat_init_for_all, joints3d_22)

        # get rotation
        def get_cont6d(positions):
            skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
            quat_params = skel.inverse_kinematics_np(positions, self.face_joint_indx, smooth_forward=True)
            return quat.quaternion_to_cont6d_np(quat_params)[:, 1:, ...]  # discard pelvis rotation
        re_rot = get_cont6d(joints3d_22)

        # get foot contacts
        def foot_detect(positions, thres):
            velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])
            # left foot
            feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
            feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
            feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
            feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)
            # right foot
            feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
            feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
            feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
            feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
            return feet_l, feet_r
        re_feet_l, re_feet_r = foot_detect(joints3d_22, feet_thre)
        re_feet_l = np.concatenate([re_feet_l, re_feet_l[-1:, ...]], axis=0)
        re_feet_r = np.concatenate([re_feet_r, re_feet_r[-1:, ...]], axis=0)

        # get velocity
        re_joint_vels = joints3d_22[1:] - joints3d_22[:-1]
        re_joint_vels = np.concatenate([re_joint_vels, re_joint_vels[-1:, ...]], axis=0)

        # compact data
        n_frames = joints3d_22.shape[0]
        intergen_262 = np.concatenate([
            joints3d_22.reshape(n_frames, -1),
            re_joint_vels.reshape(n_frames, -1),
            re_rot.reshape(n_frames, -1),
            re_feet_l, re_feet_r
        ], axis=-1)

        if is_input_torch:
            intergen_262 = torch.from_numpy(intergen_262).to(device=raw_device)

        return intergen_262

    def intergen_262_to_joints3d_22(self, intergen_262):
        shape = intergen_262.shape
        return intergen_262[..., :22*3].reshape(*shape[:-1], 22, 3)

    def uniform_skeleton(self, positions):
        l_idx1, l_idx2 = 5, 8
        fid_r, fid_l = [8, 11], [7, 10]
        face_joint_indx = [2, 1, 17, 16]
        r_hip, l_hip = 2, 1
        joints_num = 22

        src_skel = Skeleton(self.n_raw_offsets, hparam.t2m_kinematic_chain, 'cpu')
        src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
        src_offset = src_offset.numpy()
        tgt_offset = self.tgt_offsets.numpy()
        # print(src_offset)
        # print(tgt_offset)
        '''Calculate Scale Ratio as the ratio of legs'''
        src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
        tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

        scale_rt = tgt_leg_len / src_leg_len
        # print(scale_rt)
        src_root_pos = positions[:, 0]
        tgt_root_pos = src_root_pos * scale_rt

        '''Inverse Kinematics'''
        quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
        # print(quat_params.shape)

        '''Forward Kinematics'''
        src_skel.set_offset(self.tgt_offsets)
        new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
        return new_joints

    def joints3d_22_to_humanml3d_263(self, positions, feet_thre, norm=False):
        if norm:
            fid_r, fid_l = [8, 11], [7, 10]
            face_joint_indx = [2, 1, 17, 16]
            r_hip, l_hip = 2, 1

            n_raw_offsets = torch.from_numpy(hparam.t2m_raw_offsets)
            kinematic_chain = hparam.t2m_kinematic_chain

            '''Uniform Skeleton'''
            positions = self.uniform_skeleton(positions, self.tgt_offsets)

            '''Put on Floor'''
            floor_height = positions.min(axis=0).min(axis=0)[1]
            positions[:, :, 1] -= floor_height
            #     print(floor_height)

            '''XZ at origin'''
            root_pos_init = positions[0]
            root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
            positions = positions - root_pose_init_xz

            '''All initially face Z+'''
            r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
            across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
            across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
            across = across1 + across2
            across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

            # forward (3,), rotate around y-axis
            forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
            # forward (3,)
            forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

            target = np.array([[0, 0, 1]])
            root_quat_init = quat.qbetween_np(forward_init, target)
            root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init
            positions = quat.qrot_np(root_quat_init, positions)

        '''New ground truth positions'''
        global_positions = positions.copy()

        """ Get Foot Contacts """

        def foot_detect(positions, thres):
            velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

            feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
            feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
            feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
            #     feet_l_h = positions[:-1,fid_l,1]
            #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
            feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

            feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
            feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
            feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
            #     feet_r_h = positions[:-1,fid_r,1]
            #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
            feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
            return feet_l, feet_r
        #
        feet_l, feet_r = foot_detect(positions, feet_thre)
        # feet_l, feet_r = foot_detect(positions, 0.002)

        '''Quaternion and Cartesian representation'''
        r_rot = None

        def get_rifke(positions):
            '''Local pose'''
            positions[..., 0] -= positions[:, 0:1, 0]
            positions[..., 2] -= positions[:, 0:1, 2]
            '''All pose face Z+'''
            positions = quat.qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
            return positions

        def get_cont6d_params(positions):
            skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
            # (seq_len, joints_num, 4)
            quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

            '''Quaternion to continuous 6D'''
            cont_6d_params = quat.quaternion_to_cont6d_np(quat_params)
            # (seq_len, 4)
            r_rot = quat_params[:, 0].copy()
            #     print(r_rot[0])
            '''Root Linear Velocity'''
            # (seq_len - 1, 3)
            velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
            #     print(r_rot.shape, velocity.shape)
            velocity = quat.qrot_np(r_rot[1:], velocity)
            '''Root Angular Velocity'''
            # (seq_len - 1, 4)
            r_velocity = quat.qmul_np(r_rot[1:], quat.qinv_np(r_rot[:-1]))
            # (seq_len, joints_num, 4)
            return cont_6d_params, r_velocity, velocity, r_rot

        cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
        if norm:
            positions = get_rifke(positions)

        '''Root height'''
        root_y = positions[:, 0, 1:2]

        '''Root rotation and linear velocity'''
        # (seq_len-1, 1) rotation velocity along y-axis
        # (seq_len-1, 2) linear velovity on xz plane
        r_velocity = np.arcsin(r_velocity[:, 2:3])
        l_velocity = velocity[:, [0, 2]]
        #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
        root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

        '''Get Joint Rotation Representation'''
        # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
        rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

        '''Get Joint Rotation Invariant Position Represention'''
        # (seq_len, (joints_num-1)*3) local joint position
        ric_data = positions[:, 1:].reshape(len(positions), -1)

        '''Get Joint Velocity Representation'''
        # (seq_len-1, joints_num*3)
        local_vel = quat.qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                            global_positions[1:] - global_positions[:-1])
        local_vel = local_vel.reshape(len(local_vel), -1)

        data = root_data
        data = np.concatenate([data, ric_data[:-1]], axis=-1)
        data = np.concatenate([data, rot_data[:-1]], axis=-1)
        data = np.concatenate([data, local_vel], axis=-1)
        data = np.concatenate([data, feet_l, feet_r], axis=-1)

        return data

    def humanm3d_263_to_joints3d_22(self, h263):
        def recover_root_rot_pos(data):
            rot_vel = data[..., 0]
            r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
            '''Get Y-axis rotation from rotation velocity'''
            r_rot_ang[..., 1:] = rot_vel[..., :-1]
            r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

            r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
            r_rot_quat[..., 0] = torch.cos(r_rot_ang)
            r_rot_quat[..., 2] = torch.sin(r_rot_ang)

            r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
            r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
            '''Add Y-axis rotation to root position'''
            r_pos = quat.qrot(quat.qinv(r_rot_quat), r_pos)

            r_pos = torch.cumsum(r_pos, dim=-2)

            r_pos[..., 1] = data[..., 3]
            return r_rot_quat, r_pos

        r_rot_quat, r_pos = recover_root_rot_pos(h263)
        positions = h263[..., 4: 21 * 3 + 4]
        positions = positions.view(positions.shape[:-1] + (-1, 3))

        '''Add Y-axis rotation to local joints'''
        positions = quat.qrot(quat.qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

        '''Add root XZ to joints'''
        positions[..., 0] += r_pos[..., 0:1]
        positions[..., 2] += r_pos[..., 2:3]

        '''Concate root and joints'''
        positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

        return positions

    def joints3d_22_to_joints12d_22(self, joints3d_22):
        is_input_torch = False
        raw_device = 'cpu'
        if isinstance(joints3d_22, torch.Tensor):
            is_input_torch = True
            raw_device = joints3d_22.device
            joints3d_22 = joints3d_22.detach().cpu().numpy()

        face_joint_indx = [2, 1, 17, 16]
        n_raw_offsets = torch.from_numpy(hparam.t2m_raw_offsets)
        kinematic_chain = hparam.t2m_kinematic_chain

        # get rotation
        def get_cont6d(positions):
            skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
            quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)
            return quat.quaternion_to_cont6d_np(quat_params)  # keep pelvis rot
        rot6d = get_cont6d(joints3d_22)
        joint_vels = joints3d_22[1:] - joints3d_22[:-1]
        joint_vels = np.concatenate([joint_vels, joint_vels[-1:, :, :]], axis=0)  # copy the last frame's vel

        res = np.concatenate([joints3d_22, joint_vels, rot6d], axis=-1)
        if is_input_torch:
            res = torch.from_numpy(res).to(device=raw_device)
        return res

    def joints12d_22_to_joints3d_22(self, joints12d_22):
        return joints12d_22[:, :, :3]

    def joints12d_22_to_intergen_262(self, joints12d_22):
        joints3d_22 = self.joints12d_22_to_joints3d_22(joints12d_22)
        return self.joints3d_22_to_intergen_262(joints3d_22)
