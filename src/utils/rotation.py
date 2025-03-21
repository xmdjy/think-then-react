"""
This file is actually never accessed, but might be useful in the future
"""
#%%
from typing import Union
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


def dtype_maintaining_wrapper(func):
    """
    Maintain the data type of input and output (torch or numpy) for functions in numpy space
    """
    def wrapper(rotation: Union[np.ndarray, torch.Tensor], **kwargs):
        torch_input = False
        if isinstance(rotation, torch.Tensor):
            torch_input = True
            rotation = rotation.cpu().numpy()

        result = func(rotation, **kwargs)

        if torch_input:
            result = torch.from_numpy(result)
        
        return result
    return wrapper


def multi_dim_rotmat_input_wrapper(func):
    def wrapper(rotation: np.ndarray, **kwargs):
        raw_shape = rotation.shape
        if len(raw_shape) <= 3:  # ([bs,] 3, 3)
            result = func(rotation, **kwargs)
        else:
            result = func(rotation.reshape(-1, 3, 3), **kwargs).reshape(*raw_shape[: -2], -1)
        return result
    return wrapper


class RotationHelper:

    @staticmethod
    @dtype_maintaining_wrapper
    def quat_to_rotmat(q: np.ndarray):
        r = Rotation.from_quat(q)
        return r.as_matrix()
    
    @staticmethod
    @dtype_maintaining_wrapper
    @multi_dim_rotmat_input_wrapper
    def rotmat_to_quat(rotmat: np.ndarray):
        r = Rotation.from_matrix(rotmat)
        return r.as_quat()
    
    @staticmethod
    @dtype_maintaining_wrapper
    def axis_angle_to_rotmat(aa: np.ndarray):
        r = Rotation.from_rotvec(aa)
        return r.as_matrix()
    
    @staticmethod
    @dtype_maintaining_wrapper
    @multi_dim_rotmat_input_wrapper
    def rotmat_to_axis_angle(rotmat: np.ndarray):
        r = Rotation.from_matrix(rotmat)
        return r.as_rotvec()
    
    @staticmethod
    @dtype_maintaining_wrapper
    def euler_angle_to_rotmat(euler: np.ndarray, euler_format='xyz'):
        r = Rotation.from_euler(euler_format, euler)
        return r.as_matrix()
    
    @staticmethod
    @dtype_maintaining_wrapper
    @multi_dim_rotmat_input_wrapper
    def rotmat_to_euler_angle(rotmat: np.ndarray, euler_format='xyz'):
        r = Rotation.from_matrix(rotmat)
        return r.as_euler(euler_format)
 
    @staticmethod
    @dtype_maintaining_wrapper
    def sixd_to_rotmat(sixd: np.ndarray):
        rotmat = np.zeros((sixd.shape[0], 3, 3))
        rotmat[:, :, :2] = sixd.reshape(-1, 3, 2)
        rotmat[:, :, 2] = np.cross(rotmat[:, :, 0], rotmat[:, :, 1], axis=-1)
        return rotmat
    
    @staticmethod
    @dtype_maintaining_wrapper
    @multi_dim_rotmat_input_wrapper
    def rotmat_to_6d(rotmat: np.ndarray):
        sixd: np.ndarray = rotmat[None, :, 0:2]
        return sixd.reshape(-1, 6)

#%%
if __name__ == '__main__':
    # euler = np.array([0, 1, 2])
    euler = torch.tensor([[0, 1, 2]])
    rotmat = RotationHelper.euler_angle_to_rotmat(euler)
    euler1 = RotationHelper.rotmat_to_euler_angle(rotmat)

    aa = RotationHelper.rotmat_to_axis_angle(rotmat)
    quat = RotationHelper.rotmat_to_quat(rotmat)
    sixd = RotationHelper.rotmat_to_6d(rotmat)
    rot_t = RotationHelper.sixd_to_rotmat(sixd)
    rot1 = RotationHelper.quat_to_rotmat(quat)
# %%
