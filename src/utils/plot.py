from typing import Union
import copy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pickle

from src.utils.constants import JOINTS3D_22_KINEMATIC_CHAIN, EDGE22_INDICES_UNDIRCTIONAL
from src.utils.motion_representation_converter import MotionRepresentationConverter


optional_colors = ['r', 'g', 'b', 'c', 'm', 'y']
mrc = MotionRepresentationConverter()


def visualize_all_pkl(src_dir: str, file_pattern: str = '*.pkl'):
    src_dir = Path(src_dir)
    for p in src_dir.glob(file_pattern):
        animate_from_pkl(p)


def animate_from_pkl(file_path: str):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    shift = np.array([2, 0, 0])
    motions = []

    gt_action = data_dict.get('action', data_dict.get('gt_action', None))
    if gt_action is not None:
        if len(gt_action.shape) == 2 and gt_action.shape[-1] == 262:
            gt_action = mrc('i262', 'j3d', gt_action)
        motions.append(gt_action)

    gt_reaction = data_dict.get('reaction', data_dict.get('gt_reaction', None))
    if gt_reaction is not None:
        if len(gt_reaction.shape) == 2 and gt_reaction.shape[-1] == 262:
            gt_reaction = mrc('i262', 'j3d', gt_reaction)
        motions.append(gt_reaction)

    pred_reaction = data_dict.get('pred_reaction', None)
    if pred_reaction is not None:
        if len(pred_reaction.shape) == 2 and pred_reaction.shape[-1] == 262:
            pred_reaction = mrc('i262', 'j3d', pred_reaction)
        pred_action = gt_action + shift
        pred_reaction = pred_reaction + shift
        motions.append(pred_action)
        motions.append(pred_reaction)

    text = data_dict.get('caption', 'None')

    animate_multiple_joints3d_22(
        motions=motions,
        colors=optional_colors[:len(motions)],
        title=text,
        file_path=file_path.replace('pkl', 'mp4')
    )


def animate_multiple_joints3d_22(motions, colors, title, file_path, fps=20, downsample_rate=4, show_axis=False):
    motions = copy.deepcopy(motions)
    for i, m in enumerate(motions):
        motions[i] = m[::downsample_rate, ...]
        if len(motions[i].shape) == 2 and motions[i].shape[1] == 262:
            motions[i] = mrc('i262', 'j3d', motions[i])

    if isinstance(title, str):
        words = title.split(' ')
    else:
        words = []
    title = ''
    for i, word in enumerate(words):
        if i % 10 == 0 and i > 0:
            title += '\n'
        title += word + ' '
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Clear the axis before drawing new frame
    def init():
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 2)
        ax.set_zlim(-2, 2)
        if not show_axis:
            ax.set_axis_off()
        fig.suptitle(title, fontsize=10)
        return []

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        ## Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    # Update function for animation
    def update(frame):
        ax.clear()
        ax.set_xlim(-2, 2)
        ax.set_ylim(0, 2)
        ax.set_zlim(-2, 2)
        if not show_axis:
            ax.set_axis_off()
        # ax.view_init(elev=120, azim=-30, roll=90, vertical_axis='y')
        ax.view_init(vertical_axis='y')
        plot_xzPlane(-2, 2, 0, -2, 2)

        # Draw reaction skeleton
        for m, c in zip(motions, colors):
            plot_skeleton(ax, m[frame], color=c)

        return ax,

    # Function to plot a single skeleton
    def plot_skeleton(ax, joints, color):
        for chain in JOINTS3D_22_KINEMATIC_CHAIN:
            for i in range(len(chain) - 1):
                ax.plot([joints[chain[i]][0], joints[chain[i + 1]][0]],
                         [joints[chain[i]][1], joints[chain[i + 1]][1]],
                         [joints[chain[i]][2], joints[chain[i + 1]][2]],
                         '-k', lw=2)  # Plot lines between joints

        for joint in joints:
            ax.scatter(joint[0], joint[1], joint[2], c=color, s=30)  # Plot joints

    # Create the animation
    ani = FuncAnimation(fig, update, frames=np.arange(0, motions[0].shape[0]), init_func=init, blit=False)

    ani.save(file_path, fps=fps // downsample_rate)
    plt.close()


def visualize_3d_skeleton(joints: np.ndarray, save_path):
    """
    Visualizes a 3D skeleton.

    Parameters:
    - joints: A numpy array of shape (22, 3) representing the XYZ coordinates of 22 joints.
    - edge_indices: A numpy array of shape (n_edges, 2) representing the indices of connected joints.
    """
    # Create a new matplotlib figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the joints as points
    for i in range(joints.shape[0]):
        ax.scatter(joints[i, 0], joints[i, 1], joints[i, 2], color='r', s=30)

    # Plot the edges between joints
    for edge in EDGE22_INDICES_UNDIRCTIONAL:
        start_joint = joints[edge[0]]
        end_joint = joints[edge[1]]
        ax.plot([start_joint[0], end_joint[0]], [start_joint[1], end_joint[1]], [start_joint[2], end_joint[2]], color='b')

    # Set labels for the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show the plot
    fig.savefig(save_path)
    plt.close()
