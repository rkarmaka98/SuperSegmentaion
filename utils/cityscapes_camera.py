import json
import numpy as np
from scipy.spatial.transform import Rotation as R


def load_cityscapes_camera(json_path, right_camera=False):
    """Load Cityscapes camera parameters.

    Args:
        json_path (str): Path to the Cityscapes calibration JSON.
        right_camera (bool): Whether to offset the translation to model the
            right stereo camera. The baseline between cameras is 0.209313 m.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Intrinsics
    fx = data['intrinsic']['fx']
    fy = data['intrinsic']['fy']
    u0 = data['intrinsic']['u0']
    v0 = data['intrinsic']['v0']
    K = np.array([[fx, 0, u0],
                  [0, fy, v0],
                  [0,  0,  1]])

    # Extrinsics
    yaw = data['extrinsic']['yaw']       # radians
    pitch = data['extrinsic']['pitch']
    roll = data['extrinsic']['roll']
    x = data['extrinsic']['x']
    y = data['extrinsic']['y']
    z = data['extrinsic']['z']

    # Rotation matrix (yaw, pitch, roll order)
    R_cam = R.from_euler('yxz', [yaw, pitch, roll]).as_matrix()
    t = np.array([x, y, z])

    if right_camera:
        # Baseline from csCalibration.pdf: left-right offset of 0.209313 m
        t = t + np.array([0.209313, 0.0, 0.0])  # shift along X-axis

    return K, R_cam, t

def simulate_ego_motion(delta_yaw_deg=0.0, forward_cm=0.1, scale=0.1):
    delta_yaw_deg *= scale
    forward_cm *= scale
    R_delta = R.from_euler('y', np.deg2rad(delta_yaw_deg)).as_matrix()
    t_delta = np.array([forward_cm / 100.0, 0.0, 0.0])  # forward in x
    return R_delta, t_delta

def compute_homography(K, R1, t1, R2, t2, z_plane=30.0):
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    n = np.array([0, 0, 1])
    d = z_plane
    H = K @ (R_rel - np.outer(t_rel, n) / d) @ np.linalg.inv(K)
    return H
