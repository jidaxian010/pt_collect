"""
gripper_transforms.py — Shared rigid-body transforms for the ANTERO wearable gripper.

Single source of truth for:
  - IMU-to-gripper rotation matrices and position offsets
  - Tag-to-gripper homogeneous transforms
  - Tag IDs and plot colors

All other scripts should import from here rather than defining their own copies.
"""

import numpy as np

# ==========================================
# --- UTILITIES ---
# ==========================================

def build_T(r, R):
    """Build a 4x4 homogeneous transform from rotation R and translation r."""
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3]   = r
    return T

# ==========================================
# --- IMU-TO-GRIPPER TRANSFORMS ---
# ==========================================
# R_id: rotates a vector FROM frame i TO gripper frame d
# r_id: position of gripper (d) relative to IMU (i), expressed in gripper frame d

# Frame a — left RealSense camera IMU
R_ad = np.array([[ 0.0,   -0.9397, -0.3420],
                 [ 1.0,    0.0,     0.0   ],
                 [ 0.0,   -0.3420,  0.9397]])
r_ad = np.array([-0.0981, 0.0175, 0.0181])

# Frame b — right RealSense camera IMU
R_bd = np.array([[ 0.0,   -0.9397, -0.3420],
                 [ 1.0,    0.0,     0.0   ],
                 [ 0.0,    0.3420,  0.9397]])
r_bd = np.array([0.0981, 0.0175, 0.0181])

# Frame c — VectorNav (rows 0/1 swapped vs original MATLAB to match
# gripper convention: Y-axis up, confirmed by gravity alignment)
R_cd = np.array([[ 0.7070,  0.0, -0.7071],
                 [ 0.7071,  0.0,  0.7071],
                 [ 0.0,    -1.0,  0.0   ]])
r_cd = np.array([0.0696, 0.0611, 0.2011])

# ==========================================
# --- TAG -> GRIPPER TRANSFORMS ---
# (must match ARUCO_Record_Pose.py exactly)
# ==========================================

R_10 = np.array([[ 0.0,    0.0, 1.0],
                 [ 0.0,    1.0, 0.0],
                 [-1.0,    0.0, 0.0]])
R_20 = np.array([[ 0.7071, 0.0, 0.7071],
                 [ 0.0,    1.0, 0.0   ],
                 [-0.7071, 0.0, 0.7071]])
R_30 = np.array([[ 0.0,    0.0,    1.0],
                 [-0.7071, 0.7071, 0.0],
                 [-0.7071,-0.7071, 0.0]])
R_40 = np.array([[-0.7071, 0.0, 0.7071],
                 [ 0.0,    1.0, 0.0   ],
                 [-0.7071, 0.0,-0.7071]])
R_50 = np.array([[ 0.0,    0.0,    1.0],
                 [ 0.7071, 0.7071, 0.0],
                 [-0.7071, 0.7071, 0.0]])

TAG_TRANSFORMS = {
    1: build_T(r=[ 0.3927,  0.0225, -0.2148], R=R_10),
    2: build_T(r=[ 0.3622,  0.0225,  0.0978], R=R_20),
    3: build_T(r=[ 0.3927, -0.0686, -0.1958], R=R_30),
    4: build_T(r=[ 0.1932,  0.0225, -0.4576], R=R_40),
    5: build_T(r=[ 0.3927,  0.1004, -0.1639], R=R_50),
}

# ==========================================
# --- MARKER GEOMETRY ---
# ==========================================

MARKER_SIZE_METERS = 0.0662
_half = MARKER_SIZE_METERS / 2.0

# Corners in tag frame (matches OpenCV / ARUCO_Record_Pose.py ordering):
#   c0=top-left, c1=top-right, c2=bottom-right, c3=bottom-left
MARKER_CORNERS_TAG = np.array([
    [-_half,  _half, 0.0],
    [ _half,  _half, 0.0],
    [ _half, -_half, 0.0],
    [-_half, -_half, 0.0],
])

# Precompute inverse tag-to-gripper transforms: T_body_tag = inv(T_tag_body)
TAG_TRANSFORMS_INV = {}
for _tid, _T in TAG_TRANSFORMS.items():
    _Rinv = _T[:3, :3].T
    _tinv = -_Rinv @ _T[:3, 3]
    _Tinv = np.eye(4)
    _Tinv[:3, :3] = _Rinv
    _Tinv[:3, 3]  = _tinv
    TAG_TRANSFORMS_INV[_tid] = _Tinv

# ==========================================
# --- COMMON CONFIG ---
# ==========================================

TAG_IDS    = [1, 2, 3, 4, 5]
TAG_COLORS = {1: 'tab:orange', 2: 'tab:green', 3: 'tab:red',
              4: 'tab:purple', 5: 'tab:brown'}

# IMU topics in the ROS2 bag
VECTORNAV_TOPIC = '/vectornav/imu_uncompensated'
VECTORNAV_MAG_TOPIC = '/vectornav/magnetic'
LEFT_CAM_TOPIC  = '/left_camera/camera/camera/imu'
RIGHT_CAM_TOPIC = '/right_camera/camera/camera/imu'