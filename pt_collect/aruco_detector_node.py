import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np


def _build_T(t, R):
    """Build 4x4 homogeneous transform from translation (3,) and rotation (3,3)."""
    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = np.asarray(R, dtype=np.float64)
    T[0:3, 3] = np.asarray(t, dtype=np.float64).flatten()
    return T


def _rot_to_quat(R):
    """Convert 3x3 rotation matrix to unit quaternion [w, x, y, z]."""
    m = np.asarray(R, dtype=np.float64)
    tr = np.trace(m)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _quat_to_rot(q):
    """Convert unit quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def _slerp(q0, q1, alpha):
    """Spherical linear interpolation between two unit quaternions."""
    dot = np.dot(q0, q1)
    # Ensure shortest path
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        # Very close — linear interpolation + normalize
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (np.sin((1 - alpha) * theta) / sin_theta) * q0 + \
           (np.sin(alpha * theta) / sin_theta) * q1


# Known transforms: marker frame (j) -> gripper frame (k)
_R_10 = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
], dtype=np.float64)

_R_20 = np.array([
    [0.7071, 0.0, 0.7071],
    [0.0, 1.0, 0.0],
    [-0.7071, 0.0, 0.7071],
], dtype=np.float64)

_R_30 = np.array([
    [0.0, 0.0, 1.0],
    [-0.6654, 0.7465, 0.0],
    [-0.7465, -0.6654, 0.0],
], dtype=np.float64)

_R_40 = np.array([
    [-0.7071, 0.0, 0.7071],
    [0.0, 1.0, 0.0],
    [-0.7071, 0.0, -0.7071],
], dtype=np.float64)

_R_50 = np.array([
    [0.0, 0.0, 1.0],
    [0.7465, 0.6654, 0.0],
    [-0.6654, 0.7465, 0.0],
], dtype=np.float64)

TAG_TRANSFORMS = {
    1: _build_T(t=[0.3927, 0.0225, -0.2142], R=_R_10),
    2: _build_T(t=[0.3641, 0.0225, 0.0993], R=_R_20),
    3: _build_T(t=[0.3927, -0.0592, -0.2003], R=_R_30),
    4: _build_T(t=[0.1912, 0.0225, -0.4561], R=_R_40),
    5: _build_T(t=[0.3927, 0.0928, -0.1703], R=_R_50),
}


class ArucoDetectorNode(Node):

    MARKER_SIZE = 0.0662 

    # ==================== TUNING PARAMETERS ====================
    DEPTH_WEIGHT = 0.6        # depth vs PnP translation fusion (0=PnP only, 1=depth only)
    EMA_ALPHA = 0.5           # EMA blend factor (0=smooth, 1=reactive)
    OUTLIER_DIST = 0.10       # meters — per-tag outlier rejection vs median
    MAX_POS_STEP = 0.02       # meters — max position change per frame
    MAX_ROT_STEP = 0.1        # radians — max rotation change per frame (~5.7 deg)
    # =============================================================

    def __init__(self):
        super().__init__('aruco_detector')

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.depth_image = None

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 53
        params.adaptiveThreshWinSizeStep = 4
        params.adaptiveThreshConstant = 7
        params.minMarkerPerimeterRate = 0.01
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.05
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        params.cornerRefinementMaxIterations = 50
        params.cornerRefinementMinAccuracy = 0.01
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, params)

        half = self.MARKER_SIZE / 2.0
        self.obj_points = np.array([
            [-half,  half, 0.0],
            [ half,  half, 0.0],
            [ half, -half, 0.0],
            [-half, -half, 0.0],
        ], dtype=np.float64)

        self.sub_info = self.create_subscription(
            CameraInfo,
            '/right_camera/camera/camera/color/camera_info',
            self._info_cb, qos_profile_sensor_data
        )
        self.sub_depth = self.create_subscription(
            Image,
            '/right_camera/camera/camera/depth/image_rect_raw',
            self._depth_cb, qos_profile_sensor_data
        )
        self.sub_image = self.create_subscription(
            Image,
            '/right_camera/camera/camera/color/image_raw',
            self._image_cb, qos_profile_sensor_data
        )
        self.pub_annotated = self.create_publisher(
            Image, '/aruco/annotated_image', 1
        )
        self.pub_gripper = self.create_publisher(
            Image, '/aruco/gripper_pose_image', 1
        )
        self.pub_gripper_pose = self.create_publisher(
            PoseStamped, '/aruco/gripper_pose', 1
        )

        self.smooth_pos = None
        self.smooth_quat = None

        self.get_logger().info('ArUco detector node started')

    def _info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)
            self.get_logger().info('Camera intrinsics received')

    def _depth_cb(self, msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if not hasattr(self, '_depth_received'):
            self._depth_received = True
            self.get_logger().info(
                f'First depth frame: shape={self.depth_image.shape} dtype={self.depth_image.dtype}'
            )

    def _deproject_depth(self, corners_2d):
        """Get 3D translation from depth at marker center (3x3 patch median).
        Returns (x, y, z) in meters or None if depth is invalid."""
        if self.depth_image is None or self.camera_matrix is None:
            return None

        cx = int(np.mean(corners_2d[:, 0]))
        cy = int(np.mean(corners_2d[:, 1]))

        h, w = self.depth_image.shape[:2]
        # 3x3 patch, clipped to image bounds
        y0 = max(cy - 1, 0)
        y1 = min(cy + 2, h)
        x0 = max(cx - 1, 0)
        x1 = min(cx + 2, w)
        patch = self.depth_image[y0:y1, x0:x1].astype(np.float64)

        # Filter out zero (invalid) pixels
        valid = patch[patch > 0]
        if len(valid) == 0:
            return None

        depth_m = np.median(valid) * 0.001  # mm -> meters

        # Deproject pixel to 3D using intrinsics
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        ppx = self.camera_matrix[0, 2]
        ppy = self.camera_matrix[1, 2]

        x = (cx - ppx) / fx * depth_m
        y = (cy - ppy) / fy * depth_m
        z = depth_m

        return np.array([x, y, z], dtype=np.float64)

    def _image_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Sharpen for better detection
        blur = cv2.GaussianBlur(frame, (0, 0), sigmaX=5)
        sharp = cv2.addWeighted(frame, 2.0, blur, -1.0, 0)

        corners, ids, _ = self.detector.detectMarkers(sharp)

        gripper_positions = []
        gripper_rotations = []

        if ids is not None and self.camera_matrix is not None:
            # Filter to IDs 1-5
            keep = [i for i, mid in enumerate(ids.flatten()) if mid in TAG_TRANSFORMS]
            if keep:
                corners = tuple(corners[i] for i in keep)
                ids = ids[keep]

                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                for c, mid in zip(corners, ids.flatten()):
                    ok, rvec, tvec = cv2.solvePnP(
                        self.obj_points, c[0],
                        self.camera_matrix, self.dist_coeffs,
                    )
                    if not ok:
                        continue

                    R_ij, _ = cv2.Rodrigues(rvec)
                    t_pnp = tvec.flatten()

                    # Require depth — skip tag if unavailable
                    t_depth = self._deproject_depth(c[0])
                    if t_depth is None:
                        continue
                    w = self.DEPTH_WEIGHT
                    t_fused = w * t_depth + (1 - w) * t_pnp

                    # Draw marker axes with fused translation
                    rvec_draw = rvec
                    tvec_draw = t_fused.reshape(3, 1)
                    cv2.drawFrameAxes(
                        frame, self.camera_matrix, self.dist_coeffs,
                        rvec_draw, tvec_draw, self.MARKER_SIZE,
                    )

                    # Label
                    dist = np.linalg.norm(t_fused)
                    pos = tuple(c[0][0].astype(int))
                    src = 'D+P' if t_depth is not None else 'PnP'
                    cv2.putText(
                        frame, f'ID{mid} {dist:.2f}m [{src}]',
                        (pos[0], pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )

                    # Chain: camera->marker (T_ij) @ marker->gripper (T_jk) = camera->gripper (T_ik)
                    T_ij = _build_T(t_fused, R_ij)
                    T_jk = TAG_TRANSFORMS[int(mid)]
                    T_ik = T_ij @ T_jk

                    gripper_positions.append(T_ik[0:3, 3])
                    gripper_rotations.append(T_ik[0:3, 0:3])

        # Publish marker-annotated image
        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header = msg.header
        self.pub_annotated.publish(out_msg)

        # --- Gripper pose: filter, smooth, draw ---
        gripper_frame = frame.copy()
        raw_pos, raw_quat, n_tags = self._fuse_gripper_pose(
            gripper_positions, gripper_rotations
        )

        if raw_pos is not None:
            smooth_pos, smooth_quat = self._update_smooth(raw_pos, raw_quat)
            R_smooth = _quat_to_rot(smooth_quat)
            rvec_g, _ = cv2.Rodrigues(R_smooth)
            tvec_g = smooth_pos.reshape(3, 1)

            cv2.drawFrameAxes(
                gripper_frame, self.camera_matrix, self.dist_coeffs,
                rvec_g, tvec_g, self.MARKER_SIZE * 1.5,
            )
            cv2.putText(
                gripper_frame,
                f'Gripper  tags:{n_tags}  '
                f'x:{smooth_pos[0]:.3f} y:{smooth_pos[1]:.3f} z:{smooth_pos[2]:.3f}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

            # Publish gripper pose as PoseStamped
            pose_msg = PoseStamped()
            pose_msg.header = msg.header
            pose_msg.pose.position.x = float(smooth_pos[0])
            pose_msg.pose.position.y = float(smooth_pos[1])
            pose_msg.pose.position.z = float(smooth_pos[2])
            # smooth_quat is [w, x, y, z], ROS uses [x, y, z, w]
            pose_msg.pose.orientation.x = float(smooth_quat[1])
            pose_msg.pose.orientation.y = float(smooth_quat[2])
            pose_msg.pose.orientation.z = float(smooth_quat[3])
            pose_msg.pose.orientation.w = float(smooth_quat[0])
            self.pub_gripper_pose.publish(pose_msg)
        else:
            cv2.putText(
                gripper_frame, 'No markers detected',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )

        gp_msg = self.bridge.cv2_to_imgmsg(gripper_frame, encoding='bgr8')
        gp_msg.header = msg.header
        self.pub_gripper.publish(gp_msg)

    def _fuse_gripper_pose(self, positions, rotations):
        """Median-filter outlier tags, return fused (position, quaternion, count)."""
        if not positions:
            return None, None, 0

        positions = np.stack(positions)   # (N, 3)
        quats = np.array([_rot_to_quat(R) for R in rotations])  # (N, 4)

        if len(positions) >= 3:
            # Reject tags whose position is far from the median
            median = np.median(positions, axis=0)
            dists = np.linalg.norm(positions - median, axis=1)
            keep = dists < self.OUTLIER_DIST
            if np.any(keep):
                positions = positions[keep]
                quats = quats[keep]

        avg_pos = np.mean(positions, axis=0)

        # Average quaternions: pick reference, flip signs for consistency, mean + normalize
        ref = quats[0]
        for i in range(1, len(quats)):
            if np.dot(quats[i], ref) < 0:
                quats[i] = -quats[i]
        avg_quat = np.mean(quats, axis=0)
        avg_quat /= np.linalg.norm(avg_quat)

        return avg_pos, avg_quat, len(positions)

    def _update_smooth(self, raw_pos, raw_quat):
        """Apply EMA smoothing with clamped max step. Returns (smooth_pos, smooth_quat)."""
        if self.smooth_pos is None:
            self.smooth_pos = raw_pos.copy()
            self.smooth_quat = raw_quat.copy()
            return self.smooth_pos, self.smooth_quat

        # EMA target
        alpha = self.EMA_ALPHA
        target_pos = alpha * raw_pos + (1 - alpha) * self.smooth_pos
        target_quat = _slerp(self.smooth_quat, raw_quat, alpha)

        # Clamp position step
        delta_pos = target_pos - self.smooth_pos
        pos_step = np.linalg.norm(delta_pos)
        if pos_step > self.MAX_POS_STEP:
            delta_pos = delta_pos * (self.MAX_POS_STEP / pos_step)
        self.smooth_pos = self.smooth_pos + delta_pos

        # Clamp rotation step
        dot = np.clip(np.abs(np.dot(self.smooth_quat, target_quat)), 0.0, 1.0)
        rot_step = 2.0 * np.arccos(dot)
        if rot_step > self.MAX_ROT_STEP:
            # Only move max_rot_step towards target
            frac = self.MAX_ROT_STEP / rot_step
            self.smooth_quat = _slerp(self.smooth_quat, target_quat, frac)
        else:
            self.smooth_quat = target_quat

        return self.smooth_pos, self.smooth_quat


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
