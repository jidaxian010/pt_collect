import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np


def _build_T(t, R):
    """Build 4x4 homogeneous transform from translation (3,) and rotation (3,3)."""
    T = np.eye(4, dtype=np.float64)
    T[0:3, 0:3] = np.asarray(R, dtype=np.float64)
    T[0:3, 3] = np.asarray(t, dtype=np.float64).flatten()
    return T


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

    MARKER_SIZE = 0.0725  # 7.25 cm

    def __init__(self):
        super().__init__('aruco_detector')

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

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
            self._info_cb, 1
        )
        self.sub_image = self.create_subscription(
            Image,
            '/right_camera/camera/camera/color/image_raw',
            self._image_cb, 1
        )
        self.pub_annotated = self.create_publisher(
            Image, '/aruco/annotated_image', 1
        )
        self.pub_gripper = self.create_publisher(
            Image, '/aruco/gripper_pose_image', 1
        )

        self.get_logger().info('ArUco detector node started')

    def _info_cb(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d, dtype=np.float64)
            self.get_logger().info('Camera intrinsics received')

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

                    # Draw marker axes
                    cv2.drawFrameAxes(
                        frame, self.camera_matrix, self.dist_coeffs,
                        rvec, tvec, self.MARKER_SIZE,
                    )

                    # Label
                    dist = np.linalg.norm(tvec)
                    pos = tuple(c[0][0].astype(int))
                    cv2.putText(
                        frame, f'ID{mid} {dist:.2f}m',
                        (pos[0], pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )

                    # Chain: camera->marker (T_ij) @ marker->gripper (T_jk) = camera->gripper (T_ik)
                    T_ij = _build_T(tvec.flatten(), R_ij)
                    T_jk = TAG_TRANSFORMS[int(mid)]
                    T_ik = T_ij @ T_jk

                    gripper_positions.append(T_ik[0:3, 3])
                    gripper_rotations.append(T_ik[0:3, 0:3])

        # Publish marker-annotated image
        out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        out_msg.header = msg.header
        self.pub_annotated.publish(out_msg)

        # Average gripper pose and draw on separate topic
        gripper_frame = frame.copy()
        if gripper_positions:
            avg_pos = np.mean(np.stack(gripper_positions), axis=0)
            # Use first rotation (rotation averaging is non-trivial)
            R_gripper = gripper_rotations[0]
            rvec_g, _ = cv2.Rodrigues(R_gripper)
            tvec_g = avg_pos.reshape(3, 1)

            cv2.drawFrameAxes(
                gripper_frame, self.camera_matrix, self.dist_coeffs,
                rvec_g, tvec_g, self.MARKER_SIZE * 1.5,
            )
            cv2.putText(
                gripper_frame,
                f'Gripper  tags:{len(gripper_positions)}  '
                f'x:{avg_pos[0]:.3f} y:{avg_pos[1]:.3f} z:{avg_pos[2]:.3f}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
        else:
            cv2.putText(
                gripper_frame, 'No markers detected',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )

        gp_msg = self.bridge.cv2_to_imgmsg(gripper_frame, encoding='bgr8')
        gp_msg.header = msg.header
        self.pub_gripper.publish(gp_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
