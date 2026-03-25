"""
RecordImagesAR.py — RealSense recorder with live ArUco + gripper frame overlay.

Same recording functionality as RecordImages.py, but the live preview also
shows detected ArUco tags (outlines, IDs, per-tag axes) and the estimated
gripper body frame (K) projected onto the camera image.

The gripper pose is computed in real-time from solvePnP + TAG_TRANSFORMS.
When multiple tags are visible, positions are averaged and rotations are
fused via quaternion mean.

Controls: 'r' = start/stop recording (5s countdown), 'q' = quit.

Usage:
    python RecordImagesAR.py
"""

import pyrealsense2 as rs
import numpy as np
import cv2
import csv
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from threading import Thread
from queue import Queue
from scipy.spatial.transform import Rotation

from GripperTransforms import (
    TAG_TRANSFORMS, TAG_IDS, MARKER_SIZE_METERS,
)

# ==========================================
# --- CONFIGURATION ---
# ==========================================

EXPOSURE_US    = 75       # microseconds. Set to None for auto.
GAIN           = 80       # RGB gain (0-128, default 64). Set to None for auto.
CAMERA_FPS     = 30
RECORD_DELAY   = 5.0      # seconds countdown before recording starts
DEPTH_UNITS    = 0.0001   # metres per depth unit (0.1 mm resolution)
DISPLAY_SCALE  = 0.75     # preview window = camera_res * this (0.75 → 960x540)
GRIP_AXIS_LEN  = 0.12     # length of gripper frame axes on screen (metres)
TAG_AXIS_LEN   = MARKER_SIZE_METERS  # per-tag axis length (metres)

# BGR colours for tag overlays
TAG_COLORS_BGR = {
    1: (0, 165, 255),    # orange
    2: (0, 200, 0),      # green
    3: (0, 0, 255),      # red
    4: (200, 0, 200),    # purple
    5: (42, 42, 165),    # brown
}

# ==========================================
# --- NTP SYNC CHECK ---
# ==========================================

def check_ntp_sync():
    """Check Windows clock NTP offset. Returns offset in ms, or None on failure."""
    try:
        result = subprocess.run(
            ['w32tm', '/stripchart', '/computer:pool.ntp.org',
             '/samples:3', '/dataonly'],
            capture_output=True, text=True, timeout=20
        )
        lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
        for line in reversed(lines):
            match = re.search(r'([+-]?\d+\.\d+)s', line)
            if match:
                return float(match.group(1)) * 1000  # ms
    except Exception as e:
        print(f"  NTP check failed: {e}")
    return None

# ==========================================
# --- BACKGROUND WRITER ---
# ==========================================

def writer_thread_fn(q):
    """Drains the write queue, saving color (JPEG) and depth (16-bit PNG) to disk."""
    while True:
        item = q.get()
        if item is None:
            break
        color_path, depth_path, color_img, depth_img = item
        cv2.imwrite(color_path, color_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(depth_path, depth_img)   # 16-bit PNG, lossless
        q.task_done()

# ==========================================
# --- ARUCO DETECTION + GRIPPER OVERLAY ---
# ==========================================

_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_parameters = cv2.aruco.DetectorParameters()
_detector   = cv2.aruco.ArucoDetector(_dictionary, _parameters)
_clahe      = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))

_half = MARKER_SIZE_METERS / 2.0
_marker_3d = np.array([
    [-_half,  _half, 0],
    [ _half,  _half, 0],
    [ _half, -_half, 0],
    [-_half, -_half, 0],
], dtype=np.float32)


def _draw_frame_on_image(img, T_cam_frame, camera_matrix, dist_coeffs,
                         axis_len, label="", thickness=2):
    """Project a 3D coordinate frame (X=red, Y=green, Z=blue) onto the image."""
    origin = T_cam_frame[:3, 3]
    R = T_cam_frame[:3, :3]

    if origin[2] <= 0:
        return

    pts_3d = np.array([
        origin,
        origin + R[:, 0] * axis_len,
        origin + R[:, 1] * axis_len,
        origin + R[:, 2] * axis_len,
    ], dtype=np.float64)

    if np.any(pts_3d[:, 2] <= 0):
        return

    pts_2d, _ = cv2.projectPoints(
        pts_3d, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
    pts_2d = pts_2d.reshape(-1, 2).astype(int)

    o, x, y, z = tuple(pts_2d[0]), tuple(pts_2d[1]), tuple(pts_2d[2]), tuple(pts_2d[3])

    cv2.line(img, o, x, (0, 0, 255), thickness)    # X = red
    cv2.line(img, o, y, (0, 255, 0), thickness)    # Y = green
    cv2.line(img, o, z, (255, 0, 0), thickness)    # Z = blue
    cv2.circle(img, o, 5, (255, 255, 255), -1)

    if label:
        cv2.putText(img, label, (o[0] + 10, o[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def detect_and_draw(color_img, camera_matrix, dist_coeffs):
    """Detect ArUco tags, estimate gripper frame, draw both on a copy of the image."""
    display = color_img.copy()
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    gray = _clahe.apply(gray)

    corners, ids, _ = _detector.detectMarkers(gray)
    if ids is None:
        return display

    cv2.aruco.drawDetectedMarkers(display, corners, ids)

    gripper_poses = []

    for i in range(len(ids)):
        tag_id = ids[i][0]
        if tag_id not in TAG_TRANSFORMS:
            continue

        corner_px = corners[i][0]

        _, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            _marker_3d, corner_px,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE)
        rvec, tvec = rvecs[0], tvecs[0]

        # Per-tag axes (small)
        cv2.drawFrameAxes(display, camera_matrix, dist_coeffs,
                          rvec, tvec, TAG_AXIS_LEN)

        # Tag label with colour
        cx_px, cy_px = corner_px.mean(axis=0).astype(int)
        color = TAG_COLORS_BGR.get(tag_id, (255, 255, 255))
        cv2.putText(display, f"T{tag_id}", (cx_px - 10, cy_px - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Gripper pose: T_cam_grip = T_cam_tag @ TAG_TRANSFORMS[tid]
        R_cam_tag, _ = cv2.Rodrigues(rvec)
        T_cam_tag = np.eye(4)
        T_cam_tag[:3, :3] = R_cam_tag
        T_cam_tag[:3, 3]  = tvec.flatten()
        T_cam_grip = T_cam_tag @ TAG_TRANSFORMS[tag_id]
        gripper_poses.append(T_cam_grip)

    if not gripper_poses:
        return display

    # Fuse gripper pose across visible tags
    if len(gripper_poses) == 1:
        T_grip = gripper_poses[0]
    else:
        avg_pos = np.mean([T[:3, 3] for T in gripper_poses], axis=0)
        quats = [Rotation.from_matrix(T[:3, :3]).as_quat() for T in gripper_poses]
        for j in range(1, len(quats)):
            if np.dot(quats[j], quats[0]) < 0:
                quats[j] = -quats[j]
        avg_q = np.mean(quats, axis=0)
        avg_q /= np.linalg.norm(avg_q)
        T_grip = np.eye(4)
        T_grip[:3, :3] = Rotation.from_quat(avg_q).as_matrix()
        T_grip[:3, 3]  = avg_pos

    # Draw gripper body frame (K) — thick white-outlined axes
    _draw_frame_on_image(display, T_grip, camera_matrix, dist_coeffs,
                         axis_len=GRIP_AXIS_LEN, label="K", thickness=3)

    # HUD: tag count
    n_tags = len(gripper_poses)
    cv2.putText(display, f"{n_tags} tag{'s' if n_tags != 1 else ''}",
                (10, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return display

# ==========================================
# --- MAIN ---
# ==========================================

def main():
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, CAMERA_FPS)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16,  CAMERA_FPS)

    print("Starting pipeline...")
    profile = pipeline.start(config)
    align   = rs.align(rs.stream.color)

    # --- Sensor options ---
    device = profile.get_device()
    for sensor in device.query_sensors():
        if sensor.get_info(rs.camera_info.name) == 'RGB Camera':
            sensor.set_option(rs.option.auto_exposure_priority, 0)
            if EXPOSURE_US is not None:
                sensor.set_option(rs.option.enable_auto_exposure, 0)
                sensor.set_option(rs.option.exposure, EXPOSURE_US)
            if GAIN is not None:
                sensor.set_option(rs.option.gain, GAIN)

    depth_sensor = profile.get_device().first_depth_sensor()
    if depth_sensor.supports(rs.option.depth_units):
        depth_sensor.set_option(rs.option.depth_units, DEPTH_UNITS)
        print(f"Depth units set to {DEPTH_UNITS} m/unit  "
              f"(max range: {65535 * DEPTH_UNITS:.1f} m)")

    # --- Calibration data ---
    depth_sensor  = profile.get_device().first_depth_sensor()
    depth_scale   = depth_sensor.get_depth_scale()

    color_stream  = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_stream  = profile.get_stream(rs.stream.depth).as_video_stream_profile()

    color_intr    = color_stream.get_intrinsics()
    depth_intr    = depth_stream.get_intrinsics()
    d2c_extr      = depth_stream.get_extrinsics_to(color_stream)
    c2d_extr      = color_stream.get_extrinsics_to(depth_stream)

    camera_matrix = np.array([
        [color_intr.fx, 0.0,           color_intr.ppx],
        [0.0,           color_intr.fy, color_intr.ppy],
        [0.0,           0.0,           1.0           ],
    ], dtype=np.float64)
    dist_coeffs = np.array(color_intr.coeffs[:5], dtype=np.float64).reshape(5, 1)

    calibration = {
        "fps":           CAMERA_FPS,
        "exposure_us":   EXPOSURE_US,
        "depth_scale_m": depth_scale,
        "color_intrinsics": {
            "width":  color_intr.width,
            "height": color_intr.height,
            "fx":     color_intr.fx,
            "fy":     color_intr.fy,
            "cx":     color_intr.ppx,
            "cy":     color_intr.ppy,
            "model":  str(color_intr.model),
            "coeffs": list(color_intr.coeffs),
        },
        "depth_intrinsics": {
            "width":  depth_intr.width,
            "height": depth_intr.height,
            "fx":     depth_intr.fx,
            "fy":     depth_intr.fy,
            "cx":     depth_intr.ppx,
            "cy":     depth_intr.ppy,
            "model":  str(depth_intr.model),
            "coeffs": list(depth_intr.coeffs),
        },
        "depth_to_color_extrinsics": {
            "rotation":    list(d2c_extr.rotation),
            "translation": list(d2c_extr.translation),
        },
        "color_to_depth_extrinsics": {
            "rotation":    list(c2d_extr.rotation),
            "translation": list(c2d_extr.translation),
        },
    }

    print(f"Color: {color_intr.width}x{color_intr.height}  "
          f"fx={color_intr.fx:.4f}  fy={color_intr.fy:.4f}  "
          f"cx={color_intr.ppx:.4f}  cy={color_intr.ppy:.4f}")
    print(f"Depth scale: {depth_scale:.6f} m/unit")

    # --- NTP sync check ---
    print("Checking NTP sync...")
    ntp_offset_ms = check_ntp_sync()
    if ntp_offset_ms is not None:
        print(f"  NTP offset: {ntp_offset_ms:+.1f} ms")
        if abs(ntp_offset_ms) > 10:
            print("  WARNING: NTP offset > 10ms. Cross-device time sync may be degraded.")
            print("           Run 'w32tm /resync' or check NTP configuration.")
    else:
        print("  WARNING: Could not verify NTP sync. Ensure NTP is configured.")

    # --- Display size ---
    disp_w = int(1280 * DISPLAY_SCALE)
    disp_h = int(720 * DISPLAY_SCALE)

    # --- State ---
    recording         = False
    record_start_at   = None
    save_dir          = None
    color_dir         = None
    depth_dir         = None
    ts_writer         = None
    ts_file           = None
    frame_count       = 0
    record_start_wall = None
    dropped           = 0

    write_queue = Queue(maxsize=180)
    writer      = Thread(target=writer_thread_fn, args=(write_queue,), daemon=True)
    writer.start()

    print("Press 'r' to start/stop recording (5s countdown), 'q' to quit.")

    try:
        while True:
            frames  = pipeline.wait_for_frames()
            aligned = align.process(frames)

            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())   # uint8 BGR
            depth_img = np.asanyarray(depth_frame.get_data())   # uint16 Z16

            ts_color = color_frame.get_timestamp()
            ts_depth = depth_frame.get_timestamp()

            # --- ArUco detection + gripper frame overlay ---
            display = detect_and_draw(color_img, camera_matrix, dist_coeffs)

            # --- Countdown ---
            if record_start_at is not None and not recording:
                remaining = record_start_at - time.time()
                if remaining <= 0:
                    recording         = True
                    record_start_at   = None
                    record_start_wall = time.time()
                    dropped           = 0
                    frame_count       = 0

                    tag = datetime.now().strftime('%Y%m%d_%H%M%S')
                    save_dir  = f"RealSense_{tag}"
                    color_dir = os.path.join(save_dir, "color")
                    depth_dir = os.path.join(save_dir, "depth")
                    os.makedirs(color_dir, exist_ok=True)
                    os.makedirs(depth_dir, exist_ok=True)

                    calibration["record_start_unix"] = record_start_wall
                    calibration["record_start_iso"]  = datetime.fromtimestamp(
                        record_start_wall, tz=timezone.utc).isoformat()
                    calibration["ntp_offset_ms"]     = ntp_offset_ms

                    with open(os.path.join(save_dir, 'calibration.json'), 'w') as f:
                        json.dump(calibration, f, indent=2)

                    ts_file   = open(os.path.join(save_dir, 'timestamps.csv'), 'w', newline='')
                    ts_writer = csv.writer(ts_file)
                    ts_writer.writerow(['frame_index', 'wall_time_s', 'unix_time_s',
                                        'color_hw_ts_ms', 'depth_hw_ts_ms'])

                    print(f"Recording started -> {save_dir}/")
                else:
                    cv2.putText(display, f"REC in {int(remaining) + 1}s",
                                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

            # --- Record frame (saves raw images, NOT the overlay) ---
            if recording:
                now = time.time()
                wall_t = now - record_start_wall
                ts_writer.writerow([frame_count, f"{wall_t:.6f}", f"{now:.6f}",
                                    f"{ts_color:.3f}", f"{ts_depth:.3f}"])

                color_path = os.path.join(color_dir, f"{frame_count:06d}.jpg")
                depth_path = os.path.join(depth_dir, f"{frame_count:06d}.png")

                if not write_queue.full():
                    write_queue.put((color_path, depth_path,
                                     color_img.copy(), depth_img.copy()))
                else:
                    dropped += 1

                frame_count += 1
                cv2.putText(display, f"REC {frame_count}  drop {dropped}",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Resize for display
            display = cv2.resize(display, (disp_w, disp_h))
            cv2.imshow('RealSense AR — RGB+Depth Recorder', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if recording:
                    recording = False
                    ts_file.close()
                    ts_file = None
                    print(f"Recording stopped. {frame_count} frames, {dropped} dropped -> {save_dir}/")
                elif record_start_at is None:
                    record_start_at = time.time() + RECORD_DELAY
                    print(f"Recording starts in {RECORD_DELAY:.0f}s...")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if ts_file and not ts_file.closed:
            ts_file.close()
        write_queue.put(None)
        writer.join()
        print("Writer flushed. Done.")

if __name__ == "__main__":
    main()
