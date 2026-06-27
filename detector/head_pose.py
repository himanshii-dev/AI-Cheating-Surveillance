"""
head_pose.py
------------
Estimates head pose direction from facial landmarks detected by dlib.

Uses a Perspective-n-Point (PnP) solve against a canonical 3-D face model
to obtain rotation and translation vectors, then converts them to Euler
angles (pitch, yaw, roll) to determine gaze direction.
"""

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# 3-D reference points for a generic human face (in millimetres).
# These correspond to dlib landmark indices used below.
# ---------------------------------------------------------------------------
MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),           # Nose tip            – index 30
        (0.0, -330.0, -65.0),      # Chin                – index 8
        (-225.0, 170.0, -135.0),   # Left eye corner     – index 36
        (225.0, 170.0, -135.0),    # Right eye corner    – index 45
        (-150.0, -150.0, -125.0),  # Left mouth corner   – index 48
        (150.0, -150.0, -125.0),   # Right mouth corner  – index 54
    ],
    dtype=np.float64,
)

# Yaw thresholds (degrees) for left / right classification
YAW_LEFT_THRESHOLD = -15
YAW_RIGHT_THRESHOLD = 15

# Pitch thresholds (degrees) for up / down classification
PITCH_UP_THRESHOLD = -15
PITCH_DOWN_THRESHOLD = 10


def build_camera_matrix(frame_width: int, frame_height: int) -> np.ndarray:
    """
    Build an approximate camera intrinsic matrix assuming a standard webcam.

    Parameters
    ----------
    frame_width  : int – Width of the video frame in pixels.
    frame_height : int – Height of the video frame in pixels.

    Returns
    -------
    np.ndarray – 3×3 camera matrix.
    """
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return camera_matrix


def extract_2d_points(landmarks) -> np.ndarray:
    """
    Extract the six 2-D landmark coordinates used for PnP from a dlib shape.

    Parameters
    ----------
    landmarks : dlib.full_object_detection – 68-point facial landmark shape.

    Returns
    -------
    np.ndarray – Shape (6, 2) array of (x, y) pixel coordinates.
    """
    indices = [30, 8, 36, 45, 48, 54]
    points = np.array(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in indices],
        dtype=np.float64,
    )
    return points


def get_head_pose_direction(landmarks, frame_width: int, frame_height: int) -> str:
    """
    Compute the head pose and return a human-readable direction label.

    Parameters
    ----------
    landmarks    : dlib.full_object_detection – 68-point shape.
    frame_width  : int
    frame_height : int

    Returns
    -------
    str – One of: "Looking Straight", "Looking Left", "Looking Right",
                   "Looking Up", "Looking Down".
    """
    camera_matrix = build_camera_matrix(frame_width, frame_height)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    image_points_2d = extract_2d_points(landmarks)

    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points_2d,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return "Looking Straight"

    rotation_matrix, _ = cv2.Rodrigues(rotation_vec)

    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    pitch = float(np.degrees(np.arctan2(-rotation_matrix[2, 0], sy)))
    yaw   = float(np.degrees(np.arctan2( rotation_matrix[1, 0], rotation_matrix[0, 0])))

    if yaw < YAW_LEFT_THRESHOLD:
        return "Looking Left"
    elif yaw > YAW_RIGHT_THRESHOLD:
        return "Looking Right"
    elif pitch < PITCH_UP_THRESHOLD:
        return "Looking Up"
    elif pitch > PITCH_DOWN_THRESHOLD:
        return "Looking Down"
    else:
        return "Looking Straight"


def draw_pose_annotation(frame: np.ndarray, direction: str) -> None:
    """
    Draw the head-pose label onto the frame in-place.

    Parameters
    ----------
    frame     : np.ndarray – BGR video frame.
    direction : str        – Direction label from get_head_pose_direction().
    """
    color = (0, 255, 0) if direction == "Looking Straight" else (0, 165, 255)
    cv2.putText(
        frame,
        f"Head: {direction}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
