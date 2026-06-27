"""
eye_movement.py
---------------
Detects eye movement direction and whether eyes are closed using dlib's
68-point facial landmark predictor.

Eye Aspect Ratio (EAR) is used to detect blinks / closed eyes.
The centre of each iris is estimated from the eye landmark hull to determine
lateral gaze direction.
"""

import numpy as np
from scipy.spatial import distance as dist


# ---------------------------------------------------------------------------
# dlib 68-landmark indices for the left and right eyes
# ---------------------------------------------------------------------------
LEFT_EYE_INDICES = list(range(36, 42))   # landmarks 36-41
RIGHT_EYE_INDICES = list(range(42, 48))  # landmarks 42-47

# EAR below this value means the eye is considered closed
EAR_CLOSED_THRESHOLD = 0.21

# Ratio of iris-centre x offset to eye width that triggers left/right label
GAZE_RATIO_LEFT = 0.40
GAZE_RATIO_RIGHT = 0.60


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _eye_aspect_ratio(eye_points: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for a single eye.

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    Parameters
    ----------
    eye_points : np.ndarray – Shape (6, 2) array of (x, y) landmark coords.

    Returns
    -------
    float – EAR value; lower means more closed.
    """
    vertical_a = dist.euclidean(eye_points[1], eye_points[5])
    vertical_b = dist.euclidean(eye_points[2], eye_points[4])
    horizontal = dist.euclidean(eye_points[0], eye_points[3])

    if horizontal == 0:
        return 0.0

    ear = (vertical_a + vertical_b) / (2.0 * horizontal)
    return ear


def _get_eye_points(landmarks, indices: list) -> np.ndarray:
    """
    Extract (x, y) coordinates for the given landmark indices.

    Parameters
    ----------
    landmarks : dlib.full_object_detection
    indices   : list[int] – Landmark index list for one eye.

    Returns
    -------
    np.ndarray – Shape (N, 2).
    """
    return np.array(
        [(landmarks.part(i).x, landmarks.part(i).y) for i in indices],
        dtype=np.float32,
    )


def _gaze_ratio_for_eye(eye_points: np.ndarray, gray_frame: np.ndarray) -> float:
    """
    Estimate a normalised horizontal gaze ratio for one eye by analysing
    the relative position of the iris centre within the eye region.

    A value < GAZE_RATIO_LEFT  → looking left
    A value > GAZE_RATIO_RIGHT → looking right
    Otherwise                  → looking centre

    Parameters
    ----------
    eye_points : np.ndarray  – Shape (6, 2) landmark coords.
    gray_frame : np.ndarray  – Greyscale video frame.

    Returns
    -------
    float – Normalised gaze ratio in [0, 1].
    """
    # Bounding box of the eye region
    x_min = int(np.min(eye_points[:, 0]))
    x_max = int(np.max(eye_points[:, 0]))
    y_min = int(np.min(eye_points[:, 1]))
    y_max = int(np.max(eye_points[:, 1]))

    eye_width = x_max - x_min
    eye_height = y_max - y_min

    if eye_width <= 0 or eye_height <= 0:
        return 0.5

    # Crop the eye region from the greyscale frame
    eye_roi = gray_frame[y_min:y_max, x_min:x_max]

    if eye_roi.size == 0:
        return 0.5

    # Threshold the eye ROI to isolate the dark iris region
    _, threshold = __import__("cv2").threshold(
        eye_roi, 70, 255, __import__("cv2").THRESH_BINARY_INV
    )

    # Find the centroid of the thresholded region (iris proxy)
    moments = __import__("cv2").moments(threshold)
    if moments["m00"] == 0:
        return 0.5

    iris_x = moments["m10"] / moments["m00"]
    gaze_ratio = iris_x / eye_width
    return float(gaze_ratio)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_eye_status(landmarks, gray_frame: np.ndarray) -> str:
    """
    Determine the current eye status from dlib landmarks and the greyscale frame.

    Parameters
    ----------
    landmarks  : dlib.full_object_detection – 68-point shape.
    gray_frame : np.ndarray                 – Greyscale video frame.

    Returns
    -------
    str – One of: "Eyes Closed", "Looking Left", "Looking Right", "Looking Center".
    """
    left_eye_pts = _get_eye_points(landmarks, LEFT_EYE_INDICES)
    right_eye_pts = _get_eye_points(landmarks, RIGHT_EYE_INDICES)

    left_ear = _eye_aspect_ratio(left_eye_pts)
    right_ear = _eye_aspect_ratio(right_eye_pts)
    avg_ear = (left_ear + right_ear) / 2.0

    if avg_ear < EAR_CLOSED_THRESHOLD:
        return "Eyes Closed"

    left_ratio = _gaze_ratio_for_eye(left_eye_pts, gray_frame)
    right_ratio = _gaze_ratio_for_eye(right_eye_pts, gray_frame)
    avg_ratio = (left_ratio + right_ratio) / 2.0

    if avg_ratio < GAZE_RATIO_LEFT:
        return "Looking Left"
    elif avg_ratio > GAZE_RATIO_RIGHT:
        return "Looking Right"
    else:
        return "Looking Center"
