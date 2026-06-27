"""
suspicion_score.py
------------------
Rule-based suspicion scoring engine.

Each detected condition increments the running score by a fixed weight.
The score decays gradually over time when no violations are detected so
that brief anomalies do not permanently flag a candidate.
"""

# ---------------------------------------------------------------------------
# Score weights for each violation type
# ---------------------------------------------------------------------------
WEIGHTS = {
    "phone_detected":   10,
    "looking_away":      2,
    "eyes_closed":       2,
    "no_face":           5,
    "multiple_faces":    8,
}

# Score above this threshold triggers the "Suspicious Activity Detected" alert
ALERT_THRESHOLD = 15

# Amount the score decays each frame when no violations are present
DECAY_AMOUNT = 1

# Maximum possible score (caps the value to avoid runaway accumulation)
MAX_SCORE = 100


def compute_score(
    current_score: int,
    phone_detected: bool,
    head_direction: str,
    eye_status: str,
    face_count: int,
) -> tuple[int, list[str]]:
    """
    Update and return the suspicion score for the current frame.

    Parameters
    ----------
    current_score   : int  – Score carried over from the previous frame.
    phone_detected  : bool – True if a phone was found in this frame.
    head_direction  : str  – Output from head_pose.get_head_pose_direction().
    eye_status      : str  – Output from eye_movement.get_eye_status().
    face_count      : int  – Number of faces detected in this frame.

    Returns
    -------
    tuple[int, list[str]]
        new_score          : int       – Updated suspicion score.
        active_violations  : list[str] – Human-readable labels for this frame.
    """
    frame_delta = 0
    active_violations: list[str] = []

    # ---- Phone detection ------------------------------------------------
    if phone_detected:
        frame_delta += WEIGHTS["phone_detected"]
        active_violations.append("Phone Detected")

    # ---- Head pose -------------------------------------------------------
    looking_away_directions = {"Looking Left", "Looking Right", "Looking Up"}
    if head_direction in looking_away_directions:
        frame_delta += WEIGHTS["looking_away"]
        active_violations.append(f"Head {head_direction}")

    # ---- Eye status ------------------------------------------------------
    if eye_status == "Eyes Closed":
        frame_delta += WEIGHTS["eyes_closed"]
        active_violations.append("Eyes Closed")

    # ---- Face count ------------------------------------------------------
    if face_count == 0:
        frame_delta += WEIGHTS["no_face"]
        active_violations.append("No Face Detected")
    elif face_count > 1:
        frame_delta += WEIGHTS["multiple_faces"]
        active_violations.append(f"Multiple Faces ({face_count})")

    # ---- Update score ----------------------------------------------------
    if frame_delta > 0:
        new_score = min(current_score + frame_delta, MAX_SCORE)
    else:
        # Gradual decay when the candidate behaves normally
        new_score = max(current_score - DECAY_AMOUNT, 0)

    return new_score, active_violations


def is_suspicious(score: int) -> bool:
    """
    Return True if the score exceeds the alert threshold.

    Parameters
    ----------
    score : int – Current suspicion score.

    Returns
    -------
    bool
    """
    return score >= ALERT_THRESHOLD
