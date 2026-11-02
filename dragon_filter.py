"""Dragon face overlay that activates only while the mouth is open."""

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODEL_DIR / "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)

# Raise this if the effect activates while your mouth is nearly closed.
DRAGON_OPEN_THRESHOLD = 0.45


def _ensure_model():
    """Download the Face Landmarker model once, on the first run."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000:
        return MODEL_PATH

    temporary_path = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".part")
    try:
        print("Downloading the face landmark model...")
        urlretrieve(MODEL_URL, temporary_path)

        if temporary_path.stat().st_size <= 1_000:
            raise RuntimeError("The downloaded model file is unexpectedly small.")

        temporary_path.replace(MODEL_PATH)
        print("Model download complete.")
        return MODEL_PATH
    except Exception as error:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the Face Landmarker model. Check your internet "
            "connection and try again."
        ) from error


def create_face_landmarker():
    """Create a video-mode Face Landmarker with mouth-expression output."""
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(_ensure_model())),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def _jaw_open_score(blendshapes):
    """Return MediaPipe's jawOpen score, or zero if it is unavailable."""
    for category in blendshapes:
        if category.category_name == "jawOpen":
            return float(category.score)
    return 0.0


def _draw_dragon(frame, landmarks):
    """Draw dragon horns, scales, a snout, and fire around a face mesh."""
    height, width = frame.shape[:2]

    def pixel(index):
        point = landmarks[index]
        return np.array([point.x * width, point.y * height], dtype=float)

    # Face mesh indices: outer eye corners, nose tip, and upper/lower lips.
    eye_a = pixel(33)
    eye_b = pixel(263)
    nose = pixel(1)
    upper_lip = pixel(13)
    lower_lip = pixel(14)

    left_eye, right_eye = sorted([eye_a, eye_b], key=lambda point: point[0])
    eye_vector = right_eye - left_eye
    eye_distance = float(np.linalg.norm(eye_vector))
    if eye_distance < 2:
        return

    horizontal = eye_vector / eye_distance
    vertical = np.array([-horizontal[1], horizontal[0]])
    eye_center = (left_eye + right_eye) / 2
    angle = float(np.degrees(np.arctan2(horizontal[1], horizontal[0])))

    def point_from(origin, x_factor, y_factor):
        point = origin + eye_distance * (
            x_factor * horizontal + y_factor * vertical
        )
        return tuple(np.rint(point).astype(int))

    # OpenCV colors are BGR.
    dark_green = (35, 75, 20)
    dragon_green = (45, 155, 55)
    light_green = (95, 205, 95)
    gold = (20, 190, 245)
    orange = (0, 95, 255)
    flame_yellow = (0, 220, 255)
    mouth_dark = (25, 20, 65)

    # Draw two curved-looking pointed horns above the eyes.
    for side in (-1, 1):
        horn = np.array(
            [
                point_from(eye_center, side * 0.52, -0.30),
                point_from(eye_center, side * 1.13, -0.18),
                point_from(eye_center, side * 0.91, -1.22),
                point_from(eye_center, side * 0.70, -0.80),
            ],
            dtype=np.int32,
        )
        horn_highlight = np.array(
            [
                point_from(eye_center, side * 0.69, -0.38),
                point_from(eye_center, side * 0.98, -0.28),
                point_from(eye_center, side * 0.89, -0.95),
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, horn, dragon_green, cv2.LINE_AA)
        cv2.polylines(frame, [horn], True, dark_green, 2, cv2.LINE_AA)
        cv2.fillConvexPoly(frame, horn_highlight, gold, cv2.LINE_AA)

    # Add a small row of forehead scales.
    for x_factor in (-0.22, 0.0, 0.22):
        scale = np.array(
            [
                point_from(eye_center, x_factor - 0.10, -0.50),
                point_from(eye_center, x_factor, -0.76),
                point_from(eye_center, x_factor + 0.10, -0.50),
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, scale, light_green, cv2.LINE_AA)

    # Add a green dragon snout over the nose.
    snout_center = tuple(np.rint(nose).astype(int))
    snout_axes = (
        max(1, round(eye_distance * 0.39)),
        max(1, round(eye_distance * 0.22)),
    )
    cv2.ellipse(
        frame, snout_center, snout_axes, angle, 0, 360,
        dragon_green, -1, cv2.LINE_AA,
    )
    cv2.ellipse(
        frame, snout_center, snout_axes, angle, 0, 360,
        dark_green, 2, cv2.LINE_AA,
    )

    for side in (-1, 1):
        nostril = point_from(nose, side * 0.14, 0.03)
        cv2.circle(frame, nostril, max(2, round(eye_distance * 0.035)), dark_green, -1)

    # Darken the open mouth and draw a flame coming from the lower lip.
    mouth_center = (upper_lip + lower_lip) / 2
    mouth_axes = (
        max(2, round(eye_distance * 0.20)),
        max(2, round(eye_distance * 0.12)),
    )
    cv2.ellipse(
        frame,
        tuple(np.rint(mouth_center).astype(int)),
        mouth_axes,
        angle,
        0,
        360,
        mouth_dark,
        -1,
        cv2.LINE_AA,
    )

    flame_outer = np.array(
        [
            point_from(lower_lip, -0.16, 0.02),
            point_from(lower_lip, -0.35, 0.26),
            point_from(lower_lip, -0.17, 0.55),
            point_from(lower_lip, -0.08, 0.42),
            point_from(lower_lip, 0.02, 0.93),
            point_from(lower_lip, 0.18, 0.56),
            point_from(lower_lip, 0.34, 0.72),
            point_from(lower_lip, 0.30, 0.25),
            point_from(lower_lip, 0.14, 0.02),
        ],
        dtype=np.int32,
    )
    flame_inner = np.array(
        [
            point_from(lower_lip, -0.09, 0.04),
            point_from(lower_lip, -0.18, 0.30),
            point_from(lower_lip, -0.03, 0.62),
            point_from(lower_lip, 0.07, 0.35),
            point_from(lower_lip, 0.17, 0.58),
            point_from(lower_lip, 0.19, 0.20),
            point_from(lower_lip, 0.09, 0.04),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [flame_outer], orange, cv2.LINE_AA)
    cv2.fillPoly(frame, [flame_inner], flame_yellow, cv2.LINE_AA)


def apply_dragon_filter(frame_bgr, result, threshold=DRAGON_OPEN_THRESHOLD):
    """Return (frame, active, jaw_open_score); draw only for an open mouth."""
    output = frame_bgr.copy()
    face_landmarks = getattr(result, "face_landmarks", []) or []
    face_blendshapes = getattr(result, "face_blendshapes", []) or []

    highest_score = 0.0
    active = False

    for index, landmarks in enumerate(face_landmarks):
        if index >= len(face_blendshapes):
            continue

        score = _jaw_open_score(face_blendshapes[index])
        highest_score = max(highest_score, score)

        if score >= threshold:
            _draw_dragon(output, landmarks)
            active = True

    return output, active, highest_score
