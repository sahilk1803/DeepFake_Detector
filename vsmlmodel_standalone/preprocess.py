import cv2
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_frames(video_path, num_frames=20):
    """
    Extract evenly spaced frames from a video.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return []

    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

def detect_and_crop_face(frame):
    """
    Detect face in frame and crop to face.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = frame[y:y+h, x:x+w]
        return face
    return None

def preprocess_frame(face):
    """
    Resize and normalize the face image.
    """
    if face is None:
        return None
    pil_image = Image.fromarray(face)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(pil_image)


def detect_face_box(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    return faces[0]


def _region_variance(frame, box):
    x, y, w, h = box
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return 0.0
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray_roi, cv2.CV_64F)
    return float(np.std(lap))


def _eye_regions(face_box):
    x, y, w, h = face_box
    left_eye = (int(x + 0.15 * w), int(y + 0.18 * h), int(0.25 * w), int(0.14 * h))
    right_eye = (int(x + 0.60 * w), int(y + 0.18 * h), int(0.25 * w), int(0.14 * h))
    return left_eye, right_eye


def _mouth_region(face_box):
    x, y, w, h = face_box
    return (int(x + 0.28 * w), int(y + 0.55 * h), int(0.44 * w), int(0.20 * h))


def estimate_blink_score(frames, min_frames=5):
    """Estimate blink-based verification score from a short frame sequence."""
    scores = []
    for frame in frames:
        face_box = detect_face_box(frame)
        if face_box is None:
            continue
        left_eye, right_eye = _eye_regions(face_box)
        left_score = _region_variance(frame, left_eye)
        right_score = _region_variance(frame, right_eye)
        scores.append((left_score + right_score) / 2.0)

    if len(scores) < min_frames:
        return 0.5

    open_level = np.percentile(scores, 75)
    closed_level = np.percentile(scores, 25)
    threshold = (open_level + closed_level) / 2.0
    closed = [s < threshold for s in scores]
    transitions = sum(closed[i] != closed[i + 1] for i in range(len(closed) - 1))
    blinks = transitions // 2
    blink_rate = min(1.0, blinks / 2.0)
    return float(0.3 + 0.7 * blink_rate)


def estimate_mouth_motion_score(frames):
    """Estimate mouth-motion-based verification score using kinetic change in the mouth region."""
    last_mean = None
    motion_values = []

    for frame in frames:
        face_box = detect_face_box(frame)
        if face_box is None:
            continue
        mouth_box = _mouth_region(face_box)
        x, y, w, h = mouth_box
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            continue
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        current_mean = float(np.mean(gray_roi))
        if last_mean is not None:
            motion_values.append(abs(current_mean - last_mean))
        last_mean = current_mean

    if not motion_values:
        return 0.5

    avg_motion = float(np.mean(motion_values))
    return float(min(1.0, 0.2 + 0.8 * min(avg_motion / 20.0, 1.0)))


def compute_rule_confidence(frames):
    """Combine blink and mouth-motion scores into a verification confidence."""
    blink_score = estimate_blink_score(frames)
    lip_motion_score = estimate_mouth_motion_score(frames)
    rule_confidence = 0.6 * blink_score + 0.4 * lip_motion_score
    return {
        'rule_confidence': float(rule_confidence),
        'blink_score': float(blink_score),
        'lip_motion_score': float(lip_motion_score),
    }
