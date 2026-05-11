import torch
from model import DeepfakeDetector
from preprocess import extract_frames, detect_and_crop_face, preprocess_frame, compute_rule_confidence, detect_face_box, _eye_regions, _mouth_region
import os
import cv2
import base64
from io import BytesIO


def _pad_frames(frames, target_length):
    if len(frames) >= target_length:
        return frames[:target_length]
    pad_frame = torch.zeros_like(frames[0])
    return frames + [pad_frame] * (target_length - len(frames))


def extract_frames_with_highlights(video_path, num_frames=20):
    """Extract frames with highlighted regions for face, eyes, and mouth."""
    frames = extract_frames(video_path, num_frames)
    face_data = []
    
    for idx, frame in enumerate(frames):
        face_box = detect_face_box(frame)
        if face_box is None:
            continue
            
        # Create a copy to draw highlights
        frame_with_highlights = frame.copy()
        
        # Draw face bounding box in green
        x, y, w, h = face_box
        cv2.rectangle(frame_with_highlights, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Draw eye regions in blue (blink detection)
        left_eye, right_eye = _eye_regions(face_box)
        ex1, ey1, ew1, eh1 = left_eye
        ex2, ey2, ew2, eh2 = right_eye
        cv2.rectangle(frame_with_highlights, (ex1, ey1), (ex1 + ew1, ey1 + eh1), (255, 0, 0), 2)
        cv2.rectangle(frame_with_highlights, (ex2, ey2), (ex2 + ew2, ey2 + eh2), (255, 0, 0), 2)
        
        # Draw mouth region in red (lip motion detection)
        mx, my, mw, mh = _mouth_region(face_box)
        cv2.rectangle(frame_with_highlights, (mx, my), (mx + mw, my + mh), (0, 0, 255), 2)
        
        # Add labels
        cv2.putText(frame_with_highlights, f'Face {idx+1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_with_highlights, 'Blink Area', (ex1, ey1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame_with_highlights, 'Lip Motion', (mx, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Convert to base64 for web display
        _, buffer = cv2.imencode('.jpg', frame_with_highlights)
        img_str = base64.b64encode(buffer).decode()
        
        face_data.append({
            'frame_idx': idx,
            'image': f'data:image/jpeg;base64,{img_str}',
            'face_box': face_box.tolist()
        })
    
    return face_data

def predict_video(video_path, model_path='best_model.pth', num_frames=20):
    # Load model
    model = DeepfakeDetector()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Process video
    frames = extract_frames(video_path, num_frames)
    processed_frames = []
    for frame in frames:
        face = detect_and_crop_face(frame)
        processed = preprocess_frame(face)
        if processed is not None:
            processed_frames.append(processed)

    if not processed_frames:
        return "Unable to detect faces", 0.0, 0.0, 0.0, 0.0

    processed_frames = _pad_frames(processed_frames, num_frames)
    input_tensor = torch.stack(processed_frames).unsqueeze(0).to(device)  # (1, seq, C, H, W)

    verification = compute_rule_confidence(frames)
    rule_score = verification['rule_confidence']
    blink_score = verification['blink_score']
    lip_motion_score = verification['lip_motion_score']

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        model_prediction = 'FAKE' if prob > 0.5 else 'REAL'
        model_confidence = prob if prob > 0.5 else 1 - prob

    fake_rule_support = float(0.7 * (1 - lip_motion_score) + 0.3 * max(0.0, 0.5 - blink_score) * 2.0)

    if model_prediction == 'REAL' and fake_rule_support > 0.55:
        final_prediction = 'FAKE'
        final_confidence = float(min(1.0, 0.5 + 0.5 * fake_rule_support))
    else:
        final_prediction = model_prediction
        if final_prediction == 'REAL':
            final_confidence = float(0.75 * model_confidence + 0.25 * rule_score)
        else:
            final_confidence = float(0.7 * model_confidence + 0.3 * (1 - rule_score))

    return final_prediction, final_confidence, model_confidence, blink_score, lip_motion_score

if __name__ == "__main__":
    video_path = input("Enter video path: ")
    pred, final_conf, model_conf, blink_score, lip_motion_score = predict_video(video_path)
    print(f"Prediction: {pred}")
    print(f"Final confidence: {final_conf:.4f}")
    print(f"Model confidence: {model_conf:.4f}")
    print(f"Blink score: {blink_score:.4f}, Lip-motion score: {lip_motion_score:.4f}")
