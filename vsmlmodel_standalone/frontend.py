import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
import cv2
import numpy as np
from predict import predict_video, extract_frames_with_highlights

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'webm'}
VIDEO_EXTENSIONS = ALLOWED_EXTENSIONS

app = Flask(__name__)
app.secret_key = 'change_this_secret'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class DeepfakeDetector(nn.Module):
    def __init__(self, num_frames=20, hidden_size=256, num_layers=2):
        super(DeepfakeDetector, self).__init__()
        self.cnn = resnet18(weights='DEFAULT')
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        x = x.view(batch_size * seq_len, C, H, W)
        features = self.cnn(x)
        features = features.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(features)
        final_out = lstm_out[:, -1, :]
        out = self.fc(final_out)
        return out

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video_file(filename):
    return filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS


def load_model():
    model = DeepfakeDetector()
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, 'No face detected in the uploaded image.'

    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face_pil = Image.fromarray(face)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(face_pil)
    tensor = tensor.unsqueeze(0).repeat(20, 1, 1, 1).unsqueeze(0)
    return tensor, None


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    filepath = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            if not is_video_file(filename):
                flash('Only video uploads are supported. Please upload a .mp4, .mov, .avi, or .webm file.')
                return redirect(request.url)

            prediction, final_conf, model_conf, blink_score, lip_motion_score = predict_video(filepath)
            if isinstance(prediction, str):
                result = 'Deepfake' if prediction == 'FAKE' else 'Real'
                confidence = final_conf * 100
                return render_template(
                    'index.html', result=result, confidence=confidence,
                    filename=filename, is_video=True,
                    model_confidence=model_conf * 100,
                    blink_score=blink_score, lip_motion_score=lip_motion_score
                )
            flash('Unable to process the uploaded video.')
            return redirect(request.url)

            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output).item()
                if prob > 0.5:
                    result = 'Deepfake'
                    confidence = prob * 100
                else:
                    result = 'Real'
                    confidence = (1 - prob) * 100

                return render_template('index.html', result=result, confidence=confidence, filename=filename, is_video=False)

        else:
            flash('Allowed file types: png, jpg, jpeg, mp4, mov, avi, webm')
            return redirect(request.url)

    return render_template('index.html', result=result, confidence=confidence)


@app.route('/extract_faces/<filename>', methods=['GET'])
def extract_faces(filename):
    """Extract faces with highlights from the uploaded video."""
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Extract frames with highlights
        face_data = extract_frames_with_highlights(filepath, num_frames=20)
        
        if not face_data:
            return jsonify({'error': 'No faces detected in video'}), 400
        
        return jsonify({
            'success': True,
            'faces': face_data,
            'explanations': {
                'green_box': 'Face Detection - Green box highlights the detected face region',
                'blue_boxes': 'Blink Analysis - Blue boxes show eye regions used for blink detection. More blinks indicate real faces.',
                'red_box': 'Lip Motion Analysis - Red box shows mouth region for lip motion detection. More motion indicates real faces.'
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8501, debug=False)
