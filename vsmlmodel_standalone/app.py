import streamlit as st
import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

# Model definition (same as in model.py)
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

# Load model
@st.cache_resource
def load_model():
    model = DeepfakeDetector()
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_image(image):
    # Convert PIL to OpenCV
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        st.error("No face detected in the image.")
        return None
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face_pil = Image.fromarray(face)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(face_pil)
    # Repeat to 20 frames
    tensor = tensor.unsqueeze(0).repeat(20, 1, 1, 1).unsqueeze(0)  # (1, 20, 3, 224, 224)
    return tensor

st.title("Deepfake Detector")
st.write("Upload an image to classify it as real or deepfake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Classify"):
        with st.spinner("Processing..."):
            input_tensor = preprocess_image(image)
            if input_tensor is not None:
                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.sigmoid(output).item()
                    confidence = prob * 100 if prob > 0.5 else (1 - prob) * 100
                    label = "Deepfake" if prob > 0.5 else "Real"
                st.success(f"Prediction: {label}")
                st.info(f"Confidence: {confidence:.2f}%")
