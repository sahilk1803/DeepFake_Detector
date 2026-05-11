# Deepfake Detection System - Standalone Version

This is a standalone, production-ready version of the Deepfake Detection System. It contains only the necessary files to run inference (predictions) on videos and does **not** include training data, dataset folders, or training scripts.

## ✨ Features

- **Pre-trained Model**: Includes `best_model.pth` - a fully trained CNN + LSTM model
- **Two UI Options**:
  - **Streamlit App** (`app.py`) - Simple web interface for quick testing
  - **Flask App** (`frontend.py`) - Full-featured web interface with advanced analytics
- **Advanced Detection Algorithms**:
  - Deep Learning (CNN + LSTM) for spatial and temporal feature analysis
  - Blink detection for behavioral verification
  - Lip motion analysis for authenticity verification
  - Hybrid confidence scoring combining all methods
- **No Training Data Required** - Run inference immediately after setup

## 📋 Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- GPU support (optional but recommended for faster inference)

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
# Check if model file exists
dir best_model.pth  # Windows
ls best_model.pth   # macOS/Linux

# If missing, contact the project maintainer
```

### 3. Run the Application

#### Option A: Streamlit (Recommended for Quick Testing)

```bash
streamlit run app.py
```

- Opens at `http://localhost:8501`
- Upload images for classification
- Simple, intuitive interface

#### Option B: Flask (Full-Featured)

```bash
python frontend.py
```

- Opens at `http://127.0.0.1:8501`
- Supports video uploads (MP4, WebM, AVI, MOV)
- Shows detected faces with analysis
- Advanced metrics (blink score, lip motion score)
- Explanability features

#### Option C: Command Line (Script)

```bash
python predict.py
# Enter video path when prompted
```

## 📁 Project Structure

```
vsmlmodel_standalone/
├── app.py                    # Streamlit interface
├── frontend.py              # Flask web interface
├── predict.py               # Inference script
├── model.py                 # Model architecture
├── preprocess.py            # Image preprocessing utilities
├── best_model.pth          # Pre-trained model (required)
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── templates/
│   └── index.html          # Flask HTML template
├── static/
│   └── uploads/            # Uploaded files storage
└── requirements.txt
```

## 🎯 Usage Examples

### Streamlit Example

```python
streamlit run app.py
# 1. Upload an image
# 2. Click "Classify"
# 3. View results with confidence score
```

### Flask Example

```bash
python frontend.py
# 1. Open browser to http://127.0.0.1:8501
# 2. Upload a video file (MP4, WebM, AVI, MOV)
# 3. Click "Analyze Video"
# 4. View detailed analysis with detected faces
```

### Python Script Example

```python
from predict import predict_video

# Run prediction
video_path = "path/to/video.mp4"
prediction, confidence, model_conf, blink_score, lip_motion = predict_video(video_path)

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2f}")
print(f"Model Confidence: {model_conf:.2f}")
print(f"Blink Score: {blink_score:.2f}")
print(f"Lip Motion Score: {lip_motion:.2f}")
```

## 🔧 Configuration

### GPU Support

The system automatically detects GPU availability:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

For CUDA support, ensure NVIDIA GPU drivers are installed.

### Model Parameters

Default model configuration (in `model.py`):
- **CNN Backbone**: ResNet18 (pretrained)
- **LSTM Hidden Size**: 256
- **LSTM Layers**: 2
- **Input Frames**: 20
- **Dropout**: 0.5

## 📊 Understanding Results

### Confidence Scores

| Score Range | Interpretation |
|---|---|
| 90-100% | Very high confidence |
| 70-89% | High confidence |
| 50-69% | Moderate confidence |
| 30-49% | Low confidence |
| 0-29% | Very low confidence |

### Behavioral Metrics (Flask Only)

- **Blink Score** (0-1): Measures natural blinking patterns
  - > 0.7: Likely authentic
  - 0.4-0.7: Mixed signals
  - < 0.4: May indicate synthetic media

- **Lip Motion Score** (0-1): Analyzes mouth movement
  - > 0.7: Natural speech patterns
  - 0.4-0.7: Moderate movement
  - < 0.4: Limited or unnatural motion

## ⚠️ Limitations

1. **Video Quality**: Works best with clear, well-lit videos
2. **Face Visibility**: Requires visible face in at least 80% of frames
3. **Model Bias**: Trained on specific deepfake generation methods
4. **False Positives**: May flag heavily edited authentic videos
5. **Face Detection**: Uses OpenCV, may struggle with:
   - Very small faces (< 80px)
   - Extreme angles or rotations
   - Heavy makeup or masks

## 🐛 Troubleshooting

### "Model file not found: best_model.pth"
- Ensure `best_model.pth` exists in the project root directory
- Download from the project repository if missing

### "No face detected in video"
- Ensure the face is clearly visible
- Check video quality and lighting
- Try a different video clip

### "CUDA out of memory"
- Reduce batch size in preprocessing
- Use CPU instead (automatic fallback)
- Close other GPU-using applications

### "ModuleNotFoundError" when running
```bash
pip install -r requirements.txt
```

## 📝 API Reference

### predict_video()

```python
def predict_video(video_path, model_path='best_model.pth', num_frames=20):
    """
    Analyze a video for deepfake indicators.
    
    Args:
        video_path (str): Path to video file
        model_path (str): Path to trained model
        num_frames (int): Number of frames to extract
    
    Returns:
        tuple: (prediction, confidence, model_conf, blink_score, lip_motion_score)
    """
```

### DeepfakeDetector

```python
class DeepfakeDetector(nn.Module):
    def __init__(self, num_frames=20, hidden_size=256, num_layers=2):
        # CNN + LSTM architecture
```

## 🤝 Contributing

For issues, improvements, or questions:
1. Check existing documentation
2. Review error messages carefully
3. Test with different videos
4. Report with detailed information

## 📄 License

This project is provided as-is for research and educational purposes.

## 🙏 Acknowledgments

- ResNet18 backbone: PyTorch model zoo
- Face detection: OpenCV Haar Cascades
- Deep learning framework: PyTorch

## 📞 Support

For questions or issues:
1. Check the README thoroughly
2. Review the API documentation
3. Test with sample videos
4. Consult project maintainers

---

**Last Updated**: 2024
**Version**: 1.0 (Standalone)
