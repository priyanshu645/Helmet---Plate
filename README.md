# Helmet and Number Plate Detection System

A Streamlit-based computer vision project for detecting motorcycle riders in traffic videos and identifying whether the rider is wearing a helmet. The project uses a custom YOLOv3 object detection model with OpenCV and a TensorFlow/Keras CNN model for helmet classification.

> Note: The project is designed as a helmet and traffic violation detection system. The current source code mainly implements bike/rider detection and helmet/non-helmet classification. OCR-based number plate text recognition is not implemented in the current version.

## Features

- Upload and process traffic videos through a Streamlit web interface.
- Detect bike/rider regions using a custom YOLOv3 model.
- Classify helmet usage using a trained CNN model.
- Display annotated video frames with bounding boxes and labels.
- Supports `.mp4` and `.avi` video files.
- Uses OpenCV DNN with CUDA fallback handling.
- Validates required model files before running inference.

## Tech Stack

- **Python** - Main programming language
- **Streamlit** - Web application interface
- **OpenCV** - Video processing, object detection, drawing annotations
- **YOLOv3** - Object detection model
- **TensorFlow / Keras** - Helmet classification model
- **CNN** - Helmet vs. no-helmet classification
- **NumPy** - Array and image preprocessing operations
- **imutils** - Frame resizing utility

## Project Structure

```text
.
+-- source.py                    # Main Streamlit application
+-- requirements.txt             # Python dependencies
+-- yolov3-custom.cfg            # YOLOv3 model configuration
+-- yolov3-custom_7000.weights   # Custom-trained YOLOv3 weights
+-- helmet-nonhelmet_cnn.h5      # Trained CNN helmet classifier
+-- temp/                        # Temporary uploaded video storage
+-- README.md                    # Project documentation
```

## Model Files

This project requires the following model files in the project root directory:

| File | Purpose |
| --- | --- |
| `yolov3-custom.cfg` | YOLOv3 network configuration |
| `yolov3-custom_7000.weights` | Custom-trained YOLOv3 weights |
| `helmet-nonhelmet_cnn.h5` | Keras CNN model for helmet classification |

If any required file is missing, the application shows an error message before processing the video.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/Helmet-and-Number-Plate-Detection-and-Recognition.git
cd Helmet-and-Number-Plate-Detection-and-Recognition
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Place the required model files in the project root:

```text
yolov3-custom.cfg
yolov3-custom_7000.weights
helmet-nonhelmet_cnn.h5
```

## Usage

Run the Streamlit application:

```bash
streamlit run source.py
```

Then open the local Streamlit URL in your browser and upload a `.mp4` or `.avi` traffic video.

## How It Works

1. The user uploads a video using the Streamlit interface.
2. The video is saved temporarily inside the `temp/` folder.
3. OpenCV reads the video frame by frame.
4. Each frame is resized and converted into a blob for YOLOv3 inference.
5. YOLOv3 predicts bounding boxes, confidence scores, and class IDs.
6. Non-Maximum Suppression removes duplicate overlapping detections.
7. For detected bike/rider regions, the upper portion of the bounding box is cropped as the helmet region of interest.
8. The cropped region is resized to `224 x 224`, normalized, and passed to the CNN model.
9. The CNN classifies the region as `Helmet` or `No Helmet`.
10. The processed frame is displayed in the Streamlit app with bounding boxes and labels.

## Output

The application displays video frames with:

- Green bounding boxes around detected regions
- `Helmet` label for riders wearing helmets
- `No Helmet` label for riders without helmets

## Requirements

Main dependencies:

```text
streamlit==1.39.0
imutils==0.5.4
numpy==1.24.3
opencv_python==4.6.0.66
Pillow==10.0.1
tensorflow==2.13.0
tensorflow_intel==2.13.0
```

## Limitations

- Number plate OCR/text recognition is not implemented in the current source code.
- Detection accuracy depends on video quality, lighting, camera angle, and model training data.
- Helmet detection uses a cropped region from the detected rider area, which may fail for unusual rider positions.
- CPU inference can be slow for large or high-resolution videos.
- The current app processes uploaded videos, not live CCTV streams.

## Future Enhancements

- Add number plate detection and OCR using EasyOCR or Tesseract.
- Save violation records with timestamp, frame image, and detected plate number.
- Add live camera/CCTV feed support.
- Improve accuracy using YOLOv8/YOLOv11 or another modern object detection model.
- Add output video export/download support.
- Store detection history in a database.

## Applications

- Traffic rule enforcement
- Road safety monitoring
- Smart city surveillance systems
- Automated helmet violation detection
- Video-based traffic analytics

## Acknowledgements

- YOLO object detection framework
- OpenCV computer vision library
- TensorFlow/Keras deep learning framework
- Streamlit web application framework
