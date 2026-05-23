import streamlit as st
import cv2
import numpy as np
import os
import imutils
import gdown

try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_IMPORT_ERROR = None
except Exception as exc:
    load_model = None
    TENSORFLOW_IMPORT_ERROR = exc

YOLO_WEIGHTS = "yolov3-custom_7000.weights"
YOLO_CONFIG = "yolov3-custom.cfg"
HELMET_MODEL = "helmet-nonhelmet_cnn.h5"

MODEL_URLS = {
    "yolov3-custom_7000.weights": "https://drive.google.com/file/d/1r80O-sRrfOyivbofGRnNTp-fnpGzwAyZ/view?usp=drive_link",
    "helmet-nonhelmet_cnn.h5": "https://drive.google.com/file/d/1CdJfgHlBItxpIQtT2pMaI3XeCC9C0jzj/view?usp=drive_link",
}

def load_yolo():
    net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        st.info("Using CUDA backend for YOLO inference.")
    except cv2.error:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        st.info("CUDA is unavailable. Falling back to CPU inference.")
    return net


for file_name, url in MODEL_URLS.items():
    if not os.path.exists(file_name):
        with st.spinner(f"Downloading {file_name}..."):
            gdown.download(url, file_name, quiet=False, fuzzy=True)

missing_files = [path for path in [YOLO_WEIGHTS, YOLO_CONFIG, HELMET_MODEL] if not os.path.exists(path)]

st.title("Bike,Helmet and Number Plate Detection and Recognition")

if missing_files:
    st.error(
        "Missing required model files: "
        + ", ".join(missing_files)
        + ". Add them to the project folder, then rerun the app."
    )
elif TENSORFLOW_IMPORT_ERROR is not None:
    st.error(
        "TensorFlow could not be imported. This project needs a TensorFlow build that matches your Python version. "
        f"Import error: {TENSORFLOW_IMPORT_ERROR}"
    )
else:
    net = load_yolo()
    model = load_model(HELMET_MODEL)
    st.success("Models loaded successfully.")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
    if uploaded_file is not None:
        os.makedirs("temp", exist_ok=True)

        # Save the uploaded file to a temporary directory
        temp_file_path = os.path.join("temp", uploaded_file.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        video = cv2.VideoCapture(temp_file_path)

        if not video.isOpened():
            st.error("Error: Could not open video file.")
        else:
            stframe = st.empty()

            while True:
                ret, frame = video.read()

                if not ret:
                    break

                frame = imutils.resize(frame, height=500)
                height, width = frame.shape[:2]

                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = [net.forward("yolo_82"), net.forward("yolo_94"), net.forward("yolo_106")]

                confidences = []
                boxes = []
                classIds = []

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.3:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            classIds.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        if classIds[i] == 0:  # bike
                            helmet_roi = frame[max(0, y):max(0, y) + max(0, h) // 4, max(0, x):max(0, x) + max(0, w)]
                            if helmet_roi.shape[0] > 0 and helmet_roi.shape[1] > 0:
                                helmet_roi = cv2.resize(helmet_roi, (224, 224))
                                helmet_roi = np.array(helmet_roi, dtype="float32")
                                helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
                                helmet_roi = helmet_roi / 255.0
                                prediction = int(model.predict(helmet_roi, verbose=0)[0][0])
                                if prediction == 0:
                                    frame = cv2.putText(
                                        frame,
                                        "Helmet",
                                        (x, y - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 255, 0),
                                        2,
                                    )
                                else:
                                    frame = cv2.putText(
                                        frame,
                                        "No Helmet",
                                        (x, y - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7,
                                        (0, 0, 255),
                                        2,
                                    )
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                stframe.image(frame, channels="BGR", use_column_width=True)

            video.release()
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
