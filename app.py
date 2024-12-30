import gradio as gr
from huggingface_hub import snapshot_download
from ultralytics import YOLO
import os
from PIL import Image
import cv2
import numpy as np
import tempfile

# ITI107-2024s2 model path location
MODEL_REPO_ID = "ITI107-2024S2/6319250G"

# Load model
def load_model(repo_id):
    download_dir = snapshot_download(repo_id)
    path = os.path.join(download_dir, "best_int8_openvino_model")
    detection_model = YOLO(path, task="detect")
    return detection_model

# Initialize the model
detection_model = load_model(MODEL_REPO_ID)

# Student Information
My_info = "Student ID:6319250G, Name: Aung Hlaing Tun"

# Prediction function for images
def predict_image(pil_img):
    result = detection_model.predict(pil_img, conf=0.5, iou=0.5)
    img_bgr = result[0].plot()  # Annotated image
    out_pilimg = Image.fromarray(img_bgr[..., ::-1])  # Convert to RGB PIL image
    return out_pilimg

# Prediction function for videos
def predict_video(video):
    # Read the uploaded video
    cap = cv2.VideoCapture(video)
    frames = []
    temp_dir = tempfile.mkdtemp()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection on each frame
        result = detection_model.predict(frame, conf=0.5, iou=0.5)
        annotated_frame = result[0].plot()
        frames.append(annotated_frame)

    cap.release()

    # Save annotated video
    height, width, _ = frames[0].shape
    output_path = os.path.join(temp_dir, "annotated_video.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))
    
    for frame in frames:
        out.write(frame)

    out.release()
    return output_path

# UI Interface
with gr.Blocks() as interface:
    # Header
    gr.Markdown("# 🦈 😁 Two Objects Detection (Shark/Mask)")
    gr.Markdown(f"*{My_info}*")
    gr.Markdown(
        """
        *Description*: This app detects objects in images and videos using a YOLO-based OpenVINO model. 
        Upload an image or video to see the detections.
        """
    )
    
    # Image Section
    with gr.Tab("Image Detection"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Upload an Image:")
                input_image = gr.Image(type="pil", label="Input Image")
            
            with gr.Column():
                gr.Markdown("### Detection Results:")
                output_image = gr.Image(type="pil", label="Output Image")
        
        submit_btn_image = gr.Button("Detect Objects in Image")
        submit_btn_image.click(fn=predict_image, inputs=input_image, outputs=output_image)

    # Video Section
    with gr.Tab("Video Detection"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Upload a Video:")
                input_video = gr.Video(label="Input Video")
            
            with gr.Column():
                gr.Markdown("### Detection Results:")
                output_video = gr.Video(label="Output Video")
        
        submit_btn_video = gr.Button("Detect Objects in Video")
        submit_btn_video.click(fn=predict_video, inputs=input_video, outputs=output_video)

# Launch app
interface.launch(share=True)
