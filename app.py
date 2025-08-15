import streamlit as st
from ultralytics import YOLO
import cv2
import yt_dlp
import numpy as np
import tempfile
import os

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Intelligent Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide"
)
st.title("🚦 Intelligent Traffic Sign Recognition System")
st.write("""
This application uses a **YOLOv8** model to detect traffic signs in real-time.
Choose your input source from the sidebar to get started.
""")

# --- HELPER FUNCTIONS (No Changes Here) ---
def download_youtube_video(url):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': os.path.join(tempfile.gettempdir(), '%(id)s.%(ext)s'),
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info_dict)
    except Exception as e:
        st.error(f"Error downloading YouTube video: {e}")
        return None

# --- MAIN APP LOGIC ---

# Load the trained YOLOv8 model
try:
    model = YOLO('best.pt')
except Exception as e:
    st.error(f"Error loading the YOLO model: {e}")
    st.error("Please make sure the 'best.pt' file is in the correct directory.")
    st.stop()

# --- SIDEBAR FOR INPUT SELECTION ---
with st.sidebar:
    st.header("Input Source")
    input_source = st.radio(
        "Select where to get the image/video from:",
        ("Upload Image", "Upload Video", "YouTube URL")
    )

# --- PROCESSING BASED ON INPUT SOURCE ---

if input_source == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        col1, col2 = st.columns(2)
        with col1:
            st.image(opencv_image, channels="BGR", caption="Original Image", use_container_width=True)

        with st.spinner('Running detection...'):
            # This is where you can specify the resolution for maximum accuracy
            results = model.predict(opencv_image, conf=0.387, imgsz=1280)  # <-- MODIFIED LINE
            annotated_image = results[0].plot()

        with col2:
            st.image(annotated_image, channels="BGR", caption="Annotated Image", use_container_width=True)


elif input_source == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        if st.button('Start Detection on Video'):
            with st.spinner('Processing video... this might take a moment.'):
                video_capture = cv2.VideoCapture(video_path)
                st_frame = st.empty()

                while video_capture.isOpened():
                    ret, frame = video_capture.read()
                    if not ret:
                        break
                    
                    # You can experiment here. Use 1280 for accuracy, 640 for speed.
                    results = model.predict(frame, conf=0.387, imgsz=1280)  # <-- MODIFIED LINE
                    annotated_frame = results[0].plot()

                    st_frame.image(annotated_frame, channels="BGR", use_container_width=True)

                video_capture.release()
                os.remove(video_path)
                st.success('Video processing complete!')


elif input_source == "YouTube URL":
    youtube_url = st.text_input("Enter YouTube Video URL")
    if youtube_url and st.button('Start Detection on YouTube Video'):
        video_path = download_youtube_video(youtube_url)
        if video_path:
            with st.spinner('Processing video... this might take a moment.'):
                video_capture = cv2.VideoCapture(video_path)
                st_frame = st.empty()

                while video_capture.isOpened():
                    ret, frame = video_capture.read()
                    if not ret:
                        break

                    # You can experiment here. Use 1280 for accuracy, 640 for speed.
                    results = model.predict(frame, conf=0.387, imgsz=1280)  # <-- MODIFIED LINE
                    annotated_frame = results[0].plot()

                    st_frame.image(annotated_frame, channels="BGR", use_container_width=True)

                video_capture.release()
                os.remove(video_path)
                st.success('Video processing complete!')
