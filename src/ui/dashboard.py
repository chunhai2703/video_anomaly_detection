import streamlit as st
from infer import detect_anomalies
import cv2
import tempfile
import os
import numpy as np
import plotly.express as px

def launch_ui():
    st.set_page_config(page_title="Video Anomaly Detection", layout="wide")

    st.markdown("""
    <div style='background-color: #2c3e50; padding: 15px; border-radius: 8px'>
        <h1 style='color:white; text-align:center;'>Video Anomaly Detection</h1>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Upload Video / Image")
    uploaded_file = st.file_uploader(
        "Select video or image file",
        type=[
            "mp4", "avi", "mov", "mkv", "wmv", "flv", "gif",
            "jpg", "jpeg", "png", "bmp", "tif", "tiff"
        ]
    )

    threshold_slider = st.slider("Anomaly Threshold", 0.0, 1.0, 0.5, 0.0001) 
    heatmap_toggle = st.checkbox("Show Heatmap Overlay", value=True)

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(uploaded_file.read())
        tfile.flush()
        file_path = tfile.name
        tfile.close()

        # Hiển thị preview tùy loại file
        ext = os.path.splitext(uploaded_file.name)[1].lower()
        if ext in [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".gif"]:
            st.video(file_path)
        else:
            from PIL import Image
            img = Image.open(file_path)
            st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Run Detection"):
            st.info("Processing...")
            results = detect_anomalies(file_path)

            if not results:
                st.warning("No frames returned from inference.")
                return

            scores_only = [score for _, score in results]
            auto_threshold = np.percentile(scores_only, 95.0)

            st.success(f"Detection complete (auto threshold = {auto_threshold:.6f})")

            fig = px.line(y=scores_only, title="Anomaly Score Over Time", labels={'y':'Score', 'x':'Frame'})
            st.plotly_chart(fig, width='stretch')

            anomalous_frames = [(frame, score) for frame, score in results if score >= auto_threshold]
            if anomalous_frames:
                st.subheader("Anomalous Frames (auto threshold)")
                num_cols = 4
                cols = st.columns(num_cols)
                for idx, (frame, score) in enumerate(anomalous_frames):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    col = cols[idx % num_cols]
                    col.image(frame_rgb, caption=f"Frame {idx} | Score {score:.6f}", width='stretch')

        if os.path.exists(file_path):
            os.unlink(file_path)

if __name__ == "__main__":
    launch_ui()