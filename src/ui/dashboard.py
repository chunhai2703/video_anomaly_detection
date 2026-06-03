# src/ui/dashboard.py
import streamlit as st
from infer import detect_anomalies
import cv2
import tempfile
import os
import numpy as np
import plotly.express as px

def launch_ui():
    st.set_page_config(page_title="Video Anomaly Detection", layout="wide")

    # Header
    st.markdown("""
    <div style='background-color: #2c3e50; padding: 15px; border-radius: 8px'>
        <h1 style='color:white; text-align:center;'>Video Anomaly Detection</h1>
    </div>
    """, unsafe_allow_html=True)

    # Upload Panel
    st.subheader("Upload Video")
    uploaded_file = st.file_uploader("Select video file", type=["mp4", "avi", "mov"])
    threshold_slider = st.slider("Anomaly Threshold", 0.0, 1.0, 0.5, 0.0001)  # có thể giảm để match score
    heatmap_toggle = st.checkbox("Show Heatmap Overlay", value=True)

    if uploaded_file is not None:
        # Tạo file tạm với suffix đúng file gốc
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
        tfile.write(uploaded_file.read())
        tfile.flush()
        video_path = tfile.name
        tfile.close()

        st.video(video_path)

        if st.button("Run Detection"):
            st.info("Processing video...")

            # detect_anomalies trả về list [(frame, score)]
            results = detect_anomalies(video_path)

            if not results:
                st.warning("No frames returned from inference.")
                return

            # Lấy auto threshold từ dữ liệu inference
            scores_only = [score for _, score in results]
            auto_threshold = np.percentile(scores_only, 95.0)

            st.success(f"Detection complete (auto threshold = {auto_threshold:.6f})")

            # Plot anomaly scores
            fig = px.line(y=scores_only, title="Anomaly Score Over Time", labels={'y':'Score', 'x':'Frame'})
            st.plotly_chart(fig, width='stretch')

            # Chỉ hiển thị frame vượt auto threshold
            anomalous_frames = [(frame, score) for frame, score in results if score >= auto_threshold]

            if anomalous_frames:
                st.subheader("Anomalous Frames (auto threshold)")
                num_cols = 4
                cols = st.columns(num_cols)
                for idx, (frame, score) in enumerate(anomalous_frames):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    col = cols[idx % num_cols]
                    col.image(frame_rgb, caption=f"Frame {idx} | Score {score:.6f}", width='stretch')

        # Xóa file tạm sau khi xong
        if os.path.exists(video_path):
            os.unlink(video_path)

if __name__ == "__main__":
    launch_ui()