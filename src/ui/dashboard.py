from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from infer import run_inference

VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "wmv", "flv", "gif"]
IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "tif", "tiff"]


def _read_image_bgr(path: str) -> np.ndarray | None:
    if not path:
        return None
    return cv2.imread(path)


def _show_frame_grid(rows: list[dict], max_items: int) -> None:
    rows = rows[:max_items]
    if not rows:
        st.warning("Không có frame phù hợp để hiển thị.")
        return

    cols = st.columns(4)
    for i, row in enumerate(rows):
        img = _read_image_bgr(row.get("frame_path"))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = "ANOMALY" if row["is_anomaly"] else "NORMAL"
        caption = f"Frame {row['frame_index']} · {label} · score {row['score']:.6f}"
        cols[i % 4].image(img, caption=caption, use_container_width=True)


def launch_ui() -> None:
    st.set_page_config(page_title="Video Anomaly Detection", page_icon="🎥", layout="wide")

    st.title("🎥 Video Anomaly Detection")
    st.caption("Autoencoder reconstruction-based detection")

    with st.sidebar:
        st.header("Cấu hình")
        model_path = st.text_input("Model path", value="outputs/model.pth")
        st.caption("File `model.pth` đã được đặt sẵn tại `outputs/model.pth` trong bản project này.")

        threshold_mode = st.radio("Threshold", ["Auto percentile", "Manual"], horizontal=False)
        auto_percentile = st.slider("Auto percentile", 80.0, 99.9, 95.0, 0.5)
        manual_threshold = st.number_input("Manual threshold", min_value=0.0, value=0.001, step=0.0001, format="%.6f")

        st.divider()
        stride = st.slider("Frame stride", 1, 10, 2, help="2 = xử lý mỗi 2 frame. Tăng lên 3–5 nếu video dài và cần nhanh hơn.")
        batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64, 128], value=64)
        max_frames = st.number_input("Max frames (0 = không giới hạn)", min_value=0, value=0, step=100)
        top_k = st.slider("Top frame hiển thị", 4, 80, 24)
        save_video = st.checkbox("Xuất annotated video", value=False)
        max_display_width = st.select_slider("Output video width", options=[480, 640, 960, 1280], value=960)

    uploaded = st.file_uploader("Upload video hoặc ảnh", type=VIDEO_TYPES + IMAGE_TYPES)
    if uploaded is None:
        st.info("Upload file để bắt đầu. Với video dài, dùng `Frame stride = 2–5` để tăng tốc đáng kể.")
        return

    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        input_path = tmp.name

    try:
        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("Input")
            if suffix.lstrip(".") in VIDEO_TYPES:
                st.video(input_path)
            else:
                st.image(input_path, use_container_width=True)

        if not Path(model_path).exists():
            st.error(f"Không tìm thấy model: `{model_path}`")
            return

        if st.button("🚀 Run Detection", type="primary"):
            progress = st.progress(0, text="Đang đọc video và chạy model...")

            def update_progress(done: int, total: int | None) -> None:
                if total:
                    progress.progress(min(done / total, 1.0), text=f"Đã xử lý {done}/{total} frame mẫu")
                else:
                    progress.progress(0.1, text=f"Đã xử lý {done} frame mẫu")

            threshold = manual_threshold if threshold_mode == "Manual" else None
            with st.spinner("Đang chạy inference..."):
                summary = run_inference(
                    input_path=input_path,
                    model_path=model_path,
                    output_dir="outputs/inference",
                    threshold=threshold,
                    auto_threshold_percentile=auto_percentile,
                    batch_size=batch_size,
                    stride=stride,
                    max_frames=None if max_frames == 0 else int(max_frames),
                    top_k=top_k,
                    save_frames=True,
                    save_video=save_video,
                    max_display_width=max_display_width,
                    progress_callback=update_progress,
                )
            progress.progress(1.0, text="Hoàn tất")

            rows = summary["results"]
            df = pd.DataFrame(rows)
            scores = df["score"].to_numpy(dtype=float)

            with right:
                st.subheader("Kết quả")
                c1, c2, c3 = st.columns(3)
                c1.metric("Frames đã xử lý", f"{summary['processed_frames']}")
                c2.metric("Anomaly", f"{summary['anomaly_frames']}")
                c3.metric("Threshold", f"{summary['threshold']:.6f}")
                st.write(f"Device: `{summary['device']}` · Batch: `{batch_size}` · Stride: `{stride}`")

            st.subheader("Anomaly score theo thời gian")
            fig = px.line(df, x="frame_index", y="score", markers=False)
            fig.add_hline(y=summary["threshold"], line_dash="dash", annotation_text="threshold")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Top anomaly frames")
            top_rows = sorted(rows, key=lambda r: r["score"], reverse=True)
            _show_frame_grid(top_rows, top_k)

            st.subheader("Bảng kết quả")
            st.dataframe(
                df[["frame_index", "time_seconds", "score", "threshold", "is_anomaly", "frame_path"]],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download scores.csv",
                data=Path(summary["scores_csv"]).read_bytes(),
                file_name="scores.csv",
                mime="text/csv",
            )

            if summary.get("annotated_video") and Path(summary["annotated_video"]).exists():
                st.subheader("Annotated video")
                st.video(summary["annotated_video"])
                st.download_button(
                    "Download annotated_output.mp4",
                    data=Path(summary["annotated_video"]).read_bytes(),
                    file_name="annotated_output.mp4",
                    mime="video/mp4",
                )
    finally:
        try:
            os.unlink(input_path)
        except OSError:
            pass
