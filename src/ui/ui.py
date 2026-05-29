import streamlit as st


def launch_ui():

    st.set_page_config(
        page_title="Video Anomaly Detection",
        layout="wide"
    )

    st.title("🎥 Video Anomaly Detection")

    st.write(
        "Upload video to detect anomalies"
    )

    uploaded_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file is not None:

        st.video(uploaded_file)

        if st.button("Run Detection"):

            st.success("Detection Started")
