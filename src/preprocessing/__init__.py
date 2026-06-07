"""Preprocessing public API."""

from .preprocessing import preprocess_batch, preprocess_frame, resize_keep_aspect

# Backward-compatible imports for older team scripts/tests.
try:
    from .video_loader import get_video_info, load_video
    from .frame_extractor import extract_frames, save_frames
except Exception:  # pragma: no cover
    pass

# Old code used `preprocess(...)`; keep it as an alias.
preprocess = preprocess_frame
