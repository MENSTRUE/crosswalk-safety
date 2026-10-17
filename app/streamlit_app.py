from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import CrosswalkDetector
from src.pipeline import CrosswalkSafetyPipeline
from src.roi import DEFAULT_ROI


st.set_page_config(
    page_title="Crosswalk Safety",
    page_icon="🚸",
    layout="wide",
)

st.title("🚸 Crosswalk Safety")
st.caption(
    "YOLO11n-based safety-zone conflict screening for pedestrian and vehicle interactions."
)

st.warning(
    "This MVP uses a configurable safety-zone ROI proxy. "
    "A detected conflict is not automatically a traffic violation."
)

with st.sidebar:
    st.header("Inference settings")

    confidence = st.slider(
        "YOLO confidence",
        0.05, 0.95, 0.25, 0.05,
    )

    iou = st.slider(
        "YOLO IoU",
        0.10, 0.90, 0.50, 0.05,
    )

    sample_every = st.slider(
        "Sample every (seconds)",
        0.5, 10.0, 2.0, 0.5,
    )

    event_gap = st.slider(
        "Event max gap (seconds)",
        1.0, 15.0, 4.0, 1.0,
    )

    st.subheader("Default normalized ROI")
    st.code(
        json.dumps(DEFAULT_ROI.tolist(), indent=2),
        language="json",
    )

    st.caption(
        "ROI matches the v3 notebook. Coordinates are normalized [x, y]."
    )


@st.cache_resource(show_spinner="Loading YOLO11n...")
def load_detector(conf: float, iou_value: float) -> CrosswalkDetector:
    device = 0 if torch.cuda.is_available() else "cpu"

    return CrosswalkDetector(
        model_name="yolo11n.pt",
        confidence=conf,
        iou=iou_value,
        image_size=640,
        max_detections=300,
        device=device,
    )


uploaded = st.file_uploader(
    "Upload a traffic video",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded is None:
    st.info("Upload one traffic video to start.")
    st.stop()

suffix = Path(uploaded.name).suffix or ".mp4"

with tempfile.TemporaryDirectory(prefix="crosswalk_safety_") as temp_dir:
    temp_dir_path = Path(temp_dir)
    input_path = temp_dir_path / f"input{suffix}"
    output_dir = temp_dir_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path.write_bytes(uploaded.getbuffer())

    st.subheader("Input video")
    st.video(uploaded)

    if not st.button(
        "Run safety-zone screening",
        type="primary",
        use_container_width=True,
    ):
        st.stop()

    progress = st.progress(0.0)
    status = st.empty()

    def on_progress(value: float, message: str) -> None:
        progress.progress(min(max(float(value), 0.0), 1.0))
        status.caption(message)

    detector = load_detector(confidence, iou)

    pipeline = CrosswalkSafetyPipeline(
        detector=detector,
        sample_every_seconds=sample_every,
        event_max_gap_seconds=event_gap,
        normalized_roi=DEFAULT_ROI,
    )

    with st.spinner("Processing video..."):
        result = pipeline.process_video(
            input_video=input_path,
            output_dir=output_dir,
            save_annotated_video=True,
            progress_callback=on_progress,
        )

    progress.progress(1.0)
    status.success("Inference finished.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Sampled frames",
        result.summary["sampled_frames"],
    )
    c2.metric(
        "Conflict frames",
        result.summary["conflict_frames"],
    )
    c3.metric(
        "Grouped events",
        result.summary["grouped_conflict_events"],
    )
    c4.metric(
        "Conflict-frame rate",
        f"{result.summary['conflict_frame_rate'] * 100:.1f}%",
    )

    st.caption(
        "Conflict-frame rate is the proportion of sampled frames where at least "
        "one vehicle and one pedestrian were simultaneously inside the ROI. "
        "It is not a violation rate."
    )

    if result.annotated_video_path:
        annotated_path = Path(result.annotated_video_path)
        if annotated_path.exists():
            st.subheader("Annotated sampled-video preview")
            st.video(annotated_path.read_bytes())

    st.subheader("Grouped temporal conflict events")

    if result.events.empty:
        st.success("No grouped conflict events found.")
    else:
        event_view = result.events.copy()
        for col in ["start_seconds", "end_seconds", "duration_seconds"]:
            event_view[col] = event_view[col].round(2)

        st.dataframe(
            event_view,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Sampled frame results"):
        st.dataframe(
            result.frame_results,
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "Download conflict_events.csv",
        data=result.events.to_csv(index=False).encode("utf-8"),
        file_name="conflict_events.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        "Download sampled_frame_results.csv",
        data=result.frame_results.to_csv(index=False).encode("utf-8"),
        file_name="sampled_frame_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
