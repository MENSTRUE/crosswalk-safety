from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st
import torch

PROJECT_ROOT=Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0,str(PROJECT_ROOT))

from src.detector import CrosswalkDetector
from src.pipeline import CrosswalkSafetyV4Pipeline
from src.temporal_model import TemporalModelBundle

MODEL_DIR=PROJECT_ROOT/"models"
CHECKPOINT_PATH=MODEL_DIR/"crosswalk_temporal_cnn_bilstm_v4.pt"
MEAN_PATH=MODEL_DIR/"feature_mean.npy"
SCALE_PATH=MODEL_DIR/"feature_scale.npy"

st.set_page_config(
    page_title="Crosswalk Safety v4",
    page_icon="🚸",
    layout="wide",
)

st.title("🚸 Crosswalk Safety v4")
st.caption("YOLO11n bbox preprocessing + custom Temporal Conv1D-BiLSTM")

st.warning(
    "Weakly supervised safety-conflict screening. "
    "Predictions are not validated traffic-violation labels."
)

with st.sidebar:
    st.header("Model")
    st.write("Detector: **YOLO11n pretrained COCO**")
    st.write("Temporal model: **Conv1D + BiLSTM**")
    st.write("History: **8 sampled frames / 16 s**")
    st.write("Horizon: **next sampled frame / 2 s**")

    confidence=st.slider("YOLO confidence",0.05,0.95,0.25,0.05)
    iou=st.slider("YOLO IoU",0.10,0.90,0.50,0.05)

    st.divider()
    st.caption(
        "Held-out test vs weak/pseudo labels: "
        "Recall 80.84% • F1 67.28% • ROC-AUC 82.02%."
    )

missing=[
    p.name for p in [CHECKPOINT_PATH,MEAN_PATH,SCALE_PATH]
    if not p.exists()
]
if missing:
    st.error("Missing model artifacts in models/: "+", ".join(missing))
    st.stop()

@st.cache_resource(show_spinner="Loading trained models...")
def load_models(conf,iou_value):
    detector=CrosswalkDetector(
        model_name="yolo11n.pt",
        confidence=conf,
        iou=iou_value,
        image_size=640,
        max_detections=300,
        device=0 if torch.cuda.is_available() else "cpu",
    )
    temporal=TemporalModelBundle(
        checkpoint_path=CHECKPOINT_PATH,
        mean_path=MEAN_PATH,
        scale_path=SCALE_PATH,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return detector,temporal

uploaded=st.file_uploader(
    "Upload traffic video",
    type=["mp4","avi","mov","mkv"],
)

if uploaded is None:
    st.info("Upload a traffic video to start v4 inference.")
    st.stop()

st.subheader("Input")
st.video(uploaded)

if not st.button("Run v4 temporal inference",type="primary",use_container_width=True):
    st.stop()

with tempfile.TemporaryDirectory(prefix="crosswalk_v4_") as tmp:
    tmp=Path(tmp)
    input_path=tmp/("input"+(Path(uploaded.name).suffix or ".mp4"))
    output_dir=tmp/"output"
    output_dir.mkdir(parents=True,exist_ok=True)
    input_path.write_bytes(uploaded.getbuffer())

    detector,temporal=load_models(confidence,iou)
    pipeline=CrosswalkSafetyV4Pipeline(
        detector=detector,
        temporal_bundle=temporal,
    )

    progress=st.progress(0.0)
    status=st.empty()

    def on_progress(value,message):
        progress.progress(min(max(float(value),0.0),1.0))
        status.caption(message)

    with st.spinner("Running YOLO preprocessing + trained temporal model..."):
        result=pipeline.process_video(
            input_video=input_path,
            output_dir=output_dir,
            save_annotated_video=True,
            progress_callback=on_progress,
        )

    progress.progress(1.0)
    status.success("Inference complete.")

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Sampled frames",result.summary["sampled_frames"])
    c2.metric("Temporal predictions",result.summary["temporal_predictions"])

    rate=result.summary["predicted_conflict_rate"]
    c3.metric(
        "Predicted conflict rate",
        f"{rate*100:.1f}%" if rate is not None else "N/A",
    )

    agreement=result.summary["agreement_with_proxy_on_uploaded_video"]
    c4.metric(
        "Agreement vs proxy",
        f"{agreement*100:.1f}%" if agreement is not None else "N/A",
    )

    st.caption(
        "Agreement vs proxy is only a diagnostic for the uploaded video, "
        "not real-world traffic-violation accuracy."
    )

    if result.annotated_video_path:
        p=Path(result.annotated_video_path)
        if p.exists():
            st.subheader("Annotated sampled-video preview")
            st.video(p.read_bytes())

    st.subheader("Temporal predictions")

    if result.prediction_results.empty:
        st.info("Video is too short to build the required 8-frame history.")
    else:
        view=result.prediction_results.copy()
        view["predicted_next_conflict_probability"]=view[
            "predicted_next_conflict_probability"
        ].round(4)

        st.line_chart(
            view.set_index("target_seconds")[
                "predicted_next_conflict_probability"
            ]
        )

        st.dataframe(
            view,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Sampled frame details"):
        st.dataframe(
            result.frame_results,
            use_container_width=True,
            hide_index=True,
        )

    st.download_button(
        "Download temporal predictions CSV",
        data=result.prediction_results.to_csv(index=False).encode("utf-8"),
        file_name="temporal_predictions_v4.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.download_button(
        "Download sampled frame CSV",
        data=result.frame_results.to_csv(index=False).encode("utf-8"),
        file_name="sampled_frame_results_v4.csv",
        mime="text/csv",
        use_container_width=True,
    )
