from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from .feature_extractor import FEATURE_COLUMNS,FeatureExtractor
from .roi import DEFAULT_ROI,normalized_roi_to_pixels
from .visualize import annotate_frame

@dataclass(slots=True)
class PipelineResult:
    frame_results:pd.DataFrame
    prediction_results:pd.DataFrame
    summary:dict
    annotated_video_path:str|None

class CrosswalkSafetyV4Pipeline:
    def __init__(self,detector,temporal_bundle,normalized_roi=None):
        self.detector=detector
        self.temporal_bundle=temporal_bundle
        self.normalized_roi=DEFAULT_ROI.copy() if normalized_roi is None else np.asarray(normalized_roi,dtype=np.float32)

        if self.temporal_bundle.feature_columns!=FEATURE_COLUMNS:
            raise ValueError("Checkpoint feature columns do not match inference feature columns.")

    def process_video(self,input_video,output_dir,save_annotated_video=True,progress_callback=None):
        input_video=Path(input_video)
        output_dir=Path(output_dir)
        output_dir.mkdir(parents=True,exist_ok=True)

        cap=cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {input_video}")

        fps=float(cap.get(cv2.CAP_PROP_FPS))
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps<=0: fps=25.0

        sample_every=self.temporal_bundle.sample_every_seconds
        step=max(1,int(round(fps*sample_every)))
        roi_polygon_px=normalized_roi_to_pixels(self.normalized_roi,width,height)

        extractor=FeatureExtractor()
        sequence_buffer=deque(maxlen=self.temporal_bundle.sequence_length)

        writer=None
        annotated_path=None
        if save_annotated_video:
            annotated_path=output_dir/f"{input_video.stem}_v4_annotated.mp4"
            writer=cv2.VideoWriter(
                str(annotated_path),cv2.VideoWriter_fourcc(*"mp4v"),
                max(1.0,1.0/max(sample_every,1e-6)),(width,height)
            )
            if not writer.isOpened():
                writer.release(); writer=None; annotated_path=None

        frame_rows=[]
        prediction_rows=[]
        frame_index=0
        sampled_index=0

        while True:
            grabbed=cap.grab()
            if not grabbed: break

            if frame_index%step!=0:
                frame_index+=1
                continue

            ok,frame=cap.retrieve()
            if not ok:
                frame_index+=1
                continue

            timestamp=frame_index/fps
            detections=self.detector.detect(frame,roi_polygon_px)
            extracted=extractor.extract(detections,width,height)

            probability=None
            prediction=None

            # Training alignment:
            # 8 previous sampled frames -> target at the next sampled frame.
            if len(sequence_buffer)==self.temporal_bundle.sequence_length:
                history=np.stack(list(sequence_buffer),axis=0)
                probability=self.temporal_bundle.predict_probability(history)
                prediction=self.temporal_bundle.classify(probability)

                prediction_rows.append({
                    "video":input_video.name,
                    "history_end_seconds":float(timestamp-sample_every),
                    "target_seconds":float(timestamp),
                    "predicted_next_conflict_probability":float(probability),
                    "predicted_next_conflict":bool(prediction),
                    "observed_proxy_conflict":bool(extracted["proxy_conflict"]),
                    "correct_vs_proxy":bool(prediction==extracted["proxy_conflict"]),
                })

            sequence_buffer.append(extracted["vector"])

            frame_rows.append({
                "video":input_video.name,
                "sampled_index":sampled_index,
                "frame_index":frame_index,
                "timestamp_seconds":float(timestamp),
                "persons_total":extracted["persons_total"],
                "vehicles_total":extracted["vehicles_total"],
                "pedestrians_in_zone":extracted["pedestrians_in_zone"],
                "vehicles_in_zone":extracted["vehicles_in_zone"],
                "proxy_conflict":bool(extracted["proxy_conflict"]),
                "prediction_probability_for_current_frame":probability,
                "predicted_conflict_for_current_frame":prediction,
            })

            if writer is not None:
                writer.write(annotate_frame(
                    frame,detections,roi_polygon_px,
                    probability=probability,
                    threshold=self.temporal_bundle.threshold
                ))

            sampled_index+=1

            if progress_callback:
                progress_callback(
                    min(frame_index/max(frame_count,1),1.0),
                    f"Sampled {sampled_index} frames | t={timestamp:.1f}s"
                )

            frame_index+=1

        cap.release()
        if writer is not None:
            writer.release()

        frames=pd.DataFrame(frame_rows)
        predictions=pd.DataFrame(prediction_rows)

        if predictions.empty:
            proxy_agreement=predicted_rate=mean_prob=None
        else:
            proxy_agreement=float(predictions["correct_vs_proxy"].mean())
            predicted_rate=float(predictions["predicted_next_conflict"].mean())
            mean_prob=float(predictions["predicted_next_conflict_probability"].mean())

        summary={
            "video":input_video.name,
            "sampled_frames":int(len(frames)),
            "temporal_predictions":int(len(predictions)),
            "sample_every_seconds":float(sample_every),
            "sequence_length":int(self.temporal_bundle.sequence_length),
            "history_seconds":float(self.temporal_bundle.sequence_length*sample_every),
            "prediction_horizon_seconds":float(self.temporal_bundle.predict_horizon_steps*sample_every),
            "threshold":float(self.temporal_bundle.threshold),
            "predicted_conflict_rate":predicted_rate,
            "mean_predicted_probability":mean_prob,
            "agreement_with_proxy_on_uploaded_video":proxy_agreement,
            "scope_note":"Weakly supervised next-sample safety-conflict screening; not validated traffic-violation detection."
        }

        frames.to_csv(output_dir/"sampled_frame_results_v4.csv",index=False)
        predictions.to_csv(output_dir/"temporal_predictions_v4.csv",index=False)

        return PipelineResult(
            frame_results=frames,
            prediction_results=predictions,
            summary=summary,
            annotated_video_path=str(annotated_path) if annotated_path is not None and annotated_path.exists() else None
        )
