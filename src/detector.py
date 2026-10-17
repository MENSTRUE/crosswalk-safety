from __future__ import annotations
from dataclasses import dataclass
from ultralytics import YOLO
from .roi import bbox_bottom_center,point_inside_polygon

PERSON_CLASS_ID=0
VEHICLE_CLASSES={2:"car",3:"motorcycle",5:"bus",7:"truck"}
TARGET_CLASS_IDS=[PERSON_CLASS_ID,*VEHICLE_CLASSES.keys()]

@dataclass(slots=True)
class Detection:
    class_id:int
    name:str
    confidence:float
    xyxy:tuple
    anchor_px:tuple
    anchor_norm:tuple
    inside_roi:bool
    is_person:bool
    is_vehicle:bool

class CrosswalkDetector:
    def __init__(self,model_name="yolo11n.pt",confidence=.25,iou=.50,image_size=640,max_detections=300,device=None):
        self.model=YOLO(model_name)
        self.confidence=float(confidence); self.iou=float(iou)
        self.image_size=int(image_size); self.max_detections=int(max_detections); self.device=device

    def detect(self,frame,roi_polygon_px):
        h,w=frame.shape[:2]
        result=self.model.predict(frame,imgsz=self.image_size,conf=self.confidence,iou=self.iou,
                                  classes=TARGET_CLASS_IDS,max_det=self.max_detections,
                                  device=self.device,verbose=False)[0]
        dets=[]
        if result.boxes is None:
            return dets
        for box in result.boxes:
            cls=int(box.cls[0].item())
            conf=float(box.conf[0].item())
            xyxy=tuple(float(v) for v in box.xyxy[0].detach().cpu().tolist())
            anchor_px=bbox_bottom_center(xyxy)
            anchor_norm=(anchor_px[0]/w,anchor_px[1]/h)
            dets.append(Detection(
                class_id=cls,name=result.names.get(cls,str(cls)),confidence=conf,xyxy=xyxy,
                anchor_px=anchor_px,anchor_norm=anchor_norm,
                inside_roi=point_inside_polygon(anchor_px,roi_polygon_px),
                is_person=(cls==PERSON_CLASS_ID),is_vehicle=(cls in VEHICLE_CLASSES)
            ))
        return dets
