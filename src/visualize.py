from __future__ import annotations
import cv2
import numpy as np

def annotate_frame(frame,detections,roi_polygon_px,probability=None,threshold=None):
    out=frame.copy()
    overlay=out.copy()
    polygon=np.asarray(roi_polygon_px,dtype=np.int32)

    cv2.fillPoly(overlay,[polygon.reshape(-1,1,2)],(0,255,255))
    out=cv2.addWeighted(overlay,.12,out,.88,0)
    cv2.polylines(out,[polygon.reshape(-1,1,2)],True,(0,255,255),3)

    for d in detections:
        x1,y1,x2,y2=map(int,d.xyxy)
        if d.is_vehicle and d.inside_roi:
            color=(0,0,255)
        elif d.is_person and d.inside_roi:
            color=(0,255,0)
        else:
            color=(255,120,0)

        cv2.rectangle(out,(x1,y1),(x2,y2),color,2)
        cv2.circle(out,(int(d.anchor_px[0]),int(d.anchor_px[1])),4,color,-1)
        cv2.putText(
            out,f"{d.name} {d.confidence:.2f}",
            (x1,max(18,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2,cv2.LINE_AA
        )

    if probability is None:
        text="temporal model: collecting history"
        color=(0,200,255)
    else:
        predicted=probability>=threshold
        status="NEXT CONFLICT" if predicted else "NEXT NORMAL"
        text=f"{status} | probability={probability:.3f} | threshold={threshold:.3f}"
        color=(0,0,255) if predicted else (0,200,0)

    cv2.rectangle(out,(8,8),(1050,56),(0,0,0),-1)
    cv2.putText(out,text,(18,40),cv2.FONT_HERSHEY_SIMPLEX,.72,color,2,cv2.LINE_AA)
    return out
