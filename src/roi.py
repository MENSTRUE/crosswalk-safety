from __future__ import annotations
import cv2
import numpy as np

DEFAULT_ROI=np.array([[.12,.62],[.88,.62],[.98,.96],[.02,.96]],dtype=np.float32)

def normalized_roi_to_pixels(points,width,height):
    roi=np.asarray(points,dtype=np.float32)
    if roi.ndim!=2 or roi.shape[1]!=2 or len(roi)<3:
        raise ValueError("ROI must contain at least 3 points")
    if (roi<0).any() or (roi>1).any():
        raise ValueError("ROI must stay inside [0,1]")
    out=roi.copy()
    out[:,0]*=width
    out[:,1]*=height
    return np.rint(out).astype(np.int32)

def point_inside_polygon(point,polygon):
    return cv2.pointPolygonTest(np.asarray(polygon,dtype=np.float32),(float(point[0]),float(point[1])),False)>=0

def bbox_bottom_center(xyxy):
    x1,y1,x2,y2=map(float,xyxy)
    return ((x1+x2)/2.0,y2)
