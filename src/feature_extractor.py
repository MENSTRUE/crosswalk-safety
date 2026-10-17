from __future__ import annotations
import math
import numpy as np

BASE_FEATURES=[
    "persons_total","vehicles_total","pedestrians_in_zone","vehicles_in_zone",
    "person_mean_cx","person_mean_cy","person_mean_area",
    "vehicle_mean_cx","vehicle_mean_cy","vehicle_mean_area",
    "min_person_vehicle_distance","mean_person_vehicle_distance",
]
MOTION_SOURCE=[
    "persons_total","vehicles_total","pedestrians_in_zone","vehicles_in_zone",
    "person_mean_cx","person_mean_cy","vehicle_mean_cx","vehicle_mean_cy",
    "min_person_vehicle_distance",
]
DELTA_FEATURES=[f"delta_{x}" for x in MOTION_SOURCE]
FEATURE_COLUMNS=BASE_FEATURES+DELTA_FEATURES

def safe_mean(values,default=0.0):
    return float(np.mean(values)) if values else float(default)

def safe_min(values,default=1.5):
    return float(np.min(values)) if values else float(default)

class FeatureExtractor:
    def __init__(self):
        self.previous_base=None

    def reset(self):
        self.previous_base=None

    def extract(self,detections,width,height):
        persons=[d for d in detections if d.is_person]
        vehicles=[d for d in detections if d.is_vehicle]

        pcx=[]; pcy=[]; parea=[]; vcx=[]; vcy=[]; varea=[]
        panchors=[]; vanchors=[]

        for d in persons:
            x1,y1,x2,y2=d.xyxy
            bw=max(x2-x1,0.0)/width; bh=max(y2-y1,0.0)/height
            pcx.append(((x1+x2)/2.0)/width)
            pcy.append(((y1+y2)/2.0)/height)
            parea.append(bw*bh)
            panchors.append(d.anchor_norm)

        for d in vehicles:
            x1,y1,x2,y2=d.xyxy
            bw=max(x2-x1,0.0)/width; bh=max(y2-y1,0.0)/height
            vcx.append(((x1+x2)/2.0)/width)
            vcy.append(((y1+y2)/2.0)/height)
            varea.append(bw*bh)
            vanchors.append(d.anchor_norm)

        distances=[
            math.sqrt((px-vx)**2+(py-vy)**2)
            for px,py in panchors for vx,vy in vanchors
        ]

        base={
            "persons_total":float(len(persons)),
            "vehicles_total":float(len(vehicles)),
            "pedestrians_in_zone":float(sum(d.inside_roi for d in persons)),
            "vehicles_in_zone":float(sum(d.inside_roi for d in vehicles)),
            "person_mean_cx":safe_mean(pcx),
            "person_mean_cy":safe_mean(pcy),
            "person_mean_area":safe_mean(parea),
            "vehicle_mean_cx":safe_mean(vcx),
            "vehicle_mean_cy":safe_mean(vcy),
            "vehicle_mean_area":safe_mean(varea),
            "min_person_vehicle_distance":safe_min(distances,1.5),
            "mean_person_vehicle_distance":safe_mean(distances,1.5),
        }

        feat=dict(base)
        for name in MOTION_SOURCE:
            prev=0.0 if self.previous_base is None else float(self.previous_base[name])
            feat[f"delta_{name}"]=float(base[name])-prev

        self.previous_base=dict(base)

        proxy=base["pedestrians_in_zone"]>0 and base["vehicles_in_zone"]>0
        vector=np.asarray([feat[n] for n in FEATURE_COLUMNS],dtype=np.float32)

        return {
            "vector":vector,
            "proxy_conflict":bool(proxy),
            "persons_total":int(base["persons_total"]),
            "vehicles_total":int(base["vehicles_total"]),
            "pedestrians_in_zone":int(base["pedestrians_in_zone"]),
            "vehicles_in_zone":int(base["vehicles_in_zone"]),
        }
