#!/usr/bin/env python3
"""Live shirt colour recognition using calibrated HSV ranges and YOLO-pose torso isolation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from collections import deque
import cv2
import numpy as np
from torso import isolate_torso_mask

MATCH='match'; OTHER='other'; UNKNOWN='unknown'
GREEN=(105,232,125); RED=(75,75,255); AMBER=(70,200,255); WHITE=(240,245,248)
FONT=cv2.FONT_HERSHEY_SIMPLEX


class ShirtClassifier:
    """HSV shirt colour classifier using calibrated ranges from colour.json."""
    def __init__(self, ranges, min_pixels=120, match_min=.25, other_max=.10):
        if not 0 <= other_max < match_min <= 1: raise ValueError('Invalid colour thresholds')
        self.ranges=[]
        for lo,hi in ranges:
            if len(lo)!=3 or len(hi)!=3 or not all(0<=a<=b<=m for a,b,m in zip(lo,hi,(179,255,255))):
                raise ValueError('Invalid HSV range')
            self.ranges.append((np.array(lo,np.uint8),np.array(hi,np.uint8)))
        if not self.ranges:raise ValueError('No HSV ranges provided')
        self.min_pixels=min_pixels;self.match_min=match_min;self.other_max=other_max

    def classify(self,hsv,mask):
        if hsv.size==0 or mask.size==0:return UNKNOWN
        valid=mask>0; n=np.count_nonzero(valid)
        if n<self.min_pixels:return UNKNOWN
        hit=np.zeros(mask.shape,np.uint8)
        for lo,hi in self.ranges:hit |= cv2.inRange(hsv,lo,hi)
        fraction=np.count_nonzero((hit>0)&valid)/n
        return MATCH if fraction>=self.match_min else OTHER if fraction<=self.other_max else UNKNOWN


def fit(frame,width,height):
    h,w=frame.shape[:2];s=min(width/w,(height-60)/h)
    nw,nh=max(1,round(w*s)),max(1,round(h*s))
    return cv2.resize(frame,(nw,nh),interpolation=cv2.INTER_LINEAR if s>1 else cv2.INTER_AREA),nw/w,nh/h


def text(img,label,x,y,scale=.65,colour=WHITE,thick=2):
    w=img.shape[1]; x=max(0,int(x)); y=int(y)
    tw=cv2.getTextSize(label,FONT,scale,thick)[0][0]
    if tw>w-x-8:scale*=max(.1,(w-x-8)/max(1,tw))
    cv2.putText(img,label,(x,y),FONT,scale,colour,thick,cv2.LINE_AA)


def label(img,value,center,y,c):
    h,w=img.shape[:2];scale=.68;thick=2
    tw,th=cv2.getTextSize(value,FONT,scale,thick)[0]
    if tw>w-16:
        scale*= (w-16)/tw;tw,th=cv2.getTextSize(value,FONT,scale,thick)[0]
    x=max(4,min(w-tw-12,int(center-tw/2-4)));y=max(th+10,min(h-5,int(y)))
    cv2.rectangle(img,(x,y-th-7),(min(w-1,x+tw+10),y+5),(20,24,30),-1)
    text(img,value,x+5,y,scale,c,thick)


def render(frame,people,color_name,width=1100,height=780,rate=0,cap=0,offline=False):
    img,sx,sy=fit(frame,width,height);h,w=img.shape[:2]
    for box,colour in people:
        x1,y1,x2,y2=[round(v*s) for v,s in zip(box,(sx,sy,sx,sy))]
        x1=max(0,min(w-1,x1));x2=max(0,min(w-1,x2));y1=max(0,min(h-1,y1));y2=max(0,min(h-1,y2))
        c=GREEN if colour==MATCH else RED if colour==OTHER else AMBER
        cv2.rectangle(img,(x1,y1),(x2,y2),c,3,cv2.LINE_AA)
        name=color_name.upper() if colour==MATCH else 'OTHER' if colour==OTHER else 'CHECKING...'
        label(img,name,(x1+x2)/2,y1-7,c)
    status=np.full((56,w,3),25,np.uint8)
    if offline:
        status[:]=(36,75,125);text(status,'CAMERA OFFLINE - reconnecting',10,23,.7,WHITE)
    else:
        text(status,f'Looking for: {color_name}',10,24,.70,GREEN)
    text(status,f'People {len(people)} | AI {rate:.1f} FPS | Camera {cap:.1f}',10,46,.55,WHITE,1)
    return np.vstack((img,status))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',default='0');ap.add_argument('--profile',default='colour.json');ap.add_argument('--model',default='yolo11n-pose.pt')
    ap.add_argument('--imgsz',type=int,default=640);ap.add_argument('--device')
    ap.add_argument('--width',type=int,default=1100);ap.add_argument('--height',type=int,default=780)
    ap.add_argument('--match-min',type=float,default=.25);ap.add_argument('--other-max',type=float,default=.10)
    a=ap.parse_args()
    if a.width<400 or a.height<350:ap.error('preview at least 400 x 350')
    p=Path(a.profile)
    if not p.is_file():ap.error(f'{a.profile} not found — run calibrate_colour.py first')
    data=json.loads(p.read_text(encoding='utf-8'))
    ranges=data['hsv_ranges']
    color_name=data.get('name','shirt')
    print(f'[colour] loaded {p}: {color_name} {ranges}',flush=True)
    classifier=ShirtClassifier(ranges,match_min=a.match_min,other_max=a.other_max)
    from ultralytics import YOLO
    from video_source import StreamSource,open_source
    model=YOLO(a.model);src=open_source(a.source)
    window='Shirt Colour Recognition';cv2.namedWindow(window,cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
    last_at=None;last_frame=None;last_size=None;times=deque(maxlen=30)

    def show(img):
        nonlocal last_size
        size=(img.shape[1],img.shape[0]);cv2.imshow(window,img)
        if size!=last_size:cv2.resizeWindow(window,*size);last_size=size

    def keys():
        if cv2.waitKey(1)&255 in (ord('q'),27):return True
        try:return cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1
        except cv2.error:return True

    try:
        while True:
            frame,ts=src.read();now=time.monotonic()
            if frame is None:
                if not isinstance(src,StreamSource):break
                if last_at is None or now-last_at>1:
                    base=last_frame if last_frame is not None else np.zeros((480,640,3),np.uint8)
                    show(render(base,[],color_name,a.width,a.height,offline=True))
                if keys():break
                continue
            last_at=now;last_frame=frame
            args_dict=dict(imgsz=a.imgsz,conf=.30,iou=.50,verbose=False)
            if a.device:args_dict['device']=a.device
            results=model(frame,**args_dict);people=[]
            if results and results[0].boxes is not None and len(results[0].boxes):
                r=results[0];boxes=r.boxes.xyxy.cpu().numpy()
                kps=r.keypoints.data.cpu().numpy() if r.keypoints is not None and r.keypoints.data is not None else None
                hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                for i,box in enumerate(boxes):
                    torso=isolate_torso_mask(frame.shape,i,boxes,kps[i] if kps is not None and i<len(kps) else None)
                    x1,y1,x2,y2=torso.crop_bbox
                    raw=classifier.classify(hsv[y1:y2,x1:x2],torso.crop_mask)
                    if torso.tier==4 and raw==OTHER:raw=UNKNOWN
                    people.append((box,raw))
            times.append(time.monotonic());rate=(len(times)-1)/max(.001,times[-1]-times[0]) if len(times)>1 else 0
            show(render(frame,people,color_name,a.width,a.height,rate,getattr(src,'capture_fps',0)))
            if keys():break
    except KeyboardInterrupt:pass
    finally:src.close();cv2.destroyAllWindows()

if __name__=='__main__':main()
