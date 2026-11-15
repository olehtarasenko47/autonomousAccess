#!/usr/bin/env python3
"""Fresh, per-node, per-role HSV calibration. No preset or legacy colour files used."""
from __future__ import annotations
import argparse, getpass, json, os
from pathlib import Path
from urllib.parse import quote
import cv2
import numpy as np
from peer_state import NODES, ROLES
from video_source import open_source, StreamSource

SCHEMA='access-role-hsv-v1'


def source_for(node, explicit=None):
    if explicit: return explicit
    _,_,user,ip=NODES[node]
    password=os.environ.get('CAMERA_PASSWORD') or getpass.getpass(f'{user} camera password: ')
    return f'http://{user}:{quote(password,safe="")}@{ip}:8080/mjpeg'


def profile_path(node,role,root='calibrations'):
    return Path(root)/node/(role+'.json')


def ranges_from_samples(samples):
    if len(samples)<5:raise ValueError('Take at least five separate fabric samples')
    px=np.concatenate(samples,axis=0).astype(float)
    px=px[(px[:,1]>=50)&(px[:,2]>=40)]
    if len(px)<50:raise ValueError('Not enough saturated fabric pixels')
    angle=px[:,0]*2*np.pi/180
    z=np.mean(np.exp(1j*angle))
    if abs(z)<.75:raise ValueError('Mixed colours sampled; reset and sample one shirt')
    center=np.angle(z)*180/(2*np.pi)%180
    delta=(px[:,0]-center+90)%180-90
    lo,hi=np.percentile(delta,[3,97]);lo=int(np.floor(center+lo-5));hi=int(np.ceil(center+hi+5))
    if hi-lo>36:raise ValueError('Hue range too broad; reset and sample one colour only')
    sl=max(50,int(np.percentile(px[:,1],3))-25)
    vl=max(40,int(np.percentile(px[:,2],3))-30)
    start,end=lo%180,hi%180
    if start<=end:return [[[start,sl,vl],[end,255,255]]]
    return [[[start,sl,vl],[179,255,255]],[[0,sl,vl],[end,255,255]]]


def validate_ranges(ranges):
    if not isinstance(ranges,list) or not 1<=len(ranges)<=2:raise ValueError('Expected one or two HSV intervals')
    total=0
    for pair in ranges:
        if not isinstance(pair,list) or len(pair)!=2:raise ValueError('Invalid HSV interval')
        lo,hi=pair
        if not all(isinstance(v,list) and len(v)==3 for v in (lo,hi)):raise ValueError('Invalid HSV triple')
        if not all(type(a) is int and type(b) is int and 0<=a<=b<=m for a,b,m in zip(lo,hi,(179,255,255))):raise ValueError('Invalid HSV bounds')
        if lo[1]<50 or lo[2]<40:raise ValueError('Profile includes unsaturated/dark pixels')
        total+=hi[0]-lo[0]+1
    if total>37:raise ValueError('HSV hue span too broad')
    return ranges


def load_profiles(node,root='calibrations'):
    profiles={}
    for role in ROLES:
        path=profile_path(node,role,root)
        if not path.is_file():continue
        data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema')!=SCHEMA or data.get('node')!=node or data.get('role')!=role:
            raise ValueError(f'{path}: wrong schema, node or role; recalibrate')
        profiles[role]=validate_ranges(data['hsv_ranges'])
    return profiles


def save_profile(node,role,samples,root='calibrations'):
    ranges=validate_ranges(ranges_from_samples(samples));path=profile_path(node,role,root)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(dict(schema=SCHEMA,node=node,role=role,hsv_ranges=ranges),indent=2),encoding='utf-8');tmp.replace(path)
    return path,ranges


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--node',required=True,choices=NODES);p.add_argument('--role',required=True,choices=ROLES);p.add_argument('--source');p.add_argument('--profile-dir',default='calibrations');p.add_argument('--width',type=int,default=1200);p.add_argument('--height',type=int,default=800);a=p.parse_args()
    src=open_source(source_for(a.node,a.source));window=f'Calibrate {a.node}: {a.role.upper()}'
    cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE)
    samples=[];frame=None;shown=None;scale=(1.,1.);paused=False;mask_view=False;msg='Click plain fabric of ONE role. At least 5 clicks.'
    def click(event,x,y,*_):
        nonlocal msg
        if event!=cv2.EVENT_LBUTTONDOWN or shown is None:return
        h,w=shown.shape[:2]; xx=int(x/scale[0]);yy=int((y-34)/scale[1])
        if not (0<=xx<w and 0<=yy<h):return
        patch=shown[max(0,yy-4):min(h,yy+5),max(0,xx-4):min(w,xx+5)]
        px=cv2.cvtColor(patch,cv2.COLOR_BGR2HSV).reshape(-1,3)
        if np.count_nonzero((px[:,1]>=50)&(px[:,2]>=40))<10:
            msg='Sample rejected: choose saturated fabric';return
        samples.append(px);msg=f'{len(samples)} clicks; sample HSV {np.median(px,axis=0).astype(int).tolist()}'
        print(msg,flush=True)
    cv2.setMouseCallback(window,click)
    try:
        while True:
            if not paused:
                incoming,_=src.read()
                if incoming is not None:frame=incoming
                elif not isinstance(src,StreamSource):break
                elif not src.connected:frame=None
            if frame is None:
                shown=None;view=np.zeros((400,640,3),np.uint8)
                cv2.putText(view,'Camera offline - reconnecting',(20,190),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            else:
                shown=frame;view=frame.copy();h,w=view.shape[:2]
                if mask_view and len(samples)>=5:
                    try:
                        ranges=ranges_from_samples(samples);hsv=cv2.cvtColor(view,cv2.COLOR_BGR2HSV);mask=np.zeros((h,w),np.uint8)
                        for lo,hi in ranges:mask |= cv2.inRange(hsv,np.array(lo,np.uint8),np.array(hi,np.uint8))
                        view=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
                    except ValueError as e:msg=str(e)
                s=min(a.width/w,(a.height-90)/h);nw,nh=max(1,round(w*s)),max(1,round(h*s));scale=(nw/w,nh/h)
                view=cv2.resize(view,(nw,nh));top=np.full((34,nw,3),25,np.uint8);bottom=np.full((56,nw,3),25,np.uint8)
                def line(img,t,y):
                    fs=min(.58,(nw-16)/max(1,cv2.getTextSize(t,cv2.FONT_HERSHEY_SIMPLEX,1,1)[0][0]));cv2.putText(img,t,(8,y),cv2.FONT_HERSHEY_SIMPLEX,fs,(230,240,240),1,cv2.LINE_AA)
                line(top,f'{a.role.upper()} | {a.node} | {"PAUSED" if paused else "LIVE"} | {len(samples)} samples',23)
                line(bottom,msg,20);line(bottom,'P pause | M mask | U undo | R reset | S save | Q quit',44)
                view=np.vstack([top,view,bottom])
            cv2.imshow(window,view);key=cv2.waitKey(20)&255
            if key in (ord('q'),27):break
            if key==ord('p'):paused=not paused
            if key==ord('m'):mask_view=not mask_view
            if key==ord('u') and samples:samples.pop();msg=f'{len(samples)} clicks remaining'
            if key==ord('r'):samples.clear();msg='Samples reset; previous saved profile unchanged'
            if key==ord('s'):
                try:path,ranges=save_profile(a.node,a.role,samples,a.profile_dir);msg=f'Saved {path}';print(msg,ranges,flush=True)
                except ValueError as e:msg=str(e);print(msg,flush=True)
            try:
                if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:break
            except cv2.error:break
    finally:src.close();cv2.destroyAllWindows()

if __name__=='__main__':main()
