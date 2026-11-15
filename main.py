#!/usr/bin/env python3
"""ZoneGuard: one phone, one processing laptop, one role-based access zone."""
from __future__ import annotations
import argparse, os, time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from calibration import load_profiles, source_for
from peer_state import NODES, ROLES, LABELS, POLICY, State, Peer, PORT, new_event
from torso import isolate_torso_mask
from video_source import open_source, StreamSource
from telegram_alerts import TelegramAlerter

GREEN=(110,232,145); RED=(75,75,255); AMBER=(70,200,255); WHITE=(240,245,248)
FONT=cv2.FONT_HERSHEY_SIMPLEX

@dataclass(frozen=True)
class Shirt:
    role: str
    colour: str
    score: float
    uncertain: bool=False

class Classifier:
    def __init__(self,profiles,match_min=.30,other_max=.10,margin=.10):
        if not 0<=other_max<match_min<=1 or not 0<=margin<=1:raise ValueError('Invalid thresholds')
        self.profiles={role:[(np.array(lo,np.uint8),np.array(hi,np.uint8)) for lo,hi in ranges] for role,ranges in profiles.items()}
        self.match_min,self.other_max,self.margin=match_min,other_max,margin
    def classify(self,hsv,mask,tier):
        valid=mask>0;n=int(np.count_nonzero(valid))
        if n<120 or hsv.size==0:return Shirt('hacker','uncertain',0,True)
        if not self.profiles:return Shirt('hacker','uncalibrated',0,True)
        scores=[]
        for role,ranges in self.profiles.items():
            hit=np.zeros(mask.shape,np.uint8)
            for lo,hi in ranges:hit|=cv2.inRange(hsv,lo,hi)
            scores.append((float(np.count_nonzero((hit>0)&valid))/n,role))
        scores.sort(reverse=True);best,role=scores[0]
        if best<=self.other_max:
            return Shirt('hacker','other',best,tier==4)
        if best<self.match_min or (len(scores)>1 and best-scores[1][0]<self.margin):
            return Shirt('hacker','uncertain',best,True)
        if tier==4 and role!='hacker':return Shirt('hacker','uncertain',best,True)
        return Shirt(role,'purple' if role=='hacker' else role,best)


def decision(mode,shirt,inside=True):
    if not inside or mode=='public':return 'allowed'
    if shirt.uncertain:return 'review'
    return 'allowed' if shirt.role in POLICY[mode] else 'violation'


class Gate:
    """Three observations over >=0.35s; one event per track/decision/occupancy.
    IDs are ephemeral, never shared as identity or used for facial recognition.
    """
    def __init__(self):self.tracks={}
    def clear(self):self.tracks.clear()
    def update(self,track,shirt,inside,mode,now):
        verdict=decision(mode,shirt,inside);key=(shirt.role,shirt.colour,verdict,inside)
        old=self.tracks.get(track)
        if old is None or now-old['last']>1.5:
            old=dict(key=None,first=now,n=0,last=now,emitted=set());self.tracks[track]=old
        if not inside:old['emitted'].clear()
        if old['key']!=key:old.update(key=key,first=now,n=1)
        else:old['n']+=1
        old['last']=now
        stable=old['n']>=3 and now-old['first']>=.35
        if not stable and inside and mode!='public':return 'pending',False
        emit=verdict in ('review','violation') and verdict not in old['emitted']
        if emit:old['emitted'].add(verdict)
        return verdict,emit
    def prune(self,now):
        for k in list(self.tracks):
            if now-self.tracks[k]['last']>1.5:del self.tracks[k]


def text(img,s,x,y,scale=.65,c=WHITE,thick=2):
    avail=max(1,img.shape[1]-x-8);tw=cv2.getTextSize(s,FONT,scale,thick)[0][0]
    scale*=min(1,avail/max(1,tw));cv2.putText(img,s,(int(x),int(y)),FONT,scale,c,thick,cv2.LINE_AA)


def render(frame,people,node,width,height,fps,cap,peers,events,missing,split=None,offline=False,telegram_status='off'):
    zone,mode,_,_=NODES[node];h0,w0=frame.shape[:2];hh,fh=50,87
    scale=min(width/w0,(height-hh-fh)/h0);w,h=max(1,round(w0*scale)),max(1,round(h0*scale))
    img=cv2.resize(frame,(w,h),interpolation=cv2.INTER_LINEAR if scale>1 else cv2.INTER_AREA)
    sx,sy=w/w0,h/h0
    if split is not None:cv2.line(img,(round(w*split),0),(round(w*split),h-1),WHITE,2,cv2.LINE_AA)
    for box,shirt,verdict in people if not offline else []:
        x1,y1,x2,y2=[int(round(v*s)) for v,s in zip(box,(sx,sy,sx,sy))];x1,x2=np.clip([x1,x2],0,w-1);y1,y2=np.clip([y1,y2],0,h-1)
        c=RED if verdict=='violation' else AMBER if verdict in ('review','pending') else GREEN
        cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),c,2,cv2.LINE_AA)
        # Purple is identified as HACKER; other clothes have the same role.
        name=shirt.role.upper() if verdict!='pending' or shirt.role=='hacker' else 'CHECKING'
        if shirt.role=='hacker':name+=' (PURPLE)' if shirt.colour=='purple' else ' (CHECK)' if shirt.uncertain else ' (OTHER)'
        tw,th=cv2.getTextSize(name,FONT,.62,1)[0];tw=min(tw,w-10);x=max(4,min(w-tw-8,int((x1+x2-tw)/2)));y=max(20,int(y1)-7)
        cv2.rectangle(img,(x,y-18),(min(w-1,x+tw+7),y+4),(20,24,30),-1);text(img,name,x+3,y,.62,c,1)
    head=np.full((hh,w,3),25,np.uint8);head[:]=(38,55,45) if mode=='public' else (56,43,66)
    text(head,f'ZONE {zone} | {LABELS[zone].upper()} | {node}',10,20,.68,WHITE)
    allowed='EVERYONE' if mode=='public' else ', '.join(sorted(POLICY[mode])).upper()
    text(head,('LEFT PUBLIC / RIGHT: ' if split is not None else 'ALLOWED: ')+allowed,10,41,.53,WHITE,1)
    footer=np.full((fh,w,3),25,np.uint8);viol=sum(v=='violation' for _,_,v in people);reviews=sum(v in ('pending','review') for _,_,v in people)
    if offline:footer[:]=(35,70,115);msg='CAMERA OFFLINE - reconnecting';col=WHITE
    elif viol:footer[:]=(35,35,165);msg=f'ALERT: {viol} POTENTIAL ACCESS VIOLATION(S)';col=WHITE
    elif reviews:footer[:]=(35,80,115);msg=f'CHECK ACCESS: {reviews} person(s)';col=WHITE
    else:msg='ALL CLEAR' if mode!='public' else 'PUBLIC - EVERYONE ALLOWED';col=GREEN
    text(footer,msg,10,25,.72,col)
    text(footer,f'People {len(people) if not offline else 0} | AI {fps:.1f} FPS | Camera {cap:.1f} | Nodes {peers}/3 | Events {events}',10,48,.56,WHITE,1)
    text(footer,('CALIBRATE: '+', '.join(missing) if missing else 'All four roles calibrated')+f' | Telegram {telegram_status}',10,72,.54,WHITE,1)
    return np.vstack((head,img,footer))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--node',required=True,choices=NODES);p.add_argument('--source');p.add_argument('--profile-dir',default='calibrations');p.add_argument('--model',default='yolo11n-pose.pt');p.add_argument('--imgsz',type=int,default=640);p.add_argument('--device');p.add_argument('--width',type=int,default=1400);p.add_argument('--height',type=int,default=900);p.add_argument('--split',type=float,help='Optional test: left public, right monitored zone');p.add_argument('--peer',action='append',default=[],help='Other LAPTOP IP (or IP:port), repeat twice');p.add_argument('--port',type=int,default=PORT);p.add_argument('--deployment',default='durhack-access-v2');p.add_argument('--db');p.add_argument('--telegram',action='store_true');p.add_argument('--telegram-photos',action='store_true',help='Only with organiser approval');p.add_argument('--cooldown',type=float,default=60);p.add_argument('--headless',action='store_true');p.add_argument('--max-frames',type=int,default=0);a=p.parse_args()
    if a.width<400 or a.height<350 or (a.split is not None and not .1<=a.split<=.9) or a.cooldown<1:p.error('Invalid dimensions, split or cooldown')
    if a.peer and not os.getenv('ZONEGUARD_PEER_KEY'):p.error('Set ZONEGUARD_PEER_KEY on all laptops before using --peer')
    if a.telegram_photos and not a.telegram:p.error('--telegram-photos requires --telegram')
    zone,mode,_,_=NODES[a.node]
    try:profiles=load_profiles(a.node,a.profile_dir)
    except (ValueError,OSError,KeyError) as e:p.error(str(e))
    missing=[r for r in ROLES if r not in profiles];classifier=Classifier(profiles)
    if missing:print('[calibration] Missing:',', '.join(missing),'; no clearance for uncalibrated roles',flush=True)
    if a.telegram and missing and mode!='public':p.error('Calibrate all four roles before enabling Telegram on a restricted node. Preview works without --telegram.')
    if a.telegram and (not os.getenv('TELEGRAM_BOT_TOKEN') or not os.getenv('TELEGRAM_CHAT_ID')):p.error('Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID')
    from ultralytics import YOLO
    model=YOLO(a.model);src=open_source(source_for(a.node,a.source));state=None;peer=None;bot=None
    window=f'ZoneGuard | {a.node} | Zone {zone} - {LABELS[zone]}'
    try:
        state=State(a.node,a.db or str(Path('runtime')/a.deployment/(a.node+'.sqlite3')))
        peer=Peer(state,a.peer,a.port,os.getenv('ZONEGUARD_PEER_KEY'),a.deployment)
        if a.telegram:
            bot=TelegramAlerter.from_env(a.node,zone_labels={z:f'Zone {z} - {label}' for z,label in LABELS.items()},cooldown=a.cooldown,send_photos=a.telegram_photos,status_fn=lambda:state.status_lines()+[f'Replicated events: {state.event_count()}'],on_mute=peer.mute)
            bot.start()
        if not a.headless:cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE)
        gate=Gate();times=deque(maxlen=30);last_at=None;last_frame=None;last_gen=None;frames=0;offline=False;last_mute=None
        print(f'[access] {a.node}: Zone {zone} {mode}, allowed {sorted(POLICY[mode])}; {len(a.peer)} configured peers',flush=True)
        def keys():
            if a.headless:return False
            key=cv2.waitKey(1)&255
            if key in (ord('q'),27):return True
            if key==ord('c'):peer.enabled=not peer.enabled;print('[peer]', 'enabled' if peer.enabled else 'paused',flush=True)
            if key==ord('p'):print('\n'.join(state.status_lines()),flush=True)
            try:return cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1
            except cv2.error:return True
        def bot_status():return 'off' if not bot else f'queued {bot.queued()}'
        while True:
            if bot:
                until=state.mute_until(zone)
                if until!=last_mute:bot.mute_until(zone,until);last_mute=until
            frame,ts=src.read();now=time.monotonic()
            if frame is None:
                if not isinstance(src,StreamSource):break
                if last_at is None or now-last_at>1.0:
                    if not offline:gate.clear();state.update('offline');offline=True
                    if not a.headless:cv2.imshow(window,render(last_frame if last_frame is not None else np.zeros((480,640,3),np.uint8),[],a.node,a.width,a.height,0,0,state.online_count(),state.event_count(),missing,a.split,True,bot_status()))
                if keys():break
                continue
            gen=getattr(src,'last_generation',0)
            if last_gen!=gen or offline:
                gate.clear();times.clear()
                for tracker in getattr(getattr(model,'predictor',None),'trackers',[]):
                    if hasattr(tracker,'reset'):tracker.reset()
            last_gen=gen;offline=False;last_at=now;last_frame=frame;frames+=1
            kwargs=dict(persist=True,tracker='bytetrack.yaml',imgsz=a.imgsz,conf=.30,iou=.50,verbose=False)
            if a.device:kwargs['device']=a.device
            results=model.track(frame,**kwargs);people=[];pending_events=[];counts=Counter()
            if results and results[0].boxes is not None and len(results[0].boxes):
                r=results[0];boxes=r.boxes.xyxy.cpu().numpy();ids=r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else None
                kps=r.keypoints.data.cpu().numpy() if r.keypoints is not None and r.keypoints.data is not None else None
                hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);gate.prune(now)
                for i,box in enumerate(boxes):
                    torso=isolate_torso_mask(frame.shape,i,boxes,kps[i] if kps is not None and i<len(kps) else None)
                    x1,y1,x2,y2=torso.crop_bbox;shirt=classifier.classify(hsv[y1:y2,x1:x2],torso.crop_mask,torso.tier)
                    inside=a.split is None or (float(box[0])+float(box[2]))/2 >= frame.shape[1]*a.split
                    if ids is None:verdict='allowed' if not inside or mode=='public' else 'pending';emit=False
                    else:verdict,emit=gate.update(int(ids[i]),shirt,inside,mode,now)
                    if inside:counts[shirt.role if verdict!='pending' else 'hacker']+=1
                    people.append((box,shirt,verdict))
                    if emit:pending_events.append(new_event(a.node,shirt.role,shirt.colour,verdict,int(ids[i]),shirt.score,ts))
            state.update('online',counts);times.append(time.monotonic());fps=(len(times)-1)/max(.001,times[-1]-times[0]) if len(times)>1 else 0
            preview=render(frame,people,a.node,a.width,a.height,fps,getattr(src,'capture_fps',0),state.online_count(),state.event_count(),missing,a.split,False,bot_status()) if not a.headless or a.telegram_photos else None
            for event in pending_events:
                if not peer.publish(event):continue
                print(f"[access] Zone {zone}: {event['role']} ({event['shirt']}) -> {event['decision']} | {event['id']}",flush=True)
                if bot and event['decision']=='violation':
                    try:bot.intrusion(zone,frame=preview if a.telegram_photos else None,ts=event['ts'],note=f"Detected role: {event['role']}; shirt: {event['shirt']}. Please verify in person. Event: {event['id']}")
                    except (OSError,ValueError,cv2.error) as e:print('[telegram] could not queue alert:',type(e).__name__,flush=True)
            if not a.headless:cv2.imshow(window,preview)
            if keys() or (a.max_frames and frames>=a.max_frames):break
    except KeyboardInterrupt:pass
    finally:
        if bot:bot.stop()
        if peer:peer.close()
        if state:state.close()
        src.close();cv2.destroyAllWindows()

if __name__=='__main__':main()
