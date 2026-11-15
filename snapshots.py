"""Latest-only, encrypted JPEG stills for outage evidence (not a video stream)."""
from __future__ import annotations
import base64, hashlib, math, os, time
from dataclasses import dataclass

NODES=('node_1','node_2','node_3')
MAX_JPEG=24*1024
CHUNK=720  # base64 + metadata + HMAC remains below typical Ethernet MTU
MAX_CHUNKS=(MAX_JPEG+28+CHUNK-1)//CHUNK

@dataclass(frozen=True)
class Photo:
    node: str
    ts: float
    jpeg: bytes


def encode_preview(frame):
    import cv2
    if frame is None or not frame.size:return None
    h,w=frame.shape[:2]
    scale=min(1.,640/max(h,w))
    img=cv2.resize(frame,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA) if scale<1 else frame
    for _ in range(4):
        for quality in (65,50,35,25):
            ok,buf=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,quality])
            if ok and len(buf)<=MAX_JPEG:return buf.tobytes()
        h,w=img.shape[:2];img=cv2.resize(img,(max(1,int(w*.8)),max(1,int(h*.8))),interpolation=cv2.INTER_AREA)
    return None

class PhotoCache:
    def __init__(self,state):
        self.state=state
        with state.lock:
            state.db.execute('CREATE TABLE IF NOT EXISTS latest_photos (node TEXT PRIMARY KEY, ts REAL, jpeg BLOB)');state.db.commit()
    def get(self,node,before=None):
        with self.state.lock:
            row=self.state.db.execute('SELECT ts,jpeg FROM latest_photos WHERE node=?',(node,)).fetchone()
        if not row or (before is not None and row[0]>before+0.001):return None
        return Photo(node,row[0],bytes(row[1]))
    def save(self,photo):
        if (photo.node not in NODES or type(photo.ts) not in (int,float) or not math.isfinite(photo.ts)
                or not 0<photo.ts<=time.time()+300 or not isinstance(photo.jpeg,bytes)
                or not 4<=len(photo.jpeg)<=MAX_JPEG or not photo.jpeg.startswith(b'\xff\xd8') or not photo.jpeg.endswith(b'\xff\xd9')):return False
        with self.state.lock:
            cur=self.state.db.execute('INSERT INTO latest_photos VALUES (?,?,?) ON CONFLICT(node) DO UPDATE SET ts=excluded.ts,jpeg=excluded.jpeg WHERE excluded.ts>latest_photos.ts',(photo.node,photo.ts,photo.jpeg));self.state.db.commit();return cur.rowcount==1
    def capture(self,node,ts,frame):
        jpeg=encode_preview(frame)
        if jpeg:self.save(Photo(node,float(ts),jpeg))
        return self.get(node)

class PhotoTransport:
    """AEAD-encrypted snapshots, chunked into small UDP packets. One assembly/node."""
    def __init__(self,key,cache):
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        self.aead=AESGCM(hashlib.sha256(b'zoneguard-photo-v1\0'+key).digest())
        self.cache=cache;self.pending={};self.cached_packets=None
    @staticmethod
    def aad(node,ts):return f'zoneguard-photo-v1:{node}:{ts:.6f}'.encode()
    def packets(self,photo):
        if self.cached_packets and self.cached_packets[0]==photo:return self.cached_packets[1]
        nonce=os.urandom(12);data=nonce+self.aead.encrypt(nonce,photo.jpeg,self.aad(photo.node,photo.ts))
        ident=hashlib.sha256(data).hexdigest();n=(len(data)+CHUNK-1)//CHUNK
        packets=[dict(node=photo.node,ts=photo.ts,id=ident,i=i,n=n,data=base64.b64encode(data[i*CHUNK:(i+1)*CHUNK]).decode('ascii')) for i in range(n)]
        self.cached_packets=(photo,packets);return packets
    def receive(self,p):
        try:
            node,ts,ident,i,n=p['node'],p['ts'],p['id'],p['i'],p['n']
            if (set(p)!={'node','ts','id','i','n','data'} or node not in NODES or node==self.cache.state.node
                or type(ts) not in (int,float) or not math.isfinite(ts) or not 0<ts<=time.time()+300
                or not isinstance(ident,str) or len(ident)!=64 or type(i) is not int or type(n) is not int
                or not 1<=n<=MAX_CHUNKS or not 0<=i<n or not isinstance(p['data'],str) or len(p['data'])>4*((CHUNK+2)//3)):return False
            part=base64.b64decode(p['data'],validate=True)
            if not 1<=len(part)<=CHUNK or (i<n-1 and len(part)!=CHUNK):return False
            old=self.cache.get(node)
            if old and ts<=old.ts:return False
            cur=self.pending.get(node)
            if cur and ts<cur[0]:return False
            if cur is None or (ts,ident,n)!=cur[:3]:cur=(ts,ident,n,{});self.pending[node]=cur
            cur[3][i]=part
            if len(cur[3])!=n:return False
            data=b''.join(cur[3][j] for j in range(n));del self.pending[node]
            if len(data)>MAX_JPEG+28 or hashlib.sha256(data).hexdigest()!=ident:return False
            from cryptography.exceptions import InvalidTag
            try:jpeg=self.aead.decrypt(data[:12],data[12:],self.aad(node,ts))
            except (InvalidTag,ValueError):return False
            return self.cache.save(Photo(node,ts,jpeg))
        except (KeyError,TypeError,ValueError,OverflowError):return False


def evidence_for(state,event):
    """Return only a cached image at or before the incident observation, never a future image."""
    kind=event['kind']
    if kind=='POTENTIAL_TRESPASSING':return state.photos.get('node_2',event['ts'])
    if kind in ('NODE_UNREACHABLE','CAMERA_OFFLINE'):
        snap=event['snapshot']
        if snap and snap['last_seen']:return state.photos.get(event['subject'],snap['last_seen'])
    return None
