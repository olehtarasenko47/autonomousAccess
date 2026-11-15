"""Independent durable event stores and authenticated UDP gossip; no leader.
Only metadata travels between laptops. A shared key is required when peers are configured.
"""
from __future__ import annotations
import hashlib, hmac, json, math, os, socket, sqlite3, threading, time, uuid
from pathlib import Path

NODES = {'node_1': ('A', 'public', 'camera1', '10.249.58.251'),
         'node_2': ('B', 'staff', 'camera3', '10.249.37.86'),
         'node_3': ('C', 'organisers', 'camera4', '10.249.55.81')}
ROLES = ('hacker', 'volunteer', 'mentor', 'organiser')
LABELS = {'A': 'Public', 'B': 'Staff only', 'C': 'Organisers only'}
POLICY = {'public': set(ROLES), 'staff': {'volunteer', 'mentor', 'organiser'},
          'organisers': {'organiser'}}
PORT = 50565
MAX_PACKET = 8192


def new_event(node, role, shirt, decision, track, score, timestamp=None):
    return dict(id=f'{node}:{uuid.uuid4().hex}', node=node, zone=NODES[node][0],
                role=role, shirt=shirt, decision=decision, track=int(track),
                score=round(float(score), 4), ts=time.time() if timestamp is None else timestamp)


def valid_event(e):
    try:
        return (set(e) == {'id','node','zone','role','shirt','decision','track','score','ts'}
                and e['node'] in NODES and e['zone'] == NODES[e['node']][0]
                and e['id'].startswith(e['node']+':') and bool(uuid.UUID(e['id'].split(':')[1]))
                and e['role'] in ROLES and e['shirt'] in (*ROLES,'purple','other','uncertain','uncalibrated')
                and e['decision'] in ('violation','review')
                and e['role'] not in POLICY[NODES[e['node']][1]]
                and type(e['track']) is int and e['track'] >= 0
                and 0 <= e['score'] <= 1 and math.isfinite(e['ts']))
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError):
        return False


class State:
    def __init__(self, node, path):
        self.node = node
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, payload TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS mutes (zone TEXT PRIMARY KEY, payload TEXT)')
        self.db.commit()
        self.camera = 'starting'; self.counts = {}; self.heard = {}

    def add(self, e):
        if not valid_event(e): return False
        with self.lock:
            cur = self.db.execute('INSERT OR IGNORE INTO events VALUES (?,?)', (e['id'], json.dumps(e)))
            self.db.commit()
            return cur.rowcount == 1

    def events(self):
        with self.lock:
            return [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM events ORDER BY rowid')]

    def event_count(self):
        with self.lock: return self.db.execute('SELECT count(*) FROM events').fetchone()[0]

    def set_mute(self, zone, until):
        m = dict(zone=zone, until=float(until), version=time.time(), origin=self.node, id=uuid.uuid4().hex)
        self.merge_mute(m)
        return m

    def merge_mute(self, m):
        try:
            if set(m) != {'zone','until','version','origin','id'} or m['zone'] not in LABELS or m['origin'] not in NODES: return False
            if not all(math.isfinite(m[k]) and m[k] >= 0 for k in ('until','version')): return False
            uuid.UUID(m['id'])
        except (KeyError, TypeError, ValueError, AttributeError, OverflowError): return False
        with self.lock:
            old = self.db.execute('SELECT payload FROM mutes WHERE zone=?',(m['zone'],)).fetchone()
            if old:
                prev=json.loads(old[0])
                if (m['version'],m['origin'],m['id']) <= (prev['version'],prev['origin'],prev['id']): return False
            self.db.execute('INSERT OR REPLACE INTO mutes VALUES (?,?)',(m['zone'],json.dumps(m)))
            self.db.commit(); return True

    def mutes(self):
        with self.lock: return [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM mutes')]

    def mute_until(self, zone):
        return next((m['until'] for m in self.mutes() if m['zone']==zone),0)

    def update(self, camera, counts=None):
        with self.lock: self.camera, self.counts = camera, dict(counts or {})

    def heartbeat(self):
        with self.lock: return dict(node=self.node,camera=self.camera,counts=self.counts)

    def receive_heartbeat(self,p):
        try:
            n=p['node']; counts=p['counts']; cam=p['camera']
            if n not in NODES or n==self.node or cam not in ('starting','online','offline'):return
            if not isinstance(counts,dict) or any(k not in ROLES or type(v) is not int or not 0<=v<=1000 for k,v in counts.items()):return
            with self.lock: self.heard[n]=(time.monotonic(),cam,dict(counts))
        except (KeyError,TypeError,ValueError):pass

    def status_lines(self):
        with self.lock:
            rows=[]
            for n,(z,mode,_,_) in NODES.items():
                if n==self.node: cam,counts=self.camera,self.counts; peer='local'
                elif n in self.heard and time.monotonic()-self.heard[n][0]<5:
                    _,cam,counts=self.heard[n];peer='online'
                else:cam,counts,peer='unknown',{},'offline'
                rows.append(f'{n} / Zone {z}: peer {peer}, camera {cam}, visible {sum(counts.values()) if cam=="online" else "?"}')
            return rows

    def online_count(self):
        with self.lock: return 1+sum(time.monotonic()-r[0]<5 for r in self.heard.values())

    def close(self):
        with self.lock:self.db.close()


def target(s, port):
    h,sep,p=s.partition(':')
    return (h,int(p) if sep else port)


class Peer:
    def __init__(self,state,peers=(),port=PORT,key=None,deployment='durhack-access-v2'):
        self.state=state;self.targets=[target(p,PORT) for p in peers];self.enabled=True
        if self.targets and not key:raise ValueError('Set ZONEGUARD_PEER_KEY to the same secret on all laptops')
        self.key=(key or '').encode();self.deployment=deployment;self.stop=threading.Event()
        self.sock=None;self.threads=[]
        if self.targets:
            self.sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            self.sock.bind(('',port));self.sock.settimeout(.2)
            # Windows reports ICMP port-unreachable replies as WSAECONNRESET
            # on UDP receive, e.g. while other peers are still starting.
            if os.name == 'nt':
                try:
                    self.sock.ioctl(getattr(socket, 'SIO_UDP_CONNRESET', 0x9800000C), False)
                except (OSError, ValueError, AttributeError):
                    pass  # _receive also handles WSAECONNRESET below.
            self.threads=[threading.Thread(target=f,daemon=True) for f in (self._receive,self._gossip)]
            for t in self.threads:t.start()

    def packet(self,kind,payload):
        body=json.dumps(dict(protocol='access-v2',deployment=self.deployment,kind=kind,payload=payload),separators=(',',':')).encode()
        return hmac.new(self.key,body,hashlib.sha256).hexdigest().encode()+b' '+body

    def send(self,kind,payload):
        if not self.enabled or not self.sock:return
        packet=self.packet(kind,payload)
        if len(packet)>MAX_PACKET:return
        for dest in self.targets:
            try:self.sock.sendto(packet,dest)
            except OSError:pass

    def publish(self,event):
        if self.state.add(event):self.send('event',event);return True
        return False

    def mute(self,zone,until):self.send('mute',self.state.set_mute(zone,until))

    def _receive(self):
        while not self.stop.is_set():
            try:raw,_=self.sock.recvfrom(MAX_PACKET)
            except socket.timeout:continue
            except OSError as exc:
                if self.stop.is_set():break
                if isinstance(exc, ConnectionResetError) or getattr(exc, 'winerror', None) == 10054:
                    continue  # A peer was not listening yet; keep receiving.
                raise
            if not self.enabled:continue
            try:
                sig,body=raw.split(b' ',1)
                if not hmac.compare_digest(sig,hmac.new(self.key,body,hashlib.sha256).hexdigest().encode()):continue
                m=json.loads(body)
                if m['protocol']!='access-v2' or m['deployment']!=self.deployment:continue
                kind,p=m['kind'],m['payload']
                if kind=='event':self.state.add(p)
                elif kind=='mute':self.state.merge_mute(p)
                elif kind=='hello':self.state.receive_heartbeat(p)
            except (KeyError,ValueError,TypeError,AttributeError):continue

    def _gossip(self):
        while not self.stop.is_set():
            self.send('hello',self.state.heartbeat())
            if self.enabled:
                for m in self.state.mutes():self.send('mute',m)
                for e in self.state.events():
                    if self.stop.is_set():break
                    self.send('event',e)
            self.stop.wait(1)

    def close(self):
        self.stop.set()
        if self.sock:self.sock.close()
        for t in self.threads:t.join(timeout=2)
