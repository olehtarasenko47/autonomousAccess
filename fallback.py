"""Durable health observations and advisory risk flags. No images or identities."""
from __future__ import annotations
import hashlib, json, math, time, uuid
from datetime import datetime, timezone

NAMES=('node_1','node_2','node_3')
ZONES=dict(zip(NAMES,'ABC'))
ROLES=('hacker','volunteer','mentor','organiser')
KINDS=('CAMERA_OFFLINE','CAMERA_RECOVERED','NODE_UNREACHABLE','NODE_RECOVERED','POTENTIAL_TRESPASSING')
TIMEOUT=5.0

def utc(ts):
    return datetime.fromtimestamp(ts,timezone.utc).isoformat(timespec='seconds') if ts else 'no observation yet'

def valid_snapshot(s):
    try:
        return (set(s)=={'node','session','seq','camera','last_seen','counts','violations','last_violation'}
            and s['node'] in NAMES and bool(uuid.UUID(s['session'])) and type(s['seq']) is int and s['seq']>=0
            and s['camera'] in ('starting','online','offline')
            and all(type(s[k]) in (int,float) and math.isfinite(s[k]) and 0<=s[k]<1e12 for k in ('last_seen','last_violation'))
            and isinstance(s['counts'],dict) and all(k in ROLES and type(v) is int and 0<=v<=1000 for k,v in s['counts'].items())
            and type(s['violations']) is int and 0<=s['violations']<=1000)
    except (TypeError,ValueError,KeyError,AttributeError,OverflowError):return False

def incident(kind,subject,reporter,snapshot,ts=None,source=None,expires=0):
    return dict(id=uuid.uuid4().hex,kind=kind,subject=subject,reporter=reporter,zone=ZONES[subject],
                ts=time.time() if ts is None else ts,snapshot=snapshot,source=source,expires=expires)

def valid_incident(e):
    try:
        return (set(e)=={'id','kind','subject','reporter','zone','ts','snapshot','source','expires'}
          and bool(uuid.UUID(e['id'])) and e['kind'] in KINDS and e['subject'] in NAMES and e['reporter'] in NAMES
          and e['zone']==ZONES[e['subject']] and all(type(e[k]) in (int,float) and math.isfinite(e[k]) and 0<=e[k]<1e12 for k in ('ts','expires'))
          and (e['snapshot'] is None or (valid_snapshot(e['snapshot']) and e['snapshot']['node']==e['subject']))
          and (e['source'] is None or (isinstance(e['source'],str) and len(e['source'])<100))
          and (e['kind']!='POTENTIAL_TRESPASSING' or (e['source'] is not None and (e['expires']==0 or e['expires']>e['ts']))))
    except (TypeError,ValueError,KeyError,AttributeError,OverflowError):return False

class Fallback:
    def __init__(self,state):
        self.s=state;self.session=uuid.uuid4().hex;self.seq=0;self.last_seen=0;self.counts={};self.violations=0;self.last_violation=0
        self.last_saved=0;self.started=time.monotonic();self.down=set();self.reported=set();self.last_tick=0;self.risk_enabled=True
        with state.lock:
            for table,key in [('snapshots','node'),('incidents','id'),('notified','id')]:
                state.db.execute(f'CREATE TABLE IF NOT EXISTS {table} ({key} TEXT PRIMARY KEY, payload TEXT)')
            state.db.commit()
        old=self.snapshot(state.node)
        if old:
            self.last_seen=old['last_seen'];self.counts=old['counts'];self.violations=old['violations'];self.last_violation=old['last_violation']

    def snapshot(self,node):
        with self.s.lock:
            row=self.s.db.execute('SELECT payload FROM snapshots WHERE node=?',(node,)).fetchone()
            return json.loads(row[0]) if row else None

    def save_snapshot(self,s):
        if not valid_snapshot(s):return False
        with self.s.lock:
            old=self.snapshot(s['node'])
            if old and old['session']==s['session'] and old['seq']>=s['seq']:return False
            self.s.db.execute('INSERT OR REPLACE INTO snapshots VALUES (?,?)',(s['node'],json.dumps(s)));self.s.db.commit();return True

    def local_snapshot(self):
        with self.s.lock:
            return dict(node=self.s.node,session=self.session,seq=self.seq,camera=self.s.camera,last_seen=self.last_seen,
                        counts=dict(self.counts),violations=self.violations,last_violation=self.last_violation)

    def add(self,e):
        if not valid_incident(e):return False
        with self.s.lock:
            cur=self.s.db.execute('INSERT OR IGNORE INTO incidents VALUES (?,?)',(e['id'],json.dumps(e)));self.s.db.commit();return cur.rowcount==1

    def incidents(self):
        with self.s.lock:return [json.loads(r[0]) for r in self.s.db.execute('SELECT payload FROM incidents ORDER BY rowid')]

    def update(self,camera,counts,observed_at=None,violations=0):
        # Caller updates state.camera before constructing the snapshot.
        with self.s.lock:
            if camera=='online':
                self.last_seen=float(observed_at if observed_at is not None else time.time());self.counts=dict(counts);self.violations=int(violations)
                if violations:self.last_violation=self.last_seen
            self.seq+=1
            now=time.monotonic()
            if now-self.last_saved>=1:
                self.save_snapshot(self.local_snapshot());self.last_saved=now

    def camera_transition(self,old,new):
        if old==new:return
        with self.s.lock:
            snap=self.local_snapshot();self.save_snapshot(snap)
            if new=='offline':self.add(incident('CAMERA_OFFLINE',self.s.node,self.s.node,snap))
            elif old=='offline' and new=='online':self.add(incident('CAMERA_RECOVERED',self.s.node,self.s.node,snap))

    def notified(self,id):
        with self.s.lock:return bool(self.s.db.execute('SELECT 1 FROM notified WHERE id=?',(id,)).fetchone())

    def mark_notified(self,id):
        with self.s.lock:
            self.s.db.execute('INSERT OR IGNORE INTO notified VALUES (?,?)',(id,''));self.s.db.commit()

    def notification_marks(self):
        with self.s.lock:return [r[0] for r in self.s.db.execute('SELECT id FROM notified')]

    def merge_notification_mark(self,id):
        try:uuid.UUID(id)
        except (ValueError,TypeError,AttributeError):return
        self.mark_notified(id)

    def online(self,n,now=None):
        if n==self.s.node:return True
        now=time.monotonic() if now is None else now
        with self.s.lock:return n in self.s.heard and now-self.s.heard[n][0]<TIMEOUT

    def tick(self,peer,risk_window=120,now=None):
        now=time.monotonic() if now is None else now
        self.risk_enabled=risk_window>0
        if now-self.last_tick<.25:return
        self.last_tick=now
        # Every observer tracks health; a preferred surviving witness publishes.
        # Backup waits for grace to avoid simultaneous reports in normal operation.
        if peer.targets and peer.enabled:
            for subject in NAMES:
                if subject==self.s.node:continue
                alive=self.online(subject,now)
                if alive:
                    recent=[e for e in self.incidents() if e['subject']==subject and e['kind'] in ('NODE_UNREACHABLE','NODE_RECOVERED') and e['reporter']==self.s.node]
                    own_down=recent and max(recent,key=lambda e:e['ts'])['kind']=='NODE_UNREACHABLE'
                    if subject in self.reported or own_down:
                        peer.publish_incident(incident('NODE_RECOVERED',subject,self.s.node,self.snapshot(subject)))
                    self.down.discard(subject);self.reported.discard(subject);continue
                with self.s.lock:heard=self.s.heard.get(subject)
                age=now-(heard[0] if heard else self.started)
                if age<(TIMEOUT if heard else 15):continue
                self.down.add(subject)
                if subject in self.reported:continue
                candidates=[n for n in NAMES if n!=subject]
                preferred=candidates[0]
                if self.s.node!=preferred and (self.online(preferred,now) or age<(TIMEOUT+4 if heard else 19)):continue
                # If another observer already recorded this outage, don't re-announce.
                snap=self.snapshot(subject)
                recent=[e for e in self.incidents() if e['subject']==subject and e['kind'] in ('NODE_UNREACHABLE','NODE_RECOVERED')]
                if recent and max(recent,key=lambda e:e['ts'])['kind']=='NODE_UNREACHABLE':
                    if max(recent,key=lambda e:e['ts'])['reporter']==self.s.node:self.reported.add(subject)
                    continue
                peer.publish_incident(incident('NODE_UNREACHABLE',subject,self.s.node,snap));self.reported.add(subject)
        if not self.risk_enabled:return
        # Derive advisory from the last observed violating state of unavailable B.
        # A camera outage and a later network outage of the same observation share an ID.
        snap=self.upstream_risk_snapshot()
        if snap:
            source=f"node_2:{snap['session']}:{snap['last_seen']:.6f}"
            risk=incident('POTENTIAL_TRESPASSING','node_3','node_2',None,snap['last_seen'],source,0)
            risk['id']=hashlib.sha256(('fallback-risk:'+source).encode()).hexdigest()[:32]
            peer.publish_incident(risk)

    def upstream_risk_snapshot(self):
        if not self.risk_enabled:return None
        subject='node_2'
        with self.s.lock:
            snap=self.local_snapshot() if subject==self.s.node else self.snapshot(subject)
            if not snap or not snap['violations'] or not snap['last_seen']:return None
            if not self.online(subject) or snap['camera']!='online':return snap
            # A reconnect with an old snapshot is not proof monitoring recovered.
            # Require a strictly newer processed frame than the last outage baseline.
            outages=[e for e in self.incidents() if e['subject']==subject and e['kind'] in ('CAMERA_OFFLINE','NODE_UNREACHABLE') and e['snapshot'] is not None]
            if outages:
                baseline=max(e['snapshot']['last_seen'] for e in outages)
                if snap['last_seen']<=baseline:return snap
            return None

    def risk(self,node=None):
        return (node or self.s.node)=='node_3' and self.upstream_risk_snapshot() is not None

    def risk_incident_active(self,e):
        snap=self.upstream_risk_snapshot()
        return (e['kind']=='POTENTIAL_TRESPASSING' and e['expires']==0 and snap is not None
                and e['source']==f"node_2:{snap['session']}:{snap['last_seen']:.6f}")

    def notification_owner(self,e):
        if e['reporter']==self.s.node:return True
        # The risk event is derived by all nodes. If the originating node goes
        # away before marking its notification queued, a surviving witness can send.
        if e['kind']!='POTENTIAL_TRESPASSING' or e['reporter'] not in self.down:return False
        candidates=[n for n in NAMES if n!=e['reporter']]
        preferred=candidates[0]
        if self.s.node==preferred:return True
        with self.s.lock:heard=self.s.heard.get(preferred)
        return preferred in self.down and time.monotonic()-(heard[0] if heard else self.started)>=9

    def lines(self):
        rows=[]
        for n in NAMES:
            snap=self.local_snapshot() if n==self.s.node else self.snapshot(n)
            if not snap:continue
            stale=not self.online(n) or snap['camera']!='online'
            rows.append(f"{n}: last observation {utc(snap['last_seen'])}; {'STALE ' if stale else ''}counts {snap['counts']}; TRESPASSING {'YES' if snap['violations'] else 'NO'} (last observed); last violation {utc(snap['last_violation'])}; POTENTIAL_TRESPASSING {'YES' if self.risk(n) else 'NO'}")
        return rows

    def notice(self,e):
        snap=e['snapshot'];s=f"{e['kind']} | {e['subject']} | Zone {e['zone']} | {utc(e['ts'])}"
        if snap:s+=f"\nLast observation: {utc(snap['last_seen'])}\nLast known counts: {snap['counts']}\nLast known TRESPASSING: {'YES' if snap['violations'] else 'NO'}\nLast violation: {utc(snap['last_violation'])}"
        else:s+='\nNo last observation available.' if e['kind']!='POTENTIAL_TRESPASSING' else ''
        if e['kind']=='POTENTIAL_TRESPASSING':s+=f"\nZone B monitoring unavailable; last observation contained a confirmed colour-based policy violation. Zone C may be at risk; no person identified or entry confirmed. Advisory clears when Zone B resumes fresh camera observations (no expiry)."
        if e['kind']=='NODE_UNREACHABLE':s+='\nHeartbeat missing; power failure or network partition cannot be distinguished.'
        if e['kind'] in ('CAMERA_OFFLINE','NODE_UNREACHABLE'):s+='\nLast observations are stale, not a live count.'
        return s
