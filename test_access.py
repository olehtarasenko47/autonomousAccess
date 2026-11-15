import os, socket, tempfile, time, unittest
from pathlib import Path
import cv2
import numpy as np
from calibration import ranges_from_samples, load_profiles, save_profile, validate_ranges
from main import Classifier, Shirt, Gate, decision, render
from peer_state import State, Peer, new_event, valid_event, ROLES
from torso import isolate_torso_mask
from telegram_alerts import TelegramAlerter


def sample(h):return np.tile([h,180,180],(81,1))
def hue(h):return np.tile(np.array([h,180,180],np.uint8),(20,20,1))
def ranges(h):return ranges_from_samples([sample(h)]*5)
def eventually(fn, timeout=6):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        if fn():return True
        time.sleep(.05)
    return fn()

def ports(n):
    sockets=[socket.socket(socket.AF_INET,socket.SOCK_DGRAM) for _ in range(n)]
    for s in sockets:s.bind(('127.0.0.1',0))
    result=[s.getsockname()[1] for s in sockets]
    for s in sockets:s.close()
    return result

class AccessTests(unittest.TestCase):
    def setUp(self):
        self.mask=np.full((20,20),255,np.uint8)
        self.profiles={r:ranges(h) for r,h in zip(ROLES,[125,165,90,110])}
        self.c=Classifier(self.profiles)
    def test_roles(self):
        for role,h in zip(ROLES,[125,165,90,110]):
            result=self.c.classify(hue(h),self.mask,1)
            self.assertEqual(result.role,role);self.assertFalse(result.uncertain)
        self.assertEqual(self.c.classify(hue(125),self.mask,1).colour,'purple')
    def test_ordinary_clothes(self):
        for hsv in ([0,0,0],[0,0,255],[50,180,180],[20,180,180]):
            shirt=self.c.classify(np.tile(np.array(hsv,np.uint8),(20,20,1)),self.mask,1)
            self.assertEqual(shirt.role,'hacker');self.assertEqual(shirt.colour,'other');self.assertFalse(shirt.uncertain)
    def test_ambiguity_and_partial(self):
        c=Classifier({'mentor':ranges(90),'organiser':ranges(90)})
        self.assertTrue(c.classify(hue(90),self.mask,1).uncertain)
        self.assertEqual(c.classify(hue(90),self.mask,1).role,'hacker')
        self.assertEqual(Classifier({}).classify(hue(90),self.mask,1).role,'hacker')
        self.assertTrue(self.c.classify(hue(90),self.mask,4).uncertain)
        self.assertTrue(self.c.classify(hue(90),np.zeros_like(self.mask),1).uncertain)
        self.assertEqual(Classifier({'hacker':ranges(125)}).classify(hue(90),self.mask,1).role,'hacker')
    def test_policy(self):
        for r in ROLES:
            s=Shirt(r,r,1)
            self.assertEqual(decision('public',s),'allowed')
            self.assertEqual(decision('staff',s),'violation' if r=='hacker' else 'allowed')
            self.assertEqual(decision('organisers',s),'allowed' if r=='organiser' else 'violation')
        self.assertEqual(decision('staff',Shirt('hacker','uncertain',0,True)),'review')
        self.assertEqual(decision('staff',Shirt('hacker','other',0),False),'allowed')
    def test_gate(self):
        g=Gate();s=Shirt('hacker','other',0)
        for t in [0,.2]:self.assertEqual(g.update(1,s,True,'staff',t),('pending',False))
        self.assertEqual(g.update(1,s,True,'staff',.4),('violation',True))
        self.assertEqual(g.update(1,s,True,'staff',.6),('violation',False))
        self.assertEqual(g.update(2,s,True,'staff',.6),('pending',False))
        g.update(1,s,False,'staff',.8)
        for t in [1,1.2]:g.update(1,s,True,'staff',t)
        self.assertEqual(g.update(1,s,True,'staff',1.4),('violation',True))
        g.clear();self.assertEqual(g.update(1,s,True,'staff',1.6),('pending',False))
    def test_gate_never_carries_clearance(self):
        g=Gate();s=Shirt('organiser','organiser',1)
        for t in [0,.2,.4]:verdict,_=g.update(1,s,True,'organisers',t)
        self.assertEqual(verdict,'allowed')
        self.assertEqual(g.update(1,Shirt('hacker','uncertain',0,True),True,'organisers',.6),('pending',False))
    def test_calibration(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(load_profiles('node_1',d),{})
            save_profile('node_1','hacker',[sample(125)]*5,d)
            self.assertEqual(list(load_profiles('node_1',d)),['hacker'])
            self.assertEqual(load_profiles('node_2',d),{})
            self.assertEqual(len(ranges_from_samples([sample(178),sample(2),sample(179),sample(1),sample(0)])),2)
            with self.assertRaises(ValueError):ranges_from_samples([sample(h) for h in [125,125,125,10,10]])
            with self.assertRaises(ValueError):validate_ranges([[[0,60,50],[179,255,255]]])
    def test_occlusion(self):
        boxes=np.array([[10,10,100,100],[0,0,120,120]])
        t=isolate_torso_mask((130,130,3),0,boxes,None)
        self.assertEqual(np.count_nonzero(t.crop_mask),0)
    def test_preview(self):
        for shape in [(640,480,3),(480,640,3)]:
            img=np.zeros(shape,np.uint8);s=Shirt('hacker','purple',.8)
            out=render(img,[(np.array([0,0,100,150]),s,'violation')],'node_2',1400,900,10,14,3,1,[],.5)
            self.assertLessEqual(out.shape[0],900);self.assertLessEqual(out.shape[1],1400)
            self.assertAlmostEqual((out.shape[0]-137)/out.shape[1],shape[0]/shape[1],delta=.005)
    def test_event_validation(self):
        e=new_event('node_2','hacker','purple','violation',1,.8)
        self.assertTrue(valid_event(e));self.assertFalse(valid_event(new_event('node_1','hacker','purple','violation',1,.8)))
    def test_three_nodes_partition_and_mute(self):
        with tempfile.TemporaryDirectory() as d:
            ps=ports(3);ss=[State(f'node_{i+1}',Path(d)/f'{i}.sqlite') for i in range(3)]
            pp=[]
            try:
                for i,s in enumerate(ss):pp.append(Peer(s,[f'127.0.0.1:{ps[j]}' for j in range(3) if j!=i],ps[i],'test-key'))
                for s in ss:s.update('online',{'hacker':1})
                self.assertTrue(eventually(lambda:all(s.online_count()==3 for s in ss)))
                e=new_event('node_2','hacker','purple','violation',1,.8);pp[1].publish(e)
                self.assertTrue(eventually(lambda:all(s.event_count()==1 for s in ss)))
                self.assertFalse(pp[1].publish(e));pp[2].enabled=False
                pp[2].publish(new_event('node_3','mentor','mentor','violation',3,.9))
                self.assertEqual(ss[2].event_count(),2);self.assertEqual(ss[0].event_count(),1)
                pp[0].mute('C',time.time()+600);pp[2].enabled=True
                self.assertTrue(eventually(lambda:all(s.event_count()==2 and s.mute_until('C')>time.time() for s in ss)))
                pp[0].mute('C',0)
                self.assertTrue(eventually(lambda:all(s.mute_until('C')==0 for s in ss)))
                # Forged packet (bad signature) cannot mutate state.
                bad=Peer(ss[0],[],key='wrong-key').packet('event',new_event('node_3','hacker','other','violation',4,0))
                with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:sock.sendto(bad,('127.0.0.1',ps[1]))
                time.sleep(.2);self.assertEqual(ss[1].event_count(),2)
            finally:
                for p in pp:p.close()
                for s in ss:s.close()
            reopened=State('node_3',Path(d)/'2.sqlite');self.assertEqual(reopened.event_count(),2);reopened.close()
    def test_telegram_queue(self):
        with tempfile.TemporaryDirectory() as d:
            a=TelegramAlerter('test',['-1'],'node_3',zone_labels={'C':'Organisers'},outbox_dir=d,listen=False,send_photos=False,cooldown=60)
            aid=a.intrusion('C',frame=np.zeros((20,20,3),np.uint8),note='Detected role: hacker')
            self.assertTrue(aid);self.assertEqual(a.queued(),1);self.assertIsNone(a.intrusion('C'))
            self.assertFalse(list(Path(d).rglob('*.jpg')))
            sent=[]
            a.api.call=lambda method,params=None,**kw:sent.append((method,params)) or {}
            a.start();self.assertTrue(eventually(lambda:a.queued()==0));a.stop()
            self.assertEqual(sent[0][0],'sendMessage');self.assertIn('Potential access violation',sent[0][1]['text'])
            self.assertIn('Detected role: hacker',sent[0][1]['text'])

if __name__=='__main__':unittest.main(verbosity=2)
