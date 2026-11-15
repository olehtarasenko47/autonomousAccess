import tempfile, time, unittest
from pathlib import Path
import numpy as np
import cv2
from peer_state import State, Peer
from fallback import incident
from snapshots import Photo, PhotoTransport, evidence_for, MAX_JPEG
from telegram_alerts import TelegramAlerter
from test_access import eventually, ports

class SnapshotTests(unittest.TestCase):
    def test_encrypted_chunks_reassembly_and_stale(self):
        with tempfile.TemporaryDirectory() as d:
            a=State('node_2',Path(d)/'a.db');b=State('node_3',Path(d)/'b.db')
            try:
                ts=time.time()-1;img=np.random.default_rng(12).integers(0,256,(640,480,3),dtype=np.uint8)
                photo=a.photos.capture('node_2',ts,img);self.assertIsNotNone(photo);self.assertLessEqual(len(photo.jpeg),MAX_JPEG)
                self.assertIsNotNone(cv2.imdecode(np.frombuffer(photo.jpeg,np.uint8),cv2.IMREAD_COLOR))
                tx=PhotoTransport(b'key',a.photos);rx=PhotoTransport(b'key',b.photos);wrong=PhotoTransport(b'wrong',b.photos)
                parts=tx.packets(photo);self.assertGreater(len(parts),1)
                for part in reversed(parts):wrong.receive(part)
                self.assertIsNone(b.photos.get('node_2'))
                for part in reversed(parts):rx.receive(part)
                self.assertEqual(b.photos.get('node_2'),photo)
                self.assertFalse(b.photos.save(Photo('node_2',ts-1,photo.jpeg)))
                self.assertIsNone(b.photos.get('node_2',ts-1))
                self.assertTrue(all(len(Peer(a,[],key='key').packet('photo',p))<1400 for p in parts))
            finally:a.close();b.close()
    def test_real_peer_last_photo_on_failure(self):
        with tempfile.TemporaryDirectory() as d:
            ps=ports(3);ss=[State(f'node_{i+1}',Path(d)/f'{i}.db') for i in range(3)];pp=[]
            try:
                for i,s in enumerate(ss):pp.append(Peer(s,[f'127.0.0.1:{ps[j]}' for j in range(3) if j!=i],ps[i],'photo-test',snapshots=True))
                self.assertTrue(eventually(lambda:all(s.online_count()==3 for s in ss)))
                ts=time.time();ss[1].update('online',{'hacker':1},ts,1)
                pic=ss[1].photos.capture('node_2',ts,np.full((480,640,3),100,np.uint8));pp[1].send_latest_photo(force=True);pp[1].send('hello',ss[1].heartbeat())
                self.assertTrue(eventually(lambda:all(s.photos.get('node_2')==pic for s in ss)))
                pp[1].enabled=False
                self.assertTrue(eventually(lambda:ss[0].online_count()==2,7))
                ss[0].fallback.tick(pp[0]);down=next(e for e in ss[0].fallback.incidents() if e['kind']=='NODE_UNREACHABLE')
                self.assertEqual(evidence_for(ss[0],down),pic)
                risk=next(e for e in ss[0].fallback.incidents() if e['kind']=='POTENTIAL_TRESPASSING');self.assertEqual(evidence_for(ss[0],risk),pic)
                ss[1].update('offline');camera=next(e for e in ss[1].fallback.incidents() if e['kind']=='CAMERA_OFFLINE');self.assertEqual(evidence_for(ss[1],camera),pic)
                self.assertIsNone(evidence_for(ss[0],incident('NODE_UNREACHABLE','node_3','node_1',None)))
            finally:
                for p in pp:p.close()
                for s in ss:s.close()
            reopened=State('node_1',Path(d)/'0.db');self.assertEqual(reopened.photos.get('node_2'),pic);reopened.close()
    def test_telegram_photo_queue_and_separate_violations(self):
        with tempfile.TemporaryDirectory() as d:
            bot=TelegramAlerter('test',['-1'],'node_1',outbox_dir=d,listen=False,send_photos=True,cooldown=60)
            frame=np.zeros((100,100,3),np.uint8);ok,jpg=cv2.imencode('.jpg',frame);self.assertTrue(ok)
            self.assertTrue(bot.intrusion('B',frame=frame,bypass_cooldown=True));self.assertTrue(bot.intrusion('B',frame=frame,bypass_cooldown=True))
            bot.mute('B');self.assertIsNone(bot.intrusion('B',frame=frame,bypass_cooldown=True))
            self.assertTrue(bot.incident('test-outage','B','NODE_UNREACHABLE','Last image: STALE',jpeg=jpg.tobytes()))
            bot.incident('test-outage','B','NODE_UNREACHABLE','duplicate',jpeg=jpg.tobytes());self.assertEqual(bot.queued(),3)
            sent=[];bot.api.call=lambda method,params=None,**kw:sent.append((method,params,kw)) or {}
            bot.start();self.assertTrue(eventually(lambda:bot.queued()==0));bot.stop()
            self.assertEqual(len(sent),3);self.assertTrue(all(x[0]=='sendPhoto' and x[2]['files']['photo'][1] for x in sent))
            self.assertTrue(any('STALE' in x[1]['caption'] for x in sent))
    def test_no_photo_when_disabled(self):
        with tempfile.TemporaryDirectory() as d:
            bot=TelegramAlerter('test',['-1'],'node_1',outbox_dir=d,listen=False,send_photos=False)
            bot.incident('test','B','CAMERA_OFFLINE','offline',jpeg=b'example')
            self.assertFalse(list(Path(d).rglob('*.jpg')))

if __name__=='__main__':unittest.main(verbosity=2)
