import tempfile,time,unittest
from pathlib import Path
from peer_state import State,Peer,new_event
from fallback import incident, valid_incident, valid_snapshot
from test_access import eventually,ports
from telegram_alerts import TelegramAlerter

class FallbackTests(unittest.TestCase):
    def test_camera_snapshot_and_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.db';s=State('node_2',path)
            ts=time.time()-1;s.update('online',{'hacker':1,'mentor':2},ts,1)
            s.update('offline');snap=s.fallback.local_snapshot()
            self.assertTrue(valid_snapshot(snap));self.assertEqual(snap['last_seen'],ts);self.assertEqual(snap['violations'],1)
            self.assertEqual(snap['counts'],{'hacker':1,'mentor':2});self.assertEqual(s.fallback.incidents()[0]['kind'],'CAMERA_OFFLINE')
            s.update('offline');self.assertEqual(len(s.fallback.incidents()),1)
            s.update('online',{},time.time(),0);self.assertEqual(s.fallback.incidents()[-1]['kind'],'CAMERA_RECOVERED')
            self.assertEqual(s.fallback.local_snapshot()['violations'],0);self.assertEqual(s.fallback.local_snapshot()['last_violation'],ts)
            s.close();s=State('node_2',path);self.assertEqual(s.fallback.last_violation,ts);s.close()
    def test_risk_latches_until_fresh_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            s=State('node_2',Path(d)/'db');p=Peer(s)
            try:
                ts=time.time()-3600
                s.update('online',{'hacker':1},ts,1)
                p.publish(new_event('node_2','hacker','purple','violation',1,.8,ts))
                s.fallback.tick(p,now=time.monotonic()+1)
                self.assertFalse(s.fallback.risk('node_3'))
                s.update('offline')
                s.fallback.tick(p,now=time.monotonic()+2)
                risks=[e for e in s.fallback.incidents() if e['kind']=='POTENTIAL_TRESPASSING']
                self.assertEqual(len(risks),1);risk=risks[0]
                self.assertTrue(valid_incident(risk));self.assertEqual(risk['expires'],0)
                self.assertTrue(s.fallback.risk('node_3'));self.assertTrue(s.fallback.risk_incident_active(risk))
                s.fallback.tick(p,now=time.monotonic()+3600)
                self.assertTrue(s.fallback.risk('node_3'))
                self.assertEqual(len([e for e in s.fallback.incidents() if e['kind']=='POTENTIAL_TRESPASSING']),1)
                s.update('online',{'hacker':1},ts,1) # stale frame cannot clear the flag
                self.assertTrue(s.fallback.risk('node_3'))
                s.update('online',{'hacker':1},time.time(),1)
                self.assertFalse(s.fallback.risk('node_3'));self.assertFalse(s.fallback.risk_incident_active(risk))
                self.assertEqual(s.event_count(),1);self.assertIn(risk,s.fallback.incidents())
                s.update('offline');s.fallback.tick(p,now=time.monotonic()+3601)
                self.assertTrue(s.fallback.risk('node_3'))
                self.assertEqual(len([e for e in s.fallback.incidents() if e['kind']=='POTENTIAL_TRESPASSING']),2)
            finally:p.close();s.close()

    def test_no_risk_without_last_violation(self):
        with tempfile.TemporaryDirectory() as d:
            s=State('node_2',Path(d)/'db');p=Peer(s)
            try:
                s.update('online',{'mentor':1},time.time(),0);s.update('offline')
                s.fallback.tick(p)
                self.assertFalse(s.fallback.risk('node_3'))
                self.assertFalse(any(e['kind']=='POTENTIAL_TRESPASSING' for e in s.fallback.incidents()))
            finally:p.close();s.close()

    def test_network_only_outage_and_stale_reconnect(self):
        with tempfile.TemporaryDirectory() as d:
            s=State('node_3',Path(d)/'db');p=Peer(s)
            source=State('node_2',Path(d)/'source.db')
            try:
                ts=time.time()-1
                source.update('online',{'hacker':1},ts,1)
                s.receive_heartbeat(source.heartbeat())
                self.assertFalse(s.fallback.risk())
                # Simulate missing heartbeats while keeping the remote camera online.
                heard=s.heard['node_2'];s.heard['node_2']=(time.monotonic()-10,*heard[1:])
                self.assertTrue(s.fallback.risk())
                p.publish_incident(incident('NODE_UNREACHABLE','node_2','node_1',s.fallback.snapshot('node_2')))
                s.fallback.tick(p)
                self.assertTrue(any(e['kind']=='POTENTIAL_TRESPASSING' for e in s.fallback.incidents()))
                s.receive_heartbeat(source.heartbeat())
                self.assertTrue(s.fallback.risk())
                source.update('online',{'hacker':1},time.time(),1)
                s.receive_heartbeat(source.heartbeat())
                self.assertFalse(s.fallback.risk())
            finally:p.close();s.close();source.close()

    def test_peer_snapshot_outage_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            ps=ports(3);ss=[State(f'node_{i+1}',Path(d)/f'{i}.db') for i in range(3)];pp=[]
            try:
                for i,s in enumerate(ss):pp.append(Peer(s,[f'127.0.0.1:{ps[j]}' for j in range(3) if j!=i],ps[i],'test'))
                ts=time.time();ss[1].update('online',{'hacker':1},ts,1)
                self.assertTrue(eventually(lambda:all(s.online_count()==3 for s in ss)))
                self.assertTrue(eventually(lambda:ss[0].fallback.snapshot('node_2')['violations']==1))
                pp[1].publish(new_event('node_2','hacker','purple','violation',1,.8))
                self.assertTrue(eventually(lambda:ss[0].event_count()==1));ss[0].fallback.tick(pp[0]);ss[1].fallback.tick(pp[1]);
                self.assertFalse(ss[2].fallback.risk())
                ss[1].update('offline');self.assertTrue(eventually(lambda:any(e['kind']=='CAMERA_OFFLINE' for e in ss[0].fallback.incidents())))
                ss[1].fallback.tick(pp[1],now=time.monotonic()+1)
                self.assertTrue(eventually(lambda:ss[2].fallback.risk()))
                self.assertTrue(eventually(lambda:any(e['kind']=='POTENTIAL_TRESPASSING' for e in ss[0].fallback.incidents())))
                risk=next(e for e in ss[0].fallback.incidents() if e['kind']=='POTENTIAL_TRESPASSING')
                self.assertFalse(ss[0].fallback.notification_owner(risk))
                ss[1].fallback.mark_notified(risk['id'])
                self.assertTrue(eventually(lambda:ss[0].fallback.notified(risk['id'])))
                ss[1].update('offline');self.assertTrue(eventually(lambda:any(e['kind']=='CAMERA_OFFLINE' for e in ss[0].fallback.incidents())))
                self.assertEqual(ss[0].fallback.snapshot('node_2')['last_seen'],ts)
                pp[1].enabled=False
                # Wait for actual heartbeat timeout (not a fake packet).
                self.assertTrue(eventually(lambda:ss[0].online_count()==2,7))
                ss[0].fallback.tick(pp[0]);ss[2].fallback.tick(pp[2]);
                down=[e for e in ss[0].fallback.incidents() if e['kind']=='NODE_UNREACHABLE'];self.assertEqual(len(down),1)
                self.assertTrue(ss[0].fallback.notification_owner(risk));self.assertTrue(ss[0].fallback.notified(risk['id']))
                self.assertEqual(down[0]['snapshot']['last_seen'],ts);self.assertEqual(down[0]['snapshot']['violations'],1)
                self.assertTrue(eventually(lambda:any(e['kind']=='NODE_UNREACHABLE' for e in ss[2].fallback.incidents())))
                pp[1].enabled=True;ss[1].update('online',{},time.time(),0)
                self.assertTrue(eventually(lambda:ss[0].online_count()==3));ss[0].fallback.tick(pp[0]);
                self.assertTrue(eventually(lambda:any(e['kind']=='NODE_RECOVERED' for e in ss[2].fallback.incidents())))
                self.assertTrue(eventually(lambda:not any(s.fallback.risk('node_3') for s in ss)))
            finally:
                for p in pp:p.close()
                for s in ss:s.close()
    def test_telegram_operational_queue(self):
        with tempfile.TemporaryDirectory() as d:
            bot=TelegramAlerter('test',['-1'],'node_1',outbox_dir=d,listen=False,send_photos=False)
            bot.mute('B');self.assertIsNone(bot.incident('x','B','POTENTIAL_TRESPASSING','risk'))
            self.assertTrue(bot.incident('y','B','NODE_UNREACHABLE','NODE_UNREACHABLE last observation UTC'))
            bot.incident('y','B','NODE_UNREACHABLE','NODE_UNREACHABLE last observation UTC');self.assertEqual(bot.queued(),1)
            sent=[];bot.api.call=lambda method,params=None,**kw:sent.append((method,params)) or {}
            bot.start();self.assertTrue(eventually(lambda:bot.queued()==0));bot.stop()
            self.assertIn('NODE_UNREACHABLE',sent[0][1]['text'])
            self.assertIn('last observation UTC',sent[0][1]['text']);self.assertNotIn('reply_markup',sent[0][1])

if __name__=='__main__':unittest.main(verbosity=2)
