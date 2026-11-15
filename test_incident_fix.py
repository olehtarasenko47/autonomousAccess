import tempfile, time, unittest
from pathlib import Path
from fallback import CameraHealth
from peer_state import State, Peer
from telegram_alerts import TelegramAlerter
from test_access import eventually, ports

class IncidentFixTests(unittest.TestCase):
    def test_camera_flapping(self):
        c=CameraHealth(0.)
        self.assertTrue(c.frame(.1,1));self.assertFalse(c.missing(1.5));self.assertTrue(c.missing(3.2))
        self.assertFalse(c.missing(4.));self.assertFalse(c.frame(4.2,2))
        self.assertFalse(c.missing(6.));self.assertFalse(c.frame(6.1,3))
        self.assertFalse(c.frame(6.8,3));self.assertFalse(c.frame(7.5,3));self.assertFalse(c.frame(8.,3));self.assertTrue(c.frame(8.2,3))
        self.assertFalse(c.missing(9.));self.assertTrue(c.missing(11.3));self.assertFalse(c.missing(12.))
    def test_outage_risk_delivery_after_hard_failure(self):
        with tempfile.TemporaryDirectory() as d:
            ps=ports(3);ss=[State(f'node_{i+1}',Path(d)/f'{i}.db') for i in range(3)];pp=[]
            try:
                for i,s in enumerate(ss):pp.append(Peer(s,[f'127.0.0.1:{ps[j]}' for j in range(3) if j!=i],ps[i],'test'))
                ss[1].update('online',{'hacker':1},time.time(),1);pp[1].send('hello',ss[1].heartbeat())
                self.assertTrue(eventually(lambda:all(s.online_count()==3 for s in ss)))
                self.assertTrue(eventually(lambda:all(s.fallback.snapshot('node_2') and s.fallback.snapshot('node_2')['violations']==1 for s in (ss[0],ss[2]))))
                pp[1].close()  # abrupt process/network loss, no camera-offline message
                self.assertTrue(eventually(lambda:ss[0].online_count()==2 and ss[2].online_count()==2,7))
                for i in (0,2):ss[i].fallback.tick(pp[i])
                e=next(x for x in ss[0].fallback.incidents() if x['kind']=='POTENTIAL_TRESPASSING')
                self.assertTrue(ss[0].fallback.risk_incident_active(e));self.assertTrue(ss[0].fallback.notification_owner(e));self.assertFalse(ss[2].fallback.notification_owner(e))
                bot=TelegramAlerter('test',['-1'],'node_1',outbox_dir=d,listen=False,send_photos=False)
                delivered=[];bot.api.call=lambda method,params=None,**kw:delivered.append((method,params)) or {}
                self.assertTrue(bot.incident(e['id'],e['zone'],e['kind'],ss[0].fallback.notice(e)));bot.start()
                self.assertTrue(eventually(lambda:bot.queued()==0));bot.stop();self.assertEqual(len(delivered),1);self.assertIn('POTENTIAL_TRESPASSING',delivered[0][1]['text'])
                # Even a restart/replay of a previously delivered incident must not recreate it.
                bot2=TelegramAlerter('test',['-1'],'node_1',outbox_dir=d,listen=False,send_photos=False)
                bot2.incident(e['id'],e['zone'],e['kind'],'duplicate');self.assertEqual(bot2.queued(),0)
                for i in (0,2):ss[i].fallback.tick(pp[i],now=time.monotonic()+3600)
                self.assertTrue(ss[2].fallback.risk())
                self.assertEqual(len([x for x in ss[0].fallback.incidents() if x['kind']=='POTENTIAL_TRESPASSING']),1)
            finally:
                for p in pp:p.close()
                for s in ss:s.close()
    def test_recent_violation_before_gap(self):
        with tempfile.TemporaryDirectory() as d:
            s=State('node_2',Path(d)/'db');p=Peer(s)
            try:
                ts=time.time()-3;s.update('online',{'hacker':1},ts,1);s.update('online',{},ts+1,0);s.update('offline');s.fallback.tick(p)
                self.assertTrue(s.fallback.risk('node_3'))
                s.update('online',{},ts+2,0);self.assertFalse(s.fallback.risk('node_3'))
                s.update('online',{},ts+3,0);s.update('offline');self.assertFalse(s.fallback.risk('node_3'))
            finally:p.close();s.close()
if __name__=='__main__':unittest.main(verbosity=2)
