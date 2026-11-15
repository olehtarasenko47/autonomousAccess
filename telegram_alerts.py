#!/usr/bin/env python3
"""
Telegram security alerts for ONE camera node.

Each node runs its own alerter (ideally its own bot), so alerts never depend on
a central server. If the internet drops, alerts are saved to disk and sent when
it comes back, marked as late.

    from telegram_alerts import TelegramAlerter
    alerts = TelegramAlerter.from_env("node_2", zone_labels={"room_1a": "Organiser room"})
    alerts.start()
    ...
    # when the detector sees someone in a restricted zone without a staff shirt:
    alerts.intrusion("room_1a", frame=annotated_frame, confidence=0.87)

Set up once (see TELEGRAM.md):
    set TELEGRAM_BOT_TOKEN=123456:ABC...            (Windows PowerShell: $env:TELEGRAM_BOT_TOKEN="...")
    python telegram_alerts.py setup                 # shows your chat id after you message the bot
    set TELEGRAM_CHAT_ID=-1001234567890
    python telegram_alerts.py test --node node_2 --zone room_1a
    python telegram_alerts.py run --node node_2     # try /status, /mute, /test in Telegram

In Telegram:  /status  /mute <zone> [minutes]  /unmute <zone>  /test
Alert buttons: Acknowledge, Mute zone 10 min
"""
import argparse
import html
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

API_BASE = "https://api.telegram.org"
NETWORK_ERRORS = (urllib.error.URLError, OSError, TimeoutError, ConnectionError)


class TelegramError(Exception):
    def __init__(self, description, code=None, retry_after=None):
        super().__init__(f"Telegram error {code}: {description}")
        self.code, self.retry_after = code, retry_after


def _multipart(fields, files):
    boundary = uuid.uuid4().hex
    out = bytearray()
    for k, v in fields.items():
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n'
                f'{v}\r\n').encode()
    for k, (fname, data, mime) in files.items():
        out += (f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; '
                f'filename="{fname}"\r\nContent-Type: {mime}\r\n\r\n').encode()
        out += data + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


class TelegramAPI:
    """Tiny Telegram Bot API client using only the standard library."""

    def __init__(self, token, base=API_BASE, timeout=10):
        self.url = f"{base.rstrip('/')}/bot{token}/"
        self.timeout = timeout

    def call(self, method, params=None, files=None, timeout=None):
        params = {k: v for k, v in (params or {}).items() if v is not None}
        if files:
            fields = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                      for k, v in params.items()}
            body, ctype = _multipart(fields, files)
        else:
            body, ctype = json.dumps(params).encode(), "application/json"
        req = urllib.request.Request(self.url + method, data=body,
                                     headers={"Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:      # Telegram explains errors in JSON
            try:
                data = json.loads(e.read())
            except ValueError:
                raise TelegramError(e.reason, e.code)
        if not data.get("ok"):
            raise TelegramError(data.get("description", "unknown"), data.get("error_code"),
                                (data.get("parameters") or {}).get("retry_after"))
        return data["result"]


def _ago(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60}s"
    return f"{seconds // 3600} h {seconds % 3600 // 60} min"


class TelegramAlerter:
    def __init__(self, token, chat_ids, node_id, zone_labels=None, cooldown=60,
                 send_photos=True, outbox_dir="telegram_outbox", listen=True,
                 api_base=API_BASE, status_fn=None, late_after=30, log=print,
                 on_mute=None):
        """
        chat_ids    : who gets alerts (a group chat id is best: the whole team sees them)
        zone_labels : nicer names, e.g. {"room_1a": "Organiser room"}
        cooldown    : seconds before the SAME zone can alert again (no spam every frame)
        listen      : answer buttons and commands. With ONE bot shared by all cameras,
                      exactly one node listens; all nodes can send.
        status_fn   : optional function returning extra lines for /status (e.g. peers online)
        on_mute     : called as on_mute(zone, until) when someone mutes/unmutes a zone in
                      Telegram (until=0 means unmuted). Broadcast it to the other nodes,
                      which call alerts.mute_until(zone, until).
        """
        self.on_mute = on_mute
        self._warned_conflict = False
        self.api = TelegramAPI(token, api_base)
        self.chat_ids = [str(c) for c in chat_ids]
        self.node_id = node_id
        self.zone_labels = zone_labels or {}
        self.cooldown, self.send_photos, self.listen = cooldown, send_photos, listen
        self.status_fn, self.late_after, self.log = status_fn, late_after, log
        self.outbox = Path(outbox_dir) / node_id
        (self.outbox / "failed").mkdir(parents=True, exist_ok=True)

        self.online = None                # None = unknown yet
        self.offline_since = None
        self.sent = 0
        self.recent = deque(maxlen=10)    # recent alerts (for /status)
        self._last_alert, self._muted = {}, {}
        self._lock = threading.Lock()
        self._wake, self._stop = threading.Event(), threading.Event()
        self._threads = []

    @classmethod
    def from_env(cls, node_id, **kw):
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chats = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not token or not chats:
            raise SystemExit("Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID (see TELEGRAM.md)")
        # one shared bot: only the node with TELEGRAM_LISTEN=1 handles buttons/commands
        kw.setdefault("listen", os.environ.get("TELEGRAM_LISTEN", "0") == "1")
        return cls(token, [c.strip() for c in chats.split(",") if c.strip()], node_id, **kw)

    # ---------------------------------------------------------------- public --
    def start(self):
        targets = [self._sender] + ([self._listener] if self.listen else [])
        for fn in targets:
            t = threading.Thread(target=fn, daemon=True, name=fn.__name__)
            t.start()
            self._threads.append(t)
        return self

    def stop(self):
        self._stop.set()
        self._wake.set()

    def label(self, zone):
        return self.zone_labels.get(zone, zone)

    def is_muted(self, zone):
        return self._muted.get(zone, 0) > time.time()

    def mute(self, zone, minutes=10):
        self._muted[zone] = time.time() + minutes * 60

    def unmute(self, zone):
        self._muted.pop(zone, None)

    def mute_until(self, zone, until):
        """Apply a mute received from another node (until=0 or past = unmute)."""
        if until and until > time.time():
            self._muted[zone] = until
        else:
            self.unmute(zone)

    def _set_mute_from_telegram(self, zone, minutes):
        if (self.zone_labels and zone not in self.zone_labels) or not 0 <= minutes <= 1440:
            raise ValueError("Unknown zone or invalid mute duration (0–1440 minutes)")
        until = time.time() + minutes * 60 if minutes > 0 else 0
        self.mute_until(zone, until)
        if self.on_mute:
            try:
                self.on_mute(zone, until)          # tell the other cameras
            except Exception as e:
                self.log(f"[telegram] on_mute hook failed: {e}")

    def queued(self):
        return len(list(self.outbox.glob("*.json")))

    def intrusion(self, zone, frame=None, confidence=None, ts=None, note=None, force=False, bypass_cooldown=False):
        """
        Report someone in a restricted zone. Safe to call every frame: repeats
        within `cooldown` seconds and muted zones are ignored.
        Returns the alert id, or None if it was suppressed.
        Never blocks on the network - delivery happens in the background.
        """
        now = time.time()
        with self._lock:
            if not force and (self.is_muted(zone) or
                              (not bypass_cooldown and now - self._last_alert.get(zone, 0) < self.cooldown)):
                return None
            self._last_alert[zone] = now
        aid = f"{self.node_id}-{uuid.uuid4().hex}"
        item = {"id": aid, "zone": zone, "ts": ts or now, "confidence": confidence,
                "note": note, "photo": None, "done_chats": []}
        if frame is not None and self.send_photos:
            import cv2
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                photo = self.outbox / f"{aid}.jpg"
                photo.write_bytes(jpg.tobytes())
                item["photo"] = str(photo)
        tmp = self.outbox / f"{aid}.tmp"
        tmp.write_text(json.dumps(item))
        tmp.replace(self.outbox / f"{aid}.json")      # atomic: never a half-written alert
        self.recent.appendleft(dict(item))
        self._wake.set()
        return aid

    def incident(self, event_id, zone, kind, note, ts=None, jpeg=None):
        """Queue durable operational/risk notification, separate from access cooldown."""
        if kind == 'POTENTIAL_TRESPASSING' and self.is_muted(zone):return None
        aid=f"{self.node_id}-health-{event_id}"
        item={"id":aid,"zone":zone,"ts":ts or time.time(),"confidence":None,
              "note":note,"photo":None,"done_chats":[],"kind":kind}
        with self._lock:
            path=self.outbox/f"{aid}.json"
            if not path.exists():
                if jpeg is not None and self.send_photos:
                    photo=self.outbox/f'{aid}.jpg';photo.write_bytes(jpeg);item['photo']=str(photo)
                tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(item));tmp.replace(path)
        self._wake.set();return aid

    # ---------------------------------------------------------------- sending --
    def _format(self, item):
        t = datetime.fromtimestamp(item["ts"]).strftime("%H:%M:%S")
        if item.get("kind"):
            return "<b>ZoneGuard operational alert</b>\n"+html.escape(item.get("note") or item["kind"])
        lines = ["🚨 <b>Potential access violation</b>",
                 f"<b>Zone:</b> {html.escape(self.label(item['zone']))}",
                 f"<b>Camera:</b> {html.escape(self.node_id)}",
                 f"<b>Time:</b> {t}"]
        if item.get("confidence") is not None:
            lines.append(f"<b>Shirt match fraction:</b> {item['confidence']:.0%}")
        if item.get("note"):
            lines.append(html.escape(item["note"]))
        delay = time.time() - item["ts"]
        if delay > self.late_after:
            lines.append(f"<i>⏳ Sent {_ago(delay)} late: this camera was offline. "
                         f"It kept watching and saved the alert.</i>")
        return "\n".join(lines)

    def _buttons(self, item):
        return {"inline_keyboard": [[
            {"text": "✅ Acknowledge", "callback_data": f"ack:{item['id']}"[:64]},
            {"text": "🔇 Mute zone 10 min", "callback_data": f"mute:{item['zone']}"[:64]}]]}

    def _deliver(self, chat, item):
        params = {"chat_id": chat, "parse_mode": "HTML",
                  "reply_markup": self._buttons(item)}
        if item.get("kind"):params.pop("reply_markup",None)
        photo = item.get("photo")
        if photo and Path(photo).exists():
            params["caption"] = self._format(item)
            self.api.call("sendPhoto", params,
                          files={"photo": ("alert.jpg", Path(photo).read_bytes(), "image/jpeg")},
                          timeout=30)
        else:
            params["text"] = self._format(item)
            self.api.call("sendMessage", params)

    def _set_online(self, ok):
        was = self.online
        self.online = ok
        if not ok and was is not False:
            self.offline_since = time.time()
            self.log(f"[telegram] offline - alerts are being saved ({self.queued()} queued)")
        if ok and was is False:
            gone = _ago(time.time() - (self.offline_since or time.time()))
            self.log(f"[telegram] back online after {gone}")
            self._notify(f"✅ <b>{html.escape(self.node_id)}</b> reconnected after {gone} offline. "
                         f"Delivering saved alerts.")
            self.offline_since = None

    def _notify(self, text, chats=None):
        """Best-effort message (not queued)."""
        for chat in chats or self.chat_ids:
            try:
                self.api.call("sendMessage", {"chat_id": chat, "text": text,
                                              "parse_mode": "HTML"})
            except Exception as e:
                self.log(f"[telegram] could not send message: {e}")

    def _sender(self):
        backoff = 2
        while not self._stop.is_set():
            items = sorted(self.outbox.glob("*.json"))
            if not items:
                self._wake.wait(1)
                self._wake.clear()
                continue
            path = items[0]
            item = json.loads(path.read_text())
            try:
                for chat in self.chat_ids:
                    if chat in item["done_chats"]:
                        continue
                    self._deliver(chat, item)
                    item["done_chats"].append(chat)       # never send twice to one chat
                    tmp = path.with_suffix(".tmp")
                    tmp.write_text(json.dumps(item))
                    tmp.replace(path)
                self._set_online(True)
                path.unlink()
                if item.get("photo"):
                    Path(item["photo"]).unlink(missing_ok=True)
                self.sent += 1
                backoff = 2
            except TelegramError as e:
                if e.retry_after:                         # Telegram rate limit
                    self._stop.wait(e.retry_after)
                elif e.code in (400, 401, 403, 404):      # bad token/chat: retrying won't help
                    self.log(f"[telegram] alert {item['id']} rejected: {e} - moved to failed/")
                    path.replace(self.outbox / "failed" / path.name)
                else:
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, 60)
            except NETWORK_ERRORS:
                self._set_online(False)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 60)

    # -------------------------------------------------------------- commands --
    def _status_text(self):
        net = {True: "online", False: "OFFLINE", None: "connecting"}[self.online]
        lines = [f"📷 <b>{html.escape(self.node_id)}</b> (handling commands for all cameras)",
                 f"Internet: {net}",
                 f"Alerts sent: {self.sent}, waiting: {self.queued()}"]
        muted = [f"{html.escape(self.label(z))} ({_ago(u - time.time())} left)"
                 for z, u in self._muted.items() if u > time.time()]
        lines.append("Muted: " + (", ".join(muted) if muted else "none"))
        if self.status_fn:
            try:
                lines += [html.escape(str(x)) for x in self.status_fn()]
            except Exception as e:
                lines.append(f"(status error: {html.escape(str(e))})")
        if self.recent:
            lines.append("Recent alerts:")
            for it in list(self.recent)[:5]:
                t = datetime.fromtimestamp(it["ts"]).strftime("%H:%M:%S")
                lines.append(f"  {t}  {html.escape(self.label(it['zone']))}")
        return "\n".join(lines)

    def _handle_message(self, msg):
        chat = str(msg["chat"]["id"])
        text = (msg.get("text") or "").strip()
        if not text.startswith("/"):
            return
        parts = text.split()
        cmd = parts[0].split("@")[0].lower()
        args = parts[1:]
        if chat not in self.chat_ids:                 # strangers only learn their chat id
            if cmd in ("/start", "/help", "/id"):
                self._notify(f"Chat id: <code>{chat}</code>\nAdd it to TELEGRAM_CHAT_ID "
                             f"to receive alerts from {html.escape(self.node_id)}.", [chat])
            return
        if cmd in ("/start", "/help"):
            reply = (f"Security alerts from <b>{html.escape(self.node_id)}</b>\n"
                     f"/status  this camera's state\n/mute &lt;zone&gt; [min]\n"
                     f"/unmute &lt;zone&gt;\n/test  send a test alert\nChat id: <code>{chat}</code>")
        elif cmd == "/status":
            reply = self._status_text()
        elif cmd == "/mute" and args:
            mins = float(args[1]) if len(args) > 1 else 10
            self._set_mute_from_telegram(args[0], mins)
            reply = f"🔇 {html.escape(self.label(args[0]))} muted for {mins:g} min on all cameras"
        elif cmd == "/unmute" and args:
            self._set_mute_from_telegram(args[0], 0)
            reply = f"🔔 {html.escape(self.label(args[0]))} unmuted"
        elif cmd == "/test":
            self.intrusion(args[0] if args else "test_zone", note="(test alert)", force=True)
            reply = None
        else:
            reply = "Commands: /status /mute &lt;zone&gt; [min] /unmute &lt;zone&gt; /test"
        if reply:
            self._notify(reply, [chat])

    def _handle_callback(self, cb):
        data = cb.get("data", "")
        who = cb.get("from", {}).get("first_name", "someone")
        msg = cb.get("message") or {}
        chat = str(msg.get("chat", {}).get("id", ""))
        if chat not in self.chat_ids:
            return
        now = datetime.now().strftime("%H:%M")
        if data.startswith("ack:"):
            done, answer = f"✅ Acknowledged by {who} at {now}", "Acknowledged"
        elif data.startswith("mute:"):
            zone = data[5:]
            self._set_mute_from_telegram(zone, 10)
            done, answer = f"🔇 {self.label(zone)} muted 10 min by {who}", "Zone muted"
        else:
            return
        try:
            self.api.call("answerCallbackQuery", {"callback_query_id": cb["id"], "text": answer})
            self.api.call("editMessageReplyMarkup", {
                "chat_id": chat, "message_id": msg.get("message_id"),
                "reply_markup": {"inline_keyboard": [[{"text": done, "callback_data": "noop"}]]}})
        except Exception as e:
            self.log(f"[telegram] button reply failed: {e}")

    def _listener(self):
        offset = None
        while not self._stop.is_set():
            try:
                updates = self.api.call("getUpdates", {
                    "timeout": 25, "offset": offset,
                    "allowed_updates": ["message", "callback_query"]}, timeout=35)
            except TelegramError as e:
                if e.code == 409 and not self._warned_conflict:
                    self._warned_conflict = True
                    self.log("[telegram] another program is already handling this bot's buttons "
                             "and commands. With one shared bot, set TELEGRAM_LISTEN=1 on ONE "
                             "node only. (Sending alerts still works.)")
                self._stop.wait(30 if e.code == 409 else 10)
                continue
            except NETWORK_ERRORS:
                self._set_online(False)
                self._stop.wait(5)
                continue
            if self.online is not True:
                self._set_online(True)            # reached Telegram, so we're online
            for u in updates:
                offset = u["update_id"] + 1
                try:
                    if "callback_query" in u:
                        self._handle_callback(u["callback_query"])
                    elif "message" in u:
                        self._handle_message(u["message"])
                except Exception as e:
                    self.log(f"[telegram] could not handle update: {e}")


# ---------------------------------------------------------------------- CLI --
def _cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["setup", "test", "run"])
    ap.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN"))
    ap.add_argument("--chat", default=os.environ.get("TELEGRAM_CHAT_ID"))
    ap.add_argument("--node", default="node_test")
    ap.add_argument("--zone", default="restricted_area")
    ap.add_argument("--image", help="photo to attach to the test alert")
    ap.add_argument("--api", default=API_BASE, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if not a.token:
        raise SystemExit("No bot token: set TELEGRAM_BOT_TOKEN or pass --token (see TELEGRAM.md)")

    api = TelegramAPI(a.token, a.api)
    if a.action == "setup":
        me = api.call("getMe")
        print(f"Bot OK: @{me['username']}\n")
        chats = {}
        for u in api.call("getUpdates", {"timeout": 0}):
            m = u.get("message") or u.get("my_chat_member") or {}
            c = m.get("chat")
            if c:
                chats[c["id"]] = c.get("title") or c.get("first_name") or c.get("username")
        if not chats:
            print("No chats yet. Send /start to the bot (or add it to your group and send "
                  "/start@" + me["username"] + "), then run setup again.")
        for cid, name in chats.items():
            print(f"chat id {cid}   {name}")
        if chats:
            print("\nNow set TELEGRAM_CHAT_ID to the id above - copy it exactly, including the minus sign.")
        return

    if not a.chat:
        raise SystemExit("No chat id: set TELEGRAM_CHAT_ID or pass --chat (run setup first)")
    alerter = TelegramAlerter(a.token, a.chat.split(","), a.node, api_base=a.api,
                              listen=(a.action == "run"), cooldown=0)
    alerter.start()
    if a.action == "test":
        frame = None
        if a.image:
            import cv2
            frame = cv2.imread(a.image)
        aid = alerter.intrusion(a.zone, frame=frame, confidence=0.9, note="(test alert)")
        failed = alerter.outbox / "failed" / f"{aid}.json"
        for _ in range(60):
            if failed.exists():
                print("Test alert FAILED - Telegram rejected it (see the error above). "
                      "Check TELEGRAM_CHAT_ID matches what 'setup' printed, digit for digit.")
                break
            if alerter.queued() == 0:
                print("Test alert delivered - check Telegram.")
                break
            time.sleep(0.5)
        else:
            print("Not delivered yet (offline?). It stays queued in", alerter.outbox)
        alerter.stop()
        return
    print(f"{a.node}: listening. In Telegram try /status, /test, /mute <zone> 5. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        alerter.stop()


if __name__ == "__main__":
    _cli()
