# Telegram security alerts

When a camera node sees someone **without a demonstrator shirt** in a **restricted zone**, it sends an
alert to the team's Telegram group: photo, zone, camera, time, plus buttons to **Acknowledge** or
**Mute zone 10 min**.

**One bot for all cameras:** every camera node sends its own alerts with the same bot token.
Exactly **one** node is the *listener* (`TELEGRAM_LISTEN=1`): it handles the buttons and
commands for everyone, and passes mutes to the other cameras.

Why it fits the challenge:
- **No central server for alerts:** every camera node sends its own alerts directly to Telegram.
  If the listener node goes down, the other cameras still alert – only the buttons pause.
- **Works when comms fail:** no internet → alerts are saved on disk, the camera keeps watching;
  when the connection returns they are delivered, marked "sent X min late", plus a "reconnected" notice.
- **No spam:** one alert per zone per minute (`cooldown`), and zones can be muted from Telegram.

Only `telegram_alerts.py` is needed – standard library, no `pip install`.

## 1. Create the bot (2 min)
1. In Telegram, open **@BotFather** → `/newbot` → give it a name, e.g. `Durhack Security node 2`,
   and a username ending in `bot`.
2. Copy the **token** it gives you (`123456789:AA...`). **Never commit it to git** – share it
   with teammates privately (e.g. a Telegram DM).
3. Every laptop uses this same token.

## 2. Make a group and get its chat id
1. Create a Telegram group, e.g. *Durhack Security*, and add the team + the bot(s).
2. In the group send `/start@your_bot_username`.
3. In PyCharm's Terminal (PowerShell):
   ```
   $env:TELEGRAM_BOT_TOKEN="123456789:AA..."
   python telegram_alerts.py setup
   ```
   It prints `chat id -100...   Durhack Security`. Group ids start with `-` (e.g. `-5546281244`; some start with `-100`).
4. On **every** laptop set the token and chat id; on **one** laptop also set the listener:
   ```
   $env:TELEGRAM_BOT_TOKEN="123456789:AA..."
   $env:TELEGRAM_CHAT_ID="-100..."
   $env:TELEGRAM_LISTEN="1"        # ONLY on the listener laptop
   ```
   (`$env:` lasts until the terminal closes. For Run Configurations, put them under
   **Environment variables** instead.)

## 3. Test it
```
python telegram_alerts.py test --node node_2 --zone organiser_room
python telegram_alerts.py test --node node_2 --zone organiser_room --image some_photo.jpg
python telegram_alerts.py run --node node_2      # then try /status, /test, /mute organiser_room 5
```
Only ONE program per bot token can read buttons/commands at a time (`run` counts as one).
If you see "another program is already handling this bot's buttons", check that only one
laptop has `TELEGRAM_LISTEN=1` and no `run` is open elsewhere. Sending alerts is unaffected.

## 4. Use it from the detector
```python
from telegram_alerts import TelegramAlerter

alerts = TelegramAlerter.from_env(
    "node_2",
    zone_labels={"organiser_room": "Organiser room"},
    cooldown=60,                        # seconds between alerts for the same zone
    status_fn=lambda: [f"Peers online: {len(peers)}"],   # optional extra /status lines
    on_mute=lambda zone, until: network.broadcast({"type": "mute", "zone": zone, "until": until}),
)
alerts.start()

# when this node receives a mute message from the listener over the team network:
alerts.mute_until(msg["zone"], msg["until"])          # until=0 means unmuted

# inside the camera loop, for each person found in a restricted zone:
if in_restricted_zone and not wearing_demonstrator_shirt:
    alerts.intrusion("organiser_room", frame=annotated_frame, confidence=0.87)
```
`intrusion()` never blocks the camera loop – sending happens in the background, and it's safe
to call every frame (repeats are ignored during the cooldown).

## Demo script (comms failure)
1. Walk into the restricted zone without the shirt → alert with photo arrives in the group.
2. Tap **Acknowledge** → message shows "Acknowledged by …".
3. Turn the laptop's Wi-Fi **off**, walk in again → the local view still detects it.
4. Turn Wi-Fi **on** → "node_2 reconnected after 1 min" + the saved alert, marked "sent 1 min late".

## Good to know
- Photos of people are sent to Telegram – ask the organisers first, and use
  `send_photos=False` if they say no.
- Alerts that Telegram rejects (wrong token/chat) go to `telegram_outbox/<node>/failed/`.
- Add to `.gitignore`: `telegram_outbox/` and `.env`.
