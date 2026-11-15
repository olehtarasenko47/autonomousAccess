# ZoneGuard — decentralised role-aware access monitoring

One phone streams to one laptop. Each laptop independently runs YOLO11n-pose, Oleh's four-tier torso extraction, calibrated HSV classification and ByteTrack. It makes its own access decision, stores its own events, and shares event metadata with its peers. No face recognition, personal identity, or shared video feeds.

## Deployment

| Node | Camera | Camera IP | Physical zone | Allowed roles |
|---|---|---|---|---|
| node_1 | camera1 | 10.249.58.251 | A (public) | All |
| node_2 | camera3 | 10.249.37.86 | B (staff) | volunteer, mentor, organiser |
| node_3 | camera4 | 10.249.55.81 | C (organisers) | organiser |

Camera 2 is unused. Camera 3's username and `/mjpeg` path must be checked in Guardian Eye. The `--node` parameter selects both the default camera and the policy; `--source` can override the camera URL, e.g. after an IP change. All passwords are provided locally at runtime. No passwords or bot tokens are included.

`hacker` is a **role ID**, not a personal ID. Purple is displayed as `HACKER (PURPLE)`; ordinary clothes as `HACKER (OTHER)`. A missing, ambiguous, or obscured shirt never grants staff clearance and is displayed as `HACKER (CHECK)` for review in restricted areas. The other three roles only match a fresh, named calibration profile. Similar-coloured non-event clothing can still be mistaken for staff: this is a monitoring aid, not authentication or a door lock.

## Install and test

Use the existing autonomousAccess repository and branch. Copy the release files into its root (backup or commit earlier work first). Do not extract into a second nested project. This release supersedes the old `main.py`, `calibration.py`, and uses `torso.py` and `video_source.py`. `phone_stream.py`, `shirt_detector.py` and `calibrate_colour.py` from earlier variants are not used by the new runtime. Retain them on another branch if desired.

```powershell
python -m pip install -r requirements.txt
python test_access.py
```

Fresh calibration is deliberate. The release contains **no calibration samples or presets**. Remove old `colour.json`, `colour_*.json`, and any `calibrations/` directory before starting a fresh calibration. The new app does not read legacy colour files at all. New profiles are ignored by Git and stored locally. Keep the YOLO weights you already downloaded (`yolo11n-pose.pt`).

## Calibrate four roles on each camera

On each laptop, replace `node_1` below with that laptop's node. Start the corresponding Guardian Eye MJPEG stream. Set the password once for this terminal (or omit this and the program will prompt for it without echoing). The same camera password can be entered on each laptop.

```powershell
$env:CAMERA_PASSWORD = Read-Host "Camera password"
$node = "node_1"
foreach ($role in @("hacker", "volunteer", "mentor", "organiser")) {
    python calibration.py --node $node --role $role
}
```

For the hacker role sample the actual purple event shirt. For each other role, use the shirt confirmed by the event organisers. Do not assume which actual colour belongs to which staff role. Do not sample multiple colours in one session. Click at least five plain fabric patches, preferably under different lighting/angles, not logos, skin or background. P freezes/resumes, M toggles mask, U undoes a sample, R clears samples, S saves, Q closes the current role. If the hue span is too wide, saving is rejected. Repeat for all four. For one camera, files are `calibrations/node_1/{hacker,volunteer,mentor,organiser}.json`. Each is independent; per-camera calibration accounts for camera and lighting differences. Recalibrate individual roles using the same command when needed. The new app loads profiles on startup; restart after saving.

## Run one node

```powershell
python main.py --node node_1
python main.py --node node_2
python main.py --node node_3
```

Run only the matching command on each laptop. The entire camera view is the monitored zone by default: point it inside the relevant physical zone, not at an adjacent public corridor. The optional `--split 0.5` restricts the right half and leaves the left half public for a tabletop demonstration. This is presence monitoring, not evidence of a directional doorway crossing. `--width 1400 --height 900` fits the frame with aspect ratio intact; `--imgsz 480` can improve CPU performance at the cost of distant detections. `q` exits, `p` prints peer status, `c` pauses/resumes only peer communication for a fault-tolerance demonstration.

A track requires at least three consistent observations across 0.35 seconds before a restricted decision is confirmed. The display shows green for allowed, amber for pending/review, red for potential violation. A confirmed violation is recorded once per local track/occupancy episode (temporary tracker IDs are not globally unique). If a person leaves/reappears or the camera reconnects, a new alert may be produced. Ambiguous/unreadable shirt observations produce review events, not automatic Telegram alarms. With missing calibrations, local preview and review events still work, but restricted-node Telegram activation is blocked until all four role profiles are present.

## Peer-to-peer communication (three independent laptops)

Get each laptop's Wi-Fi IPv4 address with `ipconfig` (not the phone IP). All laptops must be reachable on UDP 50565; allow this port on the trusted network if Windows Firewall blocks it. Share a new random secret among the three teammates, not the camera password or Telegram token. On each laptop:

```powershell
$env:ZONEGUARD_PEER_KEY = Read-Host "Shared peer secret"
```

Then start with the other two **laptop** addresses, replacing the placeholders:

```powershell
# Laptop 1
python main.py --node node_1 --peer LAPTOP2_IP --peer LAPTOP3_IP
# Laptop 2
python main.py --node node_2 --peer LAPTOP1_IP --peer LAPTOP3_IP
# Laptop 3
python main.py --node node_3 --peer LAPTOP1_IP --peer LAPTOP2_IP
```

All use one deployment name by default. Each laptop stores SQLite state under `runtime/durhack-access-v2/`. The peers exchange signed (HMAC, not encrypted) heartbeats, immutable UUID events and mute versions. One-second gossip replays event history and latest mutes after packet loss/partitions; event IDs prevent duplication. The full event history is sent periodically, suitable for a short demo, not indefinite high-volume use. The preview displays number of reachable nodes and replicated events. P prints camera/peer status. Telegram `/status` on the command listener also reports peer status. The system is eventually consistent, not a guarantee of simultaneous state under a partition. Clocks should be synchronised. A missing camera cannot monitor its zone; other nodes keep monitoring theirs.

## Telegram integration (optional)

The uploaded team's `telegram_alerts.py` is integrated. Use one bot token and one group chat for all laptops. Obtain the token from the teammate/BotFather; never commit it. Add the bot to the group and send `/start@BOT_USERNAME`. To discover the chat ID, stop any command listener and run once:

```powershell
$env:TELEGRAM_BOT_TOKEN = Read-Host "Bot token"
python telegram_alerts.py setup
```

Then on all three laptops:

```powershell
$env:TELEGRAM_BOT_TOKEN = Read-Host "Bot token"
$env:TELEGRAM_CHAT_ID = Read-Host "Group chat ID"
$env:TELEGRAM_LISTEN = "0"
```

On **only one** laptop (suggested node_1), set `$env:TELEGRAM_LISTEN = "1"`. Add `--telegram` to each node's launch command, retaining its peer options. Example:

```powershell
python main.py --node node_2 --peer LAPTOP1_IP --peer LAPTOP3_IP --telegram
```

The restricted nodes must have all four profiles. Every node sends its own new confirmed violations directly; replicated events do not send duplicates. Default per-zone cooldown 60 seconds for Telegram, not local events. Change with `--cooldown 10` for a demo. The disk-backed queue is under `telegram_outbox/` and retries network failures. Telegram is an external delivery service, not a central processing node. If the command-listener laptop stops, the other nodes can still send alerts; command buttons pause until a listener returns. `/status`, `/mute C 10`, `/unmute C`, acknowledge buttons, and the mute button are supported. Use zone IDs A/B/C; mutes suppress future Telegram alerts only, not local monitoring or event logs. Mutes replicate and persist via peers; a partitioned node cannot know a new mute until it reconnects. Alerts queued before a mute are not cancelled. Telegram photos are OFF by default; only add `--telegram-photos` with organiser approval and clear notice to participants. No face recognition or biometric storage. The bot's delivery may occasionally duplicate an alert after an ambiguous network failure (e.g. Telegram accepted it but the acknowledgement was lost); event deduplication on peers is independent.

Test a message with `python telegram_alerts.py test --node node_2 --zone B`. Do not run `telegram_alerts.py run` alongside the designated listener.

## Demo

1. Calibrate real event shirts. Run each camera and check the same purple/ordinary participant is accepted at A but flagged at B and C. A mentor/volunteer can enter B but not C; an organiser can enter all three.
2. Confirm every laptop shows Nodes 3/3 and the same event count after sync. Press P to see peers. Receive one Telegram alert for a local violation at B/C; acknowledge it.
3. Press C on node_3 to simulate a peer partition while keeping its phone Wi-Fi stream active. Generate a local event; other nodes should not yet receive it. Press C again; event logs converge. Do not turn off the Wi-Fi that also carries the camera feed, because that would stop inference.
4. Test Telegram mute C from the group; local alerts and event history still work, but new C notifications are suppressed after the mute reaches node_3. Unmute.
5. To show camera failure, stop the phone stream. Preview shows offline and local tracks reset; other nodes continue. Restart stream, and the laptop reconnects.

## Verification and limitations

`python test_access.py` tests role classifications, calibration isolation, hue wrap and mixed-colour rejection, occlusion, permissions, duplicate track events, frame aspect ratio, real local UDP replication/partition recovery, signed packet validation, replicated mutes, database persistence and Telegram queue delivery using a fake API. It does not test live camera images, YOLO inference, actual Telegram delivery, Windows display, or eduroam reachability. The original bot's local queue is persistent, but no central coordination is needed for sending. No code guarantees exact authorisation from shirt colour; staff must verify alerts. No background camera can see someone beyond its field of view. Do not monitor without organiser approval and appropriate notice.
