# MagicMirror3 — Fresh Start Plan

## Why Starting Fresh

The existing setup (`~/MagicMirror`) works in isolation but falls apart when camera +
gestures + page navigation are combined. Root causes:

- `MMM-Face-Recognition-SMAI` is archived/unreliable
- `MMM-CameraBridge` is custom and has no clear error feedback
- No defined contract between camera events → gesture events → module reactions
- Layout is not designed for portrait 4K — modules feel cramped
- No real interactivity: gesture wiring exists but nothing responds predictably

---

## Target Experience

- Portrait 4K screen, child-first, fully German
- 4 distinct pages navigable by gesture (swipe) or touch
- Camera recognizes who is standing in front → loads the right page automatically
- Everything readable at distance, large centered cards, high contrast
- Works fully without camera (graceful degradation)

---

## Architecture

```
Magicmirror3/
├── docker-compose.yml          # orchestrates 3 services
├── .env                        # all secrets/settings in one place
├── Dockerfile                  # MagicMirror image
├── config/
│   ├── config.js               # MM config: German, portrait, 4 pages
│   └── custom.css              # portrait 4K layout styles
├── modules/                    # only the modules listed below
├── camera/
│   ├── Dockerfile              # camera service image
│   ├── main.py                 # main loop: capture → detect → classify → publish
│   ├── face_recognizer.py      # face detection + recognition (face_recognition lib)
│   ├── gesture_detector.py     # gesture detection (MediaPipe Hands)
│   ├── redis_publisher.py      # publishes structured events to Redis
│   └── dataset/                # per-person face image sets
├── modules/MMM-CameraBridge/   # rewritten: subscribes Redis → emits MM notifications
└── scripts/
    ├── docker-entrypoint.sh
    └── train-faces.sh          # helper: rebuild face encodings from dataset/
```

### Services

| Service | Role |
|---------|------|
| `magicmirror` | Runs MagicMirror² + serves browser UI |
| `camera` | Python: captures frames, runs face+gesture detection, publishes to Redis |
| `redis` | Message bus between camera and MagicMirror |

### Event Contract (camera → Redis → MagicMirror)

Camera publishes JSON to a single Redis channel (e.g. `mirror.events`):

```json
{ "type": "face", "profile": "kind1", "confidence": 0.92 }
{ "type": "face", "profile": "unknown" }
{ "type": "gesture", "name": "swipe_left" }
{ "type": "gesture", "name": "swipe_right" }
{ "type": "gesture", "name": "swipe_up" }
{ "type": "presence", "state": "away" }
{ "type": "presence", "state": "present" }
```

MMM-CameraBridge subscribes and maps these to MM notifications:

| Redis event | MM notification emitted |
|-------------|------------------------|
| `face: kind1` | `PAGE_SELECT` → page 1 (Fun) |
| `face: eltern` | `PAGE_SELECT` → page 3 (Practical) |
| `face: unknown` | stay on current page |
| `gesture: swipe_left` | `PAGE_CHANGED` +1 |
| `gesture: swipe_right` | `PAGE_CHANGED` -1 |
| `presence: away` | turn off display (HDMI off) |
| `presence: present` | turn on display |

Each step is independently testable:
```bash
redis-cli publish mirror.events '{"type":"gesture","name":"swipe_left"}'
```

---

## Pages

### Page 1 — Home (default)
- Giant clock (center top)
- Current weather + icon (center)
- "What to wear" clothing suggestion
- Today's calendar events (compact list)
- Compliment / greeting (bottom)
- Background: soft nature/seasonal slideshow

### Page 2 — Fun (kids)
- Daily Pokémon (large card, center)
- Useless/fun fact (bottom card)
- Background: colorful themed imagery

### Page 3 — Learn
- "On This Day" historical fact (large, center)
- Quiz (interactive, answerable via gesture or touch)
- Background: space/science imagery

### Page 4 — Practical (parents/older kids)
- Full calendar week view
- News feed (scrollable)
- DWD weather warnings (if active)
- Background: neutral/minimal

---

## Module List

### Keep from existing setup
| Module | Page | Notes |
|--------|------|-------|
| `MMM-pages` | all | page grouping |
| `MMM-page-indicator` | all | visible dots + page switching |
| `MMM-DailyPokemon` | Fun | set `language: "de"` |
| `MMM-OnThisDay` | Learn | inherits global `de` language |
| `MMM-Quiz` | Learn | wire QUIZ_ANSWER to gesture notifications |
| `MMM-BackgroundSlideshow` | all | per-page themed backgrounds |
| `MMM-WeatherDependentClothes` | Home | needs weather notification from default module |
| `MMM-NINA` | Home | German emergency/warning alerts |

### Default modules (no install needed)
| Module | Page |
|--------|------|
| `clock` | Home |
| `weather` (current) | Home |
| `weather` (forecast) | Home |
| `calendar` | Home + Practical |
| `compliments` | Home |
| `newsfeed` | Practical |
| `alert` | all |

### New modules to add
| Module | Page | Why replacing |
|--------|------|--------------|
| `MMM-Face-Reco-DNN` | — | replaces `MMM-Face-Recognition-SMAI` (maintained, reliable) |
| `MMM-Touch` | all | replaces `MMM-Keypress` (proper multi-touch + gesture notifications) |
| `MMM-UselessFacts` | Fun | replaces nothing, adds fun German facts |
| `MMM-CalendarExt3` | Practical | richer week/month view vs default list |
| `MMM-Remote-Control` | admin | browser UI for managing mirror without SSH |
| `MMM-ModuleScheduler` | — | time-based: hide kid pages at night, show news after 18:00 |

### Drop entirely
| Module | Reason |
|--------|--------|
| `MMM-Face-Recognition-SMAI` | archived, unreliable, poor performance |
| `MMM-voice` | fragile, replaced by gesture |
| `MMM-CameraBridge` (old) | rewrite cleanly from scratch |
| `MMM-DynamicWeather` | redundant with default weather module |
| `MMM-google-route` | requires complex Google API setup |
| `MMM-GooglePhotos` | requires complex Google API setup |
| `MMM-GoogleCalendar` | replace with default calendar (simpler, no OAuth) |
| `MMM-Keypress` | replaced by MMM-Touch |
| `MMM-Formula1` | out of scope |
| `MMM-soccer` | out of scope |
| `MMM-MinecraftServer` | out of scope |
| `MMM-MyScoreboard` | out of scope |
| `MMM-OnSpotify` | out of scope for now |

---

## Layout (Portrait 4K CSS approach)

Portrait 4K = 2160×3840px rendered at browser zoom.
Strategy: use `fullscreen_above` for backgrounds, `middle_center` for main cards,
`top_center` for clock, `lower_third` for secondary info.

Key CSS rules:
- Base font size: 28px minimum (readable at 2m distance)
- Cards: `max-width: 900px`, centered, rounded corners, semi-transparent dark bg
- Clock: 180px+ font size on Home page
- Page indicator: bottom-center, large dots
- No side columns — everything centered in portrait mode

---

## Build Order

1. `docker-compose.yml` + `.env` + `Dockerfile`
2. `config/config.js` (standard default profile)
3. `config/config.child.js` + `css/custom.css` (child-focused portrait profile)
4. `modules/MMM-CameraBridge/` (rewrite: Redis subscriber → MM notifications)
5. `camera/` (clean Python pipeline: face + gesture + Redis publisher)
6. Module installs (git clone into `modules/`)
7. `scripts/train-faces.sh` (helper for adding face profiles)
8. End-to-end test: simulate gestures via `redis-cli`, verify page switching

---

## Open Questions / Decisions to Make Before Building

- [ ] Which calendar URL to use? (iCal link for family calendar)
- [ ] Which city/coordinates for weather? (currently Landshut: 48.5442, 12.1469)
- [ ] Face profiles: what names? (e.g. kind1, kind2, mama, papa)
- [ ] Which camera device? (`/dev/video0` assumed)
- [ ] Touch screen available? (determines if MMM-Touch is primary input)
- [ ] Should quiz answers work via gesture, touch, or both?
