# MagicMirror3 — Combined Plan

## Dual Goals

### Goal A: Child-Friendly German Smart Mirror
- Portrait 4K screen, fully German, child-engaging
- 4 themed pages (Home / Fun / Learn / Practical)
- Gesture navigation via webcam (swipe, finger count)
- Face recognition for profile switching
- Custom clothing advice module
- High-contrast, readable at distance

### Goal B: Academic Project (University)
- Run on **Nvidia Jetson Nano** (or Xavier NX)
- **On-device face authentication** (no cloud) — DeepFace / FaceNet
- **On-device voice authentication** — SincNet or VGGVox
- **On-device speech command recognition** — Speech Commands model
- At least **2 external services** (weather + one more)
- Modular architecture for easy service extension
- Privacy: all biometric processing stays on-device

### How They Combine
The child-friendly UI (pages, gestures, German) is the **presentation layer**.
The academic requirements (voice auth, speech commands, Jetson) are the **input/inference layer**.
Same mirror, same architecture, both goals met.

---

## Why Starting Fresh

The existing setup (`~/MagicMirror`) works in isolation but falls apart when camera +
gestures + page navigation are combined. Root causes:

- No defined contract between camera events → gesture events → module reactions
- Layout is not designed for portrait 4K — modules feel cramped
- No real interactivity: gesture wiring exists but nothing responds predictably
- No voice input at all (academic requirement)

---

## Target Hardware

| Option | GPU | RAM | Use Case |
|--------|-----|-----|----------|
| **Jetson Nano** (primary) | 128 CUDA cores | 4GB | Face + voice inference + MagicMirror |
| **Jetson Xavier NX** (upgrade) | 384 CUDA cores | 8GB | Heavier models, faster inference |
| Raspberry Pi 4/5 (fallback) | none | 4-8GB | MagicMirror display only, no heavy inference |

**Decision:** Target Jetson Nano as primary platform. Develop on desktop PC, deploy to Jetson.

### Jetson Considerations
- Use NVIDIA container runtime for GPU access in Docker (if using containers)
- TensorRT for optimizing inference models (face, voice, speech)
- JetPack SDK provides CUDA, cuDNN, TensorRT pre-installed
- Camera: USB webcam or CSI camera module
- Microphone: USB mic or array (e.g. ReSpeaker)
- Display: HDMI to portrait 4K monitor

---

## Architecture

```
Magicmirror3/
├── PLAN.md
├── camera/
│   ├── main.py                 # main loop: capture → detect → classify → send
│   ├── face_recognizer.py      # face detection + recognition (DeepFace/FaceNet)
│   ├── gesture_detector.py     # gesture detection (MediaPipe Hands)
│   ├── voice_authenticator.py  # NEW: voice auth (SincNet/VGGVox)
│   ├── speech_commander.py     # NEW: speech command recognition
│   ├── http_sender.py          # sends events to MMM-CameraBridge via HTTP
│   ├── train.py                # build face encodings from dataset/
│   ├── train_voice.py          # NEW: build voice embeddings from audio samples
│   ├── dataset/                # per-person face image sets
│   ├── voice_dataset/          # NEW: per-person voice samples
│   └── model/                  # saved encodings/models
├── MagicMirror/
│   └── modules/
│       ├── MMM-CameraBridge/   # HTTP receiver → MM notifications
│       └── MMM-ClothingAdvice/ # weather → clothing icons + German text
├── scripts/
│   ├── start.sh
│   ├── stop.sh
│   └── train-faces.sh
└── docs/                       # NEW: academic documentation
```

### Input Pipeline (runs on Jetson GPU)

```
┌──────────────────────────────────────────────────────┐
│                   main.py (sensor loop)               │
│                                                       │
│  Camera frames ──→ face_recognizer.py (DeepFace)     │
│                ──→ gesture_detector.py (MediaPipe)    │
│                                                       │
│  Microphone    ──→ voice_authenticator.py (SincNet)   │
│                ──→ speech_commander.py (Speech Cmds)  │
│                                                       │
│  All results   ──→ http_sender.py ──→ POST /camera-event
└──────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│            MMM-CameraBridge (node_helper.js)          │
│  HTTP server on port 8082                             │
│  Receives events → emits MM notifications             │
└──────────────────────────────────────────────────────┘
```

### Event Contract (camera/voice → HTTP → MagicMirror)

```json
{ "type": "face", "profile": "kind1", "confidence": 0.92 }
{ "type": "face", "profile": "unknown" }
{ "type": "gesture", "name": "swipe_left" }
{ "type": "gesture", "name": "swipe_right" }
{ "type": "gesture", "name": "fingers_1" }
{ "type": "gesture", "name": "fingers_2" }
{ "type": "voice_auth", "profile": "kind1", "confidence": 0.88 }
{ "type": "voice_auth", "profile": "unknown" }
{ "type": "speech_command", "command": "weiter" }
{ "type": "speech_command", "command": "zurück" }
{ "type": "speech_command", "command": "wetter" }
{ "type": "speech_command", "command": "stopp" }
{ "type": "presence", "state": "away" }
{ "type": "presence", "state": "present" }
```

MMM-CameraBridge maps these to MM notifications:

| Event | MM Notification |
|-------|----------------|
| `face: kind1` | `PAGE_SELECT` → page 2 (Fun) |
| `face: eltern` | `PAGE_SELECT` → page 4 (Practical) |
| `gesture: swipe_left` | `PAGE_INCREMENT` |
| `gesture: swipe_right` | `PAGE_DECREMENT` |
| `gesture: fingers_1..4` | `PAGE_SELECT` → page N |
| `voice_auth: kind1` | same as face auth (profile switch) |
| `speech_command: weiter` | `PAGE_INCREMENT` |
| `speech_command: zurück` | `PAGE_DECREMENT` |
| `speech_command: wetter` | `PAGE_SELECT` → page 1 (Home/weather) |
| `presence: away` | screen off (HDMI CEC / backlight) |

Authentication logic: face OR voice can authenticate a user. Both feed into the same
profile-switching mechanism. Confidence threshold configurable.

---

## Pages

### Page 1 — Home (default)
- Giant clock (center top) — **fixed on all pages**
- Current weather + icon (center)
- "What to wear" clothing suggestion (MMM-ClothingAdvice)
- Compliment / greeting (bottom)

### Page 2 — Fun (kids)
- Daily Pokémon (large card, center)
- Fun fact
- Quiz (answerable via gesture or voice)

### Page 3 — Learn
- "On This Day" historical fact (large, center)
- NINA warnings (if active)

### Page 4 — Practical (parents/older kids)
- Calendar
- News feed
- Detailed weather forecast

---

## Module List

### Default modules (no install needed)
| Module | Page |
|--------|------|
| `clock` | all (fixed) |
| `weather` (current) | Home |
| `weather` (forecast) | Practical |
| `calendar` | Home + Practical |
| `compliments` | Home |
| `newsfeed` | Practical |
| `alert` | all |

### Third-party modules
| Module | Page | Notes |
|--------|------|-------|
| `MMM-pages` | all | page grouping via CSS classes |
| `MMM-page-indicator` | all | visible dots |
| `MMM-DailyPokemon` | Fun | `language: "de"` |
| `MMM-OnThisDay` | Learn | German historical facts |
| `MMM-Quiz` | Fun/Learn | gesture + voice answers |
| `MMM-Facts` | Fun | random fun facts |
| `MMM-NINA` | Learn | German emergency alerts |
| `MMM-Jast` | Practical | stock ticker — from seniors' project (Yahoo Finance) |
| `MMM-iFrame` | Practical | Mensa HS Landshut menu PDF — from seniors' project |
| `MMM-Remote-Control` | admin | browser management |

### Custom modules
| Module | Purpose |
|--------|---------|
| `MMM-CameraBridge` | HTTP receiver for camera+voice events → MM notifications |
| `MMM-ClothingAdvice` | Weather → clothing icons (OpenMoji SVGs) + German text |

---

## External Services (Academic Requirement: min. 2)

1. **OpenWeatherMap** — weather data (API key: stored in config)
2. **Yahoo Finance** — stock prices via MMM-Jast (from seniors' project, satisfies "Aktienkurse")
3. **NINA (BBK)** — German federal warning system (free, no key)
4. **Mensa HS Landshut** — canteen menu via MMM-iFrame (from seniors' project)
5. **PokeAPI** — Pokémon data (free, no key, via MMM-DailyPokemon)

---

## On-Device Models (Academic Requirement)

| Task | Model | Framework | Notes |
|------|-------|-----------|-------|
| Face detection | MTCNN or RetinaFace | TensorFlow/PyTorch | lightweight, runs on Jetson GPU |
| Face recognition | FaceNet or ArcFace | TensorFlow/PyTorch | via DeepFace library |
| Voice authentication | SincNet or VGGVox | PyTorch | speaker verification (is this kind1?) |
| Speech commands | Speech Commands v2 | TensorFlow/PyTorch | ~35 short commands, train German subset |
| Gesture detection | MediaPipe Hands | MediaPipe | already implemented |

All models run on-device. No cloud calls for any biometric data.

### TensorRT Optimization (Jetson)
- Export PyTorch models → ONNX → TensorRT for 2-5x speedup on Jetson
- Target: face+gesture at 15fps, voice auth < 2s, speech command < 500ms

---

## Build Order

### Phase 1: Get MagicMirror Working (current)
1. ✅ Module installation (MMM-pages, MMM-DailyPokemon, etc.)
2. ✅ MMM-CameraBridge + MMM-ClothingAdvice custom modules created
3. ✅ Camera pipeline (face + gesture + HTTP sender)
4. ⬜ Write merged config.js using working base settings
5. ⬜ Write safe custom.css for portrait layout
6. ⬜ Fix face_recognition Python dependency
7. ⬜ Test incrementally (one module at a time)

### Phase 2: Voice Input (academic)
8. ⬜ Add USB microphone to hardware setup
9. ⬜ Implement `voice_authenticator.py` (SincNet/VGGVox)
10. ⬜ Implement `speech_commander.py` (Speech Commands)
11. ⬜ Extend MMM-CameraBridge to handle voice_auth + speech_command events
12. ⬜ Train voice profiles (collect samples per user)
13. ⬜ Train German speech commands (weiter, zurück, wetter, stopp, etc.)

### Phase 3: Jetson Deployment
14. ⬜ Set up Jetson Nano with JetPack SDK
15. ⬜ Install MagicMirror on Jetson
16. ⬜ Optimize models with TensorRT
17. ⬜ Performance tuning (FPS, latency, memory)
18. ⬜ Hardware assembly (mirror + screen + Jetson + camera + mic)

### Phase 4: Polish & Documentation
19. ⬜ German speech command fine-tuning
20. ⬜ Multi-user testing (kids vs adults)
21. ⬜ Academic documentation (architecture, evaluation, results)

---

## Open Questions

- [ ] Which Jetson model available? (Nano 2GB/4GB or Xavier NX)
- [ ] USB microphone model? (ReSpeaker recommended for array)
- [ ] Calendar URL (iCal)?
- [ ] City/coords for weather? (currently Landshut: 48.5442, 12.1469)
- [ ] Face profile names? (kind1, kind2, mama, papa?)
- [ ] Camera device path? (`/dev/video0` assumed)
- [ ] German speech commands list? (weiter, zurück, wetter, stopp, hilfe, quiz, ...)
- [ ] Voice enrollment: how many samples per person? (recommend 10-20 utterances)
