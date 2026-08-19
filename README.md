# ARGUS Vision

A futuristic, fully on-device computer-vision assistant for the Samsung Galaxy S24 —
real-time object detection with a JARVIS-inspired (but original) augmented-reality HUD,
voice queries, object tracking, and an adaptive performance system.

```
Camera → frame preprocessing → object detector (LiteRT/GPU) → IoU tracker → HUD renderer
                                                     ↘ structured results → voice Q&A
```

## Features

- **On-device detection** — EfficientDet-Lite (COCO 80 classes) running through
  MediaPipe Tasks / LiteRT with the **GPU delegate** (Xclipse 940 on the S24),
  automatic CPU/XNNPACK fallback. Works in airplane mode.
- **Futuristic HUD** — thin cyan boxes, animated corner brackets, labels + confidence,
  track IDs, scanning line, rotating reticle, screen-edge frame, "VISION ONLINE" status.
- **Object tracking** — IoU tracker with stable IDs, box smoothing at inference rate,
  plus display-rate interpolation so the HUD glides at 60+ Hz while the camera moves.
- **Voice assistant (offline)** — tap the mic: "what do you see", "count the bottles",
  "find my phone", "what is that", "what color is the car". Answers are computed from
  the detector's structured results (+ a local dominant-color sampler) and spoken via TTS.
- **Optional cloud reasoning** — OFF by default. When explicitly enabled with your own
  OpenAI-compatible endpoint/key, a single frame is sent only for questions local logic
  can't answer. Detection itself never touches the network.
- **Adaptive performance** — inference paced by a busy-flag (never queues frames),
  thermal-status-aware throttling (`PowerManager`), FAST / BALANCED / ACCURATE modes.
- **Developer panel** — camera FPS, inference FPS + latency, model + delegate,
  CPU %, memory, thermal state, tracked-object count.
- **Model manager** — swap between built-in models; drop custom `.tflite` detection
  models (with metadata) into the app's `files/models/` directory to use them.

## Getting the APK

Every push builds a debug APK in GitHub Actions and attaches it to the
**`argus-latest` release** of this repo — download `argus-vision.apk` on your phone,
allow "install unknown apps", and install. (Or grab the artifact from the Actions run.)

## Building locally

Requirements: JDK 17, Android SDK 34, Gradle 8.9 (or Android Studio).

```bash
bash scripts/download_models.sh   # fetches EfficientDet-Lite0/2 into app assets
gradle assembleDebug              # or ./gradlew via Android Studio
```

## Architecture

| Module | Path | Role |
|---|---|---|
| Camera | `camera/CameraController.kt` | CameraX preview + 640×480 analysis stream, KEEP_ONLY_LATEST backpressure |
| Vision Engine | `vision/VisionEngine.kt` | Orchestration, pacing, thermal adaptation, stats |
| Object Detector | `vision/detector/` | `FrameDetector` interface + MediaPipe/LiteRT implementation (swappable runtime) |
| Object Tracker | `vision/tracker/IouTracker.kt` | Stable IDs, smoothing, flicker suppression |
| Model Manager | `vision/model/ModelManager.kt` | Built-in + custom model registry |
| HUD Renderer | `hud/HudOverlay.kt` | Single-pass Compose canvas, cached native paints |
| Voice Interface | `voice/` | On-device SpeechRecognizer + TTS + local query engine |
| Local/Cloud AI | `ai/CloudReasoner.kt` | Optional reasoning hook (same seam a local VLM would use) |
| Settings | `settings/SettingsStore.kt` | Persisted preferences as a StateFlow |
| Performance | `perf/PerformanceMonitor.kt` | CPU/memory sampling for the dev panel |

### Threading model

- UI thread: Compose only. Never blocked by inference.
- CameraX analyzer thread: YUV→bitmap conversion + rotation, frame submission.
- MediaPipe LIVE_STREAM thread: inference results → tracker → StateFlow.

### Future local VLM

The reasoning seam (`ai/`) is where a small on-device multimodal model plugs in.
Realistic option for the S24: **Gemma 3n E2B via MediaPipe LLM Inference**
(~3.1 GB storage, ~3–4 GB RAM while active, seconds-per-answer). It was intentionally
left out of v1 to keep the app lean; integrating it changes nothing above this seam.

## Privacy

- Camera frames stay on-device; there is no upload path in the detection pipeline.
- Microphone activates only while the mic button session is running.
- The only network code is the opt-in cloud reasoning call, disabled by default.
