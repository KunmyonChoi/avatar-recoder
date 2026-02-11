# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time 3D avatar tracking web application that uses MediaPipe for face/body/hand tracking and Three.js with VRM avatars. The avatar mirrors user movements captured via webcam, with support for screen capture and video recording.

## Build Commands

```bash
npm run dev      # Start Vite dev server at http://localhost:5173
npm run build    # Production build to dist/
npm run preview  # Preview production build
```

## Architecture

### Core Technology Stack
- **Three.js** - 3D rendering
- **@pixiv/three-vrm** - VRM avatar loading and animation
- **@mediapipe/tasks-vision** - AI-powered face, pose, and hand tracking
- **Vite** - Build tooling (no config file, uses defaults)

### Application Structure

The entire application logic is in `main.js` (~1700 lines). Key components:

**Initialization Pipeline:**
```
init() → setupScene() → setupWebcam() → setupMediaPipe() → loadAvatar() → animate()
```

**Main Animation Loop (`animate()`):**
1. Face tracking → `applyBlendshapes()` + `applyHeadRotation()`
2. Body tracking (optional) → `applyPose()` using Two-Bone IK solver
3. Hand tracking → `applyHands()` + `applyFingers()`

### Key Functions by Category

**Tracking:**
- `applyBlendshapes()` - Apply facial expressions to VRM
- `applyHeadRotation()` - Apply head rotation from face matrix
- `applyPose()` (line ~1284) - Apply body tracking via Two-Bone IK
- `solveTwoBoneIK()` (line ~1204) - Inverse kinematics solver for arms
- `applyHands()` (line ~1421) - Hand positioning and finger curls
- `mpToVRM()` (line ~1114) - MediaPipe to VRM coordinate conversion

**Screen Capture & Recording:**
- `startScreenCapture()` / `stopScreenCapture()` - Screen sharing
- `startRecording()` / `stopRecording()` / `downloadRecording()` - Video recording with mixed audio

**Audio:**
- `toggleMicrophone()` - Mic enable/disable
- `updateAudioMix()` - Mix microphone and tab audio

### Motion Smoothing

**OneEuroFilter / OneEuroFilter3D classes** - Adaptive low-pass filtering for jitter reduction on pose landmarks.

**Hysteresis logic** for arm tracking - Uses `VIS_THRESHOLD_ON` (0.65) and `VIS_THRESHOLD_OFF` (0.45) to prevent flickering when tracking confidence drops.

### Key Constants

```javascript
const VIDEO_WIDTH = 1280;
const VIDEO_HEIGHT = 720;
const LERP_SPEED = 12;           // Motion response speed
const VIS_THRESHOLD_ON = 0.65;   // Arm activation threshold
const VIS_THRESHOLD_OFF = 0.45;  // Arm deactivation threshold (hysteresis)
```

### Coordinate System

MediaPipe outputs are transformed via `mpToVRM()` which handles:
- Coordinate space conversion from MediaPipe to Three.js/VRM
- Left-right mirroring for natural avatar control

## File Structure

```
main.js          # All application logic
index.html       # Entry point with UI controls
style.css        # Styling
public/avatar.vrm # Default VRM avatar model
```

## UI Features

- **View modes:** Overlay (avatar centered) vs Debug (split-screen with landmarks)
- **Mini avatar:** Draggable 300×400px mode for screen sharing
- **Recording:** WebM video with mixed microphone + tab audio
- **Toggles:** Body tracking, hand tracking, landmark visualization
