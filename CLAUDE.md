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

The entire application logic is in `main.js` (~4700 lines; line numbers rot quickly — locate functions by name). Key components:

**Initialization Pipeline:**
```
init() → setupScene() → setupWebcam() → setupMediaPipe() → loadAvatar() → animate()
```

**Main Animation Loop (`animate()`):**
1. Face tracking → `applyBlendshapes()` + `applyHeadRotation()`
2. Hand detection (results stored, applied after pose)
3. Body tracking (optional) → `applyPose()` using Two-Bone IK solver
4. Hand application → `applyHands()` (wrist orientation + finger curls), `relaxHand()` for undetected hands

Detection is throttled to ~150ms intervals while a pen stroke is active (drawing responsiveness).

### Key Functions by Category

**Tracking:**
- `applyBlendshapes()` - Apply facial expressions to VRM
- `applyHeadRotation()` - Apply head rotation from face matrix
- `applyPose()` - Body tracking: arm IK, pelvis rotation, body sway/bob, spine/chest lean, leg IK
- `solveTwoBoneIK()` - Generic two-bone IK solver (arms and legs; per-chain plane-normal state in `ikPlaneState`)
- `solvePinnedFootLeg()` - Legs solve to feet pinned at their rest ground position (VTuber-style)
- `applyHands()` / `applyHandOrientation()` / `applyFingers()` - Wrist orientation from hand landmarks + distance-based finger curls
- `relaxHand()` - Ease hand/fingers back to rest when a hand is not detected
- `mpToVRM()` - MediaPipe world-landmark to VRM coordinate conversion (mirroring)
- `handLmToVRM()` - Hand image-landmark to VRM direction conversion (aspect-corrected)

**Body model modes (`applyPose`):**
- Fallback (hips not visible): whole-body sway from mid-shoulder position vs slow-adapting baseline
- Dance mode (hips visible, hysteresis `DANCE_VIS_ON/OFF`): hips driven by mid-hip position, spine expresses hip→shoulder lean, chest takes shoulder-line rotation — enables S-curve dancing in place

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
const VIDEO_WIDTH = 1280;         // 640 on mobile
const VIDEO_HEIGHT = 720;         // 480 on mobile
const VIDEO_ASPECT = VIDEO_WIDTH / VIDEO_HEIGHT;
const LERP_SPEED = 12;            // Motion response speed
const VIS_THRESHOLD_ON = 0.65;    // Arm activation threshold
const VIS_THRESHOLD_OFF = 0.45;   // Arm deactivation threshold (hysteresis)
const DANCE_VIS_ON = 0.6;         // Hip visibility to enter dance mode
const DANCE_VIS_OFF = 0.4;        // Hip visibility to leave dance mode
const BASELINE_ADAPT_SPEED = 0.1; // Sway/lean baseline adaptation (τ≈10s)
```

Pose model defaults to `pose_landmarker_lite` (full is too slow for realtime on desktop); override with `?poseModel=full|heavy` URL param for experiments.

### VRM version handling

The default `public/avatar.vrm` is VRM 1.0. `VRMUtils.rotateVRM0()` rotates VRM 0.x scenes 180°Y so both versions face the camera (+Z). The three-vrm normalized rig is built in the load-time frame (before that rotation), which is why `isVRM0` sign flips appear in bone-axis/rotation code — they are correct, not hacks. Thumb bones follow VRM1 naming: Metacarpal→Proximal→Distal (there is no ThumbIntermediate).

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
