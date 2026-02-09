# Prismic Eagle - MediaPipe 3D Avatar

A real-time 3D avatar application that tracks your face, body, and hands using MediaPipe and renders a VRM avatar with Three.js.

## Features

- **Face Tracking**: Real-time facial expression tracking with eye, mouth, and eyebrow movements
- **Body Tracking**: Full upper body tracking with arm and hand positions (optional, toggleable)
- **Hand Tracking**: Finger curl detection for natural hand gestures
- **Screen Capture**: Share your screen as a background behind your avatar
- **Mini Avatar Mode**: Drag and position your avatar anywhere on the screen
- **Video Recording**: Record your avatar with screen background and audio
- **Audio Mixing**: Mix microphone and tab audio with adjustable levels
- **Smooth Motion**: One Euro Filter implementation for jitter-free tracking

## Demo

https://github.com/user-attachments/assets/demo.mp4

## Getting Started

### Prerequisites

- Node.js 18+
- A webcam
- A modern browser with WebGL support (Chrome, Edge, Firefox)

### Installation

```bash
# Clone the repository
git clone https://github.com/KunmyonChoi/avatar-recoder.git
cd avatar-recoder

# Install dependencies
npm install

# Start development server
npm run dev
```

Open http://localhost:5173 in your browser.

### Using Your Own Avatar

Replace `public/avatar.vrm` with your own VRM avatar file. The application supports VRM 0.x and 1.0 formats.

You can create or download VRM avatars from:
- [VRoid Hub](https://hub.vroid.com/)
- [VRoid Studio](https://vroid.com/en/studio)
- [Ready Player Me](https://readyplayer.me/) (export as VRM)

## Usage

### Controls

| Button | Description |
|--------|-------------|
| **Show Landmarks** | Toggle face/body landmark visualization |
| **Switch to Debug** | Toggle between overlay and split-screen view |
| **Enable Body Tracking** | Toggle full body tracking (arms, hands) |
| **Select Screen** | Choose a screen or window to share as background |
| **Stop Screen** | Stop screen sharing |
| **Mini Avatar** | Toggle compact avatar mode (draggable) |
| **Enable Mic** | Enable microphone for recording |
| **Start Recording** | Start recording video with audio |
| **Stop Recording** | Stop recording and download video |

### Audio Mixer

When both microphone and tab audio are active, use the slider to adjust the mix:
- Left: More microphone audio
- Right: More tab audio

The level meters show real-time audio levels for both sources.

## Technical Details

### Dependencies

- **[MediaPipe Tasks Vision](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)** - Face, pose, and hand landmark detection
- **[Three.js](https://threejs.org/)** - 3D rendering
- **[@pixiv/three-vrm](https://github.com/pixiv/three-vrm)** - VRM avatar support
- **[Vite](https://vitejs.dev/)** - Build tool and dev server

### Architecture

```
main.js
├── MediaPipe Initialization
│   ├── FaceLandmarker
│   ├── PoseLandmarker
│   └── HandLandmarker
├── Three.js Scene
│   ├── VRM Loader
│   └── Avatar Animation
├── Screen Capture
│   └── getDisplayMedia API
├── Audio Processing
│   ├── AudioContext
│   ├── GainNode (mixing)
│   └── AnalyserNode (meters)
└── Video Recording
    └── MediaRecorder API
```

### One Euro Filter

The application uses a One Euro Filter implementation to smooth tracking data and reduce jitter while maintaining responsiveness. The filter parameters can be adjusted:

- `minCutoff`: Minimum cutoff frequency (lower = more smoothing)
- `beta`: Speed coefficient (higher = less lag during fast movements)
- `dCutoff`: Derivative cutoff frequency

## Browser Support

| Browser | Face | Body | Hands | Screen Capture | Recording |
|---------|------|------|-------|----------------|-----------|
| Chrome  | Yes  | Yes  | Yes   | Yes            | Yes       |
| Edge    | Yes  | Yes  | Yes   | Yes            | Yes       |
| Firefox | Yes  | Yes  | Yes   | Yes            | Yes       |
| Safari  | Yes  | Yes  | Yes   | Limited*       | Limited*  |

*Safari has limited support for `getDisplayMedia` and some MediaRecorder codecs.

## Building for Production

```bash
npm run build
```

The built files will be in the `dist` directory. Serve with any static file server.

```bash
npm run preview  # Preview production build locally
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) by Google
- [three-vrm](https://github.com/pixiv/three-vrm) by pixiv
- VRM avatar format by [VRM Consortium](https://vrm.dev/)
