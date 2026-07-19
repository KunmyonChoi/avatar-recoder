import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { VRMLoaderPlugin, VRMUtils, VRMExpressionPresetName } from '@pixiv/three-vrm';
import { FilesetResolver, FaceLandmarker, PoseLandmarker, HandLandmarker, DrawingUtils } from '@mediapipe/tasks-vision';

// --- Mobile Detection ---
const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
const isAndroid = /Android/i.test(navigator.userAgent);

// --- Configuration (모바일에서는 저해상도) ---
const VIDEO_WIDTH = isMobile ? 640 : 1280;
const VIDEO_HEIGHT = isMobile ? 480 : 720;
// 이미지 정규화 좌표의 x/y 스케일 차이 보정용 (x는 너비, y는 높이 기준 정규화)
const VIDEO_ASPECT = VIDEO_WIDTH / VIDEO_HEIGHT;

// --- Improved Configuration ---
const LERP_SPEED = 12; // 반응 속도 (높을수록 빠름)
const VIS_THRESHOLD_ON = 0.65;  // 활성화 임계값
const VIS_THRESHOLD_OFF = 0.45; // 비활성화 임계값 (hysteresis)
const DANCE_SWAY_GAIN = 1.4;    // 댄스 모드 골반 좌우 이동 증폭 (표현력)
const DANCE_ROLL_COUPLE = 0.8;  // 골반 이동→기울기 커플링 (rad/m) — 무게이동 표현
const DANCE_VIS_ON = 0.6;       // 골반 visibility 댄스 모드 진입 임계값
const DANCE_VIS_OFF = 0.4;      // 댄스 모드 해제 임계값 (hysteresis)
const BASELINE_ADAPT_SPEED = 0.1; // sway/lean baseline 적응 속도 (τ≈10s)
const DIST_CONF_MIN = 0.35;       // 원거리 최소 신뢰도 (이 값까지 스무딩 강화)
const RECORDING_TARGET_FPS = 30;
const RECORDING_FRAME_INTERVAL = 1000 / RECORDING_TARGET_FPS;
const RECORDING_BODY_FPS = 18;
const RECORDING_BODY_INTERVAL = 1000 / RECORDING_BODY_FPS;
const RECORDING_SCREEN_MAX_LONG_EDGE = 1920;
const RECORDING_SCREEN_MAX_SHORT_EDGE = 1080;
const RECORDING_AVATAR_WIDTH = 1280;
const RECORDING_AVATAR_HEIGHT = 720;
const RECORDING_STATS_INTERVAL = 5000;

// 거리 기반 신뢰도: 1 = 표준 거리(가까움), 작을수록 멀어서 랜드마크 노이즈가 커짐
// → pose 필터·표정·머리 회전의 스무딩을 비례 강화해 원거리 튐을 완화
let trackingDistConf = 1;
function updateDistConf(sizeRatio, deltaTime) {
    const target = THREE.MathUtils.clamp(sizeRatio, DIST_CONF_MIN, 1);
    trackingDistConf += (target - trackingDistConf) * getLerpFactor(deltaTime, 2);
}
const AVATAR_HEAD_HEIGHT = 1.39;  // 아바타 머리 기준 높이(m) — 모델 신장 차이를 흡수해 얼굴 위치 통일

// --- One Euro Filter (떨림 완화) ---
class OneEuroFilter {
    constructor(minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
        this.minCutoff = minCutoff;  // 최소 cutoff 주파수 (낮을수록 부드러움)
        this.beta = beta;            // 속도에 따른 cutoff 증가율 (높을수록 빠른 움직임에 민감)
        this.dCutoff = dCutoff;      // 미분값의 cutoff
        this.xPrev = null;
        this.dxPrev = null;
        this.tPrev = null;
    }

    smoothingFactor(te, cutoff) {
        const r = 2 * Math.PI * cutoff * te;
        return r / (r + 1);
    }

    filter(x, t) {
        if (this.tPrev === null) {
            this.xPrev = x;
            this.dxPrev = 0;
            this.tPrev = t;
            return x;
        }

        const te = t - this.tPrev;
        if (te <= 0) return this.xPrev;

        // 미분값 (속도) 계산
        const dx = (x - this.xPrev) / te;
        const alphaDx = this.smoothingFactor(te, this.dCutoff);
        const dxFiltered = alphaDx * dx + (1 - alphaDx) * this.dxPrev;

        // 적응형 cutoff 계산
        const cutoff = this.minCutoff + this.beta * Math.abs(dxFiltered);
        const alpha = this.smoothingFactor(te, cutoff);

        // 필터링된 값
        const xFiltered = alpha * x + (1 - alpha) * this.xPrev;

        this.xPrev = xFiltered;
        this.dxPrev = dxFiltered;
        this.tPrev = t;

        return xFiltered;
    }

    reset() {
        this.xPrev = null;
        this.dxPrev = null;
        this.tPrev = null;
    }
}

// 3D 좌표용 One Euro Filter
class OneEuroFilter3D {
    constructor(minCutoff = 1.0, beta = 0.007) {
        this.xFilter = new OneEuroFilter(minCutoff, beta);
        this.yFilter = new OneEuroFilter(minCutoff, beta);
        this.zFilter = new OneEuroFilter(minCutoff, beta);
    }

    filter(point, t) {
        return {
            x: this.xFilter.filter(point.x, t),
            y: this.yFilter.filter(point.y, t),
            z: this.zFilter.filter(point.z, t),
            visibility: point.visibility
        };
    }

    reset() {
        this.xFilter.reset();
        this.yFilter.reset();
        this.zFilter.reset();
    }
}

// Pose landmarks 필터 (33개 랜드마크)
let poseFilters = null;
function getFilteredPoseLandmarks(landmarks, worldLandmarks, timestamp) {
    if (!poseFilters) {
        // 필터 초기화 (minCutoff 낮을수록 부드러움, beta 높을수록 빠른 움직임에 반응)
        poseFilters = {
            landmarks: Array.from({ length: 33 }, () => new OneEuroFilter3D(1.5, 0.01)),
            worldLandmarks: Array.from({ length: 33 }, () => new OneEuroFilter3D(1.5, 0.01))
        };
    }

    // 거리 신뢰도에 따라 필터 강도 조정: 멀수록 cutoff를 낮춰 강한 스무딩 (원거리 튐 완화)
    const minCutoff = 1.5 * trackingDistConf;
    const beta = 0.01 * trackingDistConf;
    for (const group of [poseFilters.landmarks, poseFilters.worldLandmarks]) {
        for (const f3 of group) {
            f3.xFilter.minCutoff = f3.yFilter.minCutoff = f3.zFilter.minCutoff = minCutoff;
            f3.xFilter.beta = f3.yFilter.beta = f3.zFilter.beta = beta;
        }
    }

    const t = timestamp / 1000;  // 초 단위로 변환

    const filteredLandmarks = landmarks.map((lm, i) => poseFilters.landmarks[i].filter(lm, t));
    const filteredWorldLandmarks = worldLandmarks
        ? worldLandmarks.map((lm, i) => poseFilters.worldLandmarks[i].filter(lm, t))
        : null;

    return { filteredLandmarks, filteredWorldLandmarks };
}

// --- Globals ---
let scene, camera, renderer;
let orbitControls = null;
let isOrbitEnabled = false;
let video;
let faceLandmarker, poseLandmarker, handLandmarker;
let currentVrm;
let currentAvatarUrl = './avatar.vrm';
let isAvatarLoading = false;
let lastVideoTime = -1;
let lastFrameTime = performance.now();
let lastDetectionTime = 0;           // 드로잉 중 인식 스로틀링 기준 시각
let lastFaceDetectionTime = 0;       // 녹화 중 face detect 페이싱 기준
let lastBodyDetectionTime = 0;       // 녹화 중 pose/hand detect 페이싱 기준
let blendShapes = [];
let rotation = new THREE.Euler();
let currentGesture = 'neutral';
let gestureTimer = 0;
let debugCanvas, debugCtx, drawingUtils;
let DEBUG_MODE = false;              // 기본: Hide landmarks
let isDebugView = false;             // 기본: Default mode
let BODY_TRACKING_ENABLED = false;   // 기본: Face tracking만
let isCameraEnabled = true;          // 카메라 상태 (기본: ON)
let webcamStream = null;             // 웹캠 스트림

// --- Screen Capture & Recording ---
let screenStream = null;             // 화면 공유 스트림
let screenVideo = null;              // 화면 공유 비디오 엘리먼트
let mediaRecorder = null;            // 녹화기
let recordedChunks = [];             // 녹화 데이터
let isMiniAvatar = false;            // 미니 아바타 모드
let miniAvatarPosition = { x: null, y: null };  // 미니 아바타 위치
let isAvatarVisible = true;          // 아바타 표시 여부
let recordingStats = null;           // 녹화 성능 계측 집계

// --- Screen Zoom ---
let screenZoom = 1;
let screenZoomTx = 0;
let screenZoomTy = 0;
let isScreenPanning = false;
let screenPanStartX = 0;
let screenPanStartY = 0;
let screenZoomTxStart = 0;
let screenZoomTyStart = 0;
let zoomIndicatorTimer = null;

// --- Captured Screen Resolution ---
let capturedScreenWidth = 0;     // 캡쳐된 화면 실제 너비
let capturedScreenHeight = 0;    // 캡쳐된 화면 실제 높이
let prevRendererSize = null;     // 세로 모드 적응 전 렌더러 크기 저장

// --- Cross-Origin Integration (postMessage) ---
const _integrationParams = (() => {
    const p = new URLSearchParams(window.location.search);
    const mode = p.get('mode');
    const rawOrigin = p.get('origin');
    if (mode !== 'popup' || !rawOrigin) return null;
    try {
        const url = new URL(rawOrigin);
        return {
            origin: url.origin,
            sessionId: p.get('session') || null,
            autoRecord: p.get('autoRecord') === '1',
        };
    } catch {
        return null;
    }
})();
const isIntegrationMode = !!_integrationParams;

// --- Stable Window Dimensions (debounced) ---
// window.innerWidth/Height를 매 프레임 직접 읽으면 Dock 등 시스템 UI 변화에 즉시 반응해
// compositeFrame / 대화 렌더링에서 출렁임이 발생한다. 150ms debounce로 안정화.
let stableWindowWidth = window.innerWidth;
let stableWindowHeight = window.innerHeight;
let windowResizeTimer = null;
window.addEventListener('resize', () => {
    clearTimeout(windowResizeTimer);
    windowResizeTimer = setTimeout(() => {
        stableWindowWidth = window.innerWidth;
        stableWindowHeight = window.innerHeight;
    }, 150);
});

function isRecordingActive() {
    return !!(mediaRecorder && mediaRecorder.state === 'recording');
}

function shouldPauseAvatarWorkForRecording() {
    return isRecordingActive() && !isAvatarVisible && !DEBUG_MODE;
}

function getRecordingCanvasSize(sourceWidth, sourceHeight) {
    const safeSourceWidth = sourceWidth > 0 ? sourceWidth : RECORDING_AVATAR_WIDTH;
    const safeSourceHeight = sourceHeight > 0 ? sourceHeight : RECORDING_AVATAR_HEIGHT;

    if (sourceWidth <= 0 || sourceHeight <= 0) {
        return { width: RECORDING_AVATAR_WIDTH, height: RECORDING_AVATAR_HEIGHT, scale: 1 };
    }

    const isLandscape = safeSourceWidth >= safeSourceHeight;
    const maxWidth = isLandscape ? RECORDING_SCREEN_MAX_LONG_EDGE : RECORDING_SCREEN_MAX_SHORT_EDGE;
    const maxHeight = isLandscape ? RECORDING_SCREEN_MAX_SHORT_EDGE : RECORDING_SCREEN_MAX_LONG_EDGE;
    const scale = Math.min(1, maxWidth / safeSourceWidth, maxHeight / safeSourceHeight);

    return {
        width: Math.max(2, Math.round(safeSourceWidth * scale)),
        height: Math.max(2, Math.round(safeSourceHeight * scale)),
        scale,
    };
}

function getRecordingVideoBitsPerSecond(width, height) {
    const pixels = width * height;
    if (pixels <= 1280 * 720) return 4_000_000;
    if (pixels <= 1920 * 1080) return 8_000_000;
    return 12_000_000;
}

function startRecordingStats(width, height, sourceWidth, sourceHeight) {
    const screenTrack = screenStream?.getVideoTracks?.()[0];
    const trackSettings = screenTrack?.getSettings ? screenTrack.getSettings() : null;
    recordingStats = {
        startedAt: performance.now(),
        lastReportAt: performance.now(),
        width,
        height,
        sourceWidth,
        sourceHeight,
        trackSettings,
        animateFrames: 0,
        compositeFrames: 0,
        compositeSkipped: 0,
        longAnimateFrames: 0,
        longCompositeFrames: 0,
        recorderChunks: 0,
        recorderBytes: 0,
        recorderMaxGap: 0,
        lastRecorderChunkAt: 0,
        detect: {
            face: { count: 0, total: 0, max: 0 },
            hand: { count: 0, total: 0, max: 0 },
            pose: { count: 0, total: 0, max: 0 },
        },
    };
    console.info('[RecordingStats] started', {
        output: `${width}x${height}`,
        source: `${sourceWidth}x${sourceHeight}`,
        trackSettings,
    });
}

function stopRecordingStats() {
    if (!recordingStats) return;
    reportRecordingStats(true);
    recordingStats = null;
}

function recordDetectDuration(kind, duration) {
    if (!recordingStats) return;
    const bucket = recordingStats.detect[kind];
    if (!bucket) return;
    bucket.count += 1;
    bucket.total += duration;
    bucket.max = Math.max(bucket.max, duration);
}

function recordRecorderChunk(size) {
    if (!recordingStats) return;
    const now = performance.now();
    if (recordingStats.lastRecorderChunkAt) {
        recordingStats.recorderMaxGap = Math.max(recordingStats.recorderMaxGap, now - recordingStats.lastRecorderChunkAt);
    }
    recordingStats.lastRecorderChunkAt = now;
    recordingStats.recorderChunks += 1;
    recordingStats.recorderBytes += size;
}

function reportRecordingStats(force = false) {
    if (!recordingStats) return;
    const now = performance.now();
    if (!force && now - recordingStats.lastReportAt < RECORDING_STATS_INTERVAL) return;

    const elapsed = Math.max(0.001, (now - recordingStats.startedAt) / 1000);
    const avgDetect = (kind) => {
        const bucket = recordingStats.detect[kind];
        return bucket.count ? +(bucket.total / bucket.count).toFixed(1) : 0;
    };

    console.info('[RecordingStats]', {
        seconds: +elapsed.toFixed(1),
        output: `${recordingStats.width}x${recordingStats.height}`,
        source: `${recordingStats.sourceWidth}x${recordingStats.sourceHeight}`,
        animateFps: +(recordingStats.animateFrames / elapsed).toFixed(1),
        compositeFps: +(recordingStats.compositeFrames / elapsed).toFixed(1),
        compositeSkipped: recordingStats.compositeSkipped,
        longFrames: {
            animate: recordingStats.longAnimateFrames,
            composite: recordingStats.longCompositeFrames,
        },
        detectMsAvg: {
            face: avgDetect('face'),
            hand: avgDetect('hand'),
            pose: avgDetect('pose'),
        },
        detectMsMax: {
            face: +recordingStats.detect.face.max.toFixed(1),
            hand: +recordingStats.detect.hand.max.toFixed(1),
            pose: +recordingStats.detect.pose.max.toFixed(1),
        },
        recorder: {
            chunks: recordingStats.recorderChunks,
            mb: +(recordingStats.recorderBytes / 1024 / 1024).toFixed(1),
            maxGapMs: +recordingStats.recorderMaxGap.toFixed(1),
        },
        trackSettings: recordingStats.trackSettings,
    });
    recordingStats.lastReportAt = now;
}

// --- Audio ---
let micStream = null;                // 마이크 스트림
let isMicEnabled = false;            // 마이크 활성화 상태
let audioContext = null;             // 오디오 합성용
let audioDestination = null;         // 합성된 오디오 출력

// --- Audio Level Meters ---
let meterAudioContext = null;        // 레벨 미터용 AudioContext
let micAnalyser = null;              // 마이크 분석기
let tabAnalyser = null;              // 탭 오디오 분석기
let meterAnimationId = null;         // 애니메이션 ID

// --- Audio Mix ---
let micGainNode = null;              // 마이크 볼륨 조절
let tabGainNode = null;              // 탭 오디오 볼륨 조절
let audioMixValue = 50;              // 0 = Mic only, 100 = Tab only

// --- Arm Activity State (for hysteresis) ---
let leftArmActive = false;
let rightArmActive = false;

// --- Body Sway Baselines (이미지 단위, 느린 적응으로 순간적 움직임만 반영) ---
let swayBaseline = null;     // 어깨 중점 (폴백 모드: 골반이 프레임 밖)
let hipSwayBaseline = null;  // 골반 중점 (댄스 모드: hips를 직접 구동)
let leanBaseline = null;     // 골반→어깨 기울기 각 (댄스 모드: 상체 lean 중립값)
let lastSwayOffX = 0;        // 직전 프레임 골반 좌우 이동량(m) — roll 커플링용

// --- 발 사이드 스텝: 골반이 발에서 지속적으로 멀어지면 발이 한 발씩 따라옴 ---
const STEP_TRIGGER_DIST = 0.22; // 골반-발중심 수평 이격 임계(m)
const STEP_SUSTAIN_TIME = 0.45; // 임계 초과 지속 시간(s) — 춤의 왕복 sway는 걸러냄
const STEP_DURATION = 0.3;      // 한 발 이동 시간(s)
const STEP_HEIGHT = 0.06;       // 스텝 중 발 들어올림(m)
const STEP_COOLDOWN = 0.25;     // 스텝 간 최소 간격(s)
const STANCE_MISMATCH = 0.08;   // 양발 오프셋 차 허용치(m) — 초과 시 뒷발 따라붙음
let footStep = {
    offL: 0, offR: 0,      // 발 앵커의 rest 대비 월드 x 오프셋(m)
    liftL: 0, liftR: 0,    // 스텝 중 발 높이(m)
    sustainT: 0, cooldownT: 0,
    active: null           // { side, fromX, toX, t }
};
let danceMode = false;       // 골반 visibility hysteresis로 전환

// --- Hand Tracking 결과 저장 (아바타 기준 좌/우, 미검출 시 null) ---
// tasks-vision handedness 라벨은 해부학적 기준 → 미러 모드에서 좌우 스왑
let detectedHands = {
    left: null,   // 아바타 왼손 (라벨 "Right" = 사용자 오른손)
    right: null   // 아바타 오른손 (라벨 "Left" = 사용자 왼손)
};

// --- Unified Dialogue System ---
const MAX_VISIBLE_MESSAGES = 5;
let dialogueMessages = [];
let isDialogueEnabled = false;
let dialogueDisplayMode = 'single';  // 'history' or 'single'
let dialogueInputMode = isMobile ? 'typing' : 'voice';  // 모바일: typing, 데스크탑: voice
let speechRecognition = null;
let currentInterimText = '';
let messageTimeout = null;

// Legacy aliases for compatibility
let chatMessages = [];
let isChatEnabled = false;

function addChatMessage(text) {
    if (!text.trim()) return;

    const messagesContainer = document.getElementById('chat-messages');
    if (!messagesContainer) return;

    // Create new message element
    const messageEl = document.createElement('div');
    messageEl.className = 'chat-message';
    messageEl.textContent = text;

    // Append - newest at bottom
    messagesContainer.appendChild(messageEl);

    // Store message reference (oldest first, newest last)
    chatMessages.push(messageEl);

    // Remove very old messages from DOM
    updateMessageFading();
}

function updateMessageFading() {
    // 오래된 메시지(배열 앞)는 DOM에서 제거
    while (chatMessages.length > MAX_VISIBLE_MESSAGES + 1) {
        const oldMessage = chatMessages.shift();
        oldMessage.remove();
    }
}

function clearChatMessages() {
    const messagesContainer = document.getElementById('chat-messages');
    if (messagesContainer) {
        messagesContainer.innerHTML = '';
    }
    chatMessages = [];
}

function drawChatMessagesToCanvas(ctx, canvasWidth, canvasHeight) {
    if (chatMessages.length === 0) return;

    const fontSize = isMiniAvatar ? 18 : 24;
    const padding = isMiniAvatar ? 8 : 14;
    const lineHeight = fontSize + padding * 2 + 6;
    const maxWidth = isMiniAvatar ? 300 : 500;

    // 위치 계산
    let centerX, baseY;
    if (isMiniAvatar && miniAvatarPosition.x !== null) {
        // Mini avatar 모드: 아바타 머리 근처
        const scaleX = canvasWidth / stableWindowWidth;
        const scaleY = canvasHeight / stableWindowHeight;
        const avatarWidth = 300 * scaleX;
        centerX = (miniAvatarPosition.x * scaleX) + avatarWidth / 2;
        baseY = (miniAvatarPosition.y * scaleY) + 50; // 아바타 상단 근처
    } else {
        // Full avatar 모드: 입력창 위 (하단에서 약 15% 위치)
        centerX = canvasWidth / 2;
        baseY = canvasHeight * 0.82;
    }

    ctx.font = `600 ${fontSize}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // 메시지 그리기 (chatMessages: oldest first, newest last)
    const visibleMessages = chatMessages.slice(-MAX_VISIBLE_MESSAGES);
    visibleMessages.forEach((msgEl, index) => {
        const age = visibleMessages.length - 1 - index; // 마지막이 최신(age=0)
        const text = msgEl.textContent;
        const y = baseY - (age * lineHeight);

        // 투명도 계산 (위로 올라갈수록 흐려짐)
        let alpha = 1 - (age * 0.2);
        alpha = Math.max(alpha, 0.1);

        // 배경 그리기
        const textWidth = Math.min(ctx.measureText(text).width + padding * 2, maxWidth);
        const bgX = centerX - textWidth / 2;
        const bgY = y - fontSize / 2 - padding;
        const bgHeight = fontSize + padding * 2;

        ctx.fillStyle = `rgba(0, 0, 0, ${0.7 * alpha})`;
        ctx.beginPath();
        ctx.roundRect(bgX, bgY, textWidth, bgHeight, 12);
        ctx.fill();

        // 텍스트 그리기
        ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
        ctx.fillText(text, centerX, y, maxWidth - padding * 2);
    });
}

function setupChatInput() {
    const chatInput = document.getElementById('chat-input');
    if (!chatInput) return;

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.isComposing) {
            e.preventDefault();
            const text = chatInput.value;
            if (text.trim()) {
                addChatMessage(text);
                chatInput.value = '';
            }
        }
        // Prevent Escape from stopping recording while typing
        if (e.key === 'Escape') {
            e.stopPropagation();
            chatInput.blur();
        }
    });

    // Prevent space from triggering other shortcuts
    chatInput.addEventListener('keyup', (e) => {
        e.stopPropagation();
    });

    // Clear chat button
    const clearChatBtn = document.getElementById('clear-chat');
    if (clearChatBtn) {
        clearChatBtn.addEventListener('click', () => {
            clearChatMessages();
        });
    }

    // Toggle chat button
    const toggleChatBtn = document.getElementById('toggle-chat');
    if (toggleChatBtn) {
        toggleChatBtn.addEventListener('click', () => {
            toggleChatMode();
        });
    }
}

function toggleChatMode() {
    isChatEnabled = !isChatEnabled;

    const btn = document.getElementById('toggle-chat');
    if (btn) {
        btn.innerText = isChatEnabled ? 'Chat OFF' : 'Chat ON';
        btn.classList.toggle('chat-active', isChatEnabled);
    }

    document.body.classList.toggle('chat-enabled', isChatEnabled);

    // 채팅 켜면 입력창에 포커스
    if (isChatEnabled) {
        const chatInput = document.getElementById('chat-input');
        if (chatInput) {
            setTimeout(() => chatInput.focus(), 100);
        }
    }
}

function updateChatOverlayPosition() {
    const chatOverlay = document.getElementById('chat-overlay');
    if (!chatOverlay) return;

    if (isMiniAvatar && miniAvatarPosition.x !== null) {
        // Mini avatar 모드: 아바타 머리 근처에 위치
        const avatarWidth = 300;
        const overlayWidth = 300;
        const avatarX = miniAvatarPosition.x;
        const avatarY = miniAvatarPosition.y;

        chatOverlay.style.left = (avatarX + avatarWidth / 2 - overlayWidth / 2) + 'px'; // 중앙 정렬
        chatOverlay.style.top = (avatarY - 80) + 'px'; // 아바타 머리 높이
    } else {
        // Full avatar 모드: CSS 기본값 사용
        chatOverlay.style.left = '';
        chatOverlay.style.top = '';
    }
}

function updateCaptionOverlayPosition() {
    const captionOverlay = document.getElementById('caption-overlay');
    if (!captionOverlay) return;

    if (isMiniAvatar && miniAvatarPosition.x !== null) {
        // Mini avatar 모드: 아바타 몸통 위치에 표시
        const avatarWidth = 300;
        const avatarHeight = 400;
        const overlayWidth = 300;
        const avatarX = miniAvatarPosition.x;
        const avatarY = miniAvatarPosition.y;

        captionOverlay.style.left = (avatarX + avatarWidth / 2 - overlayWidth / 2) + 'px';
        captionOverlay.style.top = (avatarY + avatarHeight * 0.55) + 'px'; // 몸통 위치 (55%)
    } else {
        // Full avatar 모드: CSS 기본값 사용
        captionOverlay.style.left = '';
        captionOverlay.style.top = '';
    }
}

// --- Speech-to-Text Captions ---
let isCaptionsEnabled = false;
let isCaptionsStarting = false;  // 시작 중 race condition 방지
let networkRetryCount = 0;
let networkRetryTimer = null;
const MAX_NETWORK_RETRIES = 5;
let currentCaption = '';
let captionTimeout = null;

function initSpeechRecognition() {
    // Check browser support
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.warn('Speech Recognition not supported in this browser');
        return false;
    }

    speechRecognition = new SpeechRecognition();
    speechRecognition.continuous = true;
    speechRecognition.interimResults = true;
    speechRecognition.lang = 'ko-KR'; // Korean, can be changed

    speechRecognition.onresult = (event) => {
        // 인식 성공 시 network 재시도 카운터 리셋
        networkRetryCount = 0;
        clearTimeout(networkRetryTimer);
        networkRetryTimer = null;

        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
        }

        // Update dialogue display (unified system)
        if (finalTranscript) {
            addDialogueMessage(finalTranscript, false);
        } else if (interimTranscript) {
            addDialogueMessage(interimTranscript, true);
        }
    };

    speechRecognition.onerror = (event) => {
        if (event.error !== 'no-speech') {
            console.error('Speech recognition error:', event.error);
        }
        switch (event.error) {
            case 'not-allowed':
                alert('마이크 권한이 필요합니다. 브라우저 설정에서 마이크 권한을 허용해주세요.');
                stopCaptions();
                break;
            case 'audio-capture':
                console.warn('[Captions] Microphone is being used by another application');
                // 자동 재시도는 onend에서 처리됨
                break;
            case 'network':
                networkRetryCount++;
                if (networkRetryCount > MAX_NETWORK_RETRIES) {
                    console.error('[Captions] Too many network errors, stopping captions');
                    stopCaptions();
                    alert('음성 인식 네트워크 오류가 반복되어 Talk를 중지했습니다. 네트워크를 확인 후 다시 시도해주세요.');
                } else {
                    const delay = Math.min(1000 * Math.pow(2, networkRetryCount - 1), 16000);
                    console.warn(`[Captions] Network error (${networkRetryCount}/${MAX_NETWORK_RETRIES}), retry in ${delay}ms`);
                    clearTimeout(networkRetryTimer);
                    networkRetryTimer = setTimeout(() => {
                        if (isCaptionsEnabled && isMicEnabled) {
                            try { speechRecognition.start(); } catch (_) {}
                        }
                    }, delay);
                }
                break;
            case 'aborted':
                console.log('[Captions] Recognition aborted');
                break;
            case 'no-speech':
                // 조용히 무시: onend에서 자동 재시작됨
                break;
            default:
                console.warn('[Captions] Unhandled error:', event.error);
        }
    };

    speechRecognition.onend = () => {
        // network 에러 백오프 타이머가 있으면 즉시 재시작하지 않음
        if (networkRetryTimer) return;

        // Restart if still enabled AND mic is on (continuous mode can stop unexpectedly)
        if (isCaptionsEnabled && isMicEnabled && !isCaptionsStarting) {
            try {
                isCaptionsStarting = true;
                speechRecognition.start();
                isCaptionsStarting = false;
            } catch (e) {
                isCaptionsStarting = false;
                console.warn('Failed to restart speech recognition:', e);
            }
        } else if (isCaptionsEnabled && !isMicEnabled) {
            // Mic이 꺼졌으면 캡션도 중지
            console.log('[Captions] Mic turned off, stopping captions');
            isCaptionsEnabled = false;
            document.body.classList.remove('captions-enabled');
        }
    };

    return true;
}

function startCaptions() {
    // 이미 실행 중이거나 시작 중이면 무시
    if (isCaptionsEnabled || isCaptionsStarting) {
        console.log('[Captions] Already running or starting, skipping...');
        return;
    }

    // Mic이 꺼져있으면 시작하지 않음
    if (!isMicEnabled) {
        console.log('[Captions] Mic is OFF, cannot start speech recognition');
        return;
    }

    if (!speechRecognition && !initSpeechRecognition()) {
        alert('이 브라우저는 음성 인식을 지원하지 않습니다.');
        return;
    }

    try {
        isCaptionsStarting = true;
        speechRecognition.start();
        isCaptionsEnabled = true;
        isCaptionsStarting = false;
        document.body.classList.add('captions-enabled');

        const btn = document.getElementById('toggle-captions');
        if (btn) {
            btn.textContent = 'Captions OFF';
            btn.classList.add('captions-active');
        }
        console.log('[Captions] Started');
    } catch (e) {
        isCaptionsStarting = false;
        console.error('Failed to start speech recognition:', e);
    }
}

function stopCaptions() {
    clearTimeout(networkRetryTimer);
    networkRetryTimer = null;
    networkRetryCount = 0;

    if (speechRecognition) {
        try {
            speechRecognition.stop();
        } catch (e) {
            // Ignore
        }
    }

    isCaptionsEnabled = false;
    document.body.classList.remove('captions-enabled');
    currentCaption = '';

    const captionEl = document.getElementById('caption-text');
    if (captionEl) {
        captionEl.textContent = '';
    }

    const btn = document.getElementById('toggle-captions');
    if (btn) {
        btn.textContent = 'Captions ON';
        btn.classList.remove('captions-active');
    }

    clearTimeout(captionTimeout);
}

function toggleCaptions() {
    if (isCaptionsEnabled) {
        stopCaptions();
    } else {
        startCaptions();
    }
}

function drawCaptionToCanvas(ctx, canvasWidth, canvasHeight) {
    if (!currentCaption) return;

    const fontSize = isMiniAvatar ? 16 : 28;
    const padding = isMiniAvatar ? 10 : 16;
    const maxWidth = isMiniAvatar ? 280 : canvasWidth * 0.8;

    ctx.font = `500 ${fontSize}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    // Calculate text dimensions
    const textWidth = Math.min(ctx.measureText(currentCaption).width + padding * 2, maxWidth);
    const bgHeight = fontSize + padding * 2;

    let x, y;
    if (isMiniAvatar && miniAvatarPosition.x !== null) {
        // Mini avatar 모드: 아바타 몸통 위치에 표시
        const scaleX = canvasWidth / stableWindowWidth;
        const scaleY = canvasHeight / stableWindowHeight;
        const avatarWidth = 300 * scaleX;
        const avatarHeight = 400 * scaleY;
        x = (miniAvatarPosition.x * scaleX) + avatarWidth / 2;
        y = (miniAvatarPosition.y * scaleY) + avatarHeight * 0.6; // 몸통 위치 (60%)
    } else {
        // Full avatar 모드: 하단 중앙
        x = canvasWidth / 2;
        y = canvasHeight - 160;
    }

    // Draw background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.beginPath();
    ctx.roundRect(x - textWidth / 2, y - bgHeight / 2, textWidth, bgHeight, 8);
    ctx.fill();

    // Draw text
    ctx.fillStyle = 'white';
    ctx.fillText(currentCaption, x, y, maxWidth - padding * 2);
}

function setupCaptionsButton() {
    const toggleCaptionsBtn = document.getElementById('toggle-captions');
    if (toggleCaptionsBtn) {
        toggleCaptionsBtn.addEventListener('click', toggleCaptions);
    }
}

// --- Unified Dialogue Setup ---
function toggleDialogue() {
    isDialogueEnabled = !isDialogueEnabled;

    const btn = document.getElementById('toggle-dialogue');
    if (btn) {
        btn.innerHTML = isDialogueEnabled ? 'Talk<br>ON' : 'Talk<br>OFF';
        btn.classList.toggle('dialogue-active', isDialogueEnabled);
    }

    document.body.classList.toggle('dialogue-enabled', isDialogueEnabled);

    if (isDialogueEnabled) {
        document.body.classList.add('display-' + dialogueDisplayMode);
        document.body.classList.add('input-' + dialogueInputMode);

        if (dialogueInputMode === 'typing') {
            const input = document.getElementById('dialogue-input');
            if (input) setTimeout(() => input.focus(), 100);
        } else if (dialogueInputMode === 'voice') {
            startCaptions();
        }
    } else {
        stopCaptions();
        // Clear messages
        const container = document.getElementById('dialogue-messages');
        if (container) container.innerHTML = '';
        dialogueMessages = [];
    }
}

function addDialogueMessage(text, isInterim = false) {
    if (!text.trim()) return;

    const container = document.getElementById('dialogue-messages');
    if (!container) return;

    // 단일 모드: 기존 요소 재사용 (애니메이션 방지)
    if (dialogueDisplayMode === 'single') {
        let msgEl = container.querySelector('.dialogue-message');
        if (!msgEl) {
            msgEl = document.createElement('div');
            msgEl.className = 'dialogue-message';
            container.appendChild(msgEl);
            dialogueMessages = [msgEl];
        }
        msgEl.textContent = text;
        msgEl.classList.toggle('interim', isInterim);

        // 확정 메시지면 자동 사라짐 타이머 설정
        if (!isInterim) {
            clearTimeout(messageTimeout);
            messageTimeout = setTimeout(() => {
                msgEl.textContent = '';
            }, 4000);
        }
        return;
    }

    // 히스토리 모드: 기존 로직
    // 임시 메시지 처리
    if (isInterim) {
        const existing = container.querySelector('.interim');
        if (existing) {
            existing.textContent = text;
            return;
        }
    } else {
        const existing = container.querySelector('.interim');
        if (existing) existing.remove();
        dialogueMessages = dialogueMessages.filter(m => !m.classList.contains('interim'));
    }

    const msgEl = document.createElement('div');
    msgEl.className = 'dialogue-message' + (isInterim ? ' interim' : '');
    msgEl.textContent = text;
    container.appendChild(msgEl);
    dialogueMessages.push(msgEl);

    // 오래된 메시지 제거
    while (dialogueMessages.length > MAX_VISIBLE_MESSAGES + 1) {
        const old = dialogueMessages.shift();
        old.remove();
    }
}

function setDialogueDisplayMode(mode) {
    dialogueDisplayMode = mode;
    document.body.classList.remove('display-history', 'display-single');
    document.body.classList.add('display-' + mode);

    document.querySelectorAll('.option-btn[data-display]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.display === mode);
    });

    // 모드 변경 시 초기화
    const container = document.getElementById('dialogue-messages');
    if (container) container.innerHTML = '';
    dialogueMessages = [];
}

function setDialogueInputMode(mode) {
    const prevMode = dialogueInputMode;
    dialogueInputMode = mode;
    document.body.classList.remove('input-typing', 'input-voice');
    document.body.classList.add('input-' + mode);

    document.querySelectorAll('.option-btn[data-input]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.input === mode);
    });

    if (isDialogueEnabled) {
        if (prevMode === 'voice' && mode === 'typing') {
            stopCaptions();
        } else if (prevMode === 'typing' && mode === 'voice') {
            startCaptions();
        }
    }
}

function updateDialogueOverlayPosition() {
    const overlay = document.getElementById('dialogue-overlay');
    if (!overlay) return;

    if (isMiniAvatar && miniAvatarPosition.x !== null) {
        const avatarWidth = 300;
        overlay.style.left = (miniAvatarPosition.x + avatarWidth / 2 - 150) + 'px';
        overlay.style.top = (miniAvatarPosition.y + 220) + 'px';
    } else {
        overlay.style.left = '';
        overlay.style.top = '';
    }
}

function drawDialogueToCanvas(ctx, canvasWidth, canvasHeight) {
    if (dialogueMessages.length === 0) return;

    const fontSize = isMiniAvatar ? 16 : 22;
    const padding = isMiniAvatar ? 8 : 12;
    const lineHeight = fontSize + padding * 2 + 6;
    const maxWidth = isMiniAvatar ? 280 : 500;

    let centerX, baseY;
    if (isMiniAvatar && miniAvatarPosition.x !== null) {
        const scaleX = canvasWidth / stableWindowWidth;
        const scaleY = canvasHeight / stableWindowHeight;
        centerX = (miniAvatarPosition.x * scaleX) + 150 * scaleX;
        baseY = (miniAvatarPosition.y * scaleY) + 250 * scaleY;
    } else {
        centerX = canvasWidth / 2;
        baseY = canvasHeight * 0.82;
    }

    ctx.font = `500 ${fontSize}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const visible = dialogueDisplayMode === 'single'
        ? dialogueMessages.slice(-1)
        : dialogueMessages.slice(-MAX_VISIBLE_MESSAGES);

    visible.forEach((msgEl, index) => {
        const age = visible.length - 1 - index;
        const text = msgEl.textContent;
        const y = baseY - (age * lineHeight);
        const isInterim = msgEl.classList.contains('interim');

        let alpha = dialogueDisplayMode === 'single' ? 1 : Math.max(1 - (age * 0.2), 0.1);
        if (isInterim) alpha *= 0.7;

        const textWidth = Math.min(ctx.measureText(text).width + padding * 2, maxWidth);
        const bgX = centerX - textWidth / 2;
        const bgY = y - fontSize / 2 - padding;
        const bgHeight = fontSize + padding * 2;

        ctx.fillStyle = `rgba(0, 0, 0, ${0.7 * alpha})`;
        ctx.beginPath();
        ctx.roundRect(bgX, bgY, textWidth, bgHeight, 12);
        ctx.fill();

        ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
        ctx.fillText(text, centerX, y, maxWidth - padding * 2);
    });
}

function setupDialogue() {
    // Toggle button
    const toggleBtn = document.getElementById('toggle-dialogue');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', toggleDialogue);
    }

    // Input field
    const input = document.getElementById('dialogue-input');
    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.isComposing) {
                e.preventDefault();
                if (input.value.trim()) {
                    addDialogueMessage(input.value);
                    input.value = '';
                }
            }
            if (e.key === 'Escape') {
                e.stopPropagation();
                input.blur();
            }
        });
        input.addEventListener('keyup', (e) => e.stopPropagation());
    }

    // Clear button
    const clearBtn = document.getElementById('clear-dialogue');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            const container = document.getElementById('dialogue-messages');
            if (container) container.innerHTML = '';
            dialogueMessages = [];
        });
    }

    // Option buttons
    document.querySelectorAll('.option-btn[data-display]').forEach(btn => {
        btn.addEventListener('click', () => setDialogueDisplayMode(btn.dataset.display));
    });
    document.querySelectorAll('.option-btn[data-input]').forEach(btn => {
        btn.addEventListener('click', () => setDialogueInputMode(btn.dataset.input));
    });

    // 기본 모드 설정
    document.body.classList.add('display-' + dialogueDisplayMode);
    document.body.classList.add('input-' + dialogueInputMode);

    // 옵션 버튼 활성 상태 업데이트
    document.querySelectorAll('.option-btn[data-input]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.input === dialogueInputMode);
    });
}

// --- Initialization ---
function _postToOpener(payload) {
    if (!isIntegrationMode || !window.opener) return;
    window.opener.postMessage(payload, _integrationParams.origin);
}

function initIntegrationMode() {
    if (!isIntegrationMode) return;

    // 팝업 모드 UI 배지 표시
    const badge = document.createElement('div');
    badge.id = 'integration-badge';
    badge.textContent = `↩ ${_integrationParams.origin}`;
    document.body.appendChild(badge);

    // opener로부터 명령 수신
    window.addEventListener('message', (e) => {
        if (e.origin !== _integrationParams.origin) return;
        const { type } = e.data || {};
        if (type === 'avatar-recoder:start') {
            if (!mediaRecorder || mediaRecorder.state !== 'recording') startRecording();
        } else if (type === 'avatar-recoder:stop') {
            if (mediaRecorder && mediaRecorder.state === 'recording') stopRecording();
        } else if (type === 'avatar-recoder:cancel') {
            recordedChunks = [];
            window.close();
        }
    });

    // 창 닫힘 시 cancelled 알림
    window.addEventListener('beforeunload', () => {
        if (!mediaRecorder || mediaRecorder.state !== 'recording') {
            _postToOpener({ type: 'avatar-recoder:cancelled', sessionId: _integrationParams.sessionId });
        }
    });

    // opener에 준비 완료 신호 전송
    _postToOpener({ type: 'avatar-recoder:ready', sessionId: _integrationParams.sessionId });

    // autoRecord 옵션 처리
    if (_integrationParams.autoRecord) {
        // MediaPipe 초기화 완료를 기다린 뒤 자동 시작
        const waitAndRecord = () => {
            if (mediaRecorder === null && document.getElementById('output_canvas')) {
                startRecording();
            } else {
                setTimeout(waitAndRecord, 500);
            }
        };
        setTimeout(waitAndRecord, 2000);
    }
}

// ============================================================
// Drawing Annotation System
// ============================================================

// --- Drawing State ---
let isDrawModeOn = false;
let isBlackout = false;
let isFadeEnabled = false;

let drawCurrentTool = 'pen';   // 'pen' | 'highlighter' | 'arrow' | 'text' | 'eraser'
let drawCurrentColor = '#007aff';
const toolLastColor = {        // per-tool color memory
    pen:         '#007aff',
    highlighter: '#ffcc00',
    arrow:       '#ff3b30',
    text:        '#ffffff',
    eraser:      null,
};
let drawCurrentSize = 'M';
let drawCurrentTextStyle = null; // null | 'shadow' | 'background'
let drawCurrentFontSize = 24;

let drawStrokes = [];   // committed strokes
let drawUndoStack = []; // undo history (stroke objects)
let activeStroke = null; // stroke being drawn right now

// Arrow drawing state
let arrowStartX = null;
let arrowStartY = null;

// Text tool state
let textComposingEl = null;    // the <textarea> DOM element
let textReEditTarget = null;   // stroke being re-edited
let textLastTapTime = 0;
let textLastTapStroke = null;

let drawingCanvasEl = null;
let drawingCanvasCtx = null;
let drawRAFId = null;
let isDrawPanMode = false;        // Space held → temporary pan mode in draw mode
let isToolbarHorizontal = false;  // toolbar layout: false=vertical, true=horizontal

const DRAW_LINE_WIDTHS = {
    pen:         { S: 2,  M: 4,  L: 8  },
    highlighter: { S: 12, M: 20, L: 32 },
    arrow:       { S: 2,  M: 4,  L: 6  },
    eraser:      { S: 20, M: 40, L: 60 },
};
const DRAW_TEXT_SIZES = { S: 16, M: 24, L: 36 };

const STROKE_FADE_DELAY = 2000;    // ms before pen/arrow fade starts
const STROKE_FADE_DUR   = 600;     // ms fade duration
const TEXT_FADE_DELAY   = 10000;
let drawNextId = 1;

// ---- Utility: draw a rounded rect path (Chrome 99+ roundRect fallback) ----
function drawRoundRect(ctx, x, y, w, h, r) {
    if (ctx.roundRect) {
        ctx.roundRect(x, y, w, h, r);
    } else {
        ctx.moveTo(x + r, y);
        ctx.lineTo(x + w - r, y);
        ctx.arcTo(x + w, y, x + w, y + r, r);
        ctx.lineTo(x + w, y + h - r);
        ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
        ctx.lineTo(x + r, y + h);
        ctx.arcTo(x, y + h, x, y + h - r, r);
        ctx.lineTo(x, y + r);
        ctx.arcTo(x, y, x + r, y, r);
        ctx.closePath();
    }
}

// ---- Render a single stroke to a canvas context ----
// nx, ny normalization factors (canvas w/h)
function renderStroke(ctx, stroke, nx, ny) {
    if (stroke.renderSkip) return;
    const op = stroke.opacity;
    if (op <= 0) return;

    ctx.save();
    ctx.globalAlpha = op;

    if (stroke.type === 'pen' || stroke.type === 'highlighter') {
        if (stroke.points.length < 2) {
            ctx.restore();
            return;
        }
        const lw = DRAW_LINE_WIDTHS[stroke.type][stroke.sizeKey];
        ctx.lineWidth = lw * (nx / stableWindowWidth);
        ctx.strokeStyle = stroke.color;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        if (stroke.type === 'highlighter') {
            ctx.globalAlpha = op * 0.5;
            ctx.lineWidth = lw * (nx / stableWindowWidth);
        }
        ctx.beginPath();
        const pts = stroke.points;
        ctx.moveTo(pts[0].x * nx, pts[0].y * ny);
        if (pts.length === 2) {
            ctx.lineTo(pts[1].x * nx, pts[1].y * ny);
        } else {
            // 중간점 quadratic 곡선 연결: 샘플이 성겨도(스레드 블로킹 등) 각지지 않게
            for (let i = 1; i < pts.length - 1; i++) {
                const mx = (pts[i].x + pts[i + 1].x) / 2 * nx;
                const my = (pts[i].y + pts[i + 1].y) / 2 * ny;
                ctx.quadraticCurveTo(pts[i].x * nx, pts[i].y * ny, mx, my);
            }
            const last = pts[pts.length - 1];
            ctx.lineTo(last.x * nx, last.y * ny);
        }
        ctx.stroke();

    } else if (stroke.type === 'arrow') {
        const lw = DRAW_LINE_WIDTHS.arrow[stroke.sizeKey];
        const scaledLw = lw * (nx / stableWindowWidth);
        const x1 = stroke.x1 * nx, y1 = stroke.y1 * ny;
        const x2 = stroke.x2 * nx, y2 = stroke.y2 * ny;
        const angle = Math.atan2(y2 - y1, x2 - x1);
        const headLen = Math.max(scaledLw * 5, 12 * (nx / stableWindowWidth));

        ctx.strokeStyle = stroke.color;
        ctx.fillStyle = stroke.color;
        ctx.lineWidth = scaledLw;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        // shaft
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2 - Math.cos(angle) * headLen * 0.5, y2 - Math.sin(angle) * headLen * 0.5);
        ctx.stroke();

        // arrowhead
        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI / 6),
                   y2 - headLen * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI / 6),
                   y2 - headLen * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();

    } else if (stroke.type === 'text') {
        const scale = nx / stableWindowWidth;
        const fs = stroke.fontSize * scale;
        const x = stroke.x * nx;
        const y = stroke.y * ny;

        ctx.font = `${fs}px sans-serif`;
        ctx.textBaseline = 'top';

        const lines = stroke.lines;
        const lineHeight = fs * 1.25;
        const totalH = lines.length * lineHeight;

        // measure max line width
        let maxW = 0;
        for (const line of lines) {
            const w = ctx.measureText(line).width;
            if (w > maxW) maxW = w;
        }

        if (stroke.textStyle === 'background') {
            const padX = fs * 0.35;
            const padY = fs * 0.2;
            ctx.fillStyle = 'rgba(0,0,0,0.6)';
            ctx.beginPath();
            drawRoundRect(ctx, x - padX, y - padY, maxW + padX * 2, totalH + padY * 2, fs * 0.2);
            ctx.fill();
            ctx.fillStyle = stroke.color;
            for (let i = 0; i < lines.length; i++) {
                ctx.fillText(lines[i], x, y + i * lineHeight);
            }
        } else {
            // optional shadow
            if (stroke.textStyle === 'shadow') {
                ctx.shadowColor = 'rgba(0,0,0,0.85)';
                ctx.shadowBlur = fs * 0.3;
                ctx.shadowOffsetX = fs * 0.05;
                ctx.shadowOffsetY = fs * 0.05;
            }
            ctx.fillStyle = stroke.color;
            for (let i = 0; i < lines.length; i++) {
                ctx.fillText(lines[i], x, y + i * lineHeight);
            }
        }
    }

    ctx.restore();
}

// ---- Render active (in-progress) stroke preview ----
function renderActiveStroke(ctx, nx, ny) {
    if (!activeStroke) return;
    ctx.save();
    ctx.globalAlpha = 1;

    if (activeStroke.type === 'pen' || activeStroke.type === 'highlighter') {
        if (activeStroke.points.length < 2) { ctx.restore(); return; }
        const lw = DRAW_LINE_WIDTHS[activeStroke.type][activeStroke.sizeKey];
        ctx.lineWidth = lw * (nx / stableWindowWidth);
        ctx.strokeStyle = activeStroke.color;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        if (activeStroke.type === 'highlighter') ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.moveTo(activeStroke.points[0].x * nx, activeStroke.points[0].y * ny);
        for (let i = 1; i < activeStroke.points.length; i++) {
            ctx.lineTo(activeStroke.points[i].x * nx, activeStroke.points[i].y * ny);
        }
        ctx.stroke();

    } else if (activeStroke.type === 'arrow') {
        if (arrowStartX === null) { ctx.restore(); return; }
        const lw = DRAW_LINE_WIDTHS.arrow[activeStroke.sizeKey];
        const scaledLw = lw * (nx / stableWindowWidth);
        const x1 = arrowStartX * nx, y1 = arrowStartY * ny;
        const x2 = activeStroke.x2 * nx, y2 = activeStroke.y2 * ny;
        const angle = Math.atan2(y2 - y1, x2 - x1);
        const headLen = Math.max(scaledLw * 5, 12 * (nx / stableWindowWidth));

        ctx.strokeStyle = activeStroke.color;
        ctx.fillStyle = activeStroke.color;
        ctx.lineWidth = scaledLw;
        ctx.lineCap = 'round';

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2 - Math.cos(angle) * headLen * 0.5, y2 - Math.sin(angle) * headLen * 0.5);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(x2, y2);
        ctx.lineTo(x2 - headLen * Math.cos(angle - Math.PI / 6),
                   y2 - headLen * Math.sin(angle - Math.PI / 6));
        ctx.lineTo(x2 - headLen * Math.cos(angle + Math.PI / 6),
                   y2 - headLen * Math.sin(angle + Math.PI / 6));
        ctx.closePath();
        ctx.fill();
    }

    ctx.restore();
}

// ---- Rendering loop for live preview canvas ----
function ensureDrawingRenderLoop() {
    if (drawRAFId) return;
    drawRAFId = requestAnimationFrame(drawingRenderLoop);
}

function drawingRenderLoop() {
    drawRAFId = null;

    if (!drawingCanvasEl) return;
    const w = drawingCanvasEl.width;
    const h = drawingCanvasEl.height;

    drawingCanvasCtx.clearRect(0, 0, w, h);

    const now = performance.now();
    let anyVisible = false;

    // Update fade and remove expired strokes (reverse loop for splice safety)
    for (let i = drawStrokes.length - 1; i >= 0; i--) {
        const s = drawStrokes[i];
        if (isFadeEnabled && s.createdAt !== null) {
            const delay = s.type === 'text' ? TEXT_FADE_DELAY : STROKE_FADE_DELAY;
            const elapsed = now - s.createdAt;
            if (elapsed >= delay + STROKE_FADE_DUR) {
                drawStrokes.splice(i, 1);
                continue;
            } else if (elapsed >= delay) {
                s.opacity = 1 - (elapsed - delay) / STROKE_FADE_DUR;
            } else {
                s.opacity = 1;
            }
        } else {
            s.opacity = 1;
        }
        if (s.opacity > 0 && !s.renderSkip) anyVisible = true;
    }

    // Render all committed strokes
    for (const s of drawStrokes) {
        renderStroke(drawingCanvasCtx, s, w, h);
    }

    // Render active (in-progress) stroke
    if (activeStroke) {
        renderActiveStroke(drawingCanvasCtx, w, h);
        anyVisible = true;
    }

    // When draw mode is off and nothing is visible, we're done (canvas stays clear)
    if (!isDrawModeOn && !anyVisible) {
        drawingCanvasCtx.clearRect(0, 0, w, h);
    }

    if (activeStroke || (isFadeEnabled && drawStrokes.length > 0)) {
        ensureDrawingRenderLoop();
    }
}

// ---- Render drawing layer into composite canvas for recording ----
function renderDrawingLayer(ctx, cw, ch, now) {
    if (drawStrokes.length === 0 && !activeStroke) return;

    for (const s of drawStrokes) {
        if (s.opacity <= 0 || s.renderSkip) continue;
        // fade is already applied in the RAF loop; read opacity directly
        renderStroke(ctx, s, cw, ch);
    }

    if (activeStroke) {
        renderActiveStroke(ctx, cw, ch);
    }
}

// ---- Eraser: delete strokes that overlap the eraser circle ----
function eraseAt(nx, ny, sizeKey) {
    const radius = DRAW_LINE_WIDTHS.eraser[sizeKey] / 2;
    const normRadius = radius / stableWindowWidth;

    for (let i = drawStrokes.length - 1; i >= 0; i--) {
        const s = drawStrokes[i];
        let hit = false;

        if (s.type === 'pen' || s.type === 'highlighter') {
            for (const pt of s.points) {
                const dx = pt.x - nx, dy = pt.y - ny;
                if (dx * dx + dy * dy <= normRadius * normRadius) { hit = true; break; }
            }
        } else if (s.type === 'arrow') {
            const dx = s.x2 - nx, dy = s.y2 - ny;
            const dx2 = s.x1 - nx, dy2 = s.y1 - ny;
            if (dx * dx + dy * dy <= normRadius * normRadius ||
                dx2 * dx2 + dy2 * dy2 <= normRadius * normRadius) hit = true;
        } else if (s.type === 'text') {
            const dx = s.x - nx, dy = s.y - ny;
            if (Math.abs(dx) < 0.15 && Math.abs(dy) < 0.08) hit = true;
        }

        if (hit) {
            drawUndoStack.push({ action: 'remove', stroke: drawStrokes[i] });
            drawStrokes.splice(i, 1);
        }
    }
}

// ---- Commit the current text textarea ----
function commitTextInput(cancel) {
    if (!textComposingEl) return;
    const el = textComposingEl;
    textComposingEl = null;

    if (cancel) {
        if (textReEditTarget) {
            textReEditTarget.renderSkip = false;
            textReEditTarget = null;
        }
        el.remove();
        return;
    }

    const rawText = el.value;
    const lines = rawText.split('\n');

    const rect = drawingCanvasEl.getBoundingClientRect();
    const nx_pos = (parseFloat(el.style.left) - rect.left) / rect.width;
    const ny_pos = (parseFloat(el.style.top) - rect.top) / rect.height;

    if (textReEditTarget) {
        // update existing stroke
        textReEditTarget.lines = lines;
        textReEditTarget.renderSkip = false;
        textReEditTarget.createdAt = performance.now();
        textReEditTarget = null;
    } else {
        const stroke = {
            id: drawNextId++,
            type: 'text',
            color: drawCurrentColor,
            textStyle: drawCurrentTextStyle,
            fontSize: drawCurrentFontSize,
            lines,
            x: nx_pos,
            y: ny_pos,
            opacity: 1,
            createdAt: performance.now(),
            renderSkip: false,
        };
        drawStrokes.push(stroke);
        drawUndoStack.push({ action: 'add', stroke });
    }

    el.remove();
    ensureDrawingRenderLoop();
}

// ---- Mount a textarea for text composition ----
function mountTextInput(clientX, clientY, existingStroke) {
    if (textComposingEl) {
        commitTextInput(false);
    }

    const el = document.createElement('textarea');
    el.id = 'drawing-text-input';
    el.rows = 1;
    el.placeholder = 'Type...';
    el.style.left = clientX + 'px';
    el.style.top = clientY + 'px';
    el.style.fontSize = drawCurrentFontSize + 'px';
    el.style.color = drawCurrentColor;
    document.body.appendChild(el);
    textComposingEl = el;

    if (existingStroke) {
        el.value = existingStroke.lines.join('\n');
        textReEditTarget = existingStroke;
        existingStroke.renderSkip = true;
    }

    el.focus();

    // Auto-resize height
    const autoResize = () => {
        el.style.height = 'auto';
        el.style.height = el.scrollHeight + 'px';
    };
    el.addEventListener('input', autoResize);
    autoResize();

    // Shift+Enter = commit, Enter = newline
    el.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.shiftKey) {
            e.preventDefault();
            commitTextInput(false);
        } else if (e.key === 'Escape') {
            e.preventDefault();
            commitTextInput(true);
        }
        e.stopPropagation();
    });

    // Outside click = commit
    const outsideClick = (e) => {
        if (textComposingEl && e.target !== textComposingEl) {
            commitTextInput(false);
            document.removeEventListener('pointerdown', outsideClick, true);
        }
    };
    // Delay to avoid immediate trigger from the same tap that opened the textarea
    setTimeout(() => {
        document.addEventListener('pointerdown', outsideClick, true);
    }, 50);
}

// ---- Main drawing canvas pointer handlers ----
function onDrawPointerDown(e) {
    // Allow mini avatar drag only when NOT in draw mode
    if (isMiniAvatar && !isDrawModeOn) {
        const sceneWrapper = document.getElementById('scene-wrapper');
        if (sceneWrapper) {
            const r = sceneWrapper.getBoundingClientRect();
            if (e.clientX >= r.left && e.clientX <= r.right &&
                e.clientY >= r.top  && e.clientY <= r.bottom) {
                onDragStart(e);
                return;
            }
        }
    }

    if (!isDrawModeOn) return;

    // Right mouse button → temporary eraser (no tool change)
    if (e.button === 2) {
        e.preventDefault();
        const rect = drawingCanvasEl.getBoundingClientRect();
        const nx = (e.clientX - rect.left) / rect.width;
        const ny = (e.clientY - rect.top)  / rect.height;
        eraseAt(nx, ny, drawCurrentSize);
        activeStroke = { type: 'eraser', x: nx, y: ny };
        try { drawingCanvasEl.setPointerCapture(e.pointerId); } catch(_) {}
        document.addEventListener('pointermove', onDrawPointerMove, { passive: false });
        document.addEventListener('pointerup', onDrawPointerUpGlobal);
        ensureDrawingRenderLoop();
        return;
    }

    // Space held → pan the zoomed screen instead of drawing
    if (isDrawPanMode && screenStream && screenZoom > 1) {
        isScreenPanning = true;
        screenPanStartX = e.clientX;
        screenPanStartY = e.clientY;
        screenZoomTxStart = screenZoomTx;
        screenZoomTyStart = screenZoomTy;
        document.body.classList.add('screen-panning');
        return;
    }

    e.preventDefault();

    const rect = drawingCanvasEl.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / rect.width;
    const ny = (e.clientY - rect.top)  / rect.height;

    if (drawCurrentTool === 'eraser') {
        eraseAt(nx, ny, drawCurrentSize);
        activeStroke = { type: 'eraser', x: nx, y: ny };
        ensureDrawingRenderLoop();
        return;
    }

    if (drawCurrentTool === 'text') {
        // Check double-tap for re-edit
        const now = performance.now();
        if (now - textLastTapTime < 300) {
            // Double tap: find nearby text stroke
            const hit = drawStrokes.slice().reverse().find(s => {
                if (s.type !== 'text') return false;
                const dx = Math.abs(s.x - nx), dy = Math.abs(s.y - ny);
                return dx < 0.15 && dy < 0.08;
            });
            if (hit) {
                textLastTapTime = 0;
                mountTextInput(e.clientX, e.clientY, hit);
                return;
            }
        }
        textLastTapTime = now;
        // Single tap: new text
        if (textComposingEl) {
            commitTextInput(false);
        } else {
            mountTextInput(e.clientX, e.clientY, null);
        }
        return;
    }

    if (drawCurrentTool === 'pen' || drawCurrentTool === 'highlighter') {
        activeStroke = {
            id: drawNextId++,
            type: drawCurrentTool,
            color: drawCurrentColor,
            sizeKey: drawCurrentSize,
            points: [{ x: nx, y: ny }],
            opacity: 1,
            createdAt: null,
        };
    } else if (drawCurrentTool === 'arrow') {
        arrowStartX = nx;
        arrowStartY = ny;
        activeStroke = {
            id: drawNextId++,
            type: 'arrow',
            color: drawCurrentColor,
            sizeKey: drawCurrentSize,
            x1: nx, y1: ny, x2: nx, y2: ny,
            opacity: 1,
            createdAt: null,
        };
    }

    // Capture pointer so move/up fire even outside the canvas
    if (activeStroke) {
        try { drawingCanvasEl.setPointerCapture(e.pointerId); } catch(_) {}
        document.addEventListener('pointermove', onDrawPointerMove, { passive: false });
        document.addEventListener('pointerup', onDrawPointerUpGlobal);
        ensureDrawingRenderLoop();
    }
}

function onDrawPointerUpGlobal(e) {
    document.removeEventListener('pointermove', onDrawPointerMove);
    document.removeEventListener('pointerup', onDrawPointerUpGlobal);
    onDrawPointerUp(e);
}

function onDrawPointerMove(e) {
    if (!activeStroke) return;

    const rect = drawingCanvasEl.getBoundingClientRect();
    // 메인 스레드가 인식 연산으로 바쁠 때 브라우저가 병합해 버린 중간 샘플까지 복원 (선 끊김 완화)
    const coalesced = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
    const samples = coalesced.length > 0 ? coalesced : [e];

    if (activeStroke.type === 'eraser') {
        for (const ev of samples) {
            const nx = (ev.clientX - rect.left) / rect.width;
            const ny = (ev.clientY - rect.top)  / rect.height;
            eraseAt(nx, ny, drawCurrentSize);
            activeStroke.x = nx;
            activeStroke.y = ny;
        }
        ensureDrawingRenderLoop();
        return;
    }

    if (activeStroke.type === 'pen' || activeStroke.type === 'highlighter') {
        for (const ev of samples) {
            activeStroke.points.push({
                x: (ev.clientX - rect.left) / rect.width,
                y: (ev.clientY - rect.top)  / rect.height
            });
        }
    } else if (activeStroke.type === 'arrow') {
        activeStroke.x2 = (e.clientX - rect.left) / rect.width;
        activeStroke.y2 = (e.clientY - rect.top)  / rect.height;
    }
    ensureDrawingRenderLoop();
}

function onDrawPointerUp(e) {
    if (!activeStroke) return;
    if (activeStroke.type === 'eraser') {
        activeStroke = null;
        ensureDrawingRenderLoop();
        return;
    }

    const rect = drawingCanvasEl.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / rect.width;
    const ny = (e.clientY - rect.top)  / rect.height;

    if (activeStroke.type === 'pen' || activeStroke.type === 'highlighter') {
        if (activeStroke.points.length >= 2) {
            activeStroke.createdAt = isFadeEnabled ? performance.now() : null;
            drawStrokes.push(activeStroke);
            drawUndoStack.push({ action: 'add', stroke: activeStroke });
        }
    } else if (activeStroke.type === 'arrow') {
        activeStroke.x2 = nx;
        activeStroke.y2 = ny;
        const dx = activeStroke.x2 - activeStroke.x1;
        const dy = activeStroke.y2 - activeStroke.y1;
        if (dx * dx + dy * dy > 0.0001) {
            activeStroke.createdAt = isFadeEnabled ? performance.now() : null;
            drawStrokes.push(activeStroke);
            drawUndoStack.push({ action: 'add', stroke: activeStroke });
        }
    }

    activeStroke = null;
    ensureDrawingRenderLoop();
}

// ---- Drawing canvas resize sync ----
function syncDrawingCanvasSize() {
    if (!drawingCanvasEl) return;
    const container = document.getElementById('preview-container');
    if (!container) return;
    drawingCanvasEl.width = container.clientWidth;
    drawingCanvasEl.height = container.clientHeight;
    ensureDrawingRenderLoop();
}

// ---- Toggle draw mode ----
function toggleDrawMode(on) {
    isDrawModeOn = on !== undefined ? on : !isDrawModeOn;
    const btn = document.getElementById('toggle-draw');
    if (btn) {
        btn.innerHTML = isDrawModeOn ? 'Draw<br>ON' : 'Draw<br>OFF';
        btn.classList.toggle('draw-active', isDrawModeOn);
    }
    if (isDrawModeOn) {
        document.body.classList.add('draw-mode-on');
        updateDrawToolCursor();
    } else {
        document.body.classList.remove('draw-mode-on');
        // commit any in-progress text
        if (textComposingEl) commitTextInput(false);
    }
    ensureDrawingRenderLoop();
}

function updateDrawToolCursor() {
    document.body.classList.remove('draw-tool-eraser', 'draw-tool-text');
    if (drawCurrentTool === 'eraser') document.body.classList.add('draw-tool-eraser');
    if (drawCurrentTool === 'text')   document.body.classList.add('draw-tool-text');
}

// ---- Setup drawing system ----
function setupDrawing() {
    drawingCanvasEl = document.getElementById('drawing-canvas');
    if (!drawingCanvasEl) return;

    syncDrawingCanvasSize();
    drawingCanvasCtx = drawingCanvasEl.getContext('2d');

    // Pointer events on the drawing canvas
    drawingCanvasEl.addEventListener('pointerdown', onDrawPointerDown);
    drawingCanvasEl.addEventListener('pointermove', onDrawPointerMove);
    drawingCanvasEl.addEventListener('pointerup',   onDrawPointerUp);
    drawingCanvasEl.addEventListener('pointerleave', onDrawPointerUp);
    drawingCanvasEl.addEventListener('pointercancel', onDrawPointerUp);
    drawingCanvasEl.addEventListener('contextmenu', e => { if (isDrawModeOn) e.preventDefault(); });

    // Resize
    const ro = new ResizeObserver(() => syncDrawingCanvasSize());
    const container = document.getElementById('preview-container');
    if (container) ro.observe(container);

    // Draw lazily; the loop wakes on edits and only stays active for fades/active strokes.
    ensureDrawingRenderLoop();

    // Toggle draw button
    const toggleDrawBtn = document.getElementById('toggle-draw');
    if (toggleDrawBtn) {
        toggleDrawBtn.addEventListener('click', () => toggleDrawMode());
    }

    // Tool buttons
    document.querySelectorAll('.draw-tool-btn').forEach(btn => {
        btn.addEventListener('click', () => setDrawTool(btn.dataset.tool));
    });

    // Color buttons
    document.querySelectorAll('.draw-color-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            drawCurrentColor = btn.dataset.color;
            document.querySelectorAll('.draw-color-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            // Save to per-tool color memory
            if (Object.prototype.hasOwnProperty.call(toolLastColor, drawCurrentTool)) {
                toolLastColor[drawCurrentTool] = drawCurrentColor;
            }
            if (textComposingEl) textComposingEl.style.color = drawCurrentColor;
        });
    });

    // Size buttons
    document.querySelectorAll('.draw-size-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            drawCurrentSize = btn.dataset.size;
            document.querySelectorAll('.draw-size-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Fade toggle
    const fadeBtn = document.getElementById('draw-fade-btn');
    if (fadeBtn) {
        fadeBtn.addEventListener('click', () => {
            isFadeEnabled = !isFadeEnabled;
            fadeBtn.classList.toggle('active', isFadeEnabled);
            // Apply/remove createdAt for existing strokes
            const now = performance.now();
            if (isFadeEnabled) {
                drawStrokes.forEach(s => { if (s.createdAt === null) s.createdAt = now; });
            } else {
                drawStrokes.forEach(s => { s.opacity = 1; });
            }
            ensureDrawingRenderLoop();
        });
    }

    // Blackout toggle
    const blackoutBtn = document.getElementById('draw-blackout-btn');
    if (blackoutBtn) {
        blackoutBtn.addEventListener('click', () => toggleBlackout());
    }

    // Undo
    const undoBtn = document.getElementById('draw-undo-btn');
    if (undoBtn) {
        undoBtn.addEventListener('click', drawUndo);
    }

    // Clear
    const clearBtn = document.getElementById('draw-clear-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', drawClear);
    }

    // Text style buttons
    document.querySelectorAll('.draw-textstyle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const style = btn.dataset.textstyle;
            if (drawCurrentTextStyle === style) {
                drawCurrentTextStyle = null;
                btn.classList.remove('active');
            } else {
                drawCurrentTextStyle = style;
                document.querySelectorAll('.draw-textstyle-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            }
        });
    });

    // Font size presets
    document.querySelectorAll('.draw-fontsize-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            drawCurrentFontSize = parseInt(btn.dataset.fontsize);
            document.querySelectorAll('.draw-fontsize-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            if (textComposingEl) textComposingEl.style.fontSize = drawCurrentFontSize + 'px';
        });
    });

    setupToolbarDrag();

    // Migrate title → data-tooltip for custom styled tooltips (removes native browser tooltip)
    document.querySelectorAll('#drawing-toolbar [title]').forEach(el => {
        el.dataset.tooltip = el.getAttribute('title');
        el.removeAttribute('title');
    });

    // Fixed-position tooltip (bypasses toolbar overflow clipping)
    const tooltipEl = document.createElement('div');
    tooltipEl.id = 'draw-tooltip';
    document.body.appendChild(tooltipEl);

    let tooltipTimer = null;
    document.getElementById('drawing-toolbar').addEventListener('mouseover', e => {
        const target = e.target.closest('[data-tooltip]');
        if (!target) return;
        clearTimeout(tooltipTimer);
        tooltipTimer = setTimeout(() => {
            const rect = target.getBoundingClientRect();
            tooltipEl.textContent = target.dataset.tooltip;
            tooltipEl.className = isToolbarHorizontal ? 'tip-below' : 'tip-right';
            tooltipEl.style.visibility = 'hidden';
            const tw = tooltipEl.offsetWidth;
            const th = tooltipEl.offsetHeight;
            let top, left;
            if (isToolbarHorizontal) {
                left = rect.left + rect.width / 2 - tw / 2;
                top  = rect.bottom + 10;
            } else {
                left = rect.right + 10;
                top  = rect.top + rect.height / 2 - th / 2;
            }
            tooltipEl.style.left = `${Math.max(4, left)}px`;
            tooltipEl.style.top  = `${Math.max(4, top)}px`;
            tooltipEl.style.visibility = '';
            tooltipEl.classList.add('visible');
        }, 300);
    });

    document.getElementById('drawing-toolbar').addEventListener('mouseleave', () => {
        clearTimeout(tooltipTimer);
        tooltipEl.classList.remove('visible');
    });

    document.getElementById('drawing-toolbar').addEventListener('mouseout', e => {
        const target = e.target.closest('[data-tooltip]');
        if (target && !target.contains(e.relatedTarget)) {
            clearTimeout(tooltipTimer);
            tooltipEl.classList.remove('visible');
        }
    });
}

function setDrawTool(tool) {
    drawCurrentTool = tool;
    document.querySelectorAll('.draw-tool-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tool === tool);
    });
    // Restore this tool's last color
    if (toolLastColor[tool]) {
        drawCurrentColor = toolLastColor[tool];
        document.querySelectorAll('.draw-color-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.color === drawCurrentColor);
        });
    }
    updateDrawToolCursor();
}

function setupToolbarDrag() {
    const toolbar = document.getElementById('drawing-toolbar');
    const handle = document.getElementById('drawing-toolbar-handle');
    if (!toolbar || !handle) return;

    // Orientation toggle
    const orientBtn = document.getElementById('draw-orient-btn');
    const orientPath = document.getElementById('draw-orient-path');
    if (orientBtn) {
        orientBtn.addEventListener('mousedown', e => e.stopPropagation());
        orientBtn.addEventListener('click', () => {
            isToolbarHorizontal = !isToolbarHorizontal;
            document.body.classList.toggle('toolbar-horizontal', isToolbarHorizontal);
            if (orientPath) {
                orientPath.setAttribute('d', isToolbarHorizontal ? 'M3 12h18' : 'M12 3v18');
            }
            // Reset position so toolbar reflows from default placement
            toolbar.style.left = '';
            toolbar.style.top = '';
            toolbar.style.transform = '';
        });
    }

    let isDragging = false;
    let dragOffsetX = 0;
    let dragOffsetY = 0;

    handle.addEventListener('mousedown', (e) => {
        if (e.target.closest('#draw-orient-btn')) return;
        e.preventDefault();
        const rect = toolbar.getBoundingClientRect();
        toolbar.style.transform = 'none';
        toolbar.style.left = rect.left + 'px';
        toolbar.style.top = rect.top + 'px';
        dragOffsetX = e.clientX - rect.left;
        dragOffsetY = e.clientY - rect.top;
        isDragging = true;
        document.body.classList.add('toolbar-dragging');
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const tw = toolbar.offsetWidth;
        const th = toolbar.offsetHeight;
        const newLeft = Math.max(0, Math.min(window.innerWidth - tw, e.clientX - dragOffsetX));
        const newTop = Math.max(0, Math.min(window.innerHeight - th, e.clientY - dragOffsetY));
        toolbar.style.left = newLeft + 'px';
        toolbar.style.top = newTop + 'px';
    });

    document.addEventListener('mouseup', () => {
        if (isDragging) {
            isDragging = false;
            document.body.classList.remove('toolbar-dragging');
        }
    });
}

function toggleBlackout(force) {
    isBlackout = force !== undefined ? force : !isBlackout;
    document.body.classList.toggle('blackout-on', isBlackout);
    const btn = document.getElementById('draw-blackout-btn');
    if (btn) btn.classList.toggle('active', isBlackout);
}

function drawUndo() {
    if (drawUndoStack.length === 0) return;
    const entry = drawUndoStack.pop();
    if (entry.action === 'add') {
        const idx = drawStrokes.indexOf(entry.stroke);
        if (idx !== -1) drawStrokes.splice(idx, 1);
    } else if (entry.action === 'remove') {
        drawStrokes.push(entry.stroke);
    }
    ensureDrawingRenderLoop();
}

function drawClear() {
    drawStrokes = [];
    drawUndoStack = [];
    activeStroke = null;
    if (textComposingEl) commitTextInput(true);
    ensureDrawingRenderLoop();
}

// ============================================================
// End Drawing Annotation System
// ============================================================

async function init() {
    initIntegrationMode();

    // 모바일 모드 설정
    if (isMobile) {
        console.log('[Mobile] Mobile device detected:', isIOS ? 'iOS' : isAndroid ? 'Android' : 'Other');
        document.body.classList.add('mobile-mode');

        // 모바일: 터치로 드롭다운 토글 (호버 대신)
        const setupMobileDropdown = (containerSelector, buttonSelector) => {
            const container = document.querySelector(containerSelector);
            const button = document.querySelector(buttonSelector);
            if (container && button) {
                button.addEventListener('click', (e) => {
                    e.stopPropagation();
                    container.classList.toggle('options-open');
                });
            }
        };
        setupMobileDropdown('.dialogue-controls', '#toggle-dialogue');
        setupMobileDropdown('.dev-controls', '#toggle-dev');

        // 다른 곳 터치하면 드롭다운 닫기
        document.addEventListener('click', () => {
            document.querySelectorAll('.options-open').forEach(el => {
                el.classList.remove('options-open');
            });
        });
    }

    // Dev dropdown menu handlers
    document.querySelectorAll('.option-btn[data-dev]').forEach(btn => {
        btn.addEventListener('click', () => {
            const action = btn.dataset.dev;

            if (action === 'debug-on') {
                isDebugView = true;
                if (orbitControls) orbitControls.enabled = false;
                updateDevOptions();
                updateView();
            } else if (action === 'debug-off') {
                isDebugView = false;
                if (orbitControls) orbitControls.enabled = isOrbitEnabled;
                updateDevOptions();
                updateView();
            } else if (action === 'landmarks-on') {
                DEBUG_MODE = true;
                setBoneAxesVisible(true);
                updateDevOptions();
            } else if (action === 'landmarks-off') {
                DEBUG_MODE = false;
                setBoneAxesVisible(false);
                updateDevOptions();
                if (debugCtx) {
                    debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
                }
            }
        });
    });

    // 초기 view 상태 적용
    updateView();
    updateDevOptions();

    // 아바타 드롭다운
    // 아바타 선택 (라디오식 버튼 — Size/Pose/View 그룹과 동일한 스타일)
    const avatarModelGroup = document.getElementById('avatar-model-group');
    const syncAvatarModelButtons = () => {
        if (!avatarModelGroup) return;
        avatarModelGroup.querySelectorAll('[data-avatar-model]').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.avatarModel === currentAvatarUrl);
        });
    };
    const setAvatarModelButtonsDisabled = (disabled) => {
        if (!avatarModelGroup) return;
        avatarModelGroup.querySelectorAll('[data-avatar-model]').forEach(btn => {
            btn.disabled = disabled;
        });
    };
    if (avatarModelGroup) {
        avatarModelGroup.addEventListener('click', async (e) => {
            const btn = e.target.closest('[data-avatar-model]');
            if (!btn) return;
            const url = btn.dataset.avatarModel;
            if (isAvatarLoading || currentAvatarUrl === url) return;
            setAvatarModelButtonsDisabled(true);
            await switchAvatar(url);
            setAvatarModelButtonsDisabled(false);
            // 로드 실패 시에도 currentAvatarUrl 기준으로 active 상태 복원
            syncAvatarModelButtons();
        });
    }

    // 커스텀 VRM 파일 로드
    let customAvatarBlobUrl = null;
    const loadCustomBtn = document.getElementById('load-custom-avatar');
    const avatarFileInput = document.getElementById('avatar-file-input');

    if (loadCustomBtn && avatarFileInput) {
        loadCustomBtn.addEventListener('click', () => {
            avatarFileInput.click();
        });

        avatarFileInput.addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file || (!file.name.endsWith('.vrm') && !file.name.endsWith('.glb'))) return;

            if (customAvatarBlobUrl) URL.revokeObjectURL(customAvatarBlobUrl);
            customAvatarBlobUrl = URL.createObjectURL(file);

            setAvatarModelButtonsDisabled(true);
            loadCustomBtn.disabled = true;

            const prevUrl = currentAvatarUrl;
            currentAvatarUrl = '';
            await switchAvatar(customAvatarBlobUrl);

            loadCustomBtn.disabled = false;
            setAvatarModelButtonsDisabled(false);

            if (currentAvatarUrl === customAvatarBlobUrl) {
                // 커스텀 아바타 버튼 추가 또는 갱신 (라디오 그룹에 합류)
                let customBtn = avatarModelGroup?.querySelector('button[data-custom]');
                if (!customBtn && avatarModelGroup) {
                    customBtn = document.createElement('button');
                    customBtn.className = 'option-btn';
                    customBtn.dataset.custom = '1';
                    avatarModelGroup.appendChild(customBtn);
                }
                if (customBtn) {
                    customBtn.dataset.avatarModel = customAvatarBlobUrl;
                    const name = file.name.replace(/\.(vrm|glb)$/i, '');
                    customBtn.textContent = `★ ${name.length > 10 ? name.slice(0, 10) + '…' : name}`;
                    customBtn.title = name;
                }
                loadCustomBtn.textContent = 'Load VRM';
            } else {
                currentAvatarUrl = prevUrl;
            }
            syncAvatarModelButtons();
            avatarFileInput.value = '';
        });
    }

    const canvas = document.getElementById('output_canvas');

    window.triggerGesture = (name) => {
        currentGesture = name;
        gestureTimer = 0;
        console.log("Gesture manually triggered:", name);
    };

    // Screen capture & recording buttons
    setupScreenCaptureControls();
    setupScreenZoomAndPan();

    // Unified dialogue system
    setupDialogue();

    // Drawing annotation system
    setupDrawing();

    // 키보드 단축키: Escape로 녹화 중지 + 드로잉 단축키
    document.addEventListener('keydown', (e) => {
        // Escape: stop recording
        if (e.key === 'Escape' && mediaRecorder && mediaRecorder.state === 'recording') {
            stopRecording();
        }

        // Skip drawing shortcuts if any input/textarea is focused (except drawing-text-input)
        const activeEl = document.activeElement;
        const isInputFocused = activeEl && (
            (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA' || activeEl.tagName === 'SELECT') &&
            activeEl.id !== 'drawing-text-input'
        );
        if (isInputFocused) return;

        // B: blackout toggle
        if (e.key === 'b' || e.key === 'B') {
            toggleBlackout();
            return;
        }

        // Drawing mode must be on for these
        if (!isDrawModeOn) return;

        // C: clear
        if (e.key === 'c' || e.key === 'C') {
            drawClear();
            return;
        }

        // Undo: Cmd/Ctrl+Z
        if ((e.metaKey || e.ctrlKey) && (e.key === 'z' || e.key === 'Z')) {
            e.preventDefault();
            drawUndo();
            return;
        }

        // Tool shortcuts (not when text composing)
        if (textComposingEl) {
            // Font size with [ ] when text tool active and not composing...
            // Actually [ ] are only active when textarea is NOT focused
            return;
        }

        if (e.key === 'p' || e.key === 'P') {
            setDrawTool('pen');
        } else if (e.key === 'h' || e.key === 'H') {
            setDrawTool('highlighter');
        } else if (e.key === 'a' || e.key === 'A') {
            setDrawTool('arrow');
        } else if (e.key === 't' || e.key === 'T') {
            setDrawTool('text');
        } else if (e.key === 'e' || e.key === 'E') {
            setDrawTool('eraser');
        } else if (e.code === 'Space') {
            e.preventDefault();
            isDrawPanMode = true;
            document.body.classList.add('draw-pan-mode');
        }
    });

    document.addEventListener('keyup', (e) => {
        if (e.code === 'Space' && isDrawPanMode) {
            isDrawPanMode = false;
            document.body.classList.remove('draw-pan-mode');
        }
    });

    // 초기 오디오 미터 상태
    const micMeter = document.getElementById('mic-meter');
    const tabMeter = document.getElementById('tab-meter');
    if (micMeter) micMeter.classList.add('inactive');
    if (tabMeter) tabMeter.classList.add('inactive');

    setupScene(canvas);

    debugCanvas = document.getElementById('debug_canvas');
    if (debugCanvas) {
        debugCanvas.width = VIDEO_WIDTH;
        debugCanvas.height = VIDEO_HEIGHT;
        debugCtx = debugCanvas.getContext('2d');
        drawingUtils = new DrawingUtils(debugCtx);
    }

    setupWebcam();
    setupMediaPipe();
    isAvatarLoading = true;
    loadAvatar().finally(() => { isAvatarLoading = false; });

    // 탭 전환/최소화 시 처리
    document.addEventListener('visibilitychange', () => {
        if (document.hidden) {
            console.log('[App] Tab hidden');
            // 녹화 중이 아니면 리소스 절약 가능 (현재는 로그만)
        } else {
            console.log('[App] Tab visible');
            // AudioContext가 suspended 상태일 수 있으므로 resume
            if (meterAudioContext && meterAudioContext.state === 'suspended') {
                meterAudioContext.resume();
            }
        }
    });

    // 창 크기 변경 시 미니 아바타 및 대화창 위치 보정
    window.addEventListener('resize', () => {
        if (isMiniAvatar && miniAvatarPosition.x !== null) {
            const miniWidth = 300;
            const miniHeight = 400;

            // 창 크기를 벗어나지 않도록 클램프
            const maxX = window.innerWidth - miniWidth;
            const maxY = window.innerHeight - miniHeight;

            if (miniAvatarPosition.x > maxX || miniAvatarPosition.y > maxY) {
                miniAvatarPosition.x = Math.max(0, Math.min(miniAvatarPosition.x, maxX));
                miniAvatarPosition.y = Math.max(0, Math.min(miniAvatarPosition.y, maxY));

                const sceneWrapper = document.getElementById('scene-wrapper');
                if (sceneWrapper) {
                    sceneWrapper.style.left = miniAvatarPosition.x + 'px';
                    sceneWrapper.style.top = miniAvatarPosition.y + 'px';
                }
            }

            // 대화창 위치도 업데이트
            updateDialogueOverlayPosition();
        }
    });

    animate();
}

// ============================================================
// Screen Capture & Recording
// ============================================================
function setupScreenCaptureControls() {
    const toggleScreenBtn = document.getElementById('toggle-screen');
    const toggleMicBtn = document.getElementById('toggle-mic');
    const toggleRecordBtn = document.getElementById('toggle-record');
    const toggleCameraBtn = document.getElementById('toggle-camera');

    if (toggleScreenBtn) {
        toggleScreenBtn.addEventListener('click', toggleScreenCapture);
    }
    setupAvatarControls();
    if (toggleMicBtn) {
        toggleMicBtn.addEventListener('click', toggleMicrophone);
    }
    if (toggleRecordBtn) {
        toggleRecordBtn.addEventListener('click', toggleRecording);
    }
    if (toggleCameraBtn) {
        toggleCameraBtn.addEventListener('click', toggleCamera);
    }

    // 오디오 믹스 슬라이더
    const mixSlider = document.getElementById('audio-mix-slider');
    if (mixSlider) {
        mixSlider.addEventListener('input', (e) => {
            audioMixValue = parseInt(e.target.value);
            updateAudioMix();
        });
    }
}

function updateAudioMix() {
    // 슬라이더 값: 0 = Mic 100%, Tab 0%
    //             50 = Mic 50%, Tab 50%
    //             100 = Mic 0%, Tab 100%
    const micVolume = (100 - audioMixValue) / 100;
    const tabVolume = audioMixValue / 100;

    if (micGainNode) {
        micGainNode.gain.value = micVolume;
    }
    if (tabGainNode) {
        tabGainNode.gain.value = tabVolume;
    }

}

// 마이크 토글 중복 방지 플래그
let isMicToggling = false;

// 마이크 토글
async function toggleMicrophone() {
    // 중복 호출 방지
    if (isMicToggling) {
        console.log('[Mic] Already toggling, skipping...');
        return;
    }
    isMicToggling = true;

    console.log('[Mic] toggleMicrophone called, current state:', isMicEnabled);
    const btn = document.getElementById('toggle-mic');
    const micMeter = document.getElementById('mic-meter');

    try {
    if (isMicEnabled) {
        // 마이크 비활성화
        console.log('[Mic] Disabling microphone...');

        // 음성 인식이 실행 중이면 중지
        if (isCaptionsEnabled) {
            console.log('[Mic] Stopping captions due to mic off');
            stopCaptions();
        }

        if (micStream) {
            micStream.getTracks().forEach(track => track.stop());
            micStream = null;
        }
        isMicEnabled = false;
        micAnalyser = null;
        if (btn) {
            btn.innerHTML = 'Mic<br>OFF';
            btn.classList.remove('mic-active');
        }
        if (micMeter) {
            micMeter.classList.add('inactive');
        }
        updateAudioMeters();
        console.log('[Mic] Microphone disabled');
    } else {
        // 마이크 활성화
        console.log('[Mic] Enabling microphone...');
        try {
            micStream = await navigator.mediaDevices.getUserMedia({ audio: true });

            // 외부에서 마이크가 종료되면 (다른 앱이 점유, 장치 분리 등)
            micStream.getAudioTracks().forEach(track => {
                track.onended = () => {
                    console.warn('[Mic] Microphone track ended externally');
                    if (isMicEnabled) {
                        isMicEnabled = false;
                        micStream = null;
                        micAnalyser = null;
                        if (btn) {
                            btn.innerHTML = 'Mic<br>OFF';
                            btn.classList.remove('mic-active');
                        }
                        if (micMeter) {
                            micMeter.classList.add('inactive');
                        }
                        if (isCaptionsEnabled) {
                            stopCaptions();
                        }
                        updateAudioMeters();
                    }
                };
            });

            isMicEnabled = true;
            if (btn) {
                btn.innerHTML = 'Mic<br>ON';
                btn.classList.add('mic-active');
            }
            if (micMeter) {
                micMeter.classList.remove('inactive');
            }
            // 마이크 레벨 미터 설정
            await setupMicMeter();
            console.log('[Mic] Microphone enabled');
        } catch (err) {
            console.error('[Mic] Microphone access error:', err.name, err.message);
            if (err.name === 'NotFoundError') {
                alert('마이크를 찾을 수 없습니다. 다른 탭을 닫고 브라우저를 재시작해보세요.');
            } else if (err.name === 'NotAllowedError') {
                alert('마이크 접근 권한이 필요합니다.');
            } else {
                alert('마이크 접근 실패: ' + err.message);
            }
        }
    }
    } finally {
        isMicToggling = false;
    }
}

// 오디오 레벨 미터 설정
async function setupMicMeter() {
    if (!micStream) return;

    if (!meterAudioContext) {
        meterAudioContext = new AudioContext();
    }

    // AudioContext가 suspended 상태이면 resume
    if (meterAudioContext.state === 'suspended') {
        await meterAudioContext.resume();
    }

    micAnalyser = meterAudioContext.createAnalyser();
    micAnalyser.fftSize = 256;
    micAnalyser.smoothingTimeConstant = 0.3;

    const source = meterAudioContext.createMediaStreamSource(micStream);
    source.connect(micAnalyser);

    startMeterAnimation();
}

async function setupTabAudioMeter() {
    if (!screenStream || screenStream.getAudioTracks().length === 0) {
        const tabMeter = document.getElementById('tab-meter');
        if (tabMeter) tabMeter.classList.add('inactive');
        return;
    }

    if (!meterAudioContext) {
        meterAudioContext = new AudioContext();
    }

    if (meterAudioContext.state === 'suspended') {
        await meterAudioContext.resume();
    }

    tabAnalyser = meterAudioContext.createAnalyser();
    tabAnalyser.fftSize = 256;
    tabAnalyser.smoothingTimeConstant = 0.3;

    const audioTracks = screenStream.getAudioTracks();
    const source = meterAudioContext.createMediaStreamSource(
        new MediaStream(audioTracks)
    );
    source.connect(tabAnalyser);

    const tabMeter = document.getElementById('tab-meter');
    if (tabMeter) tabMeter.classList.remove('inactive');

    startMeterAnimation();
}

function startMeterAnimation() {
    if (meterAnimationId) return;  // 이미 실행 중
    let lastMeterUpdate = 0;

    function updateMeters() {
        const now = performance.now();
        const minInterval = isRecordingActive() ? 100 : 0; // 녹화 중 UI 미터는 10Hz로 충분
        if (now - lastMeterUpdate >= minInterval) {
            updateAudioMeters();
            lastMeterUpdate = now;
        }
        meterAnimationId = requestAnimationFrame(updateMeters);
    }
    updateMeters();
}

function stopMeterAnimation() {
    if (meterAnimationId) {
        cancelAnimationFrame(meterAnimationId);
        meterAnimationId = null;
    }
}

function updateAudioMeters() {
    // 현재 믹스 비율 계산
    const micGain = (100 - audioMixValue) / 100;
    const tabGain = audioMixValue / 100;

    // 마이크 레벨 (gain 적용)
    const micMeter = document.getElementById('mic-meter');
    if (micMeter && micAnalyser) {
        const rawLevel = getAudioLevel(micAnalyser);
        const adjustedLevel = rawLevel * micGain;
        const bar = micMeter.querySelector('.audio-meter-bar');
        if (bar) bar.style.width = adjustedLevel + '%';
    }

    // 탭 오디오 레벨 (gain 적용)
    const tabMeter = document.getElementById('tab-meter');
    if (tabMeter && tabAnalyser) {
        const rawLevel = getAudioLevel(tabAnalyser);
        const adjustedLevel = rawLevel * tabGain;
        const bar = tabMeter.querySelector('.audio-meter-bar');
        if (bar) bar.style.width = adjustedLevel + '%';
    }

    // 둘 다 없으면 애니메이션 중지
    if (!micAnalyser && !tabAnalyser) {
        stopMeterAnimation();
    }
}

function getAudioLevel(analyser) {
    const dataArray = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(dataArray);

    // 평균 볼륨 계산
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i];
    }
    const average = sum / dataArray.length;

    // 0-100% 범위로 변환 (감도 조정)
    return Math.min(100, average * 1.5);
}

function toggleAvatarSize() {
    isMiniAvatar = !isMiniAvatar;

    const sceneWrapper = document.getElementById('scene-wrapper');

    if (isMiniAvatar) {
        document.body.classList.add('mini-avatar');

        // 초기 위치 설정 (우하단)
        if (sceneWrapper) {
            if (miniAvatarPosition.x === null) {
                miniAvatarPosition.x = window.innerWidth - 320;
                miniAvatarPosition.y = window.innerHeight - 500;
            }
            sceneWrapper.style.left = miniAvatarPosition.x + 'px';
            sceneWrapper.style.top = miniAvatarPosition.y + 'px';

            // 드래그 이벤트 추가
            setupDragAndDrop(sceneWrapper);
        }
    } else {
        document.body.classList.remove('mini-avatar');

        // 드래그 이벤트 제거 및 위치 초기화
        if (sceneWrapper) {
            sceneWrapper.style.left = '';
            sceneWrapper.style.top = '';
            removeDragAndDrop(sceneWrapper);
        }
    }

    // 앵커 핸들 표시 업데이트
    updateAnchorVisibility();

    // 대화 오버레이 위치 업데이트
    updateDialogueOverlayPosition();

    // 렌더러 크기 업데이트
    setTimeout(() => {
        if (sceneWrapper && renderer && camera) {
            const newWidth = sceneWrapper.clientWidth;
            const newHeight = sceneWrapper.clientHeight;
            if (newWidth > 0 && newHeight > 0) {
                camera.aspect = newWidth / newHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(newWidth, newHeight);
            }
        }
    }, 50);
    syncAvatarOptionsUI();
}

function updateAnchorVisibility() {
    const anchor = document.getElementById('avatar-anchor');
    if (!anchor) return;
    const show = isMiniAvatar && !isAvatarVisible;
    if (show) {
        if (miniAvatarPosition.x !== null) {
            anchor.style.left = miniAvatarPosition.x + 'px';
            anchor.style.top  = miniAvatarPosition.y + 'px';
        }
        anchor.classList.add('visible');
        setupDragAndDrop(anchor);
    } else {
        anchor.classList.remove('visible');
        removeDragAndDrop(anchor);
    }
}

function toggleAvatarVisibility() {
    isAvatarVisible = !isAvatarVisible;

    const sceneWrapper = document.getElementById('scene-wrapper');
    if (sceneWrapper) {
        sceneWrapper.style.visibility = isAvatarVisible ? '' : 'hidden';
    }
    updateAnchorVisibility();
    syncAvatarOptionsUI();
}

function syncAvatarOptionsUI() {
    // Size 옵션 버튼 동기화
    document.querySelectorAll('[data-avatar-size]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.avatarSize === (isMiniAvatar ? 'mini' : 'full'));
    });
    // Pose 옵션 버튼 동기화
    document.querySelectorAll('[data-avatar-pose]').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.avatarPose === (BODY_TRACKING_ENABLED ? 'on' : 'off'));
    });
    // 메인 가시성 버튼 동기화
    const visBtn = document.getElementById('toggle-avatar-visibility');
    if (visBtn) {
        visBtn.innerHTML = isAvatarVisible ? 'Avatar<br>ON' : 'Avatar<br>OFF';
        visBtn.classList.toggle('active', !isAvatarVisible);
    }
}

function setupAvatarControls() {
    // 메인 버튼: 가시성 토글
    const visBtn = document.getElementById('toggle-avatar-visibility');
    if (visBtn) visBtn.addEventListener('click', toggleAvatarVisibility);

    // Size 옵션 버튼
    document.querySelectorAll('[data-avatar-size]').forEach(btn => {
        btn.addEventListener('click', () => {
            const shouldBeMini = btn.dataset.avatarSize === 'mini';
            if (isMiniAvatar !== shouldBeMini) toggleAvatarSize();
            else syncAvatarOptionsUI(); // 이미 같은 상태면 UI만 동기화
        });
    });

    // Pose 옵션 버튼
    document.querySelectorAll('[data-avatar-pose]').forEach(btn => {
        btn.addEventListener('click', () => {
            const enable = btn.dataset.avatarPose === 'on';
            if (BODY_TRACKING_ENABLED !== enable) {
                BODY_TRACKING_ENABLED = enable;
                if (!BODY_TRACKING_ENABLED) {
                    leftArmActive = false;
                    rightArmActive = false;
                }
            }
            syncAvatarOptionsUI();
        });
    });

    // View (OrbitControls) 버튼
    document.getElementById('orbit-controls-on')?.addEventListener('click', () => setOrbitEnabled(true));
    document.getElementById('orbit-controls-off')?.addEventListener('click', () => setOrbitEnabled(false));
}

function setOrbitEnabled(enable) {
    isOrbitEnabled = enable;
    if (orbitControls) {
        orbitControls.enabled = enable && !isDebugView;
    }
    document.getElementById('orbit-controls-on')?.classList.toggle('active', enable);
    document.getElementById('orbit-controls-off')?.classList.toggle('active', !enable);
}

// 드래그&드롭 기능
let isDragging = false;
let dragOffset = { x: 0, y: 0 };

function setupDragAndDrop(element) {
    element.addEventListener('mousedown', onDragStart);
    element.addEventListener('touchstart', onDragStart, { passive: false });
}

function removeDragAndDrop(element) {
    element.removeEventListener('mousedown', onDragStart);
    element.removeEventListener('touchstart', onDragStart);
}

function onDragStart(e) {
    if (!isMiniAvatar) return;

    isDragging = true;
    const sceneWrapper = document.getElementById('scene-wrapper');

    const clientX = e.type === 'touchstart' ? e.touches[0].clientX : e.clientX;
    const clientY = e.type === 'touchstart' ? e.touches[0].clientY : e.clientY;

    dragOffset.x = clientX - sceneWrapper.offsetLeft;
    dragOffset.y = clientY - sceneWrapper.offsetTop;

    document.addEventListener('mousemove', onDragMove);
    document.addEventListener('mouseup', onDragEnd);
    document.addEventListener('touchmove', onDragMove, { passive: false });
    document.addEventListener('touchend', onDragEnd);

    e.preventDefault();
}

function onDragMove(e) {
    if (!isDragging) return;

    const clientX = e.type === 'touchmove' ? e.touches[0].clientX : e.clientX;
    const clientY = e.type === 'touchmove' ? e.touches[0].clientY : e.clientY;

    const sceneWrapper = document.getElementById('scene-wrapper');
    if (!sceneWrapper) return;

    // 새 위치 계산
    let newX = clientX - dragOffset.x;
    let newY = clientY - dragOffset.y;

    // 화면 경계 체크
    const maxX = window.innerWidth - sceneWrapper.offsetWidth;
    const maxY = window.innerHeight - sceneWrapper.offsetHeight;

    newX = Math.max(0, Math.min(newX, maxX));
    newY = Math.max(0, Math.min(newY, maxY));

    sceneWrapper.style.left = newX + 'px';
    sceneWrapper.style.top = newY + 'px';

    // 위치 저장
    miniAvatarPosition.x = newX;
    miniAvatarPosition.y = newY;

    // 앵커 핸들 위치 동기화
    const anchor = document.getElementById('avatar-anchor');
    if (anchor && anchor.classList.contains('visible')) {
        anchor.style.left = newX + 'px';
        anchor.style.top  = newY + 'px';
    }

    // 오버레이 위치 업데이트
    updateDialogueOverlayPosition();
    updateCaptionOverlayPosition();

    e.preventDefault();
}

function onDragEnd() {
    isDragging = false;
    document.removeEventListener('mousemove', onDragMove);
    document.removeEventListener('mouseup', onDragEnd);
    document.removeEventListener('touchmove', onDragMove);
    document.removeEventListener('touchend', onDragEnd);
}

// ============================================================
// Screen Zoom
// ============================================================
function getScreenVideoRenderRect(video) {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const ew = video.offsetWidth;
    const eh = video.offsetHeight;
    if (!vw || !vh || !ew || !eh) return null;
    const videoAspect = vw / vh;
    const elemAspect = ew / eh;
    let rx, ry, rw, rh;
    if (videoAspect > elemAspect) {
        rw = ew; rh = ew / videoAspect;
        rx = 0; ry = (eh - rh) / 2;
    } else {
        rh = eh; rw = eh * videoAspect;
        rx = (ew - rw) / 2; ry = 0;
    }
    return { rx, ry, rw, rh };
}

function applyScreenZoom() {
    const screenBg = document.getElementById('screen-background');
    if (!screenBg) return;
    if (screenZoom <= 1) {
        screenBg.style.transform = '';
        screenBg.style.transformOrigin = '';
        document.body.classList.remove('screen-zoomed');
    } else {
        screenBg.style.transformOrigin = '0 0';
        screenBg.style.transform = `translate(${screenZoomTx}px, ${screenZoomTy}px) scale(${screenZoom})`;
        document.body.classList.add('screen-zoomed');
    }
    updateZoomIndicator();
}

function resetScreenZoom() {
    screenZoom = 1;
    screenZoomTx = 0;
    screenZoomTy = 0;
    applyScreenZoom();
}

function updateZoomIndicator() {
    const indicator = document.getElementById('screen-zoom-indicator');
    if (!indicator) return;
    if (screenZoom > 1) {
        indicator.textContent = `${screenZoom.toFixed(1)}×`;
        indicator.style.opacity = '1';
        clearTimeout(zoomIndicatorTimer);
        zoomIndicatorTimer = setTimeout(() => {
            // 줌 해제 후: 해상도 표시 또는 숨김
            if (capturedScreenWidth > 0) {
                indicator.textContent = `${capturedScreenWidth}×${capturedScreenHeight}`;
                indicator.style.opacity = '0.75';
            } else {
                indicator.style.opacity = '0';
            }
        }, 1500);
    } else if (capturedScreenWidth > 0) {
        // 스크린 캡쳐 중: 캡쳐 해상도 표시
        indicator.textContent = `${capturedScreenWidth}×${capturedScreenHeight}`;
        indicator.style.opacity = '0.75';
    } else {
        indicator.style.opacity = '0';
    }
}

// 캡쳐 해상도에 맞춰 Three.js 렌더러 크기 적응 (세로 모드 대응)
function adaptRendererToCapture() {
    if (!capturedScreenWidth || !capturedScreenHeight) return;
    if (!renderer || !camera) return;

    const isPortrait = capturedScreenHeight > capturedScreenWidth;
    if (!isPortrait) return;  // 가로 모드는 변경 없음

    // 미니 아바타 모드는 이미 300×400 (3:4) portrait canvas → 불필요
    if (isMiniAvatar) return;

    const sceneWrapper = document.getElementById('scene-wrapper');
    if (!sceneWrapper) return;

    // 현재 크기 저장
    prevRendererSize = {
        cssWidth: sceneWrapper.style.width,
        cssHeight: sceneWrapper.style.height,
    };

    // 브라우저 창 높이 기준으로 세로 아바타 캔버스 크기 계산
    const aspect = capturedScreenWidth / capturedScreenHeight;
    const targetHeight = Math.min(window.innerHeight * 0.85, 720);
    const targetWidth = Math.round(targetHeight * aspect);

    sceneWrapper.style.width = targetWidth + 'px';
    sceneWrapper.style.height = targetHeight + 'px';

    camera.aspect = aspect;
    camera.updateProjectionMatrix();
    renderer.setSize(targetWidth, targetHeight);

    console.log(`[Screen] Renderer adapted to portrait: ${targetWidth}×${targetHeight} (${capturedScreenWidth}×${capturedScreenHeight})`);
}

// 스크린 캡쳐 종료 시 렌더러 크기 복원
function restoreRendererFromCapture() {
    if (!prevRendererSize || !renderer || !camera) {
        prevRendererSize = null;
        return;
    }

    const sceneWrapper = document.getElementById('scene-wrapper');
    if (!sceneWrapper) { prevRendererSize = null; return; }

    sceneWrapper.style.width = prevRendererSize.cssWidth || '';
    sceneWrapper.style.height = prevRendererSize.cssHeight || '';

    // ResizeObserver가 자동 업데이트하지만 즉시 적용도 보장
    setTimeout(() => {
        if (!sceneWrapper || !renderer || !camera) return;
        const w = sceneWrapper.clientWidth;
        const h = sceneWrapper.clientHeight;
        if (w > 0 && h > 0) {
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        }
    }, 100);

    prevRendererSize = null;
    console.log('[Screen] Renderer size restored');
}

function setupScreenZoomAndPan() {
    const container = document.getElementById('preview-container');
    const screenBg = document.getElementById('screen-background');
    if (!container || !screenBg) return;

    // 마우스 휠 / 트랙패드 핀치줌 (ctrlKey)
    container.addEventListener('wheel', (e) => {
        if (!screenStream) return;
        e.preventDefault();

        const rect = container.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const W = rect.width;
        const H = rect.height;

        const zoomFactor = e.ctrlKey
            ? Math.pow(0.98, e.deltaY)   // 핀치줌: 부드럽게
            : e.deltaY < 0 ? 1.12 : 1 / 1.12; // 마우스 휠

        const newZoom = Math.max(1, Math.min(8, screenZoom * zoomFactor));

        const ratio = newZoom / screenZoom;
        const newTx = mx * (1 - ratio) + screenZoomTx * ratio;
        const newTy = my * (1 - ratio) + screenZoomTy * ratio;

        screenZoom = newZoom;
        screenZoomTx = Math.max(W * (1 - screenZoom), Math.min(0, newTx));
        screenZoomTy = Math.max(H * (1 - screenZoom), Math.min(0, newTy));

        applyScreenZoom();
    }, { passive: false });

    // 드래그 패닝 (줌 상태에서)
    screenBg.addEventListener('mousedown', (e) => {
        if (!screenStream || screenZoom <= 1) return;
        isScreenPanning = true;
        screenPanStartX = e.clientX;
        screenPanStartY = e.clientY;
        screenZoomTxStart = screenZoomTx;
        screenZoomTyStart = screenZoomTy;
        document.body.classList.add('screen-panning');
        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isScreenPanning) return;
        const W = window.innerWidth;
        const H = window.innerHeight;
        const newTx = screenZoomTxStart + (e.clientX - screenPanStartX);
        const newTy = screenZoomTyStart + (e.clientY - screenPanStartY);
        screenZoomTx = Math.max(W * (1 - screenZoom), Math.min(0, newTx));
        screenZoomTy = Math.max(H * (1 - screenZoom), Math.min(0, newTy));
        applyScreenZoom();
    });

    document.addEventListener('mouseup', () => {
        if (!isScreenPanning) return;
        isScreenPanning = false;
        document.body.classList.remove('screen-panning');
    });

    // 더블클릭으로 줌 리셋
    screenBg.addEventListener('dblclick', () => {
        if (!screenStream) return;
        resetScreenZoom();
    });
}

async function startScreenCapture() {
    try {
        // 화면 공유 요청 (오디오 포함 시도)
        screenStream = await navigator.mediaDevices.getDisplayMedia({
            video: { cursor: "always" },
            audio: true  // 탭/앱 오디오 요청
        });

        // DOM 비디오 엘리먼트에 스트림 연결
        screenVideo = document.getElementById('screen-background');
        screenVideo.srcObject = screenStream;
        screenVideo.play();

        // 캡쳐된 화면 해상도 감지 (loadedmetadata 또는 이미 준비된 경우 즉시)
        const detectCapturedResolution = () => {
            if (screenVideo.videoWidth > 0) {
                capturedScreenWidth = screenVideo.videoWidth;
                capturedScreenHeight = screenVideo.videoHeight;
                console.log(`[Screen] Captured resolution: ${capturedScreenWidth}×${capturedScreenHeight}`);
                adaptRendererToCapture();
                updateZoomIndicator();
            }
        };
        screenVideo.addEventListener('loadedmetadata', detectCapturedResolution, { once: true });
        if (screenVideo.videoWidth > 0) detectCapturedResolution();

        // 탭 오디오 여부 확인 및 레벨 미터 설정
        const hasTabAudio = screenStream.getAudioTracks().length > 0;
        if (hasTabAudio) {
            setupTabAudioMeter();
        } else {
            const tabMeter = document.getElementById('tab-meter');
            if (tabMeter) tabMeter.classList.add('inactive');
        }

        // 비디오 트랙 종료 감지
        screenStream.getVideoTracks()[0].onended = () => {
            console.log('[Screen] Video track ended');
            stopScreenCapture();
        };

        // 오디오 트랙 종료 감지 (별도로 종료될 수 있음)
        screenStream.getAudioTracks().forEach(track => {
            track.onended = () => {
                console.log('[Screen] Audio track ended');
                tabAnalyser = null;
                const tabMeter = document.getElementById('tab-meter');
                if (tabMeter) {
                    tabMeter.classList.add('inactive');
                    const bar = tabMeter.querySelector('.audio-meter-bar');
                    if (bar) bar.style.width = '0%';
                }
            };
        });

        // 카메라 프리뷰 숨기기 & 화면 공유 모드 활성화
        document.body.classList.add('screen-sharing');

        // 기본 Mini Avatar 모드로 전환
        if (!isMiniAvatar) {
            toggleAvatarSize();
        }

        // 버튼 상태 업데이트
        updateScreenCaptureButtons(true);

    } catch (err) {
        console.error("[Screen] Screen capture error:", err.name, err.message);
        if (err.name === 'NotAllowedError') {
            // 사용자가 취소하거나 권한 거부
            console.log('[Screen] User cancelled or permission denied');
        } else if (err.name === 'NotFoundError') {
            alert('화면 공유를 사용할 수 없습니다.');
        } else if (err.name === 'NotSupportedError') {
            alert('이 브라우저는 화면 공유를 지원하지 않습니다.');
        } else {
            alert('화면 공유 오류: ' + err.message);
        }
    }
}

function stopScreenCapture() {
    // 녹화 중이면 먼저 중지
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
    }

    // 스트림 정리
    if (screenStream) {
        screenStream.getTracks().forEach(track => track.stop());
        screenStream = null;
    }

    // 비디오 엘리먼트 정리
    const screenBg = document.getElementById('screen-background');
    if (screenBg) {
        screenBg.srcObject = null;
    }
    screenVideo = null;

    // 탭 오디오 미터 정리
    tabAnalyser = null;
    const tabMeter = document.getElementById('tab-meter');
    if (tabMeter) {
        tabMeter.classList.add('inactive');
        const bar = tabMeter.querySelector('.audio-meter-bar');
        if (bar) bar.style.width = '0%';
    }

    // 줌 상태 리셋
    resetScreenZoom();

    // 캡쳐 해상도 리셋 및 렌더러 복원
    capturedScreenWidth = 0;
    capturedScreenHeight = 0;
    restoreRendererFromCapture();
    updateZoomIndicator();

    // 카메라 프리뷰 다시 보이기
    document.body.classList.remove('screen-sharing');

    // 미니 아바타 모드 리셋
    if (isMiniAvatar) {
        isMiniAvatar = false;
        document.body.classList.remove('mini-avatar');
        syncAvatarOptionsUI();

        const sceneWrapper = document.getElementById('scene-wrapper');
        if (sceneWrapper) {
            sceneWrapper.style.left = '';
            sceneWrapper.style.top = '';
            removeDragAndDrop(sceneWrapper);
        }

        // 렌더러 크기 복원
        setTimeout(() => {
            if (sceneWrapper && renderer && camera) {
                const newWidth = sceneWrapper.clientWidth;
                const newHeight = sceneWrapper.clientHeight;
                if (newWidth > 0 && newHeight > 0) {
                    camera.aspect = newWidth / newHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(newWidth, newHeight);
                }
            }
        }, 50);
    }

    // 버튼 상태 업데이트
    updateScreenCaptureButtons(false);
}

// 합성 캔버스 및 녹화 관련
let compositeCanvas = null;
let compositeCtx = null;
let compositeAnimationId = null;

function startRecording() {
    const avatarCanvas = document.getElementById('output_canvas');
    const screenBg = document.getElementById('screen-background');

    if (!avatarCanvas) return;

    recordedChunks = [];

    const hasScreenSource = !!(screenBg && screenBg.srcObject && screenBg.videoWidth && screenBg.videoHeight);
    const sourceWidth = hasScreenSource ? screenBg.videoWidth : RECORDING_AVATAR_WIDTH;
    const sourceHeight = hasScreenSource ? screenBg.videoHeight : RECORDING_AVATAR_HEIGHT;
    const recordingSize = getRecordingCanvasSize(sourceWidth, sourceHeight);

    if (hasScreenSource && (capturedScreenWidth !== sourceWidth || capturedScreenHeight !== sourceHeight)) {
        capturedScreenWidth = sourceWidth;
        capturedScreenHeight = sourceHeight;
        updateZoomIndicator();
    }

    // 합성 캔버스 생성 (DOM에 추가하여 captureStream 호환성 확보)
    // 캡처 타겟의 실제 aspect ratio는 유지하고, 너무 큰 경우에만 FHD급으로 축소한다.
    compositeCanvas = document.createElement('canvas');
    compositeCanvas.width = recordingSize.width;
    compositeCanvas.height = recordingSize.height;
    compositeCanvas.style.cssText = 'position:fixed;top:-9999px;left:-9999px;pointer-events:none;';
    document.body.appendChild(compositeCanvas);
    compositeCtx = compositeCanvas.getContext('2d');

    // requestFrame 지원 브라우저에서는 합성한 프레임만 캡처한다.
    let canvasStream = compositeCanvas.captureStream(0);
    let canvasVideoTrack = canvasStream.getVideoTracks()[0];
    let requestManualFrame = typeof canvasVideoTrack?.requestFrame === 'function'
        ? () => canvasVideoTrack.requestFrame()
        : null;
    if (!requestManualFrame) {
        canvasVideoTrack?.stop();
        canvasStream = compositeCanvas.captureStream(RECORDING_TARGET_FPS);
        canvasVideoTrack = canvasStream.getVideoTracks()[0];
    }
    let lastCompositeAt = 0;
    startRecordingStats(compositeCanvas.width, compositeCanvas.height, sourceWidth, sourceHeight);

    function drawCompositeFrame() {
        const frameStart = performance.now();
        // 1. 배경 그리기 (블랙아웃이면 검정, 아니면 화면 공유)
        compositeCtx.fillStyle = '#000';
        compositeCtx.fillRect(0, 0, compositeCanvas.width, compositeCanvas.height);
        if (!isBlackout && screenBg && screenBg.srcObject && screenBg.videoWidth) {
            if (screenZoom > 1) {
                // 줌 상태: 현재 보이는 영역만 크롭해서 캔버스 전체에 그리기
                const renderRect = getScreenVideoRenderRect(screenBg);
                if (renderRect) {
                    const { rx, ry, rw, rh } = renderRect;
                    const W = stableWindowWidth;
                    const H = stableWindowHeight;
                    const vw = screenBg.videoWidth;
                    const vh = screenBg.videoHeight;

                    const visLeftNorm  = Math.max(0, (-screenZoomTx / screenZoom - rx) / rw);
                    const visRightNorm = Math.min(1, ((W - screenZoomTx) / screenZoom - rx) / rw);
                    const visTopNorm   = Math.max(0, (-screenZoomTy / screenZoom - ry) / rh);
                    const visBottomNorm = Math.min(1, ((H - screenZoomTy) / screenZoom - ry) / rh);

                    const sx = visLeftNorm * vw;
                    const sy = visTopNorm * vh;
                    const sw = (visRightNorm - visLeftNorm) * vw;
                    const sh = (visBottomNorm - visTopNorm) * vh;

                    if (sw > 0 && sh > 0) {
                        compositeCtx.drawImage(screenBg, sx, sy, sw, sh, 0, 0, compositeCanvas.width, compositeCanvas.height);
                    }
                }
            } else if (hasScreenSource) {
                // 캡처 타겟의 현재 비디오 크기 전체를 녹화 캔버스에 맞춘다.
                // 캔버스도 같은 aspect ratio라 왜곡 없이 스케일만 적용된다.
                compositeCtx.drawImage(
                    screenBg,
                    0, 0, sourceWidth, sourceHeight,
                    0, 0, compositeCanvas.width, compositeCanvas.height
                );
            } else {
                // 스크린 캡쳐 없이 녹화 (아바타만): aspect-ratio 유지해서 중앙 정렬
                const videoAspect = screenBg.videoWidth / screenBg.videoHeight;
                const canvasAspect = compositeCanvas.width / compositeCanvas.height;
                let drawWidth, drawHeight, drawX, drawY;
                if (videoAspect > canvasAspect) {
                    drawWidth = compositeCanvas.width;
                    drawHeight = drawWidth / videoAspect;
                    drawX = 0;
                    drawY = (compositeCanvas.height - drawHeight) / 2;
                } else {
                    drawHeight = compositeCanvas.height;
                    drawWidth = drawHeight * videoAspect;
                    drawX = (compositeCanvas.width - drawWidth) / 2;
                    drawY = 0;
                }
                compositeCtx.drawImage(screenBg, drawX, drawY, drawWidth, drawHeight);
            }
        }

        // 2. 아바타 캔버스 그리기 (표시 상태일 때만)
        if (isAvatarVisible && isMiniAvatar) {
            // 미니 모드: 현재 위치에 맞춰 그리기
            const miniWidth = 300;
            const miniHeight = 400;

            // 안정 캐시 값 사용 (Dock 등 시스템 UI 변화로 인한 순간 출렁임 방지)
            const winW = stableWindowWidth;
            const winH = stableWindowHeight;

            // 현재 창 크기에 맞게 위치 클램프 (창 크기 변경 대응)
            const clampedX = Math.min(miniAvatarPosition.x || 0, winW - miniWidth);
            const clampedY = Math.min(miniAvatarPosition.y || 0, winH - miniHeight);
            const safeX = Math.max(0, clampedX);
            const safeY = Math.max(0, clampedY);

            let miniX, miniY, scaledWidth, scaledHeight;

            if (hasScreenSource) {
                // 스크린 캡쳐 중: 브라우저 뷰포트 좌표 → 녹화 캔버스 좌표로 변환
                const renderRect = getScreenVideoRenderRect(screenBg);
                if (renderRect) {
                    const { rx, ry, rw, rh } = renderRect;
                    const scaleX = compositeCanvas.width / rw;
                    const scaleY = compositeCanvas.height / rh;
                    scaledWidth = miniWidth * scaleX;
                    scaledHeight = miniHeight * scaleY;

                    const rightEdge = safeX + miniWidth;
                    const bottomEdge = safeY + miniHeight;
                    miniX = ((rightEdge - rx) / rw) * compositeCanvas.width - scaledWidth;
                    miniY = ((bottomEdge - ry) / rh) * compositeCanvas.height - scaledHeight;
                } else {
                    // 렌더 rect 없으면 비율 기반 fallback
                    const scaleX = compositeCanvas.width / winW;
                    const scaleY = compositeCanvas.height / winH;
                    scaledWidth = miniWidth * scaleX;
                    scaledHeight = miniHeight * scaleY;
                    miniX = ((safeX + miniWidth) / winW) * compositeCanvas.width - scaledWidth;
                    miniY = ((safeY + miniHeight) / winH) * compositeCanvas.height - scaledHeight;
                }
            } else {
                // 스크린 캡쳐 없음: 뷰포트 비율로 매핑 (기존 로직)
                const scaleX = compositeCanvas.width / winW;
                const scaleY = compositeCanvas.height / winH;
                const uniformScale = Math.min(scaleX, scaleY);
                scaledWidth = miniWidth * uniformScale;
                scaledHeight = miniHeight * uniformScale;
                const rightEdge = safeX + miniWidth;
                const bottomEdge = safeY + miniHeight;
                miniX = (rightEdge / winW) * compositeCanvas.width - scaledWidth;
                miniY = (bottomEdge / winH) * compositeCanvas.height - scaledHeight;
            }

            compositeCtx.drawImage(avatarCanvas, miniX, miniY, scaledWidth, scaledHeight);
        } else if (isAvatarVisible) {
            // 풀 모드: 비율 유지하며 하단 정렬 (프리뷰와 동일하게)
            const avatarAspect = avatarCanvas.width / avatarCanvas.height;
            const canvasAspect = compositeCanvas.width / compositeCanvas.height;
            let drawWidth, drawHeight, drawX, drawY;

            if (avatarAspect > canvasAspect) {
                // 아바타가 더 넓음 - 좌우 맞춤, 상단 여백 (하단 정렬)
                drawWidth = compositeCanvas.width;
                drawHeight = drawWidth / avatarAspect;
                drawX = 0;
                drawY = compositeCanvas.height - drawHeight;  // 하단 정렬
            } else {
                // 아바타가 더 높음 - 상하 맞춤, 좌우 중앙 정렬
                drawHeight = compositeCanvas.height;
                drawWidth = drawHeight * avatarAspect;
                drawX = (compositeCanvas.width - drawWidth) / 2;
                drawY = 0;
            }

            compositeCtx.drawImage(avatarCanvas, drawX, drawY, drawWidth, drawHeight);
        }

        // 3. 대화 메시지 그리기
        if (isDialogueEnabled && dialogueMessages.length > 0) {
            drawDialogueToCanvas(compositeCtx, compositeCanvas.width, compositeCanvas.height);
        }

        // 4. 드로잉 레이어 합성
        renderDrawingLayer(compositeCtx, compositeCanvas.width, compositeCanvas.height, performance.now());

        if (requestManualFrame) requestManualFrame();

        if (recordingStats) {
            const frameDuration = performance.now() - frameStart;
            recordingStats.compositeFrames += 1;
            if (frameDuration > RECORDING_FRAME_INTERVAL) recordingStats.longCompositeFrames += 1;
            reportRecordingStats();
        }
    }

    // 합성 루프 시작: 캡처 목표 FPS에 맞춰 실제 합성 작업을 제한한다.
    function compositeFrame(now) {
        if (!compositeCanvas || !compositeCtx) return;
        if (now - lastCompositeAt >= RECORDING_FRAME_INTERVAL) {
            lastCompositeAt = now;
            drawCompositeFrame();
        } else if (recordingStats) {
            recordingStats.compositeSkipped += 1;
        }
        compositeAnimationId = requestAnimationFrame(compositeFrame);
    }
    compositeAnimationId = requestAnimationFrame(compositeFrame);

    // 오디오 합성
    // AudioContext 생성
    audioContext = new AudioContext();
    audioDestination = audioContext.createMediaStreamDestination();

    // 볼륨 계산
    const micVolume = (100 - audioMixValue) / 100;
    const tabVolume = audioMixValue / 100;

    // 탭/앱 오디오 추가 (화면 공유에 오디오가 있는 경우)
    if (screenStream && screenStream.getAudioTracks().length > 0) {
        const tabAudioSource = audioContext.createMediaStreamSource(
            new MediaStream(screenStream.getAudioTracks())
        );
        tabGainNode = audioContext.createGain();
        tabGainNode.gain.value = tabVolume;
        tabAudioSource.connect(tabGainNode);
        tabGainNode.connect(audioDestination);
    }

    // 마이크 오디오 추가
    if (micStream && micStream.getAudioTracks().length > 0) {
        const micAudioSource = audioContext.createMediaStreamSource(micStream);
        micGainNode = audioContext.createGain();
        micGainNode.gain.value = micVolume;
        micAudioSource.connect(micGainNode);
        micGainNode.connect(audioDestination);
    }

    // 비디오 + 오디오 스트림 합성 (오디오 소스가 있을 때만 오디오 추가)
    const hasAudioSource = (micStream && micStream.getAudioTracks().length > 0) ||
                           (screenStream && screenStream.getAudioTracks().length > 0);

    const streamTracks = [...canvasStream.getVideoTracks()];
    if (hasAudioSource) {
        streamTracks.push(...audioDestination.stream.getAudioTracks());
    }
    const combinedStream = new MediaStream(streamTracks);

    // MediaRecorder 설정 - 오디오 유무에 따라 코덱 선택
    let mimeType;
    if (hasAudioSource) {
        mimeType = 'video/webm;codecs=vp8,opus';
        if (!MediaRecorder.isTypeSupported(mimeType)) {
            mimeType = 'video/webm;codecs=vp9,opus';
        }
    } else {
        mimeType = 'video/webm;codecs=vp8';
    }
    if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'video/webm';
    }

    mediaRecorder = new MediaRecorder(combinedStream, {
        mimeType,
        videoBitsPerSecond: getRecordingVideoBitsPerSecond(compositeCanvas.width, compositeCanvas.height),
    });

    mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
            recordedChunks.push(e.data);
            recordRecorderChunk(e.data.size);
        }
    };

    mediaRecorder.onerror = (event) => {
        console.error('[Recording] MediaRecorder error:', event.error);
        alert('녹화 중 오류가 발생했습니다: ' + (event.error?.message || 'Unknown error'));
        stopRecording();
    };

    mediaRecorder.onstop = () => {
        stopRecordingStats();

        // 합성 루프 중지
        if (compositeAnimationId) {
            cancelAnimationFrame(compositeAnimationId);
            compositeAnimationId = null;
        }
        // DOM에서 캔버스 제거
        if (compositeCanvas && compositeCanvas.parentNode) {
            compositeCanvas.parentNode.removeChild(compositeCanvas);
        }
        compositeCanvas = null;
        compositeCtx = null;

        // AudioContext 정리
        if (audioContext) {
            audioContext.close();
            audioContext = null;
            audioDestination = null;
            micGainNode = null;
            tabGainNode = null;
        }

        downloadRecording();
    };

    mediaRecorder.start(100);  // 100ms마다 데이터 수집

    _postToOpener({ type: 'avatar-recoder:recording-started', sessionId: _integrationParams?.sessionId ?? null });

    // 녹화 중 컨트롤바 숨기기
    document.body.classList.add('recording');

    // 버튼 상태 업데이트
    const toggleRecordBtn = document.getElementById('toggle-record');
    if (toggleRecordBtn) {
        toggleRecordBtn.innerHTML = 'Stop<br>Record';
        toggleRecordBtn.classList.add('recording');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        // 마지막 데이터를 강제로 수집한 후 중지
        mediaRecorder.requestData();
        mediaRecorder.stop();
    }

    // 녹화 중 컨트롤바 다시 보이기
    document.body.classList.remove('recording');

    // 버튼 상태 업데이트
    const toggleRecordBtn = document.getElementById('toggle-record');
    if (toggleRecordBtn) {
        toggleRecordBtn.innerHTML = 'Start<br>Record';
        toggleRecordBtn.classList.remove('recording');
    }
}

function downloadRecording() {
    if (recordedChunks.length === 0) {
        console.warn('[Recording] No recorded data available');
        return;
    }

    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    recordedChunks = [];

    if (isIntegrationMode && window.opener) {
        const filename = `avatar-recording-${Date.now()}.webm`;
        _postToOpener({
            type: 'avatar-recoder:result',
            sessionId: _integrationParams.sessionId,
            blob,
            mimeType: 'video/webm',
            filename,
        });
        _postToOpener({ type: 'avatar-recoder:recording-stopped', sessionId: _integrationParams.sessionId });
        return;
    }

    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `avatar-recording-${Date.now()}.webm`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function updateScreenCaptureButtons(isCapturing) {
    const toggleScreenBtn = document.getElementById('toggle-screen');
    const toggleDrawBtn = document.getElementById('toggle-draw');

    if (toggleScreenBtn) {
        toggleScreenBtn.innerHTML = isCapturing ? 'Stop<br>Capture' : 'Screen<br>Capture';
        if (isCapturing) {
            toggleScreenBtn.classList.add('mic-active');
        } else {
            toggleScreenBtn.classList.remove('mic-active');
        }
    }

    // When screen capture stops, turn off blackout and clear drawings
    if (!isCapturing) {
        if (isBlackout) toggleBlackout(false);
        drawClear();
    }
}

// Screen capture toggle
function toggleScreenCapture() {
    if (screenStream) {
        stopScreenCapture();
    } else {
        startScreenCapture();
    }
}

// Recording toggle
function toggleRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
    } else {
        startRecording();
    }
}

function updateView() {
    if (isDebugView) {
        document.body.classList.add('debug-view');
    } else {
        document.body.classList.remove('debug-view');
    }
}

function updateDevOptions() {
    // Update Debug option buttons
    document.querySelectorAll('.option-btn[data-dev="debug-on"]').forEach(btn => {
        btn.classList.toggle('active', isDebugView);
    });
    document.querySelectorAll('.option-btn[data-dev="debug-off"]').forEach(btn => {
        btn.classList.toggle('active', !isDebugView);
    });

    // Update Landmarks option buttons
    document.querySelectorAll('.option-btn[data-dev="landmarks-on"]').forEach(btn => {
        btn.classList.toggle('active', DEBUG_MODE);
    });
    document.querySelectorAll('.option-btn[data-dev="landmarks-off"]').forEach(btn => {
        btn.classList.toggle('active', !DEBUG_MODE);
    });
}

function setupScene(canvas) {
    const sceneWrapper = document.getElementById('scene-wrapper');
    const width = sceneWrapper ? sceneWrapper.clientWidth : window.innerWidth / 2;
    const height = sceneWrapper ? sceneWrapper.clientHeight : window.innerHeight;

    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(30.0, width / height, 0.1, 20.0);
    camera.position.set(0.0, 1.7, 1.5);

    const light = new THREE.DirectionalLight(0xffffff, 1.0);
    light.position.set(1.0, 1.0, 1.0).normalize();
    scene.add(light);

    renderer = new THREE.WebGLRenderer({ canvas: canvas, alpha: true, antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);

    // WebGL 컨텍스트 손실 처리
    canvas.addEventListener('webglcontextlost', (event) => {
        event.preventDefault();
        console.error('[WebGL] Context lost');
        alert('그래픽 컨텍스트가 손실되었습니다. 페이지를 새로고침해주세요.');
    });

    canvas.addEventListener('webglcontextrestored', () => {
        console.log('[WebGL] Context restored');
    });

    // OrbitControls 초기화 (기본 비활성)
    orbitControls = new OrbitControls(camera, canvas);
    orbitControls.target.set(0, 1.4, 0);
    orbitControls.enableDamping = true;
    orbitControls.dampingFactor = 0.08;
    orbitControls.enabled = false;
    orbitControls.update();

    // IME 전환 등 순간적 레이아웃 변화에 의한 renderer 리사이즈 방지 (debounce 150ms)
    let rendererResizeTimer = null;
    const resizeObserver = new ResizeObserver(() => {
        if (!sceneWrapper) return;
        clearTimeout(rendererResizeTimer);
        rendererResizeTimer = setTimeout(() => {
            const newWidth = sceneWrapper.clientWidth;
            const newHeight = sceneWrapper.clientHeight;
            if (newWidth > 0 && newHeight > 0) {
                camera.aspect = newWidth / newHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(newWidth, newHeight);
            }
        }, 150);
    });
    if (sceneWrapper) resizeObserver.observe(sceneWrapper);
}

async function setupWebcam() {
    video = document.getElementById('webcam');
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: VIDEO_WIDTH, height: VIDEO_HEIGHT }
        });
        video.srcObject = webcamStream;
        await new Promise((resolve) => {
            video.onloadedmetadata = () => {
                resolve(video);
            };
        });
        video.play();

        // 외부에서 카메라가 종료되면 (다른 앱이 점유, 장치 분리 등)
        webcamStream.getVideoTracks().forEach(track => {
            track.onended = () => {
                console.warn('[Camera] Camera track ended externally');
                if (isCameraEnabled) {
                    isCameraEnabled = false;
                    webcamStream = null;
                    if (video) video.srcObject = null;
                    updateCameraButton();
                }
            };
        });

        isCameraEnabled = true;
        updateCameraButton();
    } catch (err) {
        console.error("Error accessing webcam:", err);
    }
}

// 카메라 토글
async function toggleCamera() {
    const btn = document.getElementById('toggle-camera');

    if (isCameraEnabled) {
        // 카메라 비활성화
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        if (video) {
            video.srcObject = null;
        }
        isCameraEnabled = false;
    } else {
        // 카메라 활성화
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: VIDEO_WIDTH, height: VIDEO_HEIGHT }
            });

            // 외부에서 카메라가 종료되면
            webcamStream.getVideoTracks().forEach(track => {
                track.onended = () => {
                    console.warn('[Camera] Camera track ended externally');
                    if (isCameraEnabled) {
                        isCameraEnabled = false;
                        webcamStream = null;
                        if (video) video.srcObject = null;
                        updateCameraButton();
                    }
                };
            });

            if (video) {
                video.srcObject = webcamStream;
                video.play();
            }
            isCameraEnabled = true;
        } catch (err) {
            console.error("Camera access error:", err);
            alert('카메라 접근 권한이 필요합니다.');
            return;
        }
    }
    updateCameraButton();
}

function updateCameraButton() {
    const btn = document.getElementById('toggle-camera');
    if (btn) {
        btn.innerHTML = isCameraEnabled ? 'Cam<br>ON' : 'Cam<br>OFF';
        btn.classList.toggle('camera-active', isCameraEnabled);
    }
}

async function setupMediaPipe() {
    try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm"
        );

        faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
                delegate: "GPU"
            },
            outputFaceBlendshapes: true,
            outputFacialTransformationMatrixes: true,
            runningMode: "VIDEO",
            numFaces: 1
        });

        // 기본 lite (full은 실시간 구동이 어려울 만큼 느림 — 전체 앱 프레임 저하 확인됨)
        // 정확도 실험용으로 ?poseModel=full|heavy URL 파라미터 지원
        const poseModelParam = new URLSearchParams(window.location.search).get('poseModel');
        const poseModel = (poseModelParam === 'full' || poseModelParam === 'heavy')
            ? `pose_landmarker_${poseModelParam}`
            : 'pose_landmarker_lite';
        poseLandmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/${poseModel}/float16/1/${poseModel}.task`,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numPoses: 1
        });

        handLandmarker = await HandLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numHands: 2
        });

        console.log("MediaPipe (Face, Pose, Hand) initialized");
    } catch (err) {
        console.error("MediaPipe init error:", err);
        alert('MediaPipe 초기화 실패. 인터넷 연결을 확인하고 페이지를 새로고침해주세요.');
    }
}

async function loadAvatar(url = './avatar.vrm') {
    const loader = new GLTFLoader();
    loader.register((parser) => {
        return new VRMLoaderPlugin(parser);
    });

    try {
        const gltf = await loader.loadAsync(url);
        const vrm = gltf.userData.vrm;

        VRMUtils.removeUnnecessaryVertices(gltf.scene);
        VRMUtils.removeUnnecessaryJoints(gltf.scene);
        VRMUtils.rotateVRM0(vrm);

        scene.add(vrm.scene);
        currentVrm = vrm;
        currentAvatarUrl = url;

        // 모델별 신장 차이 보정: 머리 본이 기준 높이(AVATAR_HEAD_HEIGHT)에 오도록 수직 오프셋
        // (남성 모델이 여성보다 ~17cm 커서 얼굴이 위에 보이던 문제 — 스케일 조정은
        //  다리 IK의 본 길이/월드 좌표 정합을 깨므로 오프셋 방식 사용)
        const headBone = vrm.humanoid.getNormalizedBoneNode('head');
        if (headBone) {
            vrm.scene.updateWorldMatrix(true, true);
            const headY = headBone.getWorldPosition(new THREE.Vector3()).y;
            vrm.scene.position.y = AVATAR_HEAD_HEIGHT - headY;
        }

        // Hips rest 위치 저장 (로드 직후 = rest 자세 보장, hips 이동의 기준점)
        const hipsBone = vrm.humanoid.getNormalizedBoneNode('hips');
        if (hipsBone) vrm.scene.userData.hipsRestPos = hipsBone.position.clone();

        // 디버그용 본 방향 축 (Landmarks 토글로 표시)
        setupBoneAxesHelpers(vrm);

        // 발 고정(pinning)용 rest 위치 — vrm.scene 로컬 기준이라 앵커로 scene을 옮겨도 유효
        vrm.scene.updateWorldMatrix(true, true);
        for (const side of ['left', 'right']) {
            const foot = vrm.humanoid.getNormalizedBoneNode(side + 'Foot');
            if (foot) {
                const w = foot.getWorldPosition(new THREE.Vector3());
                vrm.scene.userData[side + 'FootRestLocal'] = vrm.scene.worldToLocal(w);
            }
        }

        // 사이드 스텝 상태 초기화 (새 아바타의 rest 앵커 기준으로 재시작)
        footStep.offL = footStep.offR = footStep.liftL = footStep.liftR = 0;
        footStep.sustainT = footStep.cooldownT = 0;
        footStep.active = null;
        console.log("Avatar loaded:", url);
    } catch (err) {
        console.error("VRM load error:", err);
        alert('아바타 로딩 실패. 페이지를 새로고침해주세요.');
        throw err;
    }
}

async function switchAvatar(url) {
    if (isAvatarLoading || currentAvatarUrl === url) return;
    isAvatarLoading = true;

    if (currentVrm) {
        scene.remove(currentVrm.scene);
        VRMUtils.deepDispose(currentVrm.scene);
        currentVrm = null;
    }

    try {
        await loadAvatar(url);
    } catch (_err) {
        currentAvatarUrl = '';
    } finally {
        isAvatarLoading = false;
    }
}

function animate() {
    requestAnimationFrame(animate);

    const frameStart = performance.now();
    const currentTime = performance.now();
    const deltaTime = (currentTime - lastFrameTime) / 1000; // 초 단위
    lastFrameTime = currentTime;

    if (shouldPauseAvatarWorkForRecording()) {
        if (recordingStats) {
            recordingStats.animateFrames += 1;
            reportRecordingStats();
        }
        return;
    }

    if (video && video.readyState >= 2) {
        if (DEBUG_MODE && debugCtx) {
            debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
        }

        // 펜 획을 긋는 동안은 인식을 ~150ms 주기로 낮춰 메인 스레드를 입력 처리에 양보
        // (detectForVideo가 프레임당 수십 ms 블로킹 → pointermove 유실로 선이 끊기는 문제 완화)
        const throttleDetection = activeStroke && (currentTime - lastDetectionTime) < 150;
        const videoFrameChanged = video.currentTime !== lastVideoTime;
        const recording = isRecordingActive();
        const canDetect = videoFrameChanged && !throttleDetection && !poseFrozen;
        const shouldRunFace = canDetect && (
            !recording || currentTime - lastFaceDetectionTime >= RECORDING_FRAME_INTERVAL
        );
        const shouldRunBody = canDetect && BODY_TRACKING_ENABLED && (
            !recording || currentTime - lastBodyDetectionTime >= RECORDING_BODY_INTERVAL
        );

        // poseFrozen: 디버그 프리즈 중에는 트래킹 적용을 멈추고 수동 본 편집 상태 유지
        if (shouldRunFace || shouldRunBody) {
            lastVideoTime = video.currentTime;
            lastDetectionTime = currentTime;

            // 1. Detect Face
            if (shouldRunFace && faceLandmarker) {
                const detectStart = performance.now();
                const results = faceLandmarker.detectForVideo(video, currentTime);
                recordDetectDuration('face', performance.now() - detectStart);
                lastFaceDetectionTime = currentTime;

                // Body tracking이 꺼져 있으면 얼굴 폭으로 거리 신뢰도 갱신 (멀수록 스무딩 강화)
                if (!BODY_TRACKING_ENABLED && results.faceLandmarks && results.faceLandmarks.length > 0) {
                    const fl = results.faceLandmarks[0];
                    const faceW = Math.abs(fl[454].x - fl[234].x) * VIDEO_ASPECT;
                    updateDistConf(faceW / 0.2, deltaTime);
                }

                if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
                    applyBlendshapes(results.faceBlendshapes[0], deltaTime);
                }
                if (results.facialTransformationMatrixes && results.facialTransformationMatrixes.length > 0) {
                    applyHeadRotation(results.facialTransformationMatrixes[0], deltaTime);
                }
                if (DEBUG_MODE && drawingUtils && results.faceLandmarks) {
                    for (const landmarks of results.faceLandmarks) {
                        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#90EE90A0", lineWidth: 2 });
                    }
                }
            }

            // 2. Body Tracking이 활성화된 경우에만 Hand/Pose 처리
            if (shouldRunBody) {
                lastBodyDetectionTime = currentTime;
                // Hand tracking 결과 초기화
                detectedHands.left = null;
                detectedHands.right = null;

                let handResults = null;
                if (handLandmarker) {
                    const detectStart = performance.now();
                    handResults = handLandmarker.detectForVideo(video, currentTime);
                    recordDetectDuration('hand', performance.now() - detectStart);
                    if (handResults.landmarks && handResults.landmarks.length > 0) {
                        // 아바타 기준 좌우로 저장 — tasks-vision 라벨은 해부학적 기준이라
                        // 미러 모드에서 사용자 왼손("Left")이 아바타 오른손이 됨
                        for (let i = 0; i < handResults.landmarks.length; i++) {
                            const handedness = handResults.handednesses[i][0];
                            const landmarks = handResults.landmarks[i];

                            if (handedness.categoryName === 'Left') {
                                detectedHands.right = landmarks;
                            } else {
                                detectedHands.left = landmarks;
                            }
                        }

                        if (DEBUG_MODE && drawingUtils) {
                            for (const landmark of handResults.landmarks) {
                                drawingUtils.drawConnectors(landmark, HandLandmarker.HAND_CONNECTIONS, { color: "#FF0000", lineWidth: 2 });
                                drawingUtils.drawLandmarks(landmark, { color: "#00FF00", lineWidth: 1 });
                            }
                        }
                    }
                }

                // Pose tracking
                let poseDetected = false;
                let framePoseLandmarks = null;
                let frameWorldLandmarks = null;
                if (poseLandmarker) {
                    const detectStart = performance.now();
                    const poseResults = poseLandmarker.detectForVideo(video, currentTime);
                    recordDetectDuration('pose', performance.now() - detectStart);
                    if (poseResults.landmarks && poseResults.landmarks.length > 0) {
                        const rawLandmarks = poseResults.landmarks[0];
                        const rawWorldLandmarks = poseResults.worldLandmarks ? poseResults.worldLandmarks[0] : null;

                        // One Euro Filter 적용 (떨림 완화)
                        const { filteredLandmarks, filteredWorldLandmarks } = getFilteredPoseLandmarks(
                            rawLandmarks, rawWorldLandmarks, currentTime
                        );
                        framePoseLandmarks = filteredLandmarks;
                        frameWorldLandmarks = filteredWorldLandmarks;

                        applyPose(filteredLandmarks, filteredWorldLandmarks, deltaTime);
                        poseDetected = true;

                        if (DEBUG_MODE && drawingUtils) {
                            drawingUtils.drawLandmarks(filteredLandmarks, { radius: 1, color: "white" });
                            drawingUtils.drawConnectors(filteredLandmarks, PoseLandmarker.POSE_CONNECTIONS, { color: "white", lineWidth: 2 });
                        }
                    }
                }

                if (!poseDetected) {
                    resetPose(deltaTime);
                }

                // 손 처리는 pose 이후 실행 — 손목 회전 게이팅(armActive)이
                // 이전 프레임이 아닌 현재 프레임의 팔 활성 상태를 사용하도록
                if (handLandmarker) {
                    if (handResults && handResults.landmarks && handResults.landmarks.length > 0) {
                        applyHands(handResults.landmarks, handResults.handednesses, deltaTime);
                    }

                    // 이번 프레임에 검출되지 않은 손은 rest 자세로 복귀
                    if (!detectedHands.left) relaxHand('left', deltaTime);
                    if (!detectedHands.right) relaxHand('right', deltaTime);
                }

                // 모션 레코딩: 입력(랜드마크)과 출력(본 회전)을 프레임 단위로 기록
                // (본 적용이 모두 끝난 시점에 캡처해야 입력↔출력이 같은 프레임으로 짝지어짐)
                captureMotionFrame(framePoseLandmarks, frameWorldLandmarks, currentTime);
            } else if (!BODY_TRACKING_ENABLED) {
                // Body tracking 비활성화 시 팔을 자연스럽게 내림
                resetPose(deltaTime);
            }
        }
    }

    if (currentVrm) {
        currentVrm.update(deltaTime);
    }

    if (orbitControls && orbitControls.enabled) {
        orbitControls.update();
    }

    renderer.render(scene, camera);

    if (recordingStats) {
        const frameDuration = performance.now() - frameStart;
        recordingStats.animateFrames += 1;
        if (frameDuration > RECORDING_FRAME_INTERVAL) recordingStats.longAnimateFrames += 1;
        reportRecordingStats();
    }
}

// ============================================================
// 통일된 좌표 변환 함수
// MediaPipe 좌표계 → VRM 좌표계
// MediaPipe: X(오른쪽+), Y(아래+), Z(카메라에서 멀어지는 방향+)
// VRM: X(오른쪽+), Y(위+), Z(캐릭터 전방+)
// 미러링: 사용자가 거울을 보는 것처럼 좌우 반전
// ============================================================
function mpToVRM(landmark, mirror = true) {
    return new THREE.Vector3(
        mirror ? -landmark.x : landmark.x,  // X: 미러링
        -landmark.y,                         // Y: 축 방향 반전
        -landmark.z                          // Z: 깊이 방향 반전
    );
}

// deltaTime 기반 부드러운 보간 계수 계산
function getLerpFactor(deltaTime, speed = LERP_SPEED) {
    return 1 - Math.exp(-speed * deltaTime);
}

// 부모 본의 월드 회전을 가져오는 헬퍼
function getParentWorldQuaternion(bone) {
    const worldQuat = new THREE.Quaternion();
    if (bone.parent) {
        bone.parent.getWorldQuaternion(worldQuat);
    }
    return worldQuat;
}

// 월드 회전을 로컬 회전으로 변환
function worldToLocalQuaternion(worldQuat, parentWorldQuat) {
    const parentInverse = parentWorldQuat.clone().invert();
    return parentInverse.multiply(worldQuat.clone());
}

function resetPose(deltaTime) {
    if (!currentVrm) return;

    const factor = getLerpFactor(deltaTime, 8); // 느리게 복귀

    const rightUpperArm = currentVrm.humanoid.getNormalizedBoneNode('rightUpperArm');
    const leftUpperArm = currentVrm.humanoid.getNormalizedBoneNode('leftUpperArm');
    const rightLowerArm = currentVrm.humanoid.getNormalizedBoneNode('rightLowerArm');
    const leftLowerArm = currentVrm.humanoid.getNormalizedBoneNode('leftLowerArm');

    // T-Pose에서 팔을 내린 상태로
    // VRM 0.x는 rotateVRM0로 scene이 180°Y 회전되어 normalized bone의 Z 방향이 반전됨
    const zDir = currentVrm.meta?.metaVersion === '0' ? -1 : 1;
    const relaxRight = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, Math.PI * 0.45 * zDir, 'XYZ'));
    const relaxLeft = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -Math.PI * 0.45 * zDir, 'XYZ'));
    const neutralLower = new THREE.Quaternion(); // 아래팔은 펴진 상태

    if (rightUpperArm) rightUpperArm.quaternion.slerp(relaxRight, factor);
    if (leftUpperArm) leftUpperArm.quaternion.slerp(relaxLeft, factor);
    if (rightLowerArm) rightLowerArm.quaternion.slerp(neutralLower, factor);
    if (leftLowerArm) leftLowerArm.quaternion.slerp(neutralLower, factor);

    // 다리는 곧게 선 자세(rest)로 복귀
    for (const boneName of ['rightUpperLeg', 'rightLowerLeg', 'leftUpperLeg', 'leftLowerLeg']) {
        const bone = currentVrm.humanoid.getNormalizedBoneNode(boneName);
        if (bone) bone.quaternion.slerp(neutralLower, factor);
    }

    // Hips 회전/위치 복귀
    const hips = currentVrm.humanoid.getNormalizedBoneNode('hips');
    if (hips) {
        hips.quaternion.slerp(neutralLower, factor);
        const restPos = currentVrm.scene.userData.hipsRestPos;
        if (restPos) hips.position.lerp(restPos, factor);
    }

    // Chest도 중립 복귀 (댄스 모드에서 어깨 라인 회전을 받던 본)
    const chest = currentVrm.humanoid.getNormalizedBoneNode('chest');
    if (chest) chest.quaternion.slerp(neutralLower, factor);

    // 손목/손가락도 rest로 복귀 — body tracking OFF 시 applyHands/relaxHand가
    // 실행되지 않아 마지막 포즈(주먹 등)로 고착되는 것 방지
    relaxHand('left', deltaTime);
    relaxHand('right', deltaTime);

    // 활성 상태 리셋
    leftArmActive = false;
    rightArmActive = false;
    swayBaseline = null;
    hipSwayBaseline = null;
    leanBaseline = null;
    lastSwayOffX = 0;
    danceMode = false;

    // 사이드 스텝 상태: 진행 중 스텝 취소, 발 앵커는 rest로 서서히 복귀
    footStep.active = null;
    footStep.sustainT = 0;
    footStep.liftL = footStep.liftR = 0;
    footStep.offL = THREE.MathUtils.lerp(footStep.offL, 0, factor);
    footStep.offR = THREE.MathUtils.lerp(footStep.offR, 0, factor);
    for (const key of ['right', 'left', 'rightLeg', 'leftLeg']) {
        ikPlaneState[key].twistFlip = false;
        ikPlaneState[key].pronation = 0;
        ikPlaneState[key].wristTwist = null;
    }
}

// --- 본 방향 디버그 축 (Landmarks ON일 때 표시) ---
// 각 normalized bone의 로컬 축을 그림: X=빨강, Y=초록, Z=파랑
// VRM rest 기준 해석: 손바닥 = -Y(초록 반대쪽), 팔/손가락 방향 = ±X(빨강), 전방 = +Z(파랑)
let boneAxesHelpers = [];
const AXES_DEBUG_BONES = [
    ['hips', 0.12], ['chest', 0.12], ['head', 0.1],
    ['leftShoulder', 0.06], ['rightShoulder', 0.06],
    ['leftUpperArm', 0.08], ['leftLowerArm', 0.08], ['leftHand', 0.09],
    ['rightUpperArm', 0.08], ['rightLowerArm', 0.08], ['rightHand', 0.09],
    ['leftUpperLeg', 0.08], ['leftLowerLeg', 0.08], ['leftFoot', 0.08],
    ['rightUpperLeg', 0.08], ['rightLowerLeg', 0.08], ['rightFoot', 0.08],
];

function setupBoneAxesHelpers(vrm) {
    boneAxesHelpers = []; // 이전 아바타의 헬퍼는 scene dispose와 함께 제거됨
    for (const [name, size] of AXES_DEBUG_BONES) {
        const bone = vrm.humanoid.getNormalizedBoneNode(name);
        if (!bone) continue;
        const helper = new THREE.AxesHelper(size);
        helper.material.depthTest = false; // 메쉬 안쪽 본도 항상 보이도록
        helper.renderOrder = 999;
        helper.visible = DEBUG_MODE;
        bone.add(helper);
        boneAxesHelpers.push(helper);
    }
}

function setBoneAxesVisible(visible) {
    for (const h of boneAxesHelpers) h.visible = visible;
}

// --- 포즈 프리즈 & 본 수동 편집 (디버깅용) ---
// Landmarks ON 상태에서 키보드로 조작:
//   F  : 포즈 프리즈 토글 (트래킹 적용 중지, 렌더링은 유지)
//   N  : 편집할 본 순환 선택 (Shift+N 역방향) → TransformControls 기즈모로 회전
//   Esc: 본 선택 해제
//   D  : 현재 본 상태 덤프 (콘솔 + 클립보드) — 트래킹 결과/수동 개선치를
//        JSON으로 뽑아 비교·분석에 활용하는 재귀 보정 워크플로우용
// 주의: B(블랙아웃)·P(펜 도구) 등 기존 앱 단축키와 겹치지 않는 키만 사용
let poseFrozen = false;
let boneEditControls = null;
let boneEditIndex = -1;

function ensureBoneEditControls() {
    if (boneEditControls || !camera || !renderer) return boneEditControls;
    boneEditControls = new TransformControls(camera, renderer.domElement);
    boneEditControls.setMode('rotate');
    boneEditControls.setSpace('local');
    boneEditControls.setSize(0.6);
    boneEditControls.addEventListener('dragging-changed', (e) => {
        if (orbitControls) orbitControls.enabled = !e.value && isOrbitEnabled;
    });
    // three r169+: TransformControls는 Object3D가 아니므로 getHelper()를 scene에 추가
    const gizmo = boneEditControls.getHelper ? boneEditControls.getHelper() : boneEditControls;
    scene.add(gizmo);
    return boneEditControls;
}

function selectDebugBone(step) {
    if (!currentVrm) return;
    const controls = ensureBoneEditControls();
    if (!controls) return;
    const names = AXES_DEBUG_BONES.map(([n]) => n)
        .filter(n => currentVrm.humanoid.getNormalizedBoneNode(n));
    if (names.length === 0) return;
    boneEditIndex = (boneEditIndex + step + names.length) % names.length;
    const name = names[boneEditIndex];
    controls.attach(currentVrm.humanoid.getNormalizedBoneNode(name));
    console.log(`[debug] 본 선택: ${name} (N: 다음, Shift+N: 이전, Esc: 해제, 기즈모 드래그로 회전)`);
}

function deselectDebugBone() {
    if (boneEditControls) {
        // 기즈모 조작으로 미세 비정규가 남지 않도록 해제 시 정규화
        boneEditControls.object?.quaternion?.normalize();
        boneEditControls.detach();
    }
    boneEditIndex = -1;
}

function dumpPoseDebug() {
    if (!currentVrm) return;
    const bones = {};
    for (const [name] of AXES_DEBUG_BONES) {
        const bone = currentVrm.humanoid.getNormalizedBoneNode(name);
        if (!bone) continue;
        const q = bone.quaternion;
        const e = new THREE.Euler().setFromQuaternion(q, 'XYZ');
        bones[name] = {
            quat: [q.x, q.y, q.z, q.w].map(v => +v.toFixed(4)),
            eulerDeg: [e.x, e.y, e.z].map(v => +THREE.MathUtils.radToDeg(v).toFixed(1))
        };
    }
    const hips = currentVrm.humanoid.getNormalizedBoneNode('hips');
    const dump = {
        frozen: poseFrozen,
        model: currentAvatarUrl,
        vrmVersion: currentVrm.meta?.metaVersion ?? '1',
        states: { leftArmActive, rightArmActive, danceMode },
        hipsPos: hips ? [hips.position.x, hips.position.y, hips.position.z].map(v => +v.toFixed(4)) : null,
        bones
    };
    const text = JSON.stringify(dump, null, 2);
    console.log('[debug] pose dump:\n' + text);
    navigator.clipboard?.writeText(text).then(
        () => console.log('[debug] 클립보드에 복사됨 — 그대로 붙여넣어 분석에 사용'),
        () => {}
    );
}

// --- 모션 레코딩 (디버깅용): 입력 랜드마크 + 출력 본 회전 시계열 기록 ---
// R로 시작/정지. 정지 시 JSON 다운로드 — 오프라인에서 solver 수정안을 같은 입력으로
// 재생·비교할 수 있어, "정지 자세는 맞는데 움직임에서 깨지는" 문제 분석에 사용
let motionRecording = false;
let motionRecordBuffer = null;
let motionRecordStart = 0;
const MOTION_RECORD_MAX_FRAMES = 3000; // ~100초 @30fps 자동 정지

function captureMotionFrame(landmarks, worldLandmarks, t) {
    if (!motionRecording || !motionRecordBuffer || !currentVrm) return;

    const rnd = (v) => +v.toFixed(4);
    const packPose = (arr) => arr ? arr.map(l => [rnd(l.x), rnd(l.y), rnd(l.z), rnd(l.visibility ?? 1)]) : null;
    const packHand = (arr) => arr ? arr.map(l => [rnd(l.x), rnd(l.y), rnd(l.z)]) : null;

    const bones = {};
    for (const [name] of AXES_DEBUG_BONES) {
        const b = currentVrm.humanoid.getNormalizedBoneNode(name);
        if (b) bones[name] = [rnd(b.quaternion.x), rnd(b.quaternion.y), rnd(b.quaternion.z), rnd(b.quaternion.w)];
    }

    motionRecordBuffer.frames.push({
        t: +(t - motionRecordStart).toFixed(1),
        pose: packPose(landmarks),
        world: packPose(worldLandmarks),
        handL: packHand(detectedHands.left),
        handR: packHand(detectedHands.right),
        states: { leftArmActive, rightArmActive, danceMode },
        bones
    });

    if (motionRecordBuffer.frames.length >= MOTION_RECORD_MAX_FRAMES) {
        console.log('[debug] 모션 레코딩 최대 길이 도달 — 자동 정지');
        stopMotionRecording();
    }
}

function startMotionRecording() {
    motionRecordBuffer = {
        meta: {
            model: currentAvatarUrl,
            vrmVersion: currentVrm?.meta?.metaVersion ?? '1',
            videoWidth: VIDEO_WIDTH,
            videoHeight: VIDEO_HEIGHT,
            startedAt: new Date().toISOString()
        },
        frames: []
    };
    motionRecordStart = performance.now();
    motionRecording = true;
    console.log('[debug] 모션 레코딩 시작 — R로 정지하면 JSON 다운로드');
}

function stopMotionRecording() {
    motionRecording = false;
    if (!motionRecordBuffer || motionRecordBuffer.frames.length === 0) {
        console.log('[debug] 기록된 프레임 없음');
        motionRecordBuffer = null;
        return;
    }
    const blob = new Blob([JSON.stringify(motionRecordBuffer)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `motion-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    a.click();
    URL.revokeObjectURL(url);
    console.log(`[debug] 모션 레코딩 저장: ${motionRecordBuffer.frames.length} 프레임`);
    motionRecordBuffer = null;
}

document.addEventListener('keydown', (e) => {
    if (!DEBUG_MODE) return;
    const tag = e.target?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || e.target?.isContentEditable) return;

    if (e.key === 'r' || e.key === 'R') {
        if (motionRecording) stopMotionRecording();
        else startMotionRecording();
    } else if (e.key === 'f' || e.key === 'F') {
        poseFrozen = !poseFrozen;
        console.log(`[debug] 포즈 프리즈: ${poseFrozen ? 'ON — N으로 본 선택, D로 덤프' : 'OFF'}`);
        if (!poseFrozen) deselectDebugBone();
    } else if (e.key === 'd' || e.key === 'D') {
        dumpPoseDebug();
    } else if (e.key === 'n') {
        selectDebugBone(1);
    } else if (e.key === 'N') {
        selectDebugBone(-1);
    } else if (e.key === 'Escape') {
        deselectDebugBone();
    }
});

// --- Debug 3D ---
let debugGroup;
function updateDebug3D(worldLandmarks) {
    if (!DEBUG_MODE) {
        if (debugGroup) debugGroup.visible = false;
        return;
    }

    if (!debugGroup) {
        debugGroup = new THREE.Group();
        scene.add(debugGroup);
        for (let i = 0; i < 33; i++) {
            const geom = new THREE.SphereGeometry(0.02, 8, 8);
            const mat = new THREE.MeshBasicMaterial({ color: 0xff0000 });
            const sphere = new THREE.Mesh(geom, mat);
            debugGroup.add(sphere);
        }
    }
    debugGroup.visible = true;

    if (worldLandmarks) {
        // MediaPipe world landmark는 골반 중점이 원점이라 그대로 그리면 스켈레톤이
        // 월드 원점(발밑)에 몰림 → 아바타 hips 위치로 옮겨 아바타와 겹쳐 비교 가능하게
        const origin = new THREE.Vector3(0, 1.0, 0);
        const hips = currentVrm?.humanoid.getNormalizedBoneNode('hips');
        if (hips) hips.getWorldPosition(origin);

        for (let i = 0; i < worldLandmarks.length; i++) {
            const l = worldLandmarks[i];
            const sphere = debugGroup.children[i];
            if (sphere) {
                const pos = mpToVRM(l);
                sphere.position.copy(pos).add(origin);
            }
        }
    }
}

// ============================================================
// 개선된 Two-Bone IK Solver
// ============================================================
// 팔다리가 거의 일직선일 때 planeNormal이 노이즈로 요동치는 것을 막기 위한 체인별 상태
const ikPlaneState = {
    right: { normal: null, twistFlip: false, pronation: 0 },
    left: { normal: null, twistFlip: false, pronation: 0 },
    rightLeg: { normal: null, twistFlip: false, pronation: 0 },
    leftLeg: { normal: null, twistFlip: false, pronation: 0 }
};

// 본 회전을 swing(방향)과 twist(본 축 비틀림)로 분해해 서로 다른 속도로 수렴시킴.
// relax 자세와 IK 해의 twist 규약이 달라(≈180°) 팔을 들고 내릴 때 어깨가 홱 도는
// 현상을 완화 — 방향은 즉시 따라가고 비틀림은 천천히 따라잡음.
function slerpSwingTwist(bone, qTarget, axis, swingFactor, twistFactor) {
    const qCur = bone.quaternion;

    // 방어선: 본 쿼터니언이 퇴화(zero/NaN/큰 비정규)하면 어떤 slerp로도 회복이
    // 불가능해 영구 고착됨(실측: 팔이 T-pose에 얼어붙는 증상) — 감지 시 identity로 복구
    const lenSq = qCur.lengthSq();
    if (!isFinite(lenSq) || lenSq < 0.25 || lenSq > 4) {
        qCur.identity();
    } else if (Math.abs(lenSq - 1) > 1e-3) {
        qCur.normalize(); // 미세 드리프트는 즉시 세척 (자기 증폭 차단)
    }

    const delta = qCur.clone().invert().multiply(qTarget); // 로컬 프레임 잔여 회전

    // delta = swing ∘ twist 분해 (twist: axis 성분 사영)
    const d = delta.x * axis.x + delta.y * axis.y + delta.z * axis.z;
    const twist = new THREE.Quaternion(axis.x * d, axis.y * d, axis.z * d, delta.w);
    if (twist.lengthSq() < 1e-10) {
        twist.set(0, 0, 0, 1);
    } else {
        twist.normalize();
    }
    const swing = delta.clone().multiply(twist.clone().invert());

    const partialSwing = new THREE.Quaternion().slerp(swing, swingFactor);
    const partialTwist = new THREE.Quaternion().slerp(twist, twistFactor);
    bone.quaternion.copy(qCur).multiply(partialSwing).multiply(partialTwist);
    // 결과가 유한하지 않으면(목표에 NaN 유입 등) 오염 대신 이전 정규값 유지
    if (!isFinite(bone.quaternion.lengthSq())) {
        bone.quaternion.copy(qCur);
    } else {
        bone.quaternion.normalize();
    }
}

// boneAxis는 rig 로컬(로드 시점 프레임) 기준 rest 방향.
// localForwardZ: rig 로컬 전방의 Z 부호 — VRM1은 +1, VRM0은 rig가 rotateVRM0 이전
// 프레임(모델이 -Z를 향하던 시점) 기준이라 -1. hinge 축 계산에 사용되며,
// 틀리면 twist 정렬이 팔다리를 180° 비틀어 무릎/발이 뒤로 돌아감.
function solveTwoBoneIK(upperBone, lowerBone, upperLength, lowerLength, targetPos, polePos, boneAxis, deltaTime, planeState, localForwardZ = 1) {
    if (!upperBone || !lowerBone) return;

    const factor = getLerpFactor(deltaTime);
    const a = upperLength;
    const b = lowerLength;

    // 1. 목표 거리 제한 (팔 길이 범위 내로)
    let dist = targetPos.length();
    const epsilon = 0.001;
    const maxLen = a + b - epsilon;
    const minLen = Math.abs(a - b) + epsilon;

    if (dist > maxLen) {
        targetPos.setLength(maxLen);
        dist = maxLen;
    } else if (dist < minLen) {
        targetPos.setLength(minLen);
        dist = minLen;
    }

    // 2. 코사인 법칙으로 어깨 각도 계산
    const c = dist;
    const cosShoulderAngle = (a * a + c * c - b * b) / (2 * a * c);
    const shoulderAngle = Math.acos(THREE.MathUtils.clamp(cosShoulderAngle, -1, 1));

    // 3. 코사인 법칙으로 팔꿈치 각도 계산
    const cosElbowAngle = (a * a + b * b - c * c) / (2 * a * b);
    const elbowAngle = Math.acos(THREE.MathUtils.clamp(cosElbowAngle, -1, 1));

    // 4. IK 평면 계산 (어깨→손목 방향과 팔꿈치 방향으로 정의)
    const dirToTarget = targetPos.clone().normalize();
    const dirToPole = polePos.clone().normalize();

    // 평면 법선 계산
    let planeNormal = new THREE.Vector3().crossVectors(dirToTarget, dirToPole);

    // 팔이 거의 일직선이면(sin < ~0.2) 측정된 pole이 노이즈에 지배되므로
    // 마지막 유효 normal을 유지해 팔 떨림/뒤집힘을 방지
    if (planeNormal.lengthSq() < 0.04) {
        if (planeState && planeState.normal) {
            planeNormal.copy(planeState.normal);
        } else {
            planeNormal.crossVectors(dirToTarget, new THREE.Vector3(0, 1, 0));
            if (planeNormal.lengthSq() < 0.0001) {
                planeNormal.crossVectors(dirToTarget, new THREE.Vector3(0, 0, 1));
            }
        }
    }
    planeNormal.normalize();
    if (planeState) {
        planeState.normal = planeState.normal ? planeState.normal.copy(planeNormal) : planeNormal.clone();
    }

    // 5. Upper Arm (어깨) 방향 계산
    // 목표 방향에서 shoulderAngle만큼 회전
    const qBend = new THREE.Quaternion().setFromAxisAngle(planeNormal, shoulderAngle);
    const upperDir = dirToTarget.clone().applyQuaternion(qBend).normalize();

    // 6. Upper Arm 회전 계산
    const qUpper = new THREE.Quaternion().setFromUnitVectors(boneAxis, upperDir);

    // 팔꿈치/무릎 방향(hinge) 정렬 — rig 로컬 전방 기준으로 계산
    const hingeAxis = new THREE.Vector3().crossVectors(boneAxis, new THREE.Vector3(0, 0, localForwardZ)).normalize();
    const currentHinge = hingeAxis.clone().applyQuaternion(qUpper);
    const qTwist = new THREE.Quaternion().setFromUnitVectors(currentHinge, planeNormal);
    const qUpperFinal = qTwist.multiply(qUpper);

    // 7. 부모 좌표계를 고려한 로컬 회전 적용
    const parentWorldQuat = getParentWorldQuaternion(upperBone);
    const qUpperLocal = worldToLocalQuaternion(qUpperFinal, parentWorldQuat);

    // 8. Lower Bone (팔꿈치/무릎) 회전 계산
    // hinge joint: 굽힘 각도 = π - elbowAngle (펴진 상태가 π)
    // qUpperFinal이 로컬 hingeAxis를 world planeNormal에 정렬시켰으므로,
    // 로컬 hingeAxis 축 -bendAngle 회전 = world planeNormal 축 -bendAngle (pole 반대쪽으로 굽힘)
    const bendAngle = Math.PI - elbowAngle;
    const qLowerLocal = new THREE.Quaternion().setFromAxisAngle(hingeAxis, -bendAngle);

    // 8.5 비틀림 가지 선택 — hinge 정렬 해는 ±180° 비틀림 차이의 두 가지(world 자세
    // 동일)가 존재. 팔을 들면 정렬 비틀림이 ~240°까지 누적되어 어깨 mesh가 감기므로
    // (모션 레코딩 실측: +32°→+240°, 손목 역보정 ~180°), rest에 가까운 가지를
    // hysteresis(진입 105° / 해제 35°)로 선택. 상·하위에 같은 180°를 반대로 곱해
    // 곱(=팔의 world 자세·팔꿈치 평면)은 정확히 보존됨.
    // 오프라인 재생 검증: 이 레코딩에서 전환 1회, 비틀림 범위 -63°~+103°로 제한.
    if (planeState) {
        const qs = qUpperLocal.clone();
        if (qs.w < 0) qs.set(-qs.x, -qs.y, -qs.z, -qs.w);
        const dAxis = qs.x * boneAxis.x + qs.y * boneAxis.y + qs.z * boneAxis.z;
        const twistAbsDeg = Math.abs(2 * Math.atan2(dAxis, qs.w)) * (180 / Math.PI);
        if (!planeState.twistFlip && twistAbsDeg > 105) {
            planeState.twistFlip = true;
        } else if (planeState.twistFlip && twistAbsDeg < 35) {
            planeState.twistFlip = false;
        }
        if (planeState.twistFlip) {
            const qPi = new THREE.Quaternion().setFromAxisAngle(boneAxis, Math.PI);
            qUpperLocal.multiply(qPi);
            qLowerLocal.premultiply(qPi); // 180° 회전은 자기 역원 — 곱 보존
        }

        // 8.6 전완 회내(pronation) 적용 — 손목 비틀림 초과분을 전완 자체 롤로 이관
        // (applyHandOrientation의 서보가 갱신). postmultiply는 전완 방향은 유지하면서
        // 롤만 바꾸므로 트래킹 정확도에 영향 없음. 팔꿈치 경계의 가지 보정 덩어리도
        // 이 롤이 상쇄해 체인 전체에 비틀림이 분산됨.
        if (planeState.pronation && isFinite(planeState.pronation)) {
            qLowerLocal.multiply(new THREE.Quaternion().setFromAxisAngle(boneAxis, planeState.pronation));
        }
    }

    // 방향은 기존 속도로, 본 축 비틀림은 느리게 수렴.
    // 하위 본에도 같은 twist 속도 제한을 걸어 가지 전환 시 상·하위가 함께 굴러
    // 위상이 어긋나지 않게 함 (재분배 실험 실패의 원인이었던 비대칭 스무딩 방지)
    slerpSwingTwist(upperBone, qUpperLocal, boneAxis, factor, getLerpFactor(deltaTime, 3));
    slerpSwingTwist(lowerBone, qLowerLocal, boneAxis, factor, getLerpFactor(deltaTime, 3));
}

function applyPose(landmarks, worldLandmarks, deltaTime) {
    if (!currentVrm) return;

    updateDebug3D(worldLandmarks);

    const factor = getLerpFactor(deltaTime);
    // VRM 0.x는 rotateVRM0으로 scene이 180°Y 회전 → normalized bone의 X·Z 방향 반전
    const isVRM0 = currentVrm.meta?.metaVersion === '0';

    // 거리 신뢰도 갱신: 화면 속 어깨 폭이 표준(0.25) 대비 작을수록 멀리 있는 것
    if (landmarks) {
        const shoulderWImg = Math.abs(landmarks[11].x - landmarks[12].x) * VIDEO_ASPECT;
        updateDistConf(shoulderWImg / 0.25, deltaTime);
    }

    // 랜드마크를 VRM 좌표계로 변환하는 헬퍼
    const getPos = (index) => {
        const source = worldLandmarks || landmarks;
        return mpToVRM(source[index]);
    };

    // 본 가져오기
    const rUpper = currentVrm.humanoid.getNormalizedBoneNode('rightUpperArm');
    const rLower = currentVrm.humanoid.getNormalizedBoneNode('rightLowerArm');
    const rHand = currentVrm.humanoid.getNormalizedBoneNode('rightHand');

    const lUpper = currentVrm.humanoid.getNormalizedBoneNode('leftUpperArm');
    const lLower = currentVrm.humanoid.getNormalizedBoneNode('leftLowerArm');
    const lHand = currentVrm.humanoid.getNormalizedBoneNode('leftHand');

    // --- Hips 회전 (골반 롤/요) ---
    // 회전 부호는 spine과 동일한 규약: Y축 회전은 rotateVRM0(180°Y)에 불변이라 yaw는 버전 공통,
    // Z축(roll)만 VRM0에서 반전
    const hips = currentVrm.humanoid.getNormalizedBoneNode('hips');
    const hipsQuat = new THREE.Quaternion();
    if (hips && landmarks) {
        const mpLHip = landmarks[23];
        const mpRHip = landmarks[24];
        const hipVis = Math.min(mpLHip.visibility ?? 1.0, mpRHip.visibility ?? 1.0);

        // 댄스 모드: 골반이 안정적으로 보이면 골반/어깨 2-포인트 모델 활성 (hysteresis)
        const wasDanceMode = danceMode;
        if (!danceMode && hipVis > DANCE_VIS_ON) {
            danceMode = true;
        } else if (danceMode && hipVis < DANCE_VIS_OFF) {
            danceMode = false;
        }
        if (wasDanceMode !== danceMode) {
            // 모드 전환 시 기준점이 달라지므로 baseline 재초기화로 점프 방지
            swayBaseline = null;
            hipSwayBaseline = null;
            leanBaseline = null;
        }

        // 골반이 프레임 밖이면 노이즈 회전 방지 — 중립 유지
        // (hipsQuat이 identity로 남아 상체가 어깨 회전 전체를 받음)
        if (danceMode) {
            // roll = 측정된 hip 라인 기울기 + 무게이동 커플링(골반을 옆으로 민 만큼 기울임)
            // — 골반 춤에서 hip 라인 y차는 작아 측정 roll만으로는 표현이 약함(실측 ±5°)
            const rollH = ((mpRHip.y - mpLHip.y) * 1.2 + lastSwayOffX * DANCE_ROLL_COUPLE) * (isVRM0 ? -1 : 1);
            const yawH = (mpRHip.z - mpLHip.z) * 1.0;
            hipsQuat.setFromEuler(new THREE.Euler(0, yawH, rollH, 'XYZ'));
        }
        hips.quaternion.slerp(hipsQuat, factor * 0.5);
    }

    // --- Spine/Chest 회전 (상체) ---
    // 댄스 모드: spine = 골반→어깨 기울기(lean), chest = 어깨 라인 회전 잔여분
    //   → 골반과 어깨가 반대로 움직이는 S자(제자리 춤) 표현 가능
    // 폴백 모드: spine이 어깨 회전에서 골반 회전을 뺀 잔여분을 받는 기존 방식
    const spine = currentVrm.humanoid.getNormalizedBoneNode('spine');
    const chest = currentVrm.humanoid.getNormalizedBoneNode('chest');
    if (spine && landmarks) {
        const mpLeft = landmarks[11];  // 왼쪽 어깨
        const mpRight = landmarks[12]; // 오른쪽 어깨

        // 어깨 라인 회전 (미러링: 좌우 반전)
        const dy = mpRight.y - mpLeft.y;
        const dz = mpRight.z - mpLeft.z;

        const roll = dy * 1.2 * (isVRM0 ? -1 : 1);  // Z축 회전 (VRM 0.x: 방향 반전)
        const yaw = dz * 1.0;   // Y축 회전 (어깨 회전)

        const qShoulder = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, yaw, roll, 'XYZ'));

        if (danceMode) {
            // 골반→어깨 벡터의 좌우 기울기 각 (이미지 단위, aspect 보정)
            const aspect = VIDEO_ASPECT;
            const vX = ((mpLeft.x + mpRight.x) / 2 - (landmarks[23].x + landmarks[24].x) / 2) * aspect;
            const uY = (landmarks[23].y + landmarks[24].y) / 2 - (mpLeft.y + mpRight.y) / 2; // 이미지 y는 아래+ → 어깨가 위면 양수
            const leanAngle = Math.atan2(vX, Math.max(uY, 1e-4));

            // 중립 기울기 baseline (카메라 기울기·개인 자세 흡수)
            if (leanBaseline === null) leanBaseline = leanAngle;
            const alphaL = getLerpFactor(deltaTime, BASELINE_ADAPT_SPEED);
            leanBaseline += (leanAngle - leanBaseline) * alphaL;

            // 미러링(-x)과 Rz(+θ가 상단을 -X로 기울임)의 부호가 상쇄되어 이미지 각도를 그대로 사용
            const lean = THREE.MathUtils.clamp(leanAngle - leanBaseline, -Math.PI / 6, Math.PI / 6);
            const qLean = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, lean * (isVRM0 ? -1 : 1)));

            // 어깨 라인 회전은 (hips × spine) 위에 얹는 잔여분
            const qResidual = hipsQuat.clone().multiply(qLean).invert().multiply(qShoulder);
            if (chest) {
                spine.quaternion.slerp(qLean, factor * 0.5);
                chest.quaternion.slerp(qResidual, factor * 0.5);
            } else {
                // chest 본이 없는 모델은 spine에 합성
                spine.quaternion.slerp(qLean.clone().multiply(qResidual), factor * 0.5);
            }
        } else {
            // hips가 이미 골반 회전을 반영하므로, spine은 어깨 회전에서 골반 회전을 뺀 잔여분만 적용
            const q = hipsQuat.clone().invert().multiply(qShoulder);
            spine.quaternion.slerp(q, factor * 0.5); // 상체는 더 부드럽게
            if (chest) chest.quaternion.slerp(IDENTITY_QUAT, factor * 0.3);
        }
    }

    // --- Visibility 체크 (Hysteresis 적용) ---
    // MediaPipe Left Wrist (15) → Avatar Right Arm (미러링)
    // MediaPipe Right Wrist (16) → Avatar Left Arm (미러링)

    const leftWristVis = landmarks[15].visibility ?? 1.0;
    const rightWristVis = landmarks[16].visibility ?? 1.0;

    // Hysteresis: 켜질 때는 높은 임계값, 꺼질 때는 낮은 임계값
    if (!leftArmActive && leftWristVis > VIS_THRESHOLD_ON) {
        leftArmActive = true;
    } else if (leftArmActive && leftWristVis < VIS_THRESHOLD_OFF) {
        leftArmActive = false;
    }

    if (!rightArmActive && rightWristVis > VIS_THRESHOLD_ON) {
        rightArmActive = true;
    } else if (rightArmActive && rightWristVis < VIS_THRESHOLD_OFF) {
        rightArmActive = false;
    }

    // --- Avatar Right Arm ← MediaPipe Left Body(11,13,15) + Hand(0) ---
    if (rUpper && rLower && rHand && leftArmActive) {
        const upperLen = rLower.position.length();
        const lowerLen = rHand.position.length();

        const mpShoulder = getPos(11);
        const mpElbow = getPos(13);

        // Body wrist만 사용 (Hand landmarks는 좌표계가 다름)
        const mpWrist = getPos(15);

        // MediaPipe 팔 길이로 스케일 계산
        const mpArmLen = mpShoulder.distanceTo(mpElbow) + mpElbow.distanceTo(mpWrist);
        const avatarArmLen = upperLen + lowerLen;
        const scale = avatarArmLen / mpArmLen;

        // 어깨 기준 상대 위치
        const target = new THREE.Vector3().subVectors(mpWrist, mpShoulder).multiplyScalar(scale);
        const pole = new THREE.Vector3().subVectors(mpElbow, mpShoulder).multiplyScalar(scale);

        solveTwoBoneIK(rUpper, rLower, upperLen, lowerLen, target, pole, new THREE.Vector3(isVRM0 ? 1 : -1, 0, 0), deltaTime, ikPlaneState.right, isVRM0 ? -1 : 1);
    } else if (rUpper && !leftArmActive) {
        // 팔 내리기 — 방향은 부드럽게, 비틀림은 더 느리게 (복귀 시 어깨 스핀 완화)
        const relaxQuat = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, Math.PI * 0.45 * (isVRM0 ? -1 : 1), 'XYZ'));
        const neutralQuat = new THREE.Quaternion();
        slerpSwingTwist(rUpper, relaxQuat, new THREE.Vector3(1, 0, 0), factor * 0.3, getLerpFactor(deltaTime, 2));
        if (rLower) rLower.quaternion.slerp(neutralQuat, factor * 0.3);
        ikPlaneState.right.twistFlip = false;
        ikPlaneState.right.pronation = THREE.MathUtils.lerp(ikPlaneState.right.pronation, 0, factor * 0.3);
        ikPlaneState.right.wristTwist = null;
    }

    // --- Avatar Left Arm ← MediaPipe Right Body(12,14,16) + Hand(0) ---
    if (lUpper && lLower && lHand && rightArmActive) {
        const upperLen = lLower.position.length();
        const lowerLen = lHand.position.length();

        const mpShoulder = getPos(12);
        const mpElbow = getPos(14);

        // Body wrist만 사용 (Hand landmarks는 좌표계가 다름)
        const mpWrist = getPos(16);

        const mpArmLen = mpShoulder.distanceTo(mpElbow) + mpElbow.distanceTo(mpWrist);
        const avatarArmLen = upperLen + lowerLen;
        const scale = avatarArmLen / mpArmLen;

        const target = new THREE.Vector3().subVectors(mpWrist, mpShoulder).multiplyScalar(scale);
        const pole = new THREE.Vector3().subVectors(mpElbow, mpShoulder).multiplyScalar(scale);

        solveTwoBoneIK(lUpper, lLower, upperLen, lowerLen, target, pole, new THREE.Vector3(isVRM0 ? -1 : 1, 0, 0), deltaTime, ikPlaneState.left, isVRM0 ? -1 : 1);
    } else if (lUpper && !rightArmActive) {
        // 팔 내리기 — 방향은 부드럽게, 비틀림은 더 느리게 (복귀 시 어깨 스핀 완화)
        const relaxQuat = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -Math.PI * 0.45 * (isVRM0 ? -1 : 1), 'XYZ'));
        const neutralQuat = new THREE.Quaternion();
        slerpSwingTwist(lUpper, relaxQuat, new THREE.Vector3(1, 0, 0), factor * 0.3, getLerpFactor(deltaTime, 2));
        if (lLower) lLower.quaternion.slerp(neutralQuat, factor * 0.3);
        ikPlaneState.left.twistFlip = false;
        ikPlaneState.left.pronation = THREE.MathUtils.lerp(ikPlaneState.left.pronation, 0, factor * 0.3);
        ikPlaneState.left.wristTwist = null;
    }

    // ============================================================
    // Hips 이동 (몸 좌우 sway / 상하 bob) — 어깨 기준 추정
    // 어깨는 상반신 프레이밍에서도 항상 보이므로 전신 촬영이 필요 없음.
    // 느린 적응 baseline 대비 오프셋으로 순간적 몸 움직임만 아바타에 반영
    // ============================================================
    const hipsRestPos = currentVrm.scene.userData.hipsRestPos;
    if (hips && hipsRestPos && landmarks) {
        const shoulderVis = Math.min(landmarks[11].visibility ?? 1.0, landmarks[12].visibility ?? 1.0);
        const aspect = VIDEO_ASPECT;

        // 기준점 선택 (이미지 단위, x는 aspect 보정으로 y와 등방화):
        // 댄스 모드 = 골반 중점 → 골반 sway가 hips를 직접 구동 (상체 lean은 spine이 별도 표현)
        // 폴백 모드 = 어깨 중점 → 몸 전체가 한 덩어리로 sway
        let refX = null, refY = null, baseline = null;
        if (danceMode) {
            refX = (landmarks[23].x + landmarks[24].x) / 2 * aspect;
            refY = (landmarks[23].y + landmarks[24].y) / 2;
            if (!hipSwayBaseline) hipSwayBaseline = { x: refX, y: refY };
            baseline = hipSwayBaseline;
        } else if (shoulderVis > 0.5) {
            refX = (landmarks[11].x + landmarks[12].x) / 2 * aspect;
            refY = (landmarks[11].y + landmarks[12].y) / 2;

            if (!swayBaseline) swayBaseline = { x: refX, y: refY };
            baseline = swayBaseline;
        }

        if (baseline) {
            // 느린 적응: 순간적 sway는 표현, 장기적 위치 변화는 중립화
            const alpha = getLerpFactor(deltaTime, BASELINE_ADAPT_SPEED);
            baseline.x += (refX - baseline.x) * alpha;
            baseline.y += (refY - baseline.y) * alpha;

            // 이미지 단위 → 미터: 영상 속 어깨 폭과 아바타 어깨 폭(양쪽 upperArm 거리)의 비율
            // 측면 자세에서는 어깨 폭이 단축되어 스케일이 폭주하므로 분모 하한 + 스케일 상한으로 제한
            const shoulderWidthImg = Math.abs(landmarks[11].x - landmarks[12].x) * aspect;
            let scale = 0;
            if (rUpper && lUpper && shoulderWidthImg > 1e-4) {
                const rW = rUpper.getWorldPosition(new THREE.Vector3());
                const lW = lUpper.getWorldPosition(new THREE.Vector3());
                scale = Math.min(rW.distanceTo(lW) / Math.max(shoulderWidthImg, 0.08), 3.0);
            }

            // 미러링(-x), 이미지 y(아래+) → 월드 y(위+); 댄스 모드는 골반 sway 증폭 + 범위 확대
            // (레코딩 실측: 골반 춤 입력은 뚜렷한데 루트 이동 ±10cm + 반대편 lean이
            //  상쇄되어 보여 표현이 약함 → 게인 증폭과 roll 커플링으로 강화)
            const swayGain = danceMode ? DANCE_SWAY_GAIN : 1;
            const maxSwayX = danceMode ? 0.35 : 0.25;
            const offX = THREE.MathUtils.clamp(-(refX - baseline.x) * scale * swayGain, -maxSwayX, maxSwayX);
            const offY = THREE.MathUtils.clamp(-(refY - baseline.y) * scale, -0.3, 0.1);
            lastSwayOffX = offX; // 다음 프레임 골반 roll 커플링에 사용

            // rig 로컬 x는 VRM0(scene 180°Y 회전)에서 월드와 반대
            const xDir = isVRM0 ? -1 : 1;
            hips.position.x = THREE.MathUtils.lerp(hips.position.x, hipsRestPos.x + offX * xDir, factor * 0.5);
            hips.position.y = THREE.MathUtils.lerp(hips.position.y, hipsRestPos.y + offY, factor * 0.5);
        } else {
            hips.position.lerp(hipsRestPos, factor * 0.3);
        }
    }

    // ============================================================
    // 다리: 발을 지면 rest 위치에 고정하는 Two-Bone IK
    // hips가 이동/회전하면 다리가 자연스럽게 굽혀져 발 착지 유지 (VTuber 스타일)
    // 골반이 발에서 지속적으로 멀어지면(실제 옆걸음) 발이 한 발씩 따라오는 사이드 스텝
    // ============================================================
    if (hips && hipsRestPos) updateFootStepping(hips, hipsRestPos, isVRM0, deltaTime);
    currentVrm.scene.updateWorldMatrix(true, false);
    solvePinnedFootLeg('right', deltaTime);
    solvePinnedFootLeg('left', deltaTime);
}

// 사이드 스텝 상태 머신: 골반-발중심 이격이 임계를 '지속' 초과하면(왕복 sway 제외)
// 이동 방향 쪽 발부터 smoothstep으로 이동 + sin 곡선으로 들어올림.
// 한 발 이동 후 스탠스가 어긋나면 뒷발이 따라붙음. baseline이 골반을 중립으로
// 되돌리면 반대 방향 이격이 생겨 발도 자동으로 되돌아옴 (자기 수렴).
function updateFootStepping(hips, hipsRestPos, isVRM0, deltaTime) {
    const xDir = isVRM0 ? -1 : 1;
    const hipsOffX = (hips.position.x - hipsRestPos.x) * xDir; // 월드 기준 골반 이동량

    const fs = footStep;
    if (fs.active) {
        fs.active.t += deltaTime;
        const p = Math.min(fs.active.t / STEP_DURATION, 1);
        const s = p * p * (3 - 2 * p); // smoothstep
        const off = THREE.MathUtils.lerp(fs.active.fromX, fs.active.toX, s);
        const lift = STEP_HEIGHT * Math.sin(Math.PI * p);
        if (fs.active.side === 'left') { fs.offL = off; fs.liftL = lift; }
        else { fs.offR = off; fs.liftR = lift; }
        if (p >= 1) {
            if (fs.active.side === 'left') fs.liftL = 0; else fs.liftR = 0;
            fs.active = null;
            fs.cooldownT = STEP_COOLDOWN;
            fs.sustainT = 0;
        }
        return;
    }

    fs.cooldownT = Math.max(0, fs.cooldownT - deltaTime);
    const feetMid = (fs.offL + fs.offR) / 2;
    const gap = hipsOffX - feetMid;

    if (Math.abs(gap) > STEP_TRIGGER_DIST) fs.sustainT += deltaTime;
    else fs.sustainT = 0;

    if (fs.cooldownT > 0) return;

    if (fs.sustainT > STEP_SUSTAIN_TIME) {
        // 이동 방향 쪽 발부터 (아바타 왼발이 월드 +X 쪽)
        const side = gap > 0 ? 'left' : 'right';
        fs.active = { side, fromX: side === 'left' ? fs.offL : fs.offR, toX: hipsOffX, t: 0 };
    } else if (Math.abs(fs.offL - fs.offR) > STANCE_MISMATCH) {
        // 스탠스가 어긋나 있으면 골반에서 먼 발(뒷발)이 따라붙음
        const dL = Math.abs(fs.offL - hipsOffX);
        const dR = Math.abs(fs.offR - hipsOffX);
        const side = dL > dR ? 'left' : 'right';
        fs.active = { side, fromX: side === 'left' ? fs.offL : fs.offR, toX: hipsOffX, t: 0 };
    }
}

// 발 고정 다리 IK: hips 현재 위치에서 로드 시 저장한 발 rest 위치로 다리를 풀어줌
// 다리 rest 방향은 -Y라 VRM 버전과 무관하게 boneAxis 동일 (Y축은 rotateVRM0 영향 없음)
function solvePinnedFootLeg(side, deltaTime) {
    if (!currentVrm) return;

    const upper = currentVrm.humanoid.getNormalizedBoneNode(side + 'UpperLeg');
    const lower = currentVrm.humanoid.getNormalizedBoneNode(side + 'LowerLeg');
    const foot = currentVrm.humanoid.getNormalizedBoneNode(side + 'Foot');
    const footRestLocal = currentVrm.scene.userData[side + 'FootRestLocal'];
    if (!upper || !lower || !foot || !footRestLocal) return;

    const upperLen = lower.position.length();
    const lowerLen = foot.position.length();

    // 사이드 스텝 오프셋/들어올림 반영 (scene-local x는 VRM0에서 월드와 반대)
    const isVRM0Local = currentVrm.meta?.metaVersion === '0';
    const anchorLocal = footRestLocal.clone();
    anchorLocal.x += (side === 'left' ? footStep.offL : footStep.offR) * (isVRM0Local ? -1 : 1);
    anchorLocal.y += side === 'left' ? footStep.liftL : footStep.liftR;

    // scene의 updateWorldMatrix는 호출부(applyPose)에서 프레임당 1회 수행
    const footTarget = currentVrm.scene.localToWorld(anchorLocal);
    const hipJoint = upper.getWorldPosition(new THREE.Vector3());
    const target = footTarget.sub(hipJoint);

    // 무릎은 앞(+Z, 카메라 방향)으로 굽힘: pole은 타깃 중간점에 전방 오프셋
    const pole = target.clone().multiplyScalar(0.5)
        .addScaledVector(new THREE.Vector3(0, 0, 1), (upperLen + lowerLen) * 0.4);

    // VRM0는 rig 로컬 전방이 -Z (hinge 축이 반대가 되어 발이 180° 돌아가는 것 방지)
    const isVRM0 = currentVrm.meta?.metaVersion === '0';
    solveTwoBoneIK(upper, lower, upperLen, lowerLen, target, pole,
        new THREE.Vector3(0, -1, 0), deltaTime, ikPlaneState[side + 'Leg'], isVRM0 ? -1 : 1);
}

// ============================================================
// 완전히 새로 작성된 손 & 손가락 처리
// - 거리 기반 손가락 curl (단순하고 안정적)
// - 손바닥 방향을 Hand bone에 직접 적용
// ============================================================

// 손가락 설정 (MCP, TIP 인덱스)
const FINGER_CONFIG = {
    Thumb:  { mcp: 1, tip: 4, isThumb: true },
    Index:  { mcp: 5, tip: 8, isThumb: false },
    Middle: { mcp: 9, tip: 12, isThumb: false },
    Ring:   { mcp: 13, tip: 16, isThumb: false },
    Little: { mcp: 17, tip: 20, isThumb: false }
};

// 디버그 로그 쓰로틀링
let lastDebugTime = 0;
const DEBUG_INTERVAL = 2000;

function applyHands(landmarksArray, handednesses, deltaTime) {
    if (!currentVrm) return;

    // 원거리에서는 손 랜드마크 노이즈가 커지므로 스무딩 강화
    const factor = getLerpFactor(deltaTime, 15 * trackingDistConf);
    const now = performance.now();
    const shouldLog = (now - lastDebugTime) > DEBUG_INTERVAL;

    for (let i = 0; i < landmarksArray.length; i++) {
        const landmarks = landmarksArray[i];
        const handedness = handednesses[i][0];

        // tasks-vision handedness 라벨은 해부학적 기준: "Left" = 사용자의 왼손.
        // 미러 모드에서 사용자의 왼손은 아바타의 오른손이 됨.
        const isAvatarRightHand = handedness.categoryName === 'Left';
        const prefix = isAvatarRightHand ? 'right' : 'left';

        // 아바타 오른팔은 MediaPipe left body(leftArmActive)가 구동 — 팔이 활성일 때만 손목 회전 적용
        const armActive = isAvatarRightHand ? leftArmActive : rightArmActive;

        if (armActive) {
            // 손 랜드마크 기반 실제 손목 방향 적용
            applyHandOrientation(prefix, landmarks, factor, deltaTime);
        } else {
            // 내려간 팔에 맞지 않는 손목 회전이 남지 않도록 rest로 복귀
            const handBone = currentVrm.humanoid.getNormalizedBoneNode(prefix + 'Hand');
            if (handBone) handBone.quaternion.slerp(IDENTITY_QUAT, factor * 0.3);
        }

        // 손가락 처리 (거리 기반)
        applyFingers(prefix, landmarks, factor);

        if (shouldLog && i === 0) {
            lastDebugTime = now;
        }
    }
}

// Hand landmark(이미지 정규화 좌표)를 VRM 방향 벡터용 좌표로 변환
// x는 영상 너비, y는 높이 기준 정규화라 스케일이 달라 aspect 보정 필요 (z는 x와 유사 스케일)
// 부호 변환은 mpToVRM과 동일 (미러링 + y/z 반전)
function handLmToVRM(lm) {
    const aspect = VIDEO_ASPECT;
    return new THREE.Vector3(-lm.x * aspect, -lm.y, -lm.z * aspect);
}

// 손 랜드마크로부터 실제 손목 방향(손가락 방향 + 손바닥 법선)을 계산해 Hand bone에 적용
// deltaTime이 주어지면 손목 비틀림 초과분을 전완 회내 서보로 이관
function applyHandOrientation(prefix, landmarks, factor, deltaTime) {
    if (!currentVrm) return;

    const handBone = currentVrm.humanoid.getNormalizedBoneNode(prefix + 'Hand');
    if (!handBone) return;

    const isRight = prefix === 'right';
    const isVRM0 = currentVrm.meta?.metaVersion === '0';

    const wrist = handLmToVRM(landmarks[0]);
    const indexMcp = handLmToVRM(landmarks[5]);
    const middleMcp = handLmToVRM(landmarks[9]);
    const pinkyMcp = handLmToVRM(landmarks[17]);

    const fingerDir = new THREE.Vector3().subVectors(middleMcp, wrist);
    const toIndex = new THREE.Vector3().subVectors(indexMcp, wrist);
    const toPinky = new THREE.Vector3().subVectors(pinkyMcp, wrist);
    if (fingerDir.lengthSq() < 1e-8) return;
    fingerDir.normalize();

    // 손바닥 법선: 오른손은 index×pinky, 왼손은 반대 (미러링된 좌표계 기준 chirality)
    const palmNormal = isRight
        ? new THREE.Vector3().crossVectors(toIndex, toPinky)
        : new THREE.Vector3().crossVectors(toPinky, toIndex);
    if (palmNormal.lengthSq() < 1e-8) return;
    palmNormal.normalize();

    // Rest 자세: 손가락 ±X 방향, 손바닥 -Y (rig는 rotateVRM0 이전 프레임 기준이라 VRM0는 X 반전)
    const boneAxis = new THREE.Vector3((isRight ? -1 : 1) * (isVRM0 ? -1 : 1), 0, 0);
    const qAim = new THREE.Quaternion().setFromUnitVectors(boneAxis, fingerDir);

    // 손바닥 방향 twist 정렬 (fingerDir에 수직인 성분만 사용해 순수 twist 보장)
    const palmProj = palmNormal.clone().addScaledVector(fingerDir, -palmNormal.dot(fingerDir));
    if (palmProj.lengthSq() > 1e-6) {
        palmProj.normalize();
        const currentPalm = new THREE.Vector3(0, -1, 0).applyQuaternion(qAim);
        const qTwist = new THREE.Quaternion().setFromUnitVectors(currentPalm, palmProj);
        qAim.premultiply(qTwist);
    }

    const parentWorldQuat = getParentWorldQuaternion(handBone);
    const qLocal = worldToLocalQuaternion(qAim, parentWorldQuat);
    handBone.quaternion.slerp(qLocal, factor * 0.5);

    // 손목 비틀림 서보: 손목 로컬 twist가 ±60°를 넘는 초과분을 전완 회내(pronation)
    // 상태로 이관 (다음 프레임 IK가 전완 롤로 적용 → 손목 twist가 그만큼 감소).
    // 실측(모션 레코딩): 손목이 최대 ~180°를 홀로 감당 → 팔꿈치·손목 경계 감김의 원인
    const state = ikPlaneState[prefix];
    if (state && deltaTime !== undefined) {
        const qs = qLocal.clone();
        if (qs.w < 0) qs.set(-qs.x, -qs.y, -qs.z, -qs.w);
        // 측정 축을 팔 체인의 boneAxis 부호 규약과 일치시킴 — 고정 +X로 측정하면
        // boneAxis가 -X인 조합(VRM1 오른팔 등)에서 서보 피드백 부호가 뒤집혀 발산함
        // (옆으로 벌린 팔 레코딩 실측: 전완 62°→191° 폭주, 손목이 동량 역방향)
        const axisSign = (isRight ? -1 : 1) * (isVRM0 ? -1 : 1);
        let wristTwist = 2 * Math.atan2(qs.x * axisSign, qs.w); // hand 본 축 기준 twist

        // 연속화(unwrap): 손목 twist가 ±180° 경계에 있으면 최단 표현의 부호가
        // 프레임마다 널뛰어 서보 적분이 자기 상쇄됨(모션 레코딩 실측) —
        // 이전 프레임 값에 가장 가까운 표현을 선택해 부호를 안정화
        if (state.wristTwist != null) {
            let delta = wristTwist - state.wristTwist;
            while (delta > Math.PI) delta -= 2 * Math.PI;
            while (delta < -Math.PI) delta += 2 * Math.PI;
            wristTwist = state.wristTwist + delta;
            // 과도 누적 방지 (1바퀴 초과 시 재래핑)
            if (wristTwist > 2 * Math.PI) wristTwist -= 2 * Math.PI;
            else if (wristTwist < -2 * Math.PI) wristTwist += 2 * Math.PI;
        }
        // 유한성 가드: NaN이 서보 상태에 들어오면 IK 전체로 오염이 번지므로 즉시 리셋
        if (!isFinite(wristTwist)) {
            state.wristTwist = null;
            return;
        }
        state.wristTwist = wristTwist;

        const limit = Math.PI / 3; // 손목 허용 비틀림 ±60°
        const overflow = wristTwist - THREE.MathUtils.clamp(wristTwist, -limit, limit);
        const target = THREE.MathUtils.clamp(state.pronation + overflow, -2.1, 2.1);
        state.pronation += (target - state.pronation) * getLerpFactor(deltaTime, 4);
        if (!isFinite(state.pronation)) state.pronation = 0;
    }
}

// 손이 화면에서 사라졌을 때 손목/손가락을 rest 자세로 복귀
const IDENTITY_QUAT = new THREE.Quaternion();
function relaxHand(prefix, deltaTime) {
    if (!currentVrm) return;

    const factor = getLerpFactor(deltaTime, 5); // 천천히 복귀

    const handBone = currentVrm.humanoid.getNormalizedBoneNode(prefix + 'Hand');
    if (handBone) handBone.quaternion.slerp(IDENTITY_QUAT, factor);

    for (const fingerName of Object.keys(FINGER_CONFIG)) {
        // 엄지는 Metacarpal→Proximal→Distal (three-vrm에 ThumbIntermediate 없음)
        const boneTypes = fingerName === 'Thumb'
            ? ['Metacarpal', 'Proximal', 'Distal']
            : ['Proximal', 'Intermediate', 'Distal'];
        for (const boneType of boneTypes) {
            const bone = currentVrm.humanoid.getNormalizedBoneNode(prefix + fingerName + boneType);
            if (bone) bone.quaternion.slerp(IDENTITY_QUAT, factor);
        }
    }
}

// 손가락 설정 (MCP, PIP, DIP, TIP 인덱스)
const FINGER_JOINTS = {
    Thumb:  { mcp: 1, pip: 2, dip: 3, tip: 4 },
    Index:  { mcp: 5, pip: 6, dip: 7, tip: 8 },
    Middle: { mcp: 9, pip: 10, dip: 11, tip: 12 },
    Ring:   { mcp: 13, pip: 14, dip: 15, tip: 16 },
    Little: { mcp: 17, pip: 18, dip: 19, tip: 20 }
};

// 거리 기반 손가락 처리
function applyFingers(prefix, landmarks, factor) {
    if (!currentVrm) return;

    const isRight = prefix === 'right';

    // 이미지 정규화 좌표는 x/y 스케일이 달라(1280 vs 720) 거리 계산 전에 aspect 보정
    const aspect = VIDEO_ASPECT;
    const lmVec = (i) => new THREE.Vector3(landmarks[i].x * aspect, landmarks[i].y, landmarks[i].z * aspect);

    for (const [fingerName, config] of Object.entries(FINGER_CONFIG)) {
        const { isThumb } = config;
        const joints = FINGER_JOINTS[fingerName];

        // 각 관절 위치
        const mcpPos = lmVec(joints.mcp);
        const pipPos = lmVec(joints.pip);
        const dipPos = lmVec(joints.dip);
        const tipPos = lmVec(joints.tip);

        // 각 세그먼트의 실제 길이 (굽혀도 변하지 않음)
        const seg1 = mcpPos.distanceTo(pipPos);  // MCP-PIP
        const seg2 = pipPos.distanceTo(dipPos);  // PIP-DIP
        const seg3 = dipPos.distanceTo(tipPos);  // DIP-TIP
        const totalLength = seg1 + seg2 + seg3;  // 손가락 전체 길이 (고정)
        if (totalLength < 1e-6) continue;

        // MCP에서 TIP까지 직선 거리 (굽히면 줄어듦)
        const straightDist = mcpPos.distanceTo(tipPos);

        // Curl raw: 펴진 손가락도 자연 굴곡으로 ~0.03-0.08, 주먹은 ~0.6 (엄지는 ~0.35)
        // 실측 범위를 0..1로 리매핑 — 기존 pow(x,0.7)*1.5 증폭은 편 손가락도 30% 굽혀 보이게 했음
        const rawCurl = 1 - (straightDist / totalLength);
        let curl = isThumb
            ? THREE.MathUtils.clamp((rawCurl - 0.04) / 0.30, 0, 1)
            : THREE.MathUtils.clamp((rawCurl - 0.07) / 0.50, 0, 1);

        // 각 관절에 curl 적용
        if (isThumb) {
            applyThumbCurl(prefix, curl, factor);
        } else {
            applyFingerCurl(prefix, fingerName, curl, factor);
        }
    }
}

// 일반 손가락 curl 적용
function applyFingerCurl(prefix, fingerName, curl, factor) {
    const boneTypes = ['Proximal', 'Intermediate', 'Distal'];
    const isRight = prefix === 'right';
    // rig의 rest 방향이 VRM0는 반대라 curl 회전 부호도 반전 (팔의 zDir 보정과 동일한 이유)
    const dir = (isRight ? 1 : -1) * (currentVrm.meta?.metaVersion === '0' ? -1 : 1);

    // 각 관절의 최대 굽힘 각도
    const maxAngles = [Math.PI * 0.45, Math.PI * 0.55, Math.PI * 0.45];

    boneTypes.forEach((boneType, idx) => {
        const boneName = prefix + fingerName + boneType;
        const bone = currentVrm.humanoid.getNormalizedBoneNode(boneName);

        if (bone) {
            const angle = curl * maxAngles[idx];
            const rotation = new THREE.Quaternion().setFromEuler(new THREE.Euler(
                0,
                0,
                angle * dir
            ));
            bone.quaternion.slerp(rotation, factor);
        }
    });
}

// 엄지 curl 적용
// 주의: three-vrm 표준 엄지 본은 Metacarpal→Proximal→Distal ('ThumbIntermediate'는 존재하지 않음)
function applyThumbCurl(prefix, curl, factor) {
    const isRight = prefix === 'right';
    // rig의 rest 방향이 VRM0는 반대라 회전 부호도 반전
    const dir = (isRight ? -1 : 1) * (currentVrm.meta?.metaVersion === '0' ? -1 : 1);

    const metacarpal = currentVrm.humanoid.getNormalizedBoneNode(prefix + 'ThumbMetacarpal');
    const proximal = currentVrm.humanoid.getNormalizedBoneNode(prefix + 'ThumbProximal');
    const distal = currentVrm.humanoid.getNormalizedBoneNode(prefix + 'ThumbDistal');

    // 엄지는 손바닥에서 비스듬히 나오므로 Y축 회전으로 손바닥 안쪽으로 접힘
    const maxAngles = { metacarpal: 0.35, proximal: 0.45, distal: 0.4 };

    if (metacarpal) {
        const angle = curl * Math.PI * maxAngles.metacarpal;
        const rotation = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, angle * dir, 0));
        metacarpal.quaternion.slerp(rotation, factor);
    }

    if (proximal) {
        const angle = curl * Math.PI * maxAngles.proximal;
        const rotation = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, angle * dir, 0));
        proximal.quaternion.slerp(rotation, factor);
    }

    if (distal) {
        const angle = curl * Math.PI * maxAngles.distal;
        const rotation = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, angle * dir, 0));
        distal.quaternion.slerp(rotation, factor);
    }
}

// ============================================================
// 표정 처리
// ============================================================
// VTuber 용도로 표정을 실측보다 과장해서 표현:
// deadzone 이하의 노이즈는 무시하고, 중간 강도를 curve(<1)로 끌어올린 뒤
// EXPRESSION_EXAGGERATION 배율을 곱해 0..1로 클램프
const EXPRESSION_EXAGGERATION = 1.35; // 전체 과장 배율 (1.0 = 실측 그대로)

function exaggerate(score, deadzone = 0.05, curve = 0.75) {
    const s = Math.max(0, (score - deadzone) / (1 - deadzone));
    return THREE.MathUtils.clamp(Math.pow(s, curve) * EXPRESSION_EXAGGERATION, 0, 1);
}

function applyBlendshapes(blendShapesData, deltaTime) {
    if (!currentVrm) return;

    // 표정은 빠르게 반응하되, 원거리에서는 blendshape 노이즈가 커지므로 스무딩 강화
    const factor = getLerpFactor(deltaTime, 15 * trackingDistConf);

    const presetName = VRMExpressionPresetName;
    const expressions = currentVrm.expressionManager;

    const getScore = (name) => {
        const shape = blendShapesData.categories.find(s => s.categoryName === name);
        return shape ? shape.score : 0;
    };

    // ============================================================
    // 1. 입모양 (Lip Sync) 목표값 계산 — 적용은 아래에서 일괄 수행
    // 입모양은 과장 없이 실측값 사용 (과장 시 부자연스럽다는 피드백 반영)
    // ============================================================

    // 입 벌림 (あ) - jawOpen을 직접 사용
    let aaScore = getScore('jawOpen');

    // 둥근 입 - "우"(Ou)와 "오"(Oh) 구분:
    // 입을 둥글게 모은 상태(pucker/funnel)에서 턱이 벌어질수록 Ou → Oh로 전환
    // (Oh 미지원 모델은 기존처럼 전부 Ou로)
    const mouthPucker = getScore('mouthPucker');
    const mouthFunnel = getScore('mouthFunnel');
    const roundness = Math.max(mouthPucker, mouthFunnel * 0.8);
    const hasOh = expressions.getValue(presetName.Oh) !== null;
    const openWeight = hasOh ? THREE.MathUtils.clamp((aaScore - 0.05) / 0.25, 0, 1) : 0;
    let ouScore = roundness * (1 - openWeight);
    let ohScore = roundness * openWeight;
    if (ohScore > 0) {
        // Oh morph가 자체적으로 턱을 벌리므로 Aa의 중복 벌림을 완화
        aaScore *= (1 - ohScore * 0.6);
    }

    // 입 넓히기 (い) - mouthStretch 사용
    const mouthStretchL = getScore('mouthStretchLeft');
    const mouthStretchR = getScore('mouthStretchRight');
    let ihScore = ((mouthStretchL + mouthStretchR) / 2) * 0.5;

    // ============================================================
    // 2. 눈 (독립적으로 동작)
    // ============================================================

    // 눈 깜빡임 - deadzone을 높여 뜬 눈은 또렷하게, 감는 동작은 빠르게 완결
    // 미러 모드: 사용자 오른눈 → 아바타 왼눈 (팔/손과 동일한 좌우 반전 — 윙크 방향 일치)
    let blinkL = exaggerate(getScore('eyeBlinkRight'), 0.2, 0.7);
    let blinkR = exaggerate(getScore('eyeBlinkLeft'), 0.2, 0.7);
    const currentBlinkL = expressions.getValue(presetName.BlinkLeft) ?? 0;
    const currentBlinkR = expressions.getValue(presetName.BlinkRight) ?? 0;

    // 윙크 분리: 좌우 깜빡임 차이가 크면(한쪽 눈 윙크) 크로스토크를 제거해
    // 감는 쪽은 완전히 감고 뜬 쪽은 완전히 뜨게
    const winkStrength = THREE.MathUtils.clamp((Math.abs(blinkL - blinkR) - 0.2) / 0.3, 0, 1);
    if (winkStrength > 0) {
        if (blinkL > blinkR) {
            blinkL = THREE.MathUtils.lerp(blinkL, 1, winkStrength);
            blinkR = THREE.MathUtils.lerp(blinkR, 0, winkStrength);
        } else {
            blinkR = THREE.MathUtils.lerp(blinkR, 1, winkStrength);
            blinkL = THREE.MathUtils.lerp(blinkL, 0, winkStrength);
        }
    }

    // 눈 찡그림 (웃을 때) - eyeSquint를 깜빡임에 더해줌 (좌우 반전 매핑,
    // 완전히 감지 않도록 제한하되 윙크 중에는 분리 결과를 침범하지 않음)
    const eyeSquintL = getScore('eyeSquintRight') * (1 - winkStrength);
    const eyeSquintR = getScore('eyeSquintLeft') * (1 - winkStrength);
    const squintBlinkL = Math.max(blinkL, Math.min(blinkL + eyeSquintL * 0.4, 0.9));
    const squintBlinkR = Math.max(blinkR, Math.min(blinkR + eyeSquintR * 0.4, 0.9));
    expressions.setValue(presetName.BlinkLeft, THREE.MathUtils.lerp(currentBlinkL, squintBlinkL, factor));
    expressions.setValue(presetName.BlinkRight, THREE.MathUtils.lerp(currentBlinkR, squintBlinkR, factor));

    // 놀람 (눈 크게 뜨기) - 모델에 Surprised 표정이 있을 때만 적용
    const eyeWide = (getScore('eyeWideLeft') + getScore('eyeWideRight')) / 2;
    const surprisedScore = eyeWide > 0.25 ? exaggerate(eyeWide, 0.25, 0.8) * 0.8 : 0;
    const currentSurprised = expressions.getValue(presetName.Surprised);
    if (currentSurprised !== null) {
        expressions.setValue(presetName.Surprised, THREE.MathUtils.lerp(currentSurprised, surprisedScore, factor));
    }

    // 입 morph 합계 정규화: 여러 viseme을 동시에 최대치로 걸면 일부 모델(특히 VRM0)의
    // 입술 위·턱 아래 mesh가 찢어져 빈 공간/흐릿한 부분이 보임 → 합이 1을 넘으면 비례 축소
    const mouthSum = aaScore + ouScore + ihScore + ohScore;
    if (mouthSum > 1) {
        const s = 1 / mouthSum;
        aaScore *= s;
        ouScore *= s;
        ihScore *= s;
        ohScore *= s;
    }

    const setMouth = (name, target) => {
        const current = expressions.getValue(name);
        if (current === null) return; // 모델에 없는 표정은 스킵
        expressions.setValue(name, THREE.MathUtils.lerp(current, target, factor));
    };
    setMouth(presetName.Aa, aaScore);
    setMouth(presetName.Ou, ouScore);
    setMouth(presetName.Ih, ihScore);
    setMouth(presetName.Oh, ohScore);

    // ============================================================
    // 3. 표정 (감정) - 입모양(lip sync)을 완전히 덮지 않는 선에서 과장
    // ============================================================

    // 웃음 - 임계값을 낮추고 상한을 올려 잘 웃는 캐릭터로.
    // 단, 입을 크게 벌릴수록 감쇠 — 웃음 morph가 입 벌림 morph와 중첩되면
    // 일부 모델에서 mesh가 깨지므로 (입모양 우선)
    const smileL = getScore('mouthSmileLeft');
    const smileR = getScore('mouthSmileRight');
    const smileScore = (smileL + smileR) / 2;
    const happyScore = (smileScore > 0.2 ? Math.min(smileScore * 0.8, 0.8) : 0) * (1 - aaScore * 0.5);
    const currentHappy = expressions.getValue(presetName.Happy) ?? 0;
    expressions.setValue(presetName.Happy, THREE.MathUtils.lerp(currentHappy, happyScore, factor));

    // 슬픔 (눈썹 올림)
    const browInnerUp = getScore('browInnerUp');
    const browDownL = getScore('browDownLeft');
    const browDownR = getScore('browDownRight');
    if (browInnerUp > 0.25) {
        const sadScore = Math.min(browInnerUp * 0.7, 0.6);
        const currentSad = expressions.getValue(presetName.Sad) ?? 0;
        expressions.setValue(presetName.Sad, THREE.MathUtils.lerp(currentSad, sadScore, factor));
    } else {
        const currentSad = expressions.getValue(presetName.Sad) ?? 0;
        expressions.setValue(presetName.Sad, THREE.MathUtils.lerp(currentSad, 0, factor));
    }

    // 화남 (눈썹 찌푸림)
    const angryScore = (browDownL + browDownR) / 2;
    if (angryScore > 0.25) {
        const currentAngry = expressions.getValue(presetName.Angry) ?? 0;
        expressions.setValue(presetName.Angry, THREE.MathUtils.lerp(currentAngry, Math.min(angryScore * 0.6, 0.6), factor));
    } else {
        const currentAngry = expressions.getValue(presetName.Angry) ?? 0;
        expressions.setValue(presetName.Angry, THREE.MathUtils.lerp(currentAngry, 0, factor));
    }

    expressions.update();
}

// ============================================================
// 머리 회전
// ============================================================
function applyHeadRotation(matrix, deltaTime) {
    if (!currentVrm) return;

    // 원거리에서는 얼굴 행렬 노이즈가 커지므로 스무딩 강화
    const factor = getLerpFactor(deltaTime, 10 * trackingDistConf);

    const m = new THREE.Matrix4().fromArray(matrix.data);
    const rot = new THREE.Quaternion().setFromRotationMatrix(m);

    // Euler로 변환하여 축별 조정
    const euler = new THREE.Euler().setFromQuaternion(rot, 'YXZ');
    const isVRM0 = currentVrm.meta?.metaVersion === '0';

    // 180°Y conjugation(rotateVRM0)은 X·Z 회전 부호를 반전(Y는 불변),
    // 미러링은 Y·Z를 반전(X는 불변) →
    //   VRM1: y·z만 반전 / VRM0: x·y 반전, z는 두 반전이 상쇄되어 그대로
    if (isVRM0) euler.x *= -1;
    euler.y *= -1;
    if (!isVRM0) euler.z *= -1;

    // 회전 범위 제한 (과도한 회전 방지)
    euler.x = THREE.MathUtils.clamp(euler.x, -Math.PI / 4, Math.PI / 4);
    euler.y = THREE.MathUtils.clamp(euler.y, -Math.PI / 3, Math.PI / 3);
    euler.z = THREE.MathUtils.clamp(euler.z, -Math.PI / 6, Math.PI / 6);

    const mirrorRot = new THREE.Quaternion().setFromEuler(euler);

    const head = currentVrm.humanoid.getNormalizedBoneNode('head');
    if (head) {
        head.quaternion.slerp(mirrorRot, factor);
    }

    // Neck도 약간 회전 (더 자연스러운 움직임)
    const neck = currentVrm.humanoid.getNormalizedBoneNode('neck');
    if (neck) {
        const neckEuler = new THREE.Euler(
            euler.x * 0.3,
            euler.y * 0.3,
            euler.z * 0.3,
            'YXZ'
        );
        const neckRot = new THREE.Quaternion().setFromEuler(neckEuler);
        neck.quaternion.slerp(neckRot, factor * 0.5);
    }
}

init();
