import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils, VRMExpressionPresetName } from '@pixiv/three-vrm';
import { FilesetResolver, FaceLandmarker, PoseLandmarker, HandLandmarker, DrawingUtils } from '@mediapipe/tasks-vision';

// --- Configuration ---
const VIDEO_WIDTH = 1280;
const VIDEO_HEIGHT = 720;

// --- Improved Configuration ---
const LERP_SPEED = 12; // 반응 속도 (높을수록 빠름)
const VIS_THRESHOLD_ON = 0.65;  // 활성화 임계값
const VIS_THRESHOLD_OFF = 0.45; // 비활성화 임계값 (hysteresis)

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

    const t = timestamp / 1000;  // 초 단위로 변환

    const filteredLandmarks = landmarks.map((lm, i) => poseFilters.landmarks[i].filter(lm, t));
    const filteredWorldLandmarks = worldLandmarks
        ? worldLandmarks.map((lm, i) => poseFilters.worldLandmarks[i].filter(lm, t))
        : null;

    return { filteredLandmarks, filteredWorldLandmarks };
}

// --- Globals ---
let scene, camera, renderer;
let video;
let faceLandmarker, poseLandmarker, handLandmarker;
let currentVrm;
let lastVideoTime = -1;
let lastFrameTime = performance.now();
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

// --- Hand Tracking 결과 저장 (Body Tracking과 통합용) ---
let detectedHands = {
    left: null,   // MediaPipe Left Hand → Avatar Right
    right: null   // MediaPipe Right Hand → Avatar Left
};

// --- Unified Dialogue System ---
const MAX_VISIBLE_MESSAGES = 5;
let dialogueMessages = [];
let isDialogueEnabled = false;
let dialogueDisplayMode = 'single';  // 'history' or 'single'
let dialogueInputMode = 'voice';     // 'typing' or 'voice'
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
        const scaleX = canvasWidth / window.innerWidth;
        const scaleY = canvasHeight / window.innerHeight;
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
        console.error('Speech recognition error:', event.error);
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
                console.warn('[Captions] Network error, will retry...');
                // 자동 재시도는 onend에서 처리됨
                break;
            case 'aborted':
                console.log('[Captions] Recognition aborted');
                break;
            default:
                console.warn('[Captions] Unhandled error:', event.error);
        }
    };

    speechRecognition.onend = () => {
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
        const scaleX = canvasWidth / window.innerWidth;
        const scaleY = canvasHeight / window.innerHeight;
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
        const scaleX = canvasWidth / window.innerWidth;
        const scaleY = canvasHeight / window.innerHeight;
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
}

// --- Initialization ---
async function init() {
    // Dev dropdown menu handlers
    document.querySelectorAll('.option-btn[data-dev]').forEach(btn => {
        btn.addEventListener('click', () => {
            const action = btn.dataset.dev;

            if (action === 'debug-on') {
                isDebugView = true;
                updateDevOptions();
                updateView();
            } else if (action === 'debug-off') {
                isDebugView = false;
                updateDevOptions();
                updateView();
            } else if (action === 'landmarks-on') {
                DEBUG_MODE = true;
                updateDevOptions();
            } else if (action === 'landmarks-off') {
                DEBUG_MODE = false;
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

    const toggleBodyBtn = document.getElementById('toggle-body');
    if (toggleBodyBtn) {
        toggleBodyBtn.addEventListener('click', () => {
            BODY_TRACKING_ENABLED = !BODY_TRACKING_ENABLED;
            toggleBodyBtn.innerHTML = BODY_TRACKING_ENABLED ? "Pose<br>ON" : "Pose<br>OFF";
            // Body tracking 비활성화 시 팔 상태 리셋
            if (!BODY_TRACKING_ENABLED) {
                leftArmActive = false;
                rightArmActive = false;
            }
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

    // Unified dialogue system
    setupDialogue();

    // 키보드 단축키: Escape로 녹화 중지
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && mediaRecorder && mediaRecorder.state === 'recording') {
            stopRecording();
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
    loadAvatar();

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
    const toggleAvatarSizeBtn = document.getElementById('toggle-avatar-size');
    const toggleMicBtn = document.getElementById('toggle-mic');
    const toggleRecordBtn = document.getElementById('toggle-record');
    const toggleCameraBtn = document.getElementById('toggle-camera');

    if (toggleScreenBtn) {
        toggleScreenBtn.addEventListener('click', toggleScreenCapture);
    }
    if (toggleAvatarSizeBtn) {
        toggleAvatarSizeBtn.addEventListener('click', toggleAvatarSize);
    }
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

    function updateMeters() {
        updateAudioMeters();
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

    const btn = document.getElementById('toggle-avatar-size');
    const sceneWrapper = document.getElementById('scene-wrapper');

    if (btn) {
        btn.innerHTML = isMiniAvatar ? 'Avatar<br>Mini' : 'Avatar<br>Full';
    }

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

    // 대화 오버레이 위치 업데이트
    updateDialogueOverlayPosition();

    e.preventDefault();
}

function onDragEnd() {
    isDragging = false;
    document.removeEventListener('mousemove', onDragMove);
    document.removeEventListener('mouseup', onDragEnd);
    document.removeEventListener('touchmove', onDragMove);
    document.removeEventListener('touchend', onDragEnd);
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

    // 카메라 프리뷰 다시 보이기
    document.body.classList.remove('screen-sharing');

    // 미니 아바타 모드 리셋
    if (isMiniAvatar) {
        isMiniAvatar = false;
        document.body.classList.remove('mini-avatar');
        const btn = document.getElementById('toggle-avatar-size');
        if (btn) btn.innerHTML = 'Avatar<br>Full';

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

    // 합성 캔버스 생성 (DOM에 추가하여 captureStream 호환성 확보)
    compositeCanvas = document.createElement('canvas');
    compositeCanvas.width = 1920;
    compositeCanvas.height = 1080;
    compositeCanvas.style.cssText = 'position:fixed;top:-9999px;left:-9999px;pointer-events:none;';
    document.body.appendChild(compositeCanvas);
    compositeCtx = compositeCanvas.getContext('2d');

    // 합성 루프 시작
    function compositeFrame() {
        // 1. 배경 그리기 (화면 공유가 있으면)
        if (screenBg && screenBg.srcObject) {
            // 비디오를 캔버스 중앙에 맞춰 그리기
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

            compositeCtx.fillStyle = '#000';
            compositeCtx.fillRect(0, 0, compositeCanvas.width, compositeCanvas.height);
            compositeCtx.drawImage(screenBg, drawX, drawY, drawWidth, drawHeight);
        } else {
            compositeCtx.fillStyle = '#000';
            compositeCtx.fillRect(0, 0, compositeCanvas.width, compositeCanvas.height);
        }

        // 2. 아바타 캔버스 그리기
        if (isMiniAvatar) {
            // 미니 모드: 현재 위치에 맞춰 그리기
            const miniWidth = 300;
            const miniHeight = 400;

            // 현재 창 크기에 맞게 위치 클램프 (창 크기 변경 대응)
            const clampedX = Math.min(miniAvatarPosition.x || 0, window.innerWidth - miniWidth);
            const clampedY = Math.min(miniAvatarPosition.y || 0, window.innerHeight - miniHeight);
            const safeX = Math.max(0, clampedX);
            const safeY = Math.max(0, clampedY);

            // 균일한 스케일 사용 (비율 유지)
            const scaleX = compositeCanvas.width / window.innerWidth;
            const scaleY = compositeCanvas.height / window.innerHeight;
            const uniformScale = Math.min(scaleX, scaleY);

            const scaledWidth = miniWidth * uniformScale;
            const scaledHeight = miniHeight * uniformScale;

            // 위치 계산: 하단/우측 경계 기준으로 정렬
            // 미니 아바타의 우측 끝이 창 우측에 있으면 녹화에서도 우측에
            // 미니 아바타의 하단 끝이 창 하단에 있으면 녹화에서도 하단에
            const rightEdge = safeX + miniWidth;
            const bottomEdge = safeY + miniHeight;

            // X 위치: 우측 경계 기준으로 계산
            const miniX = (rightEdge / window.innerWidth) * compositeCanvas.width - scaledWidth;
            // Y 위치: 하단 경계 기준으로 계산
            const miniY = (bottomEdge / window.innerHeight) * compositeCanvas.height - scaledHeight;

            compositeCtx.drawImage(avatarCanvas, miniX, miniY, scaledWidth, scaledHeight);
        } else {
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
        drawDialogueToCanvas(compositeCtx, compositeCanvas.width, compositeCanvas.height);

        compositeAnimationId = requestAnimationFrame(compositeFrame);
    }
    compositeFrame();

    // 합성 캔버스 스트림 캡처 (30fps)
    const canvasStream = compositeCanvas.captureStream(30);

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

    mediaRecorder = new MediaRecorder(combinedStream, { mimeType });

    mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
            recordedChunks.push(e.data);
        }
    };

    mediaRecorder.onerror = (event) => {
        console.error('[Recording] MediaRecorder error:', event.error);
        alert('녹화 중 오류가 발생했습니다: ' + (event.error?.message || 'Unknown error'));
        stopRecording();
    };

    mediaRecorder.onstop = () => {
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
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `avatar-recording-${Date.now()}.webm`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    URL.revokeObjectURL(url);
    recordedChunks = [];
}

function updateScreenCaptureButtons(isCapturing) {
    const toggleScreenBtn = document.getElementById('toggle-screen');
    const toggleRecordBtn = document.getElementById('toggle-record');

    if (toggleScreenBtn) {
        toggleScreenBtn.innerHTML = isCapturing ? 'Stop<br>Capture' : 'Screen<br>Capture';
        if (isCapturing) {
            toggleScreenBtn.classList.add('mic-active');
        } else {
            toggleScreenBtn.classList.remove('mic-active');
        }
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
    camera.position.set(0.0, 1.4, 1.5);

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

    const resizeObserver = new ResizeObserver(() => {
        if (sceneWrapper) {
            const newWidth = sceneWrapper.clientWidth;
            const newHeight = sceneWrapper.clientHeight;
            if (newWidth > 0 && newHeight > 0) {
                camera.aspect = newWidth / newHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(newWidth, newHeight);
            }
        }
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

        poseLandmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
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

async function loadAvatar() {
    const loader = new GLTFLoader();
    loader.register((parser) => {
        return new VRMLoaderPlugin(parser);
    });

    try {
        const url = './avatar.vrm';
        const gltf = await loader.loadAsync(url);
        const vrm = gltf.userData.vrm;

        VRMUtils.removeUnnecessaryVertices(gltf.scene);
        VRMUtils.removeUnnecessaryJoints(gltf.scene);

        scene.add(vrm.scene);
        currentVrm = vrm;
        console.log("Avatar loaded");
    } catch (err) {
        console.error("VRM load error:", err);
        alert('아바타 로딩 실패. 페이지를 새로고침해주세요.');
    }
}

function animate() {
    requestAnimationFrame(animate);

    const currentTime = performance.now();
    const deltaTime = (currentTime - lastFrameTime) / 1000; // 초 단위
    lastFrameTime = currentTime;

    if (video && video.readyState >= 2) {
        if (DEBUG_MODE && debugCtx) {
            debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
        }

        if (video.currentTime !== lastVideoTime) {
            lastVideoTime = video.currentTime;

            // 1. Detect Face
            if (faceLandmarker) {
                const results = faceLandmarker.detectForVideo(video, currentTime);
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
            if (BODY_TRACKING_ENABLED) {
                // Hand tracking 결과 초기화
                detectedHands.left = null;
                detectedHands.right = null;

                if (handLandmarker) {
                    const handResults = handLandmarker.detectForVideo(video, currentTime);
                    if (handResults.landmarks && handResults.landmarks.length > 0) {
                        // Hand 결과 저장 (Pose에서 사용)
                        for (let i = 0; i < handResults.landmarks.length; i++) {
                            const handedness = handResults.handednesses[i][0];
                            const landmarks = handResults.landmarks[i];

                            // MediaPipe Left → Avatar Right, MediaPipe Right → Avatar Left
                            if (handedness.categoryName === 'Left') {
                                detectedHands.left = landmarks;
                            } else {
                                detectedHands.right = landmarks;
                            }
                        }

                        // 손가락 처리
                        applyHands(handResults.landmarks, handResults.handednesses, deltaTime);

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
                if (poseLandmarker) {
                    const poseResults = poseLandmarker.detectForVideo(video, currentTime);
                    if (poseResults.landmarks && poseResults.landmarks.length > 0) {
                        const rawLandmarks = poseResults.landmarks[0];
                        const rawWorldLandmarks = poseResults.worldLandmarks ? poseResults.worldLandmarks[0] : null;

                        // One Euro Filter 적용 (떨림 완화)
                        const { filteredLandmarks, filteredWorldLandmarks } = getFilteredPoseLandmarks(
                            rawLandmarks, rawWorldLandmarks, currentTime
                        );

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
            } else {
                // Body tracking 비활성화 시 팔을 자연스럽게 내림
                resetPose(deltaTime);
            }
        }
    }

    if (currentVrm) {
        currentVrm.update(deltaTime);
    }

    renderer.render(scene, camera);
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
    // VRM 기본: T-Pose (팔이 수평)
    // 내린 상태: Z축 회전
    const relaxRight = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, Math.PI * 0.45, 'XYZ'));
    const relaxLeft = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -Math.PI * 0.45, 'XYZ'));
    const neutralLower = new THREE.Quaternion(); // 아래팔은 펴진 상태

    if (rightUpperArm) rightUpperArm.quaternion.slerp(relaxRight, factor);
    if (leftUpperArm) leftUpperArm.quaternion.slerp(relaxLeft, factor);
    if (rightLowerArm) rightLowerArm.quaternion.slerp(neutralLower, factor);
    if (leftLowerArm) leftLowerArm.quaternion.slerp(neutralLower, factor);

    // 활성 상태 리셋
    leftArmActive = false;
    rightArmActive = false;
}

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
        for (let i = 0; i < worldLandmarks.length; i++) {
            const l = worldLandmarks[i];
            const sphere = debugGroup.children[i];
            if (sphere) {
                const pos = mpToVRM(l);
                sphere.position.copy(pos);
            }
        }
    }
}

// ============================================================
// 개선된 Two-Bone IK Solver
// ============================================================
function solveTwoBoneIK(upperBone, lowerBone, upperLength, lowerLength, targetPos, polePos, boneAxis, deltaTime) {
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

    if (planeNormal.lengthSq() < 0.0001) {
        // 팔꿈치가 직선상에 있는 경우 대체 벡터 사용
        planeNormal.crossVectors(dirToTarget, new THREE.Vector3(0, 1, 0));
        if (planeNormal.lengthSq() < 0.0001) {
            planeNormal.crossVectors(dirToTarget, new THREE.Vector3(0, 0, 1));
        }
    }
    planeNormal.normalize();

    // 5. Upper Arm (어깨) 방향 계산
    // 목표 방향에서 shoulderAngle만큼 회전
    const qBend = new THREE.Quaternion().setFromAxisAngle(planeNormal, shoulderAngle);
    const upperDir = dirToTarget.clone().applyQuaternion(qBend).normalize();

    // 6. Upper Arm 회전 계산
    const qUpper = new THREE.Quaternion().setFromUnitVectors(boneAxis, upperDir);

    // 팔꿈치 방향(hinge) 정렬
    const hingeAxis = new THREE.Vector3().crossVectors(boneAxis, new THREE.Vector3(0, 0, 1)).normalize();
    const currentHinge = hingeAxis.clone().applyQuaternion(qUpper);
    const qTwist = new THREE.Quaternion().setFromUnitVectors(currentHinge, planeNormal);
    const qUpperFinal = qTwist.multiply(qUpper);

    // 7. 부모 좌표계를 고려한 로컬 회전 적용
    const parentWorldQuat = getParentWorldQuaternion(upperBone);
    const qUpperLocal = worldToLocalQuaternion(qUpperFinal, parentWorldQuat);

    upperBone.quaternion.slerp(qUpperLocal, factor);

    // 8. Lower Arm (팔꿈치) 회전 계산
    // 팔꿈치는 단순히 구부러지는 각도만 적용 (hinge joint)
    // 팔꿈치 각도: π - elbowAngle (펴진 상태가 π)
    const bendAngle = Math.PI - elbowAngle;

    // 로컬 X축 기준 회전 (팔꿈치는 한 축으로만 회전)
    const qLowerLocal = new THREE.Quaternion().setFromAxisAngle(
        new THREE.Vector3(0, 1, 0), // VRM에서 팔꿈치 회전축
        -bendAngle * (boneAxis.x < 0 ? 1 : -1) // 좌우 팔 방향에 따라 부호 조정
    );

    lowerBone.quaternion.slerp(qLowerLocal, factor);
}

function applyPose(landmarks, worldLandmarks, deltaTime) {
    if (!currentVrm) return;

    updateDebug3D(worldLandmarks);

    const factor = getLerpFactor(deltaTime);

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

    // --- Spine 회전 (상체 기울기) ---
    const spine = currentVrm.humanoid.getNormalizedBoneNode('spine');
    if (spine && landmarks) {
        const mpLeft = landmarks[11];  // 왼쪽 어깨
        const mpRight = landmarks[12]; // 오른쪽 어깨

        // 미러링 적용: 좌우 반전
        const dy = mpRight.y - mpLeft.y;  // 반전
        const dz = mpRight.z - mpLeft.z;  // 반전

        const roll = dy * 1.2;  // Z축 회전 (좌우 기울기)
        const yaw = dz * 1.0;   // Y축 회전 (어깨 회전)

        const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, yaw, roll, 'XYZ'));
        spine.quaternion.slerp(q, factor * 0.5); // 상체는 더 부드럽게
    }

    // --- Visibility 체크 (Hysteresis 적용) ---
    // MediaPipe Left Wrist (15) → Avatar Right Arm (미러링)
    // MediaPipe Right Wrist (16) → Avatar Left Arm (미러링)

    const leftWristVis = landmarks[15].visibility ?? 1.0;
    const rightWristVis = landmarks[16].visibility ?? 1.0;
    const leftWristY = landmarks[15].y;
    const rightWristY = landmarks[16].y;

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

        solveTwoBoneIK(rUpper, rLower, upperLen, lowerLen, target, pole, new THREE.Vector3(-1, 0, 0), deltaTime);
    } else if (rUpper && !leftArmActive) {
        // 팔 내리기
        const relaxQuat = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, Math.PI * 0.45, 'XYZ'));
        const neutralQuat = new THREE.Quaternion();
        rUpper.quaternion.slerp(relaxQuat, factor * 0.3);
        if (rLower) rLower.quaternion.slerp(neutralQuat, factor * 0.3);
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

        solveTwoBoneIK(lUpper, lLower, upperLen, lowerLen, target, pole, new THREE.Vector3(1, 0, 0), deltaTime);
    } else if (lUpper && !rightArmActive) {
        // 팔 내리기
        const relaxQuat = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -Math.PI * 0.45, 'XYZ'));
        const neutralQuat = new THREE.Quaternion();
        lUpper.quaternion.slerp(relaxQuat, factor * 0.3);
        if (lLower) lLower.quaternion.slerp(neutralQuat, factor * 0.3);
    }
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

    const factor = getLerpFactor(deltaTime, 15);
    const now = performance.now();
    const shouldLog = (now - lastDebugTime) > DEBUG_INTERVAL;

    for (let i = 0; i < landmarksArray.length; i++) {
        const landmarks = landmarksArray[i];
        const handedness = handednesses[i][0];

        // 손 위치로 좌우 판단 (wrist x 좌표)
        const wristX = landmarks[0].x;  // 0~1, 왼쪽이 0, 오른쪽이 1

        // 미러링: Body tracking과 일치하도록
        // MediaPipe "Left" = 사용자의 왼손 → Avatar의 왼손 (미러 모드)
        // MediaPipe "Right" = 사용자의 오른손 → Avatar의 오른손 (미러 모드)
        const isAvatarRightHand = handedness.categoryName === 'Right';
        const prefix = isAvatarRightHand ? 'right' : 'left';

        // Hand bone에 고정 회전 적용 (손바닥이 앞을 향하도록)
        applyHandFixedRotation(prefix, factor);

        // 손가락 처리 (거리 기반)
        applyFingers(prefix, landmarks, factor);

        if (shouldLog && i === 0) {
            lastDebugTime = now;
        }
    }
}

// Hand bone에 고정 회전 적용 (손바닥이 앞을 향하도록)
function applyHandFixedRotation(prefix, factor) {
    if (!currentVrm) return;

    const handBone = currentVrm.humanoid.getNormalizedBoneNode(prefix + 'Hand');
    if (!handBone) return;

    const isRight = prefix === 'right';

    // T-Pose에서 손바닥은 아래를 향함
    // 손바닥이 앞(카메라)을 향하려면 X축으로 -90도 회전
    // Y축으로 약간 안쪽 회전 추가 (팔 들었을 때 보정)
    const yTwist = isRight ? -0.3 : 0.3;  // 약 17도 안쪽으로

    const handRot = new THREE.Quaternion().setFromEuler(new THREE.Euler(
        -Math.PI / 2,  // X축: 손바닥을 앞으로
        yTwist,        // Y축: 약간 안쪽으로
        0
    ));

    handBone.quaternion.slerp(handRot, factor * 0.3);
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

    for (const [fingerName, config] of Object.entries(FINGER_CONFIG)) {
        const { mcp, tip, isThumb } = config;
        const joints = FINGER_JOINTS[fingerName];

        // 각 관절 위치
        const mcpPos = new THREE.Vector3(landmarks[joints.mcp].x, landmarks[joints.mcp].y, landmarks[joints.mcp].z);
        const pipPos = new THREE.Vector3(landmarks[joints.pip].x, landmarks[joints.pip].y, landmarks[joints.pip].z);
        const dipPos = new THREE.Vector3(landmarks[joints.dip].x, landmarks[joints.dip].y, landmarks[joints.dip].z);
        const tipPos = new THREE.Vector3(landmarks[joints.tip].x, landmarks[joints.tip].y, landmarks[joints.tip].z);

        // 각 세그먼트의 실제 길이 (굽혀도 변하지 않음)
        const seg1 = mcpPos.distanceTo(pipPos);  // MCP-PIP
        const seg2 = pipPos.distanceTo(dipPos);  // PIP-DIP
        const seg3 = dipPos.distanceTo(tipPos);  // DIP-TIP
        const totalLength = seg1 + seg2 + seg3;  // 손가락 전체 길이 (고정)

        // MCP에서 TIP까지 직선 거리 (굽히면 줄어듦)
        const straightDist = mcpPos.distanceTo(tipPos);

        // Curl: 직선 거리 / 전체 길이
        // 펴진 상태: straightDist ≈ totalLength → curl ≈ 0
        // 굽힌 상태: straightDist << totalLength → curl → 1
        let curl = 1 - (straightDist / totalLength);
        curl = THREE.MathUtils.clamp(curl, 0, 1);

        // curl 값 증폭 (더 민감하게)
        curl = Math.pow(curl, 0.7) * 1.5;
        curl = THREE.MathUtils.clamp(curl, 0, 1);

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
                isRight ? angle : -angle
            ));
            bone.quaternion.slerp(rotation, factor);
        }
    });
}

// 엄지 curl 적용
function applyThumbCurl(prefix, curl, factor) {
    const isRight = prefix === 'right';

    const proximal = currentVrm.humanoid.getNormalizedBoneNode(prefix + 'ThumbProximal');
    const intermediate = currentVrm.humanoid.getNormalizedBoneNode(prefix + 'ThumbIntermediate');
    const distal = currentVrm.humanoid.getNormalizedBoneNode(prefix + 'ThumbDistal');

    // 엄지는 손바닥에서 비스듬히 나오므로 복합 회전 필요
    // 손바닥이 앞을 향하는 상태에서 엄지가 손바닥 안쪽으로 접히도록
    if (proximal) {
        const angle = curl * Math.PI * 0.4;
        // Y축 회전으로 손바닥 안쪽으로 접힘
        const rotation = new THREE.Quaternion().setFromEuler(new THREE.Euler(
            0,
            isRight ? -angle : angle,  // Y축 회전
            0
        ));
        proximal.quaternion.slerp(rotation, factor);
    }

    if (intermediate) {
        const angle = curl * Math.PI * 0.45;
        const rotation = new THREE.Quaternion().setFromEuler(new THREE.Euler(
            0,
            isRight ? -angle : angle,
            0
        ));
        intermediate.quaternion.slerp(rotation, factor);
    }

    if (distal) {
        const angle = curl * Math.PI * 0.4;
        const rotation = new THREE.Quaternion().setFromEuler(new THREE.Euler(
            0,
            isRight ? -angle : angle,
            0
        ));
        distal.quaternion.slerp(rotation, factor);
    }
}

// ============================================================
// 표정 처리
// ============================================================
function applyBlendshapes(blendShapesData, deltaTime) {
    if (!currentVrm) return;

    const factor = getLerpFactor(deltaTime, 15); // 표정은 빠르게 반응

    const presetName = VRMExpressionPresetName;
    const expressions = currentVrm.expressionManager;

    const getScore = (name) => {
        const shape = blendShapesData.categories.find(s => s.categoryName === name);
        return shape ? shape.score : 0;
    };

    // ============================================================
    // 1. 입모양 (Lip Sync) - 표정과 독립적으로 동작
    // ============================================================

    // 입 벌림 (あ) - jawOpen을 직접 사용
    const jawOpen = getScore('jawOpen');
    const currentAa = expressions.getValue(presetName.Aa) ?? 0;
    expressions.setValue(presetName.Aa, THREE.MathUtils.lerp(currentAa, jawOpen, factor));

    // 입 모으기 (う) - mouthPucker 사용
    const mouthPucker = getScore('mouthPucker');
    const mouthFunnel = getScore('mouthFunnel');
    const ouScore = Math.max(mouthPucker, mouthFunnel * 0.7);
    const currentOu = expressions.getValue(presetName.Ou) ?? 0;
    expressions.setValue(presetName.Ou, THREE.MathUtils.lerp(currentOu, ouScore, factor));

    // 입 넓히기 (い) - mouthStretch 사용
    const mouthStretchL = getScore('mouthStretchLeft');
    const mouthStretchR = getScore('mouthStretchRight');
    const ihScore = (mouthStretchL + mouthStretchR) / 2;
    const currentIh = expressions.getValue(presetName.Ih) ?? 0;
    expressions.setValue(presetName.Ih, THREE.MathUtils.lerp(currentIh, ihScore * 0.5, factor));

    // ============================================================
    // 2. 눈 (독립적으로 동작)
    // ============================================================

    // 눈 깜빡임 - 직접 제어
    const blinkL = getScore('eyeBlinkLeft');
    const blinkR = getScore('eyeBlinkRight');
    const currentBlinkL = expressions.getValue(presetName.BlinkLeft) ?? 0;
    const currentBlinkR = expressions.getValue(presetName.BlinkRight) ?? 0;
    expressions.setValue(presetName.BlinkLeft, THREE.MathUtils.lerp(currentBlinkL, blinkL, factor));
    expressions.setValue(presetName.BlinkRight, THREE.MathUtils.lerp(currentBlinkR, blinkR, factor));

    // 눈 찡그림 (웃을 때) - eyeSquint 사용
    const eyeSquintL = getScore('eyeSquintLeft');
    const eyeSquintR = getScore('eyeSquintRight');
    // 눈 찡그림은 눈 깜빡임에 약간 더해줌 (완전히 감지 않도록 제한)
    const squintBlinkL = Math.min(blinkL + eyeSquintL * 0.3, 0.8);
    const squintBlinkR = Math.min(blinkR + eyeSquintR * 0.3, 0.8);
    expressions.setValue(presetName.BlinkLeft, THREE.MathUtils.lerp(currentBlinkL, squintBlinkL, factor));
    expressions.setValue(presetName.BlinkRight, THREE.MathUtils.lerp(currentBlinkR, squintBlinkR, factor));

    // ============================================================
    // 3. 표정 (감정) - 입모양에 영향 주지 않도록 약하게 적용
    // ============================================================

    // 웃음 - Happy 표정을 약하게 적용 (입모양 override 방지)
    const smileL = getScore('mouthSmileLeft');
    const smileR = getScore('mouthSmileRight');
    const smileScore = (smileL + smileR) / 2;
    // Happy 표정은 0.3 이상일 때만, 최대 0.5까지만 적용 (입모양 우선)
    const happyScore = smileScore > 0.3 ? Math.min(smileScore * 0.5, 0.5) : 0;
    const currentHappy = expressions.getValue(presetName.Happy) ?? 0;
    expressions.setValue(presetName.Happy, THREE.MathUtils.lerp(currentHappy, happyScore, factor));

    // 슬픔 (눈썹 올림) - 약하게 적용
    const browInnerUp = getScore('browInnerUp');
    const browDownL = getScore('browDownLeft');
    const browDownR = getScore('browDownRight');
    if (browInnerUp > 0.3) {
        const sadScore = Math.min(browInnerUp * 0.5, 0.4);
        const currentSad = expressions.getValue(presetName.Sad) ?? 0;
        expressions.setValue(presetName.Sad, THREE.MathUtils.lerp(currentSad, sadScore, factor));
    } else {
        const currentSad = expressions.getValue(presetName.Sad) ?? 0;
        expressions.setValue(presetName.Sad, THREE.MathUtils.lerp(currentSad, 0, factor));
    }

    // 화남 (눈썹 찌푸림)
    const angryScore = (browDownL + browDownR) / 2;
    if (angryScore > 0.3) {
        const currentAngry = expressions.getValue(presetName.Angry) ?? 0;
        expressions.setValue(presetName.Angry, THREE.MathUtils.lerp(currentAngry, angryScore * 0.4, factor));
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

    const factor = getLerpFactor(deltaTime, 10);

    const m = new THREE.Matrix4().fromArray(matrix.data);
    const rot = new THREE.Quaternion().setFromRotationMatrix(m);

    // Euler로 변환하여 축별 조정
    const euler = new THREE.Euler().setFromQuaternion(rot, 'YXZ');

    // 미러링: Y축(좌우 회전)과 Z축(좌우 기울기) 반전
    // X축(위아래 끄덕임)은 그대로
    euler.y *= -1;
    euler.z *= -1;

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
