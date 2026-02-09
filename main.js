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

// --- Initialization ---
async function init() {
    const toggleDebugBtn = document.getElementById('toggle-debug');
    if (toggleDebugBtn) {
        toggleDebugBtn.addEventListener('click', () => {
            DEBUG_MODE = !DEBUG_MODE;
            toggleDebugBtn.innerText = DEBUG_MODE ? "Hide Landmarks" : "Show Landmarks";
            if (!DEBUG_MODE && debugCtx) {
                debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
            }
        });
    }

    // 초기 view 상태 적용
    updateView();

    const toggleBtn = document.getElementById('toggle-view');
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            isDebugView = !isDebugView;
            updateView();
        });
    }

    const toggleBodyBtn = document.getElementById('toggle-body');
    if (toggleBodyBtn) {
        toggleBodyBtn.addEventListener('click', () => {
            BODY_TRACKING_ENABLED = !BODY_TRACKING_ENABLED;
            toggleBodyBtn.innerText = BODY_TRACKING_ENABLED ? "Disable Body Tracking" : "Enable Body Tracking";
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

    animate();
}

// ============================================================
// Screen Capture & Recording
// ============================================================
function setupScreenCaptureControls() {
    const selectScreenBtn = document.getElementById('select-screen');
    const stopScreenBtn = document.getElementById('stop-screen');
    const toggleAvatarSizeBtn = document.getElementById('toggle-avatar-size');
    const toggleMicBtn = document.getElementById('toggle-mic');
    const startRecordBtn = document.getElementById('start-record');
    const stopRecordBtn = document.getElementById('stop-record');

    if (selectScreenBtn) {
        selectScreenBtn.addEventListener('click', startScreenCapture);
    }
    if (stopScreenBtn) {
        stopScreenBtn.addEventListener('click', stopScreenCapture);
    }
    if (toggleAvatarSizeBtn) {
        toggleAvatarSizeBtn.addEventListener('click', toggleAvatarSize);
    }
    if (toggleMicBtn) {
        toggleMicBtn.addEventListener('click', toggleMicrophone);
    }
    if (startRecordBtn) {
        startRecordBtn.addEventListener('click', startRecording);
    }
    if (stopRecordBtn) {
        stopRecordBtn.addEventListener('click', stopRecording);
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

// 마이크 토글
async function toggleMicrophone() {
    const btn = document.getElementById('toggle-mic');
    const micMeter = document.getElementById('mic-meter');

    if (isMicEnabled) {
        // 마이크 비활성화
        if (micStream) {
            micStream.getTracks().forEach(track => track.stop());
            micStream = null;
        }
        isMicEnabled = false;
        micAnalyser = null;
        if (btn) {
            btn.innerText = 'Enable Mic';
            btn.classList.remove('mic-active');
        }
        if (micMeter) {
            micMeter.classList.add('inactive');
        }
        updateAudioMeters();
    } else {
        // 마이크 활성화
        try {
            micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            isMicEnabled = true;
            if (btn) {
                btn.innerText = 'Disable Mic';
                btn.classList.add('mic-active');
            }
            if (micMeter) {
                micMeter.classList.remove('inactive');
            }
            // 마이크 레벨 미터 설정
            setupMicMeter();
        } catch (err) {
            console.error('Microphone access error:', err);
            alert('마이크 접근 권한이 필요합니다.');
        }
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
        btn.innerText = isMiniAvatar ? 'Full Avatar' : 'Mini Avatar';
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

        // 스트림 종료 감지
        screenStream.getVideoTracks()[0].onended = () => {
            stopScreenCapture();
        };

        // 카메라 프리뷰 숨기기 & 화면 공유 모드 활성화
        document.body.classList.add('screen-sharing');

        // 버튼 상태 업데이트
        updateScreenCaptureButtons(true);

    } catch (err) {
        console.error("Screen capture error:", err);
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
        if (btn) btn.innerText = 'Mini Avatar';

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

    // 합성 캔버스 생성
    compositeCanvas = document.createElement('canvas');
    compositeCanvas.width = 1920;
    compositeCanvas.height = 1080;
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
            // 화면 비율에 맞춰 위치 변환
            const scaleX = compositeCanvas.width / window.innerWidth;
            const scaleY = compositeCanvas.height / window.innerHeight;
            const miniX = (miniAvatarPosition.x || 0) * scaleX;
            const miniY = (miniAvatarPosition.y || 0) * scaleY;
            const scaledWidth = miniWidth * scaleX;
            const scaledHeight = miniHeight * scaleY;
            compositeCtx.drawImage(avatarCanvas, miniX, miniY, scaledWidth, scaledHeight);
        } else {
            // 풀 모드: 전체 화면
            compositeCtx.drawImage(avatarCanvas, 0, 0, compositeCanvas.width, compositeCanvas.height);
        }

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

    // 비디오 + 오디오 스트림 합성
    const combinedStream = new MediaStream([
        ...canvasStream.getVideoTracks(),
        ...audioDestination.stream.getAudioTracks()
    ]);

    // MediaRecorder 설정
    const options = { mimeType: 'video/webm;codecs=vp9,opus' };
    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
        options.mimeType = 'video/webm';
    }

    mediaRecorder = new MediaRecorder(combinedStream, options);

    mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
            recordedChunks.push(e.data);
        }
    };

    mediaRecorder.onstop = () => {
        // 합성 루프 중지
        if (compositeAnimationId) {
            cancelAnimationFrame(compositeAnimationId);
            compositeAnimationId = null;
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

    // 버튼 상태 업데이트
    const startBtn = document.getElementById('start-record');
    const stopBtn = document.getElementById('stop-record');
    if (startBtn) {
        startBtn.disabled = true;
        startBtn.classList.remove('recording');
    }
    if (stopBtn) {
        stopBtn.disabled = false;
        stopBtn.classList.add('recording');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    }

    // 버튼 상태 업데이트
    const startBtn = document.getElementById('start-record');
    const stopBtn = document.getElementById('stop-record');
    if (startBtn && screenStream) {
        startBtn.disabled = false;
    }
    if (stopBtn) {
        stopBtn.disabled = true;
        stopBtn.classList.remove('recording');
    }
}

function downloadRecording() {
    if (recordedChunks.length === 0) return;

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
    const selectBtn = document.getElementById('select-screen');
    const stopBtn = document.getElementById('stop-screen');
    const toggleAvatarBtn = document.getElementById('toggle-avatar-size');
    const startRecordBtn = document.getElementById('start-record');
    const stopRecordBtn = document.getElementById('stop-record');

    if (selectBtn) selectBtn.disabled = isCapturing;
    if (stopBtn) stopBtn.disabled = !isCapturing;
    if (toggleAvatarBtn) toggleAvatarBtn.disabled = !isCapturing;
    if (startRecordBtn) startRecordBtn.disabled = !isCapturing;
    if (stopRecordBtn) stopRecordBtn.disabled = true;
}

function updateView() {
    const btn = document.getElementById('toggle-view');
    if (isDebugView) {
        document.body.classList.add('debug-view');
        if (btn) btn.innerText = 'Switch to Default';
    } else {
        document.body.classList.remove('debug-view');
        if (btn) btn.innerText = 'Switch to Debug';
    }
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
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: VIDEO_WIDTH, height: VIDEO_HEIGHT }
        });
        video.srcObject = stream;
        await new Promise((resolve) => {
            video.onloadedmetadata = () => {
                resolve(video);
            };
        });
        video.play();
    } catch (err) {
        console.error("Error accessing webcam:", err);
    }
}

async function setupMediaPipe() {
    try {
        const filesetResolver = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
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

    // 입 벌림
    const jawOpen = getScore('jawOpen');
    const currentAa = expressions.getValue(presetName.Aa) ?? 0;
    expressions.setValue(presetName.Aa, THREE.MathUtils.lerp(currentAa, jawOpen, factor));

    // 눈 깜빡임
    const blinkL = getScore('eyeBlinkLeft');
    const blinkR = getScore('eyeBlinkRight');
    const currentBlinkL = expressions.getValue(presetName.BlinkLeft) ?? 0;
    const currentBlinkR = expressions.getValue(presetName.BlinkRight) ?? 0;
    expressions.setValue(presetName.BlinkLeft, THREE.MathUtils.lerp(currentBlinkL, blinkL, factor));
    expressions.setValue(presetName.BlinkRight, THREE.MathUtils.lerp(currentBlinkR, blinkR, factor));

    // 웃음
    const smileL = getScore('mouthSmileLeft');
    const smileR = getScore('mouthSmileRight');
    const happyScore = (smileL + smileR) / 2;
    const currentHappy = expressions.getValue(presetName.Happy) ?? 0;
    expressions.setValue(presetName.Happy, THREE.MathUtils.lerp(currentHappy, happyScore, factor));

    // 슬픔 (눈썹 올림)
    const browInnerUp = getScore('browInnerUp');
    if (browInnerUp > 0.3) {
        const currentSad = expressions.getValue(presetName.Sad) ?? 0;
        expressions.setValue(presetName.Sad, THREE.MathUtils.lerp(currentSad, browInnerUp, factor));
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
