import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils, VRMExpressionPresetName } from '@pixiv/three-vrm';
import { FilesetResolver, FaceLandmarker, PoseLandmarker, HandLandmarker, DrawingUtils } from '@mediapipe/tasks-vision';

// --- Configuration ---
const VIDEO_WIDTH = 1280;
const VIDEO_HEIGHT = 720;

// --- Globals ---
let scene, camera, renderer;
let video;
let faceLandmarker, poseLandmarker, handLandmarker;
let currentVrm;
let lastVideoTime = -1;
let blendShapes = [];
let rotation = new THREE.Euler();
let currentGesture = 'neutral';
let gestureTimer = 0;
let debugCanvas, debugCtx, drawingUtils;
let DEBUG_MODE = true; // Always draw landmarks
let isDebugView = true;

// --- Initialization ---
async function init() {
    // Layout Toggle Logic
    const toggleBtn = document.getElementById('toggle-view');
    if (toggleBtn) {
        // Set initial state
        updateView();

        toggleBtn.addEventListener('click', () => {
            isDebugView = !isDebugView;
            // DEBUG_MODE = isDebugView; // Decoupled: Always draw landmarks
            updateView();
        });
    }
    const canvas = document.getElementById('output_canvas');

    // Expose gesture trigger (kept for manual override fallback)
    window.triggerGesture = (name) => {
        currentGesture = name;
        gestureTimer = 0;
        console.log("Gesture manually triggered:", name);
    };

    // 1. Setup Scene (Sync)
    setupScene(canvas);

    // Setup Debug Canvas
    debugCanvas = document.getElementById('debug_canvas');
    if (debugCanvas) {
        debugCanvas.width = VIDEO_WIDTH;
        debugCanvas.height = VIDEO_HEIGHT;
        debugCtx = debugCanvas.getContext('2d');
        drawingUtils = new DrawingUtils(debugCtx);
    }

    // 2. Start Async Setups (Parallel, non-blocking)
    setupWebcam();
    setupMediaPipe();
    loadAvatar();

    // 3. Start Loop Immediately
    animate();
}

// --- View Toggle ---
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

    // Use ResizeObserver to handle layout changes (e.g. Flexbox transitions)
    const resizeObserver = new ResizeObserver(() => {
        if (sceneWrapper) {
            const newWidth = sceneWrapper.clientWidth;
            const newHeight = sceneWrapper.clientHeight;
            // Prevent 0x0 errors
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

        // Face
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

        // Pose
        poseLandmarker = await PoseLandmarker.createFromOptions(filesetResolver, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
                delegate: "GPU"
            },
            runningMode: "VIDEO",
            numPoses: 1
        });

        // Hand
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
    if (video && video.readyState >= 2) { // HAVE_CURRENT_DATA
        if (DEBUG_MODE && debugCtx) {
            debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
        }

        if (video.currentTime !== lastVideoTime) {
            lastVideoTime = video.currentTime;

            // 1. Detect Face
            if (faceLandmarker) {
                const results = faceLandmarker.detectForVideo(video, currentTime);
                if (results.faceBlendshapes && results.faceBlendshapes.length > 0) {
                    applyBlendshapes(results.faceBlendshapes[0]);
                }
                if (results.facialTransformationMatrixes && results.facialTransformationMatrixes.length > 0) {
                    applyHeadRotation(results.facialTransformationMatrixes[0]);
                }
                // Debug Draw
                if (DEBUG_MODE && drawingUtils && results.faceLandmarks) {
                    for (const landmarks of results.faceLandmarks) {
                        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: "#C0C0C070", lineWidth: 1 });
                    }
                }
            }

            // 2. Detect Pose
            let poseDetected = false;
            if (poseLandmarker) {
                const poseResults = poseLandmarker.detectForVideo(video, currentTime);
                if (poseResults.landmarks && poseResults.landmarks.length > 0) {
                    // Pass World Landmarks if available
                    const worldLandmarks = poseResults.worldLandmarks ? poseResults.worldLandmarks[0] : null;
                    applyPose(poseResults.landmarks[0], worldLandmarks);
                    poseDetected = true;

                    // Debug Draw
                    if (DEBUG_MODE && drawingUtils) {
                        for (const landmark of poseResults.landmarks) {
                            drawingUtils.drawLandmarks(landmark, { radius: 1, color: "white" });
                            drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS, { color: "white", lineWidth: 2 });
                        }
                    }
                }
            }

            // 3. Detect Hands
            if (handLandmarker) {
                const handResults = handLandmarker.detectForVideo(video, currentTime);
                if (handResults.landmarks && handResults.landmarks.length > 0) {
                    applyHands(handResults.landmarks, handResults.handednesses);

                    // Debug Draw
                    if (DEBUG_MODE && drawingUtils) {
                        for (const landmark of handResults.landmarks) {
                            drawingUtils.drawConnectors(landmark, HandLandmarker.HAND_CONNECTIONS, { color: "#FF0000", lineWidth: 2 });
                            drawingUtils.drawLandmarks(landmark, { color: "#00FF00", lineWidth: 1 });
                        }
                    }
                }
            }

            // Fallback if no pose detected (Arms Down)
            if (!poseDetected) {
                resetPose();
            }
        }
    }

    // Update VRM
    if (currentVrm) {
        currentVrm.update(0.016);
    }

    renderer.render(scene, camera);
}

function resetPose() {
    if (!currentVrm) return;
    const rightUpperArm = currentVrm.humanoid.getNormalizedBoneNode('rightUpperArm');
    const leftUpperArm = currentVrm.humanoid.getNormalizedBoneNode('leftUpperArm');

    // Smoothly return to Neutral (Arms Down)
    // Using previously tested values: +/- 1.4 rads
    const relaxRight = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, -1.4, 'XYZ'));
    const relaxLeft = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, 1.4, 'XYZ'));

    if (rightUpperArm) rightUpperArm.quaternion.slerp(relaxRight, 0.05); // Slower return
    if (leftUpperArm) leftUpperArm.quaternion.slerp(relaxLeft, 0.05);
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
        // Create 33 spheres for Body Landmarks
        for (let i = 0; i < 33; i++) {
            const geom = new THREE.SphereGeometry(0.02, 8, 8);
            const mat = new THREE.MeshBasicMaterial({ color: 0xff0000 });
            const sphere = new THREE.Mesh(geom, mat);
            debugGroup.add(sphere);
        }
    }
    debugGroup.visible = true;

    // Update positions
    if (worldLandmarks) {
        for (let i = 0; i < worldLandmarks.length; i++) {
            const l = worldLandmarks[i];
            const sphere = debugGroup.children[i];
            if (sphere) {
                // Mirroring: -x, -y, -z (Same as getPos)
                sphere.position.set(-l.x, -l.y, -l.z);
            }
        }
    }
}

function applyPose(landmarks, worldLandmarks) {
    if (!currentVrm) return;

    updateDebug3D(worldLandmarks);

    // Use World Landmarks if available (meters). If not, fallback to normalized.
    // MediaPipe poseLandmarker result: { landmarks: [...], worldLandmarks: [...] }

    // Helper to get vector from landmark
    const getPos = (index) => {
        // Prefer World Landmarks for accurate 3D direction
        const l = worldLandmarks ? worldLandmarks[index] : landmarks[index];
        return new THREE.Vector3(-l.x, -l.y, -l.z);
    };

    // Check if we have World Landmarks (visibility property usually exists on normalized ones too)
    // 11: Left Shoulder, 12: Right Shoulder
    const leftShoulder = getPos(11);
    const rightShoulder = getPos(12);
    const leftElbow = getPos(13);
    const rightElbow = getPos(14);
    const leftWrist = getPos(15);
    const rightWrist = getPos(16);



    // --- Spine/Chest Rotation ---
    const spine = currentVrm.humanoid.getNormalizedBoneNode('spine');
    if (spine) {
        // Calculate Roll (Tilt) and Yaw (Turn) from Shoulders in Avatar Space.
        // Landmarks (normalized): 
        // 11 (Left Shoulder), 12 (Right Shoulder).
        // Coordinates: x (0..1, Left..Right), y (0..1, Top..Bottom), z (relative depth).
        // Mirroring: User Left (Screen Left) -> Avatar Left (Screen Left)??
        // Wait, standard mirroring: I raise my Left Hand (LEFT side of screen).
        // Avatar raises its Right Hand (LEFT side of screen).
        // The getPos function already does: x = -l.x.

        // Let's use getPos results which are already mirrored.
        // leftShoulder = getPos(11) -> Avatar's Right Shoulder pos.
        // rightShoulder = getPos(12) -> Avatar's Left Shoulder pos.
        // Wait, check getPos logic again.
        // l.x (0..1). getPos returns -l.x (-1..0).
        // 11 (Left) x~0.7. getPos -> -0.7.
        // 12 (Right) x~0.3. getPos -> -0.3.
        // So 12 (Right) is greater than 11 (Left).
        // Vector 12 - 11 = (-0.3) - (-0.7) = 0.4. (Points +X).
        // Avatar Right is -X. Avatar Left is +X.
        // So Vector 11->12 (Right Shoulder to Left Shoulder in MP) maps to Avatar Right->Left? 

        // Let's simplify. 
        // We want the spine to mimic the user.
        // User Tilts Left (Left shoulder Goes Down).
        // Avatar should Tilt Left (Screen Left side goes down).
        // MP: Left(11).y > Right(12).y.
        // getPos: -y. So Left.y < Right.y.
        // Vector R(12) - L(11): dy = R.y - L.y > 0.
        // Angle atan2(dy, dx). Positive.
        // +Z rotation is usually Tilt Left.

        // Let's rely on the landmarks directly to avoid confusion.
        const mpLeft = landmarks[11];
        const mpRight = landmarks[12];

        // dy: Left.y - Right.y. (Positive = Left is Lower).
        const dy = mpLeft.y - mpRight.y;
        const dx = mpRight.x - mpLeft.x; // Positive? 12(0.3) - 11(0.7) = -0.4.

        // Let's just use raw difference scaling.
        // Tilt (Roll):
        // If Left is Lower (dy > 0), we want +Z rotation (Tilt Left).
        // Mirror Mode: Invert to match user.
        const roll = dy * 1.5;

        // Turn (Yaw):
        // If Left is Closer (z < 0). 
        // User Turn Right (Right shoulder back).
        // Mirror Mode: Invert to match user.
        const dz = mpLeft.z - mpRight.z; // Left - Right.
        const yaw = -dz * 1.5;

        const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(0, yaw, roll, 'XYZ'));
        spine.quaternion.slerp(q, 0.2);
    }

    // Apply to Avatar
    // MP Left -> Avatar Right
    applyLimbRotation('rightUpperArm', 'rightLowerArm', leftShoulder, leftElbow, leftWrist, new THREE.Vector3(-1, 0, 0));

    // MP Right -> Avatar Left
    applyLimbRotation('leftUpperArm', 'leftLowerArm', rightShoulder, rightElbow, rightWrist, new THREE.Vector3(1, 0, 0));
}

function applyLimbRotation(upperName, lowerName, shoulder, elbow, wrist, restDir) {
    const upper = currentVrm.humanoid.getNormalizedBoneNode(upperName);
    const lower = currentVrm.humanoid.getNormalizedBoneNode(lowerName);

    const qIdentity = new THREE.Quaternion();

    if (upper) {
        const targetDir = new THREE.Vector3().subVectors(elbow, shoulder).normalize();

        // Robust Rotation: Quaternion.setFromUnitVectors is unstable for twist.
        // Solution: Use lookAt or construct basis.
        // Bone Axis: restDir (e.g. -1,0,0 for Right Arm).
        // Target Axis: targetDir.
        // Up Hint: (0, 1, 0) - World Up.

        // We want to rotate `restDir` to `targetDir` while minimizing twist relative to Up.
        // Shortest arc (setFromUnitVectors) often works for arms unless they go straight up/down.
        // But for elbows, twist matters.

        const qSwing = new THREE.Quaternion().setFromUnitVectors(restDir, targetDir);
        upper.quaternion.slerp(qSwing, 0.5); // Smooth update
    }

    if (lower) {
        // Lower Arm -> Rotation relative to Upper Arm.
        // Vector E->W
        const foreArmDir = new THREE.Vector3().subVectors(wrist, elbow).normalize();
        const upperArmDir = new THREE.Vector3().subVectors(elbow, shoulder).normalize();

        // Calculate the bend rotation (from straight to current).
        // Straight = upperArmDir. Target = foreArmDir.
        const qBend = new THREE.Quaternion().setFromUnitVectors(upperArmDir, foreArmDir);
        lower.quaternion.slerp(qBend, 0.5);
    }
}


function applyHands(landmarksArray, handednesses) {
    if (!currentVrm) return;

    for (let i = 0; i < landmarksArray.length; i++) {
        const landmarks = landmarksArray[i];
        const handedness = handednesses[i][0];

        // Mirroring: MP Left -> Avatar Right
        const isRightHand = handedness.categoryName === 'Left';
        const prefix = isRightHand ? 'right' : 'left';

        applyFinger(prefix, 'Thumb', landmarks, 1, 4);
        applyFinger(prefix, 'Index', landmarks, 5, 8);
        applyFinger(prefix, 'Middle', landmarks, 9, 12);
        applyFinger(prefix, 'Ring', landmarks, 13, 16);
        applyFinger(prefix, 'Little', landmarks, 17, 20);
    }
}

function applyFinger(handPrefix, fingerName, landmarks, startIndex, tipIndex) {
    const getVec = (idx) => new THREE.Vector3(landmarks[idx].x, landmarks[idx].y, landmarks[idx].z);

    const wrist = getVec(0);
    const mcp = getVec(startIndex);
    const tip = getVec(tipIndex);

    // Heuristic: Distance from Tip to Wrist
    // When fist is closed, Tip is close to Wrist (or closer to MCP).
    // Full length ~ Wrist->MCP + MCP->Tip.
    const fullLen = wrist.distanceTo(mcp) + mcp.distanceTo(tip);
    const currDist = wrist.distanceTo(tip);

    // Normalized Distance: 1.0 (straight) -> ~0.3 (fist)
    let curl = (fullLen - currDist) / fullLen;

    // Remap: 
    // Open (dist ~ 1.0*Len) -> Curl 0
    // Closed (dist ~ 0.4*Len) -> Curl 1
    // currDist/fullLen: 1.0 -> 0.4
    // curl = 1 - ratio. 0 -> 0.6.
    // Map 0..0.6 to 0..1

    let ratio = currDist / fullLen;
    // ratio > 0.9 -> Open (Curl 0)
    // ratio < 0.5 -> Closed (Curl 1)

    curl = (0.9 - ratio) / 0.4; // Map 0.9..0.5 to 0..1
    curl = Math.max(0, Math.min(1, curl));

    // Thumb is special (curls different direction often)
    // But for simple "Fist" detection, X-curl usually works for all if rigged standard.
    // Thumb might need Z-curl too.

    // Rotation Axis:
    // Standard Unity/VRM Humanoid: +X Rot = Curl In? Or -X?
    // Let's try -X for Curl In (Right Hand). 
    // If Left Hand (Mirrored), Mirror axis? 
    // VRM Local Axes are usually consistent (X is curl).
    // Let's try -1.5 rad on X.

    const angle = -1.6 * curl; // -90 deg curl

    const fingerRot = new THREE.Quaternion().setFromEuler(new THREE.Euler(
        angle, // X
        0,
        0
    ));

    const bones = ['Proximal', 'Intermediate', 'Distal'];
    bones.forEach(boneType => {
        const boneName = handPrefix + fingerName + boneType;
        const bone = currentVrm.humanoid.getNormalizedBoneNode(boneName);
        if (bone) {
            bone.quaternion.slerp(fingerRot, 0.3);
        }
    });
}


function applyBlendshapes(blendShapesData) {
    if (!currentVrm) return;

    const presetName = VRMExpressionPresetName;
    const expressions = currentVrm.expressionManager;

    const getScore = (name) => {
        const shape = blendShapesData.categories.find(s => s.categoryName === name);
        return shape ? shape.score : 0;
    };

    const jawOpen = getScore('jawOpen');
    expressions.setValue(presetName.Aa, jawOpen);

    const blinkL = getScore('eyeBlinkLeft');
    const blinkR = getScore('eyeBlinkRight');
    expressions.setValue(presetName.BlinkLeft, blinkL);
    expressions.setValue(presetName.BlinkRight, blinkR);

    const smileL = getScore('mouthSmileLeft');
    const smileR = getScore('mouthSmileRight');
    const happyScore = (smileL + smileR) / 2;
    expressions.setValue(presetName.Happy, happyScore);

    const browInnerUp = getScore('browInnerUp');
    if (browInnerUp > 0.5) expressions.setValue(presetName.Sad, browInnerUp);

    expressions.update();
}

function applyHeadRotation(matrix) {
    if (!currentVrm) return;

    const m = new THREE.Matrix4().fromArray(matrix.data);
    const rot = new THREE.Quaternion().setFromRotationMatrix(m);

    // Convert to Euler to invert axes for mirror effect
    const euler = new THREE.Euler().setFromQuaternion(rot);

    // Invert Y (Yaw) and Z (Roll) for mirror effect
    // We might need to adjust X (Pitch) depending on camera/model match, but usually X is fine if we just want L/R mirroring.
    // MediaPipe matrix is often slightly different. 
    // Experimentally: 
    // - Invert Y to fix "looking left turns right"
    // - Invert Z to fix "tilting left tilts right"
    euler.y *= -1;
    euler.z *= -1;

    const mirrorRot = new THREE.Quaternion().setFromEuler(euler);

    const head = currentVrm.humanoid.getNormalizedBoneNode('head');
    if (head) {
        head.quaternion.slerp(mirrorRot, 0.5);
    }
}

init();
