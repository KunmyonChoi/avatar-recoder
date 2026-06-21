# Avatar Recorder — Cross-Origin Integration API

앱 URL: `https://avatar-recoder.netlify.app`

## 1. 팝업 열기

```js
const recorderWindow = window.open(
  'https://avatar-recoder.netlify.app' +
    '?mode=popup' +
    '&origin=' + encodeURIComponent(window.location.origin) +
    '&session=' + encodeURIComponent(mySessionId),   // 선택
  'avatar-recoder',
  'width=1280,height=800,noopener=0'                 // noopener=0 필수 (opener 참조 유지)
);
```

### URL 파라미터

| 파라미터    | 필수 | 설명 |
|------------|------|------|
| `mode`     | ✅   | 반드시 `popup` |
| `origin`   | ✅   | 호출 앱의 origin (예: `https://float-slide-editor.netlify.app`) |
| `session`  | –    | 세션 식별자. 모든 응답 메시지에 그대로 반환됨 |
| `autoRecord` | –  | `1`로 설정 시 앱 준비 완료 후 자동 녹화 시작 |

---

## 2. 메시지 수신 (avatar-recorder → 호출 앱)

`window.addEventListener('message', handler)` 로 수신합니다.
`e.origin`이 `https://avatar-recoder.netlify.app`인지 반드시 검증하세요.

```js
window.addEventListener('message', (e) => {
  if (e.origin !== 'https://avatar-recoder.netlify.app') return;
  const { type, sessionId, blob, mimeType, filename } = e.data;

  switch (type) {
    case 'avatar-recoder:ready':
      // 앱 초기화 완료. 이제 start 명령 전송 가능
      break;

    case 'avatar-recoder:recording-started':
      // 녹화가 시작됨
      break;

    case 'avatar-recoder:recording-stopped':
      // 녹화 중지됨 (result 메시지와 함께 옴)
      break;

    case 'avatar-recoder:result':
      // blob: Blob (video/webm), filename: string
      const url = URL.createObjectURL(blob);
      // 업로드하거나 <video>에 연결하거나 다운로드
      break;

    case 'avatar-recoder:cancelled':
      // 사용자가 창을 닫거나 cancel 명령으로 취소됨
      break;

    case 'avatar-recoder:error':
      // e.data.message: 오류 설명
      break;
  }
});
```

### 수신 메시지 타입 목록

| type | 데이터 필드 | 설명 |
|------|------------|------|
| `avatar-recoder:ready` | `sessionId` | 초기화 완료, 명령 수신 대기 중 |
| `avatar-recoder:recording-started` | `sessionId` | 녹화 시작됨 |
| `avatar-recoder:recording-stopped` | `sessionId` | 녹화 중지됨 |
| `avatar-recoder:result` | `sessionId`, `blob`, `mimeType`, `filename` | 녹화 결과 파일 |
| `avatar-recoder:cancelled` | `sessionId` | 취소 또는 창 닫힘 |
| `avatar-recoder:error` | `sessionId`, `message` | 오류 발생 |

---

## 3. 명령 전송 (호출 앱 → avatar-recorder)

```js
// 녹화 시작
recorderWindow.postMessage(
  { type: 'avatar-recoder:start' },
  'https://avatar-recoder.netlify.app'
);

// 녹화 중지 (결과 blob이 result 메시지로 반환됨)
recorderWindow.postMessage(
  { type: 'avatar-recoder:stop' },
  'https://avatar-recoder.netlify.app'
);

// 취소 (결과 없이 창 닫힘)
recorderWindow.postMessage(
  { type: 'avatar-recoder:cancel' },
  'https://avatar-recoder.netlify.app'
);
```

---

## 4. 전체 통합 예시

```js
let recorderWindow = null;

function openRecorder(sessionId) {
  const params = new URLSearchParams({
    mode: 'popup',
    origin: window.location.origin,
    session: sessionId,
  });
  recorderWindow = window.open(
    `https://avatar-recoder.netlify.app?${params}`,
    'avatar-recoder',
    'width=1280,height=800,noopener=0'
  );
}

window.addEventListener('message', (e) => {
  if (e.origin !== 'https://avatar-recoder.netlify.app') return;
  const { type, sessionId, blob, filename } = e.data;

  if (type === 'avatar-recoder:ready') {
    // 준비 완료 → 바로 녹화 시작
    recorderWindow.postMessage(
      { type: 'avatar-recoder:start' },
      'https://avatar-recoder.netlify.app'
    );
  }

  if (type === 'avatar-recoder:result') {
    // 예: FormData로 서버 업로드
    const formData = new FormData();
    formData.append('video', blob, filename);
    fetch('/api/upload', { method: 'POST', body: formData });

    recorderWindow?.close();
  }

  if (type === 'avatar-recoder:cancelled') {
    console.log('Recording cancelled for session:', sessionId);
  }
});
```

---

## 5. 보안 주의사항

- `window.open()`의 feature string에 `noopener=0`을 명시해야 `window.opener` 참조가 유지됩니다 (기본값은 `noopener`).
- 수신 메시지의 `e.origin`을 `https://avatar-recoder.netlify.app`으로 반드시 검증하세요.
- `blob` 객체는 수신 탭의 메모리에만 존재합니다. `URL.createObjectURL()` 사용 후 `URL.revokeObjectURL()`로 해제하세요.
- `session` 파라미터를 활용해 복수의 동시 세션을 구분할 수 있습니다.
