# 프론트 연결용 틀 — 대화 이어가기

수정·취소 대상이 여러 건일 때 **"어느 민원인가요?" → "두 번째 거요"** 를 이어서 처리하는 API 와,
그걸 화면에 붙이기 위한 최소한의 틀입니다. 실제 화면(디자인·컴포넌트)은 아직 없습니다.

| 파일 | 내용 |
|---|---|
| `chatClient.js` | 서버와 주고받는 부분만 담은 JS 클래스 (`send` / `choose` / `restore` / `reset`). 프레임워크 무관 |
| `demo.html` | 동작 확인용 최소 화면. 서버가 `http://서버/chat-demo/demo.html` 로 내줌 (`CHAT_DEMO_ENABLED=true`) |

## 흐름

```
사용자  "가로등 민원 취소해 주세요"
  POST /api/v1/chat/message {text}                       (session_id 없이 → 새 세션)
서버    state="awaiting_selection", reply="취소할 민원이 3건 있습니다. 어느 민원인가요? 1. … 2. … 3. …"
        choices=[{no:1, complaint_id:37, label:"37번 민원 (국토교통 / …)"}, …]
화면    reply 말풍선 + choices 버튼 표시

사용자  (말로) "두 번째 거요"     또는   (버튼) 2번 버튼 클릭
  POST {session_id, text:"두 번째 거요"}       POST {session_id, selected_complaint_id: 41}
서버    Qwen-Agent 가 해석 → 41번 선택          해석 없이 41번
        → 처음 요청(취소, 사유 등)을 41번 민원에 실행
        state="idle", kind="selected", reply="취소 완료 - 41번 민원 …", tool_result={executed:true, …}
화면    reply 말풍선, 버튼 제거
```

## API

### `POST /api/v1/chat/message`

요청

```json
{ "session_id": "", "user_id": "kim01", "text": "가로등 민원 취소해 주세요", "selected_complaint_id": null }
```

- `session_id` — 처음엔 비우고, 응답의 값을 다음 요청부터 그대로 보냅니다.
- `text` / `selected_complaint_id` — 둘 중 하나는 있어야 합니다. 버튼을 누른 경우 `choices[].complaint_id` 를 넣습니다.
- `user_id` — 세션이 이 사용자에게 묶입니다. 다른 사용자가 같은 `session_id` 를 쓰면 403.
  (**TODO**: 실제 로그인이 생기면 요청 본문 대신 인증 토큰에서 꺼내도록 바꿔야 합니다)

응답 (주요 필드)

| 필드 | 화면에서 할 일 |
|---|---|
| `session_id` | 저장해 두고 다음 요청에 보냄 |
| `reply` | 말풍선으로 표시 |
| `state` | `awaiting_selection` 이면 `choices` 를 버튼으로, `idle` 이면 버튼 제거 |
| `choices[]` | `label` 을 버튼 글자로, 누르면 `complaint_id` 를 `selected_complaint_id` 로 전송 |
| `kind` | `selection_asked`(되물음) `selected`(골라서 실행) `cancelled`(그만둠) `reasked`(못 알아들어 다시 물음) `gave_up`(여러 번 실패해 종료) `analyzed`(일반 처리) |
| `tool_result` | 민원 DB 실행 결과 — 성공 표시, 상세 카드 등에 사용 (`/analyze` 와 같은 형식) |
| `analysis` | 이번 턴에 분류를 돌렸으면 `/analyze/text` 와 같은 형식의 결과 (의도·카테고리 표시용) |

### `GET /api/v1/chat/sessions/{session_id}?user_id=…`

새로고침 후 복원용. 대화 기록(`messages`)과 고르는 중이면 `choices` 를 돌려줍니다. 없는 세션이면 404.

### `DELETE /api/v1/chat/sessions/{session_id}?user_id=…`

"새 대화" 버튼용.

### 오류

모든 오류는 서버 공통 형식 `{success:false, error_code, message, detail}` 입니다.
`SESSION_NOT_FOUND`(404, 세션 만료·삭제 → 새 세션으로 다시 보내면 됨), `SESSION_FORBIDDEN`(403),
`VALIDATION_ERROR`(422), `MODEL_UNAVAILABLE`(503).

## 화면 쪽 TODO

- [ ] 말풍선 목록 / 입력창 / 로딩 표시 (한 턴에 수 초, 첫 요청은 모델 로드로 더 김)
- [ ] `state === "awaiting_selection"` 일 때 `choices` 버튼, 누르면 `client.choose(id)`
- [ ] 세션 저장 (`chatClient` 의 `storage` 옵션) 과 화면 진입 시 `client.restore()`
- [ ] 로그인 연동 후 `user_id` 대신 인증 정보 사용 (서버도 같이 수정)
- [ ] 다른 주소에서 프론트를 띄우면 서버 `.env` 의 `CORS_ALLOW_ORIGINS` 에 그 주소 추가
- [ ] 운영 배포 시 서버 `.env` 의 `CHAT_DEMO_ENABLED=false`
