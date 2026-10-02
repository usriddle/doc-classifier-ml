/**
 * 민원 대화 API 클라이언트 (프론트 연결용 틀)
 *
 * 프레임워크와 무관한 순수 JS 입니다. React/Vue 등에서 그대로 import 하거나 복사해서 쓰세요.
 * 화면(말풍선·버튼) 그리는 부분은 프론트 쪽에서 구현하고, 여기서는 서버와 주고받는 것만 담당합니다.
 *
 * 흐름 (자세한 설명은 front_stub/README.md)
 *   1. send("가로등 민원 취소해 주세요")         -> state="awaiting_selection", choices=[...]
 *   2-a. send("두 번째 거요")                    -> 말로 고름 (서버의 Qwen-Agent 가 해석)
 *   2-b. choose(choices[1].complaint_id)        -> 버튼으로 고름 (해석 없이 바로 실행)
 *   3. 응답 reply 를 말풍선으로, state 가 awaiting_selection 이면 choices 를 버튼으로 표시
 *
 * session_id 는 이 객체가 기억합니다. 새로고침 후에도 이어가려면 storage 옵션을 주세요.
 */

/**
 * @typedef {Object} ChatChoice
 * @property {number} no            목록 순번 (1부터)
 * @property {number} complaint_id  버튼을 누르면 choose() 에 넘길 값
 * @property {string} label         버튼에 보여 줄 한 줄
 * @property {string} category
 * @property {string} content
 * @property {string} location
 * @property {string} status
 * @property {string} created_at
 */

/**
 * @typedef {Object} ChatResponse
 * @property {string} session_id
 * @property {boolean} session_created
 * @property {"idle"|"awaiting_selection"} state
 * @property {string} kind          analyzed | selection_asked | selected | cancelled | reasked | gave_up | no_pending
 * @property {string} reply         말풍선에 보여 줄 문장
 * @property {string|null} pending_action   "수정" | "취소" | null
 * @property {ChatChoice[]} choices
 * @property {Object|null} tool_result      민원 DB 실행 결과 (executed, ok, message, data, ...)
 * @property {Object|null} analysis         /analyze/text 와 같은 형식 (의도·카테고리 등)
 * @property {Object|null} resolution       대기 중 답을 어떻게 해석했는지 (method, kind, note)
 */

export class MinwonChatClient {
  /**
   * @param {Object} options
   * @param {string} options.baseUrl     예) "http://localhost:8000/api/v1"
   * @param {string} [options.userId]    로그인 사용자 식별자 (TODO: 실제 인증이 생기면 토큰으로 대체)
   * @param {Storage} [options.storage]  window.sessionStorage 등을 주면 session_id 를 저장해 새로고침 후에도 이어감
   */
  constructor({ baseUrl, userId = "", storage = null }) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.userId = userId;
    this.storage = storage;
    this.storageKey = `minwon-chat-session:${userId || "default"}`;
    this.sessionId = storage ? storage.getItem(this.storageKey) || "" : "";
  }

  /** 사용자가 입력한 말 보내기 */
  async send(text) {
    return this._post({ text });
  }

  /** 후보 버튼을 눌렀을 때 (choices[].complaint_id) */
  async choose(complaintId) {
    return this._post({ selected_complaint_id: complaintId });
  }

  /** 화면을 다시 열었을 때 대화 기록·고르는 중 상태 복원. 세션이 없으면 null */
  async restore() {
    if (!this.sessionId) return null;
    const url = `${this.baseUrl}/chat/sessions/${encodeURIComponent(this.sessionId)}?user_id=${encodeURIComponent(this.userId)}`;
    const res = await fetch(url);
    if (res.status === 404) {
      this._setSession("");
      return null;
    }
    return this._json(res);
  }

  /** '새 대화' 버튼 */
  async reset() {
    if (this.sessionId) {
      const url = `${this.baseUrl}/chat/sessions/${encodeURIComponent(this.sessionId)}?user_id=${encodeURIComponent(this.userId)}`;
      await fetch(url, { method: "DELETE" }).catch(() => {});
    }
    this._setSession("");
  }

  // ---------------------------------------------------------------------------
  async _post(body) {
    const res = await fetch(`${this.baseUrl}/chat/message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: this.sessionId, user_id: this.userId, ...body }),
    });
    if (res.status === 404 && this.sessionId) {
      // 서버에서 세션이 지워졌음 (DB 초기화 등) - 새 세션으로 한 번 다시 보냄
      this._setSession("");
      return this._post(body);
    }
    /** @type {ChatResponse} */
    const data = await this._json(res);
    this._setSession(data.session_id);
    return data;
  }

  async _json(res) {
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      // 서버 공통 오류 형식: { success:false, error_code, message, detail }
      const err = new Error(data.message || `HTTP ${res.status}`);
      err.code = data.error_code;
      err.status = res.status;
      throw err;
    }
    return data;
  }

  _setSession(id) {
    this.sessionId = id || "";
    if (this.storage) {
      if (this.sessionId) this.storage.setItem(this.storageKey, this.sessionId);
      else this.storage.removeItem(this.storageKey);
    }
  }
}

// TODO(프론트): 화면 연결 시 구현할 부분
//   - renderMessage(role, text)          : 말풍선 추가
//   - renderChoices(choices, onPick)     : state === "awaiting_selection" 일 때 버튼 목록, 누르면 client.choose(id)
//   - clearChoices()                     : state === "idle" 이면 버튼 제거
//   - 로딩 표시                          : 한 턴에 수 초 걸릴 수 있음 (첫 요청은 모델 로드로 더 김)
//   - 오류 표시                          : err.code (SESSION_FORBIDDEN, MODEL_UNAVAILABLE 등)
