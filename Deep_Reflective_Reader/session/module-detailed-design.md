# session Detailed Design

## SESSION-01: generation-bound ordinary QA (2026-11-18, locally verified)

`ReadingSession` has immutable scalar fields (session ID, document name, opaque
generation and anchor) plus detached, bounded history lists. The manager owns current
snapshots; callers cannot mutate its state through returned references. Creation,
document switch and explicit reset allocate a new generation, including A→B→A.
A same-document update retains the generation and merges into current history,
not the request's older history. Same-generation concurrent turns apply in completion
order; this is a lifecycle fence, not a turn-order scheduler.

Get/create, read, reset, and compare-and-update use one short in-process lock.
Update verifies captured session ID, document and generation against the current
snapshot before changing any field; it returns false without mutation if stale.
Chunk lookup happens outside the lock; no preparation, retrieval, model call or
retry is performed while holding it. Empty/missing-record results still append
the question and preserve the current anchor. Histories retain the configured
positive limit. Reads return detached snapshots, so tests/callers seed progress
through `update_session`, and inspect current state through `get_session`.

QA captures its snapshot before preparation. A late successful answer is returned
normally, but its stale state update is discarded without another model call.
Reset does not cancel work or recreate history on completion. This is memory-only,
single-process coordination, not database persistence, multiworker synchronization
or the separate critical-thinking session system.

Acceptance: Event-controlled actual manager/coordinator/FAISS/context and isolated
ASGI ask tests for switch, ABA, reset, independent sessions and subsequent local
routing; same-document concurrent history, fixed identity, isolated histories,
positive limits, empty/missing results and sequential controls. Local tests do not grant external
approval or authorize Docker/push.

## 1. Module Purpose

`session/` 管理閱讀 session 的 in-memory 狀態，追蹤 active chunk、近期 chunk 與近期提問，支援 local reading context 的連續性。 **[Code-Confirmed]**

## 2. Position in Overall Architecture

- Peripheral / Support Layer（QA runtime state）

## 3. Key Files

| File | Responsibility | Notes |
|---|---|---|
| `session/reading_session.py` | session state DTO | active_chunk_index/recent history **[Code-Confirmed]** |
| `session/session_manager.py` | session lifecycle 與更新 | create/reset/update flow **[Code-Confirmed]** |

## 4. Main Responsibilities

1. 管理 session 建立/重置/讀取。 **[Code-Confirmed]**
2. 在 ask 結果後更新 active chunk 與 recent history。 **[Code-Confirmed]**

## 5. Non-Responsibilities

1. 不負責持久化到 DB/檔案（目前 memory-only）。 **[Code-Confirmed]**
2. 不負責 scope 判斷演算法。 **[Code-Confirmed]**

## 6. Important Data Structures / Contracts

- `ReadingSession`
- `SessionManager`
- `SessionUpdateResult`

## 7. Module Relationships

- used by: `app/qa_coordinator.py`, `context/context_orchestrator.py`
- depends on: `retrieval/faiss_index_bundle.py`, `retrieval/search_metadata.py`

## 8. Main Flows Involving This Module

1. QA ask flow session get/create。
2. ask 完成後根據最佳 result 更新 active chunk。 **[Code-Confirmed]**

## 9. Persistence / Side Effects

- read/write persistence：否（in-memory）
- side effect：session store mutation

## 10. Known Legacy / Compatibility Behavior

No known legacy compatibility responsibility.

## 11. Current Risks

1. risk：session state 不持久，重啟即失
- why：local continuity 會中斷
- guardrail：視產品需求決定是否持久化 **[Needs Confirmation]**

2. risk：同 session_id 跨 doc 重用語義
- why：若重置規則誤用可能造成上下文錯配
- guardrail：SESSION-01 的 switch/reset generation fencing（已本地驗證，待精確 SHA 外審）

## 12. Open Questions for Maintainer

1. 是否要支援持久化 session（目前未見需求）？ **[Needs Confirmation]**

## 13. Suggested Next Documentation Improvements

1. 補 session lifecycle state diagram。
