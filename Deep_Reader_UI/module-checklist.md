# Deep_Reader_UI Checklist

- [x] MAIN-06 dependency: submit backend-bound rename command identity and retain unknown-outcome retries.
  Evidence: `test-rename-command-service.mjs` verifies duplicate dispatch, transport/503 retry, new command after completion/conflict; `test-rename-command-browser.mjs` passes at 1280px/390px with screenshots. `npm run build` and 21 controller groups pass. Existing broad Library workflow times out before reaching rename; it is not counted as passing.
  Timestamp: 2026-11-11

## Navigation during interaction generation (2026-11-11)

- [x] UI-AUDIT-R5: Separate presentation reset from document source invalidation.
  Owner: Deep_Reader_UI. Status: Implemented and locally verified; external re-review required.
  Acceptance: Insights/Summary/Quiz/Critical thinking generate once across unchanged A→B→A; completion while away rereads saved state, unknown failure does not auto-retry; unrelated document invalidation preserves work. Retain all 16 R4 cases and R1–R3 regressions.
  Evidence: 24/40 red cases reproduce duplicate generation; 40/40 pass after correction at 1280/390. Original external script reports one generation per kind. R4 16/16, R1 activation, 21 learning controller groups, settings/display/lifecycle, four R2/R3 file/PostgreSQL API tests, type/build and diff checks pass. R5/R4 also pass on a hash-verified real-book excerpt; screenshots inspected. See /tmp/audit-6f6e75d/ and design for limits. No deployment or push.
  Timestamp: 2026-11-11

## Source invalidation during generation (2026-11-11)

- [x] UI-AUDIT-R4: Isolate interaction generation across source invalidation.
  Owner: Deep_Reader_UI. Status: Implemented and locally verified; external re-review required.
  Acceptance: Delayed old Insights/Summary/Quiz/Critical thinking cannot fill a new source after save/reset; ordinary close/reopen still joins one generation; R2/R3 remain valid.
  Evidence: Red reproduction across four kinds/two widths plus actual controllers; 16 source-reset cases pass with fixtures and a hash-verified English book excerpt. R1 activation, 21 learning controller groups, R2/R3 settings/display/lifecycle at 1280/390, four file/PostgreSQL API cases, type/build and diff checks pass. Screenshots inspected; no Docker or push.
  Timestamp: 2026-11-11

## External review corrections (2026-11-10)

- [x] UI-AUDIT-E7073: Correct R1 generation ownership, R2 settings-stale retry and R3 explicit reimport reapply.
  Owner: Deep_Reader_UI. Status: Implemented and locally verified; external re-review required.
  Acceptance: No duplicate generation on close/reopen; current-source/settings retry with uncertain-cost consent and old-worker fencing; explicit CAS reapply after same-name reimport, no read mutation.
  Evidence: R1/R2/R3 red regressions reproduced before fixes; 21 controller groups, 1280/390 interaction/settings/display/lifecycle and actual preference/learning API mode-control suites pass. Type/build passes; 448 file/PostgreSQL backend tests and two unchanged real-book excerpts with offline providers pass. Desktop/mobile inspected. See design for limits; no Docker deployment or push.
  Timestamp: 2026-11-10

## Reader reading-mode control (2026-11-10)

- [x] Show the current reading mode and open its settings directly from the reader.
  Owner: Deep_Reader_UI. Status: Implemented and verified after documentation governance.
  Acceptance: Compact authoritative mode label/entry in normal/fullscreen reader; reuse mode list/detail and guarded preference saves, return to the same page with a fresh mode read, preserve draft/focus/scroll and reject stale owner responses. No Library detour or hidden generation on settings open.
  Evidence: `test-reader-mode-control.mjs` passes at 1280/390 using actual App/production preference and learning routes, temporary file storage and an offline provider: mode roundtrip, no Library detour, keyboard/focus, draft guard, duplicate prevention, rejected/unknown saves, read recovery, late owner responses, page/source-container retention, fullscreen and wrapping. Two provider calls total across widths; reused thereafter. Type/build plus reading-settings, Library modes, learning display/lifecycle regressions pass; screenshots inspected. Local Madame Bovary smoke confirms mode display/settings open-close with zero write/generate requests. See design for evidence limits; no paid model or production preference save.
  Timestamp: 2026-11-10

## Shared sentence translation (2026-11-09)

- [x] Verify unchanged ordinary/learning presentation with shared translation reuse.
  Owner: Deep_Reader_UI. Status: Implemented and verified.
  Acceptance: Both modes reuse compatible aligned output; ordinary aggregation and learning sentence pairs retain their UI behavior. No automatic legacy migration or duplicate aggregate storage.
  Evidence: 470 backend tests (35 shared cases); actual API/React at 1280/390 uses four offline calls for two paragraphs across both modes. Existing display/inline suites and type/build pass; backed-up English/Chinese excerpts remain unchanged.
  Follow-up (2026-11-10): User performed learning-to-normal saved-translation flow; normal read returned 200 in 666.2 ms with zero generation requests/batch events in the captured window. Reference-block identity/revision and aggregate were verified separately by API/SQL. Reverse direction remains earlier automated evidence; no new manual UI layout or semantic claim. See [manual acceptance](../Deep_Reflective_Reader/section_tasks/module-detailed-design.md#manual-reuse-acceptance-2026-11-10).
  Timestamp: 2026-11-10


## Library reading-mode labels (2026-11-09)

- [x] Show authoritative reading-mode labels on first Library open and refresh them across lifecycle changes.
  Owner: Deep_Reader_UI. Status: Implemented and verified.
  Acceptance: Visible rows read preferences with at most two concurrent requests; loading and failure stay explicit, with read-only Retry. Reopen, Refresh, settings return, identity changes and filter removal invalidate observations; stale results cannot overwrite current labels. No settings mutation or content generation.
  Evidence: `npm run check`; new Library mode regression at 1280/390 (200 books, initially 7/3 visible reads); reading-settings, Library identity and readiness suites pass. Desktop/mobile screenshots inspected; local API read-only smoke displays labels without mutations.
  Timestamp: 2026-11-09

## Persistent current-page learning progress (2026-11-09)

- [x] Keep overall learning progress visible when collapsed and guard forward pagination.
  Owner: Deep_Reader_UI. Status: Implemented and verified.
  Acceptance: Confirmed completion counts and an accessible progress bar remain outside paragraph details. Next/Last are disabled during initialization, queued/running tasks, wait timeout and unconfirmed observation; verified terminal failures release navigation without fake completion. No extra polling, generation or server-state authority.
  Evidence: `npm run check`; 19 controller groups; reader/display/lifecycle suites at 1280/390; held mode/wait, collapsed and keyboard progress, guarded forward controls with no off-page commands, stale responses, refresh/terminal failure and stable focus/scroll. Desktop/mobile screenshots inspected. Actual API/temporary file/offline provider Reader passes at both widths with four calls and zero reload regeneration.
  Timestamp: 2026-11-09

## Vocabulary filtering and completion observation (2026-11-09)

- [x] Display filtered vocabulary and replace periodic polling with completion waiting.
  Owner: Deep_Reader_UI. Status: Implemented and verified.
  Acceptance: Retain server-validated hints, expose optional selection details and distinguish all-invalid from no hints. One wait per observation; timeout requires Check progress, with no automatic paid retry. Abort and stale guards remain.
  Evidence: 16 controller groups; reader/display/lifecycle at 1280/390, keyboard details and inspected screenshots; type/build; real API/file/offline-provider Reader (4 calls, reload adds none); captured real-book API/SQL/Reader replay.
  Timestamp: 2026-11-09

## Language Learning Reading Mode Queue (2026-10-28)

**LLRN-17/18/19/21 implemented and verified (4/4 UI-owned tasks).** [Shared dependency queue](../Deep_Reflective_Reader/section_tasks/module-checklist.md#language-learning-reading-mode-queue-2026-10-28)；[canonical design](../Deep_Reflective_Reader/section_tasks/module-detailed-design.md#language-learning-reading-mode-plan-2026-10-28)。本檔只持有 Deep_Reader_UI 的唯一 checklist；LLRN-22 系統驗收由 scripts 擁有；依 2026-10-29 maintainer 指示，模型語義評分移出該任務。

### Learning reader compatibility fixes

- [x] Apply the saved learning mode to backend paragraph blocks.
  Owner: Deep_Reader_UI. Status: Implemented and verified.
  Acceptance: Current-page source selection and sentence/vocabulary rendering use backend block identity rather than a `content` type whitelist; invalid identities remain rejected and ordinary translation entries remain hidden in learning mode.
  Evidence: Paragraph-only `test-learning-reader.mjs` reproduced the failure before the fix and passes at 1280/390 afterward. Controller suite passes 16 groups, including mixed/omitted types and duplicate/missing/foreign identities; display/lifecycle suites and `npm run check` pass. Desktop/mobile screenshots inspected. Real local Madame Bovary smoke: 9 paragraph blocks, 12 sentence rows and successful preference/learning reads at both widths, with generation intercepted before backend dispatch and no model calls or user preference/source writes.
  Timestamp: 2026-11-09

### LLRN-17

- [x] LLRN-17: Workspace and per-book reading settings UI.
  Owner: Deep_Reader_UI. Status: Implemented and verified (settings UI only).
  Depends on: LLRN-01, LLRN-16.
  Deliverable: 集中母語入口；Library 每書單選普通/語言學習模式、有效原文校正、適用能力分級；型別 service 讀寫後端設定。 閱讀模式入口採可擴充 list/detail，桌面雙欄、手機單欄進退；共用 shell 與模式專屬設定 feature 分開。
  Acceptance: 母語等於原文時不提供學習該語言選項，解釋禁用原因；未設齊不啟用；切書/取消/失敗不串設定，設定持久化不散落 localStorage。 Candidate 與 active 分開，按儲存並套用才切換；dirty draft 保護、不可用原因與未知模式處理明確，普通模式無學習必填欄位。
  Evidence: `src/features/reading-settings/`、`src/services/ReadingPreferencesService.ts` 與 Library/App wiring；`npm run check` 通過。`scripts/test-reading-settings.mjs` 在 1280/390 Actual StrictMode App 通過：19-language backend catalog、純 model/DTO checks、candidate/active、dirty close/back/switch、same-language、source/level、strict identity、CAS/rejected/unknown recovery、double-click、held read/save close/reopen、workspace invalidation、keyboard/focus/scroll 與 200-row no-fanout（253/239ms）。實際 LLRN-16 API + temporary file storage 在兩寬度保存/reload/同語失效通過，provider 未呼叫；不等同 live model 或 SQL UI 驗收。Library identity/readiness 回歸通過；舊 workflow fixture 在 hierarchy 等待逾時，未宣稱全套 Library 通過。
  Timestamp: 2026-10-28

### LLRN-18

- [x] LLRN-18: Current-page learning controller.
  Owner: Deep_Reader_UI. Status: Implemented and verified (controller/recovery scope; display delivered separately in LLRN-19).
  Depends on: LLRN-16, LLRN-17.
  Deliverable: Feature-owned model/controller/typed service：當前閱讀頁先讀保存結果，僅有效 missing 明確送生成；有界併發、命令去重與 operation reconciliation。
  Acceptance: 選用模式授權閱讀頁流程，無整書/下一頁預生成。Strict Mode/翻頁/切書/關閉不重複請求或套用舊狀態；read error/unknown 不轉新付費重試；每頁兩類輸出獨立。 僅 active reader mode 的 controller 啟動；候選預覽不排隊，切模式釋放舊 controller 並阻擋晚到資料。
  Evidence: `src/features/learning-reader/`、`LearningService.ts` 與 App/ReaderContentView wiring；type/build 通過。`test-learning-controller.mjs` 14 組案例通過：50-block reads、2-workflow bound、read/source/settings guards、saved/empty reuse、dedup、abort、partial/unknown、receipt recovery、bounded polls、retry consent、refresh 與獨立結果保留。`test-learning-reader.mjs` 在 1280/390 Actual StrictMode App 通過 held/reordered responses、翻頁/切書/來源替換/模式往返、Library 暫停、離頁後 receipt recovery、無下一頁生成、keyboard recovery/scroll；截圖已檢查。`test-learning-reader-api.mjs` 在兩寬度使用真正 LLRN-16 API/lifespan、temporary file storage、固定 offline provider：兩個可見段落共 4 commands/provider calls，reload/reuse 不新增生成。Settings、reader guards 與 Library readiness 回歸通過；未呼叫真模型、未改使用者資料庫，不宣稱逐句 UI/SQL browser/語義品質驗收。
  Follow-up evidence (LLRN-22): current saved/succeeded results remain reusable after a settings roundtrip without polling the historical completed operation. Explicit refresh uses current revisions; active/failed/unknown recovery remains guarded. Controller suite now passes 15 groups, plus type/build and 1280/390 actual learning API/disposable SQL with captured live output replay. No new provider calls in replay; scripts-owned LLRN-22 now accepts system behavior only, with historical semantic observations retained.
  Timestamp: 2026-10-29

### LLRN-19

- [x] LLRN-19: Sentence-pair reading display.
  Owner: Deep_Reader_UI. Status: Implemented and verified (display/actions; system integration evidence belongs to scripts, semantic scoring is outside current scope).
  Depends on: LLRN-18.
  Deliverable: 每句原文下接母語譯文，段落後接詞語提示，再接既有 Insight；學習模式隱藏普通 Translate/View translations 入口與普通 inline translation 區域。 共用 reader 依 mode ID 組裝 typed feature 呈現/actions，避免 App 或單一 controller 累積所有模式分支。
  Acceptance: 長句自然換行，語言方向/輔助技術可辨識；原文在失敗/等待時仍可閱讀。翻譯與目前等級提示各自 refresh，舊結果到新提交前保留；普通模式恢復既有 session display 偏好。
  Evidence: `learning-reader/block-view.tsx`、context/model/controller/session 與 reader wiring；type/build 通過。`test-learning-display.mjs` 在 1280/390 Actual StrictMode App 通過逐句與空白對齊、詞語先於 Insight、empty/partial/failed/unknown/read-error、混語/RTL/長字串、HTML 文字安全、刷新保留舊結果、stable DOM/scroll/keyboard focus、Cancel-first 重複費用確認及新命令 retry；模式往返恢復普通 session 偏好且無重複呈現/普通 reads。Mounted presentation probe 驗證只高亮原文及取消後保留譯文節點；桌面/手機截圖已檢查。LLRN-18 14 組 controller、當頁 lifecycle、普通 inline/focus/session 與 source-review navigation 回歸通過；真 API＋temporary file/offline provider 在兩寬度驗證逐句/譯文/詞語 DOM，兩可見段落共 4 commands/calls，reload 重用。未呼叫付費模型，未宣稱 SQL browser/部署/語義品質或 LLRN-20/21/22 完成。
  Timestamp: 2026-10-29

### LLRN-21

- [x] LLRN-21: Reader lifecycle and accessibility acceptance.
  Owner: Deep_Reader_UI. Status: Implemented and verified (controlled HTTP browser acceptance plus isolated file-backed API/offline provider smoke).
  Depends on: LLRN-17, LLRN-18, LLRN-19, LLRN-20.
  Deliverable: 以真正 App 加可控 HTTP 在桌面/手機驗收 Library 設定到當頁生成/閱讀/刷新/恢復完整流程。
  Acceptance: 涵蓋 Strict Mode、雙擊、快速切換、關閉重開、正常模式回復、source edit/Undo、混語、空提示與有限併發；截圖與鍵盤檢查不能由 build 代替。
  Evidence required: Type/build、browser traces/request counts、1280/390 截圖、焦點/長文/scroll 證據；mock 通過不宣稱真實 API/模型通過。 模式 list/detail 鍵盤、active/candidate 區分、手機返回、dirty 切換/取消/關閉、錯誤後保留設定及未知 mode；以測試用第三種 mode 驗證 list/shell 擴充，不把它發布為新產品模式。
  Evidence: `accept-learning-reader.mjs` passes type/build and ten controller/browser suites, including 14 controller groups and new `test-learning-lifecycle.mjs` at 1280/390. Continuous settings → reading → refresh → edit/Undo → normal flow makes seven unique commands per width over three source tokens, no next-page generation. Test-only third-mode keyboard/step/draft lifecycle exposed and fixed confirmation unmounting the active panel; Keep editing now retains local steps and hidden controls are excluded from Tab/accessibility navigation. Existing fault, concurrency, mode, ordinary translation/session, source navigation and mixed/RTL/long-text regressions pass. Actual learning API/temp file/offline provider smoke makes four calls for two visible blocks, reload reuse passes and fixture cleans up. Traces, request counts, reviewed screenshots, hashes and logs: `Deep_Reflective_Reader/data/test-books/learning-reader/llrn21-20261028/`. Existing bundle warning retained; no SQL browser/live-model/semantic-quality claim. Backend progress is unchanged.
  Timestamp: 2026-10-28

## One-click Interaction Activation (2026-10-27)

- [x] UI-ACTIVATE-01: Generate missing Insights, Summary, Quiz and Critical thinking after one explicit selection.
  Acceptance: Read first; only valid missing state generates once. Preserve saved content/answers, no generation on failed/stale/mismatched reads or passive navigation. Deduplicate, cancel obsolete requests, suppress late results; retry reconciles saved state. Note and explicit Refresh remain unchanged.
  Evidence: npm run check passes; test-interaction-activation.mjs passes at 1280/390 with actual React Strict Mode and controlled APIs: four one-click flows, saved sessions/answers, 11 read-failure/mismatch states per action, no automatic retry, explicit retry reconciliation, close during read/generate, deduplication, target/document switching and Note draft protection. Summary stale-to-explicit-Refresh also passes a focused rerun at both widths. test-reader-guards.mjs passes 42 identity cases and lifecycle recovery; test-hierarchy-interaction-menus.mjs passes both widths with the real 451-unit layout and mocked saved interactions. Desktop/mobile screenshots inspected. No live model generation or document mutation was performed. Additional legacy test-reader-recovery.mjs stops later at Enhance because its activity-status fixture is missing; not claimed as a full pass.
  Timestamp: 2026-10-27

## Irreversible Ignore All OCR Flags (2026-10-27)

- [x] UI-IGNOREALL-01: Ignore every pending OCR flag with explicit irreversible confirmation.
  Acceptance: Whole-document atomic scope, count/warning/Cancel-first confirmation, idempotent retry and stale-request safety; no body deletion or hidden reload. Prior and future Undo cannot revive bulk-ignored flags. Single Ignore remains reversible.
  Evidence: IgnoreAllRegionsDialog and ReadingContentActions implement whole-document confirmation and same-command retry. npm run check and test-ignore-all-regions.mjs, test-source-review-navigation.mjs and test-cleanup-history.mjs pass at 1280/390. Browser coverage includes 25 regions across pages, Cancel-first focus, duplicate clicks, failed/lost-response retry, one logical commit, no body/layout reload, disabled empty action/Undo, unmount and late-response suppression; screenshots inspected.
  Timestamp: 2026-10-27


## OCR Review Action Visibility (2026-10-27)

- [x] UI-OCRREVIEW-01: Restrict document review actions to PDFs with persisted OCR evidence.
  Acceptance: Full OCR and actual Auto OCR remain visible, including after resolution; native/fast PDF, non-PDF, missing/stale evidence remain hidden. Preserve ordinary body edit/delete and stale-response safety. No extraction or persistence on reads.
  Evidence: npm run check and test-ocr-review-visibility.mjs, test-reading-content-requests.mjs, test-source-review-navigation.mjs and test-cleanup-history.mjs pass at 1280/390. OCR-only entry/Undo, completed review, missing capability/loading, late response isolation, ordinary text Edit and review-close keyboard focus verified; screenshots inspected. Existing mocked OCR fixtures now declare the capability.
  Timestamp: 2026-10-27


## Extracted Text Review Terminology (2026-10-27)

- [x] UI-TEXTREVIEW-01: Clarify the extracted-text checking and correction module.
  Acceptance: Entry Review extracted text, dialog Extracted text review, tab Flagged areas; concise purpose description and consistent accessible/refresh/confirmation/empty-state copy. Keep unresolved/ignored semantics accurate and backend contracts unchanged. Verify build and existing desktop/mobile review/navigation/history flows, keyboard tabs and fitted screenshots.
  Evidence: ReadingContentActions, SourceReviewBlockRepair and related refresh/result copy updated after design governance. npm run check, test-source-review-navigation.mjs, test-source-review-state.mjs and test-cleanup-history.mjs passed; 1280px/390px screenshots inspected for title/description/tab fit. Existing keyboard-tab, navigation, draft/scroll retention, history and snapshot-guard coverage passed.
  Timestamp: 2026-10-27

## Chapter / Section Tree Navigation (2026-10-26)

- [x] UI-NAV-TREE-01: Replace the chapter list presentation with an accessible MUI tree and readable long titles.
  Acceptance: At most chapter/section levels, existing single-section merge and action targets, controlled reading selection, keyboard expansion/navigation, fixed action width and fully wrapped titles. Local document-scoped expansion only; no task-unit nodes, backend mutation or eager content reads. Preserve bounded scrolling and verify narrow/large layouts.
  Evidence: @mui/x-tree-view 8.29.3, NavigationTreeItem and hierarchy-navigation/view.tsx; npm run check and test-hierarchy-tree.mjs pass at 1280/390/320px. Header, real-layout interaction-menu and pane-resize regressions pass; screenshots inspected. The 1,100-node fixture mounts 100 chapters initially and 110 nodes after one expansion, with no expansion API/storage writes. Existing 451-unit merged row still selects the correct section and retains its chapter menu.
  Timestamp: 2026-10-26

## Inline Translation Display (2026-10-26)

- [x] UI-TRINLINE-01: Select saved translation languages for display below the current block.
  Historical scope: Initial per-block/current-section selection is superseded by UI-TRINLINE-02; retain this item as initial delivery evidence.
  Acceptance: Saved drawer has accessible Translations/Reading display tabs and multiple language checkboxes. Render original body, optional block Insight, then selected translations; preserve paragraphs, partial status and narrow-screen wrapping. Per-block reading-session state survives drawer close and same-section pagination, resets on scope/source changes and responds to server invalidation even with the drawer closed. No hidden generation, persistence or whole-book reads.
  Evidence: useInlineTranslations and InlineTranslations remain feature-owned; App only supplies the getter to the reader. npm run check and mounted controller tests pass, including closed-drawer invalidation and edit/Undo non-resurrection. test-translation-inline.mjs passes actual App/reader/DTO with controlled HTTP at 1280/390 for multiselect, body/Insight order and dismissal, page retention, block isolation, authoritative saved-read updates/removal and scope reset with zero generate calls. Existing focus/recovery suites pass at both widths. Desktop/mobile screenshots inspected; backend contracts unchanged, no live books mutated.
  Timestamp: 2026-10-26

- [x] UI-TRINLINE-02: Make Reading display language selection application-session scoped.
  Acceptance: One language preference survives drawer/block/page/section/book changes and applies to existing read results and future persisted translations; full reload/unmount resets it. Separate source-bound current-scope cache from preference. Read only rendered-page blocks, at most two background requests, no automatic generation or preference writes. Preserve preferences through missing/invalidation/errors, offer read retry, prevent stale cross-source/book results, and allow clearing any selected language.
  Additional acceptance: Selected languages appear exclusively inline; unchecked ones remain in Translations. Highlight follows the selected target between locations, including generation/reuse, checkbox changes and reload. Explain an all-inline drawer and avoid focusing behind an open modal.
  Evidence required: Type/build, mounted lifecycle/session regression, actual-App 1280/390 tests for scope transitions, future generation, global deselection, source invalidation, failed/stale reads, bounded reads and reload reset; inspect screenshots. Update the superseded per-block/current-section lifetime only with passing evidence.
  Evidence: Design/checklist updated before code; session preference, current-scope snapshot/read queue and exclusive focus destinations are implemented in block-translation. npm run check and controller/session-lifecycle/inline/focus/recovery suites pass. Actual-App 1280/390 controlled-HTTP checks cover cross-book same-ID isolation, future explicit generation, global deselection, reload reset, retry, exclusive placement and focus transfer; held reads prove at most two background requests and stale source/scope suppression. Settings/highlight screenshots inspected; no backend or live-book writes.
  Timestamp: 2026-10-26

## Persisted Block Translation (2026-10-26)

Implemented and verified: **5/5 UI tasks (100%)**; shared track 11/11.
Backend 6/6 is tracked by its owners, not counted as UI completion.
[Shared order](../Deep_Reflective_Reader/section_tasks/module-checklist.md#persisted-block-translation-queue-2026-10-26).
[UI design](module-detailed-design.md#persisted-block-translation-plan-2026-10-26).

- [x] TRUI-07: Add typed translation services and feature model.
  Depends on: TRUI-05/06. Acceptance: RestClient read/options/generate, block-only identity, optional profile-default source, registry authority, target-code grouping and explicit saved/partial/failure states. No raw source, local persistence, independent allowlist or hidden generation on read.
  Evidence required: DTO/model tests for minimal payloads, unsupported/malformed data, grouping, missing profiles, registry changes, HTTP/business errors and AbortSignal forwarding.
  Evidence: TranslationService and block-translation/model provide strict target/result checks, minimal payloads, registry-driven selectors, partial/untranslatable distinction and language-key replacement. Mounted model/DTO tests and TypeScript build pass; no browser persistence or hidden generation.
  Timestamp: 2026-10-26
- [x] TRUI-08: Add block Translate menu action and searchable language Modal.
  Depends on: TRUI-07. Acceptance: Normal block menu only; profile-default From, required To, selection-only searchable combos from registry codes, disabled invalid submit, Cancel and View translations. Stable classes/labels and existing MUI/notification conventions; no parent-level/review-menu leakage.
  Evidence required: Defaults, search by name/code, arbitrary text rejection, empty/unknown language, loading/failure/cancel, keyboard/focus and desktop/mobile fit.
  Evidence: Normal block menu owns Translate/View translations; MUI Autocomplete is selection-only with backend profile defaults and no To default. Real App checks reject free-form input, exercise cancel and capture fitted Modal screenshots at 1280/390; parent/review menus are unchanged.
  Distinct-language correction: To excludes From; changing From clears a conflicting To. Controller/model/DTO guards block equal-code submission before saved reuse or HTTP. Mounted controller tests and actual MUI recovery tests at 1280/390 pass, including disabled submission and restored options; npm run check passes.
  Timestamp: 2026-10-26
- [x] TRUI-09: Implement request ownership and lifecycle-safe result delivery.
  Depends on: TRUI-08. Acceptance: Submit immediately closes Modal and transfers request to feature controller; prevent duplicates and open drawer only on persisted success. Abort/generation/target/source guards on close/unmount/switch. Reconcile saved results after unknown outcomes; no stale updates, timeout, storage or polling.
  Evidence required: Held requests, double clicks, close/reopen, unmount, block/document switch, source mutation, reordered responses, provider/commit failure and response loss; no stale drawer or unhandled promise.
  Evidence: feature controller explicitly owns submitted generation, aborts obsolete work and guards generation/target/source. test-translation-controller covers StrictMode, cancellation, reordered reads, duplicates, unknown-outcome same-ID retry, partial/untranslatable, definitive rejection, edit and unmount. Real held-request navigation produces no stale drawer or page errors.
  Timestamp: 2026-10-26
- [x] TRUI-10: Render persisted multi-language drawer and source-change invalidation.
  Depends on: TRUI-07/09 and backend TRUI-03. Acceptance: Stable target-code sections, source/time/status/text, DB reload/reopen, same-language reuse without generation or sibling loss, Add translation and bounded long content. Honor no-op versus real changes: mandatory translation/summary clear, optional other metadata. Update edit confirmation and consume server invalidation scope without full layout refresh.
  Evidence required: Multiple languages, regeneration success/failure, partial/untranslatable, saved-read retry, unchanged edit preserving records, changed Keep/Clear and block deletion, sibling preservation, long-text/keyboard screenshots.
  Evidence: bounded scrollable drawer reads saved language sections, replaces one language and retains siblings. App consumes summary/translation invalidation separately from other metadata; same-text editor avoids refresh, real changes use bounded block refresh. Two-language reload/regeneration and no-op/change Keep pass on real SQL at 1280/390; backend retention matrix and frontend failure/lifecycle tests pass.
  Superseding evidence: Same To submission now displays saved data without generate, including changed/missing From when no equal pair is selected; backend handles cache hits from direct API calls. Controller/DTO/build and real-book desktop/mobile acceptance 20261026T113558Z-5d5c71b0 pass. Earlier regeneration semantics are historical, not current behavior.
  Distinct-language boundary: Selected From must differ from To even for saved-target reuse. View translations preserves access to historical records; controller regression verifies zero generate calls on rejected equal pairs.
  Timestamp: 2026-10-26
- [x] TRUI-11: Accept integrated translation/editing workflows using backed-up real books.
  Depends on: TRUI-06/08/09/10. Acceptance: Actual App/current API/disposable PostgreSQL and configured LLM block tests using immutable Madame Bovary/Red Roulette backups. Prove two persisted target languages after reload, same-language reuse without generation, registry/profile controls and no-op/change/delete retention matrix. Run lifecycle regressions, build, accessibility and 1280/390 screenshots.
  Evidence required: Private hashes/API/SQL/browser reports, bounded request counts, unchanged production books/profiles and removal of disposable resources. No skipped gate claimed as pass; distinguish response validity from literary accuracy and deployment from implementation.
  Evidence: accept_persisted_translation_books --live --browser plus test-translation-real.mjs passes actual App/current API/configured model/disposable SQL on both immutable backup excerpts. Evidence: backend data/test-books/persisted-translation/20261026T070158Z-06e6f6e0; profile/backup hashes unchanged, schemas removed. Build, controller/DTO, bounded-refresh and 1280/390/320 pagination regressions pass. Existing Vite is running at 127.0.0.1:5173; production Docker was not restarted and translation semantics are not literary-quality certification.
  Reuse acceptance: 20261026T113558Z-5d5c71b0 passes both backup excerpts at 1280/390 with no repeat-language generate request, unchanged saved records and direct API reuse. Source-change invalidation remains covered; no production data modified.
  Timestamp: 2026-10-26

## Block-Level Source Review Correction (2026-10-26)

- [x] SRBLOCK-02: Put Edit/Delete/Keep it on each source-review-block and accept the incident workflow.
  Acceptance: No region-level mutation menu; independently editable blocks; server-authorized main-edit protection; Keep confirmation removes review entries but preserves body/Notes. Stable keyed DOM, bounded paging, targeted refresh, same-command replay and lifecycle guards. Actual Madame Bovary Page 3 backup/API/SQL/browser checks at desktop/mobile plus regression/build.
  Evidence: Actual App/FastAPI/disposable PostgreSQL passes at 1280/390 on the natural Page 3 region with 20 canonical related blocks. Repeat edits change only the selected block, keep manual flags false and preserve sibling DOM/content; Keep cancel/replay/reopen and scoped Delete pass without layout/section reloads. Editor lifecycle/Unicode/duplicate-submit, source-review state/navigation and reading-refresh suites pass; build passes with existing bundle warning. Final private report: data/test-books/srrepair07/20261026T044959Z-7b1702d7 under the backend; backup hashes unchanged, schema removed. Current API rebuilt/healthy; read-only live browser confirms 18 visible block menus (two pre-existing hidden blocks), zero region menus and the Keep confirmation; no live book mutations.
  Timestamp: 2026-10-26

## Guarded Source Region Repair (2026-10-26)

Historical acceptance of region-wide behavior, superseded by SRBLOCK. It must not
be reported as acceptance of the corrected block-level workflow.

UI tasks in the shared seven-task
[SRREPAIR queue](../Deep_Reflective_Reader/document_structure/module-checklist.md#guarded-source-region-repair-2026-10-26).
Backend owns 01..04/07; no duplicate checkboxes or completion inferred from SRMAP/RCR.
UI-owned implementation: 2/2 complete (100%); shared SRREPAIR track 7/7 accepted.
Integrated real-state receipt is owned by backend 07, not counted again here.
[Design](module-detailed-design.md#guarded-source-region-repair-2026-10-26).

- [x] SRREPAIR-05: Add guarded region draft editing, Apply and Delete text controls.
  Depends on: backend SRREPAIR-04. Acceptance: Typed services and existing MVC/editor/confirmation components; original/draft/applied text remain distinct. Render authoritative eligibility/reasons, block IDs, explicit interaction Keep/Clear and destructive confirmation. Older servers remain read-only; preserve Ignore/navigation and do not set manual flags or infer writable ranges.
  Evidence required: Component/browser tests for draft versus Apply, repeat repair, prohibited body-edited/deleted targets, multiple related blocks, failures, duplicate clicks, confirmation, legacy responses and centralized notifications.
  Evidence: SourceRegionRepair provides server-authorized Edit/Delete menus, independent Save draft/Apply, Keep/Clear and destructive confirmation. Typed ReadingContentService owns transport and definite-versus-unknown failures. Mounted editor tests pass at 1280/390 for Unicode limits, dirty cancellation, double submission, consent, late target switch and unmount; actual App/API/disposable SQL acceptance on the dark-water backup verifies repeat repair, delete/recover, unchanged flags and disabled controls after reader Edit. npm run check passes with the existing bundle-size warning.
  Timestamp: 2026-10-26
- [x] SRREPAIR-06: Preserve review position with targeted updates and lifecycle-safe requests.
  Depends on: SRREPAIR-05. Acceptance: Merge authoritative target blocks and affected related rows only; no whole layout fetch/modal remount. Preserve tab/expansion/scroll/drafts/focus and current navigation highlights. Abort/generation guards, source/revision binding, unknown-outcome reconciliation and committed-read retry; bounded lists and no browser persistence or polling.
  Evidence required: Actual App/browser tests at 1280/390 for long lists, late responses, close/reopen/switch, cross-block updates, hidden recovery, shared-region changes, scroll/focus and zero unnecessary layout reads. Build passes; real API/SQL acceptance remains backend-owned SRREPAIR-07 with UI support.
  Evidence: Existing keyed rows/expansion are retained; editor exit restores trigger focus and scroll. Regions render 20 rows, with bounded related/refresh paging. Draft Save performs no GET; region body changes publish only affected reader DTOs and lazily invalidate history. Actual dark-water desktop/mobile flow verifies connected row identity, scroll/focus, no region-triggered layout/section fetch, same-ID response-loss retry and local delete/recovery. Reading-refresh, cleanup-history, source-review-navigation/state and editor browser suites pass, including stale reads, abort/switch/unmount and legacy contracts. No persistence or polling added.
  Timestamp: 2026-10-26

## Repeatable Cleanup Repair (2026-10-25)

UI follow-up 2/2; shared RCR track 4/4 (100%, RCR-01/03/04/05). Execution order and backend tasks are in
[the owner checklist](../Deep_Reflective_Reader/document_structure/module-checklist.md#repeatable-cleanup-repair-2026-10-25).
Do not count the earlier CRUI acceptance as these corrections.

- [x] RCR-04: Reapply a cleanup record and refresh only affected body blocks.
  Acceptance: Keep draft Save separate from explicit Restore/apply. Consume backend eligibility even for restored records; never set manually_edited locally. Replace authoritative target blocks in the existing reader controller, recompute local pagination and retain focus/selection. Refresh history and affected interaction states; no task-layout request, whole-section reload or content concatenation. Preserve history even when navigation is unavailable, including recoverable cleanup-hidden blocks. Preserve cancellation, dirty drafts, same-ID retry and read-only refresh retry after committed success.
  Evidence required: Actual component/browser tests for repeat edits, targeted read/request counts, no duplicate insertion, own/foreign changes, cross-block restoration, stale responses, close/switch, failure/retry and long histories at 1280/390.
  Evidence: ReadingContentService/readingRefreshModel/reader controller consume revision-bound pages and atomically merge server DTOs/tombstones, sibling hashes and generated-state invalidation. Cleanup Restore/Undo do not fetch layout/whole-section content; Save only updates history. Same-ID response-loss recovery and read-only refresh retry remain separate. Actual App controlled-transport tests pass at 1280/390: three repeated applications, no duplicated text, zero layout/section reloads on restore, preserved page/expansion/focus/unsaved Note, backend edit-protection presentation, retained non-navigable history, 23-item pagination, conflicts and legacy behavior. Mounted controller tests pass 25-block pagination, six failure/lifecycle cases and 13 merge/contract checks including cross-unit/hidden recovery. Source-review navigation/state and reading-request regression suites pass. Build and diff checks pass; screenshots reviewed. These controlled fixtures did not themselves establish isolated incident-book acceptance; the subsequent RCR-05 evidence is recorded below. No backend code, live book mutation or deployment in RCR-04.
  Timestamp: 2026-10-25
- [x] RCR-05: Accept the corrected flow against an isolated copy of the incident data.
  Acceptance: Back up and read-verify the current document containing 19652:content:2; exercise edits/restores only in disposable storage, not the production book. Prove saved draft versus applied body, multiple replacements, false body-edit flag, block-specific refresh, subsequent body Edit protection and retained non-navigable history. Run real API/SQL and desktop/mobile browser acceptance; preserve source/backup hashes. Supplement with fixtures for absent natural cases and report them honestly.
  Evidence required: Source fingerprints/private report, API/SQL state assertions, UI target content and request counters, screenshots, build and regressions. No live LLM/OCR or production deletion is needed.
  Evidence: accept-repeatable-cleanup.py and test-repeatable-cleanup-real.mjs pass actual App/FastAPI/disposable PostgreSQL at 1280/390 using a byte-verified backup of document 68, retaining original IDs, revision 5 and legacy Restore/Undo evidence. Each viewport proves three replacements, Save without body mutation, false backend manual-edit flags, exactly three targeted reads with zero layout/section reloads, and five naturally non-navigable records retained. Body interactive Edit sets the durable flag; direct Restore returns 409 with identical SQL fingerprint. Document/snapshot reading JSONB agree; canonical hierarchy, source/profile and backups remain unchanged; temporary schema removed. Private evidence: data/test-books/rcr05/20261025T150648Z-06b7c9e3 under the backend. 264 file/SQL regression tests, cleanup-history/reading-refresh/reading-content-requests browser suites and build pass; desktop/mobile screenshots reviewed. Library/status and disabled generated availability are isolated test boundaries; supplemental fixtures cover absent failure/cross-block/Note cases. No production writes, LLM/OCR, deployment or new product behavior in this task.
  Timestamp: 2026-10-25

## Per-Item Cleanup Recovery (2026-10-25)

Historical CRUI delivery; RCR above supersedes one-shot Restore and full-layout
refresh. The old test results do not establish the new acceptance criteria.

Ordered UI tasks depend on backend CLEAN-RESTORE support. This track does not
reopen SCLEAN or claim model accuracy. Progress: 5/5 (100%).

- [x] CRUI-01: Define typed cleanup record IDs, canonical block links and server restore eligibility; older servers remain read-only.
  Evidence: ReadingContentService models record IDs, canonical links, eligibility and backend manually_edited; browser legacy responses remain read-only without runtime errors.
  Timestamp: 2026-10-25
- [x] CRUI-02: Add bounded per-record restoration-text editing with explicit save, immutable original quote and dirty-close protection; no browser persistence.
  Evidence: CleanupAudit/useCleanupHistory separate saved restoration text from original quote, paginate 10 records, preserve dirty drafts and use only component state. Desktop/mobile tests cover save, restored-record editing, discard and pagination.
  Timestamp: 2026-10-25
- [x] CRUI-03: Restore through the existing mutation service with CAS/idempotency, truthful errors and one authoritative reading refresh; never concatenate body text in UI.
  Evidence: Existing reading-content mutation port handles cleanup_edit/cleanup_restore. Browser tests verify lost-response same-ID replay, duplicate-click protection, conflict draft retention, one restore refresh and no layout refresh for draft saves; backend file/SQL tests verify transactional guards.
  Timestamp: 2026-10-25
- [x] CRUI-04: Navigate using verified current block identities and existing measured reader pagination, with abort/generation guards and no guessed targets.
  Evidence: Specific-block read and snapshot/hierarchy checks precede existing reader navigation. Tests cover actual reader focus, pending navigation cancellation and unavailable/legacy targets at 1280/390.
  Timestamp: 2026-10-25
- [x] CRUI-05: Verify desktop/mobile editor, restore, navigation, conflicts, duplicate clicks, lifecycle, legacy fallback and long histories; run build and backend support tests.
  Evidence: test-cleanup-history.mjs, test-source-review-navigation.mjs and test-reading-content-requests.mjs pass at 1280/390; screenshots inspected. Backend regression 266 tests plus existing audit suite 29 tests pass, including disposable PostgreSQL and read-only backup-book shared-engine checks. npm run check passes with the existing bundle-size warning. Browser HTTP fixtures are not live-book/model acceptance; no production book writes or deployment.
  Timestamp: 2026-10-25

## OCR Preservation Policy Reference (2026-10-25)

Backend OCR-KEEP supersedes earlier SRMAP/CLEAN claims that every proven
never-in-body region should be hidden. Quality-excluded legacy text and nonblank
recognition gaps remain server-visible; new low-quality text stays in body with
review links. UI consumes server eligibility without empty-block filtering or a
second editor. This is a requirements correction, not another completed UI task.
Historical browser receipts below retain their tested scope, not the old policy.
Timestamp: 2026-10-25

## Read-Only Source Review Backlog (2026-10-25)

Completed: **3/3 UI tasks accepted (100%)**. Shared SRMAP track: 8/8, with
five backend-owned tasks and three UI-owned tasks; no remaining task in this track.
Historical review/cleanup completion is not substituted for this acceptance.
[Shared execution order](../Deep_Reflective_Reader/document_structure/module-checklist.md#source-review-mapping-backlog-2026-10-25).
[UI design](module-detailed-design.md#read-only-source-review-navigation-2026-10-25).

- [x] SRMAP-05: Replace Source review inline editing with verified body navigation.
  Priority: P1. Depends on: backend SRMAP-04. Owner: Deep_Reader_UI.
  Acceptance: Remove review block Edit/Delete menus; retain read-only current text, block IDs, collapsible Original and confirmed Ignore. Consume implemented typed server targets only; one/multiple related blocks navigate through existing chapter/section/content selection and reader pagination, not PDF page_index. Existing body three-dot actions are the sole editor. No insertion, substring editing, fuzzy quote search, whole-book prefetch, Prepare or hidden task-layout mutation. Unknown/all-deleted targets cannot navigate to guessed or empty IDs.
  Evidence required: Actual component tests for one/multiple/partial/shared blocks, reader page differing from PDF page, resize/re-pagination, stale target, all-deleted links and conservative old-server handling. Existing inline-review tests must be revised, not retained as current acceptance.
  Evidence: Read-only SourceReviewBlocks and sourceReviewModel consume verified targets; controller/view navigation uses section selection and measured pagination, with block focus maintained through resize. Snapshot identity and target text are revalidated after content fetch before publication/focus. No inline review editor, insertion, fuzzy navigation or Prepare. Actual App/StrictMode navigation tests pass at 1280/390 for shared/partial/unknown/deleted/legacy targets, long/shrinking lists, stale IDs/text and cancellation.
  Timestamp: 2026-10-25

- [x] SRMAP-06: Refresh authoritative review state and display manual modification history.
  Priority: P1. Depends on: SRMAP-05. Owner: Deep_Reader_UI.
  Acceptance: Reselect/reopen/explicit refresh loads current text and backend manual-change state/time; body mutation uses the existing single read-only layout/content refresh path. Never equate modified/deleted with resolved; last-block deletion remains visible until user Ignore. Render server-filtered queue/count/guard truth, not local empty-block filtering. Edit Keep/Clear explicitly names Note; Delete/Undo explain irreversible interaction removal. Preserve dirty drafts, selection/focus, lazy lists and collapsible previews; central notifications, stable DOM identity, abort/generation guards on close/switch and no manual mutation-count limit.
  Evidence required: Same-row reselection, edit/delete/Undo/restart, timestamps unaffected by cleanup/Ignore, duplicate-read/mutation counters, failed refresh, conflict, rapid switching, unmount, stale responses, long-list scroll and no uncaught rejection. No optimistic success after unknown commit.
  Evidence: useSourceReviewBlocks refreshes on activation/reselection, guards dispatch and latest snapshots, aborts superseded work and clamps pagination. Server manual-change timestamps/deleted counts survive reload; all-deleted rows retain explicit Ignore. Stale cleanup audit is discarded on identity/revision change. Ignore confirmation describes acknowledgment, not body removal. Twelve delayed-response cases, nine display-normalization comparisons, 42 identity guards and existing request/pagination regressions pass; body drafts, Keep/Clear Note, text-only Undo and one layout refresh are preserved.
  Timestamp: 2026-10-25

- [x] SRMAP-08: Accept the read-only review workflow against real books and backend state.
  Priority: P1. Depends on: SRMAP-05/06 and backend SRMAP-07. Owner: Deep_Reader_UI; backend/scripts support real-source acceptance.
  Acceptance: Actual App/API/disposable PostgreSQL at desktop 1280 and mobile 390, using verified backups from backend acceptance. Select region -> navigate to actual body page -> edit/save or delete -> reopen/reselect -> inspect current text/manual time -> explicit Ignore. Verify proven excluded regions hidden without fabricated audit, unknown/partial mappings retained, all-deleted row retained, Note policies, one layout refresh, Undo and source immutability. Test no inline review editor/manual block insertion and no automatic Ignore from body mutation.
  Evidence required: Source hashes/private reports, reviewed region-block correspondence, screenshots/focus/scroll checks, request counts, regression and type/build results. Controlled fixtures supplement but cannot replace a real included-region case. Missing corpus or mapping evidence stays Needs Confirmation; do not count earlier injected-span browser success as this acceptance. No production data deletion or deployment implied.
  Evidence: Updated accept-source-review.py/test-source-review.mjs pass actual App/FastAPI/disposable SQL at 1280/390 in synthetic and hash-verified sample2 modes. Real-page text/subquote evidence has no injected canonical_span; no naturally flagged positive link exists in the corpus, and Original preview is a UI fixture. Tests prove navigation, body edit/save/delete, current text/time, Keep/Clear physical Note, retained deletion, explicit Ignore, text-only Undo, retry and one layout refresh per mutation. Twelve screenshots inspected; source/canonical/backup hashes unchanged. Private manifest/logs/screenshots: Deep_Reflective_Reader/data/test-books/srmap08/. npm run check passes with existing bundle warning; independent review findings resolved and controller/type checks pass. No production changes or deployment.
  Timestamp: 2026-10-25

## Request Wait Policy Follow-up (2026-10-24)

New UI follow-up 1/1 (100%), separate from completed SCLEAN-08/09.
[Shared WAIT order](../Deep_Reflective_Reader/llm/module-checklist.md#offline-preflight-follow-up-2026-10-24):
current combined progress is maintained in that owner index, not duplicated here.
Backend aggregate must not count this UI task.
Timestamp: 2026-10-24

- [x] WAIT-01: Remove frontend fixed API request deadlines without weakening lifecycle safety.
  Priority: P0; first implementation task. Owner: Deep_Reader_UI.
  Acceptance: Remove both current 120-second abort timers in Smart cleanup and Smart structuring; audit other app request paths for equivalent deadlines. Pending preflight/execute/lookup/resume/read-refresh must not fail or retry merely due to elapsed time. Retain owner-scoped AbortController, cancellation/close, generation/identity guards, busy locks, drafts/recovery handles, actual transport-error reconciliation and backend consent expiry. Do not remove polling/debounce/queue timers or change backend/provider timeouts.
  Evidence required: Desktop/mobile controlled-time tests past 120 seconds for pending then successful responses, cancellation/unmount/switch and late response isolation; actual network/gateway failures, expired consent, reload/reopen and zero duplicate mutations. Update old timeout tests to retain recovery coverage; build/type checks pass. No claim that backend cold-start latency is fixed by this task.
  Evidence: Removed both timer-triggered aborts; source audit finds only the unrelated zero-delay upload queue timeout. Cleanup retains lifecycle controllers, consent expiry and opaque recovery handles. Structuring execution/read-refresh now share dialog-owned cancellation with generation/identity guards; Cancel remains available while waiting. Cancelled submission becomes unknown and requires authoritative reconciliation; cancelled reading refresh retains its handle without treating an unacknowledged legacy preparation as success. App forwards AbortSignal to existing services without a new timeout or backend policy.
  Validation: npm run check passes (existing bundle-size warning). test-smart-cleanup-contract.mjs passes 36 malformed response cases. test-enhancement-confirmation.mjs, test-enhancement-outcomes.mjs, test-smart-cleanup.mjs, test-smart-cleanup-recovery.mjs and test-cleanup-flow.mjs pass at 1280/390; held responses past 120 seconds cover preflight/execute/read refresh/lookup, alongside later success, close/switch/reload recovery, consent expiry, network/503 faults, duplicate-submit and draft preservation. Long-wait screenshots inspected. Cleanup suites use actual App/FastAPI/disposable PostgreSQL with a deterministic provider; enhancement transport faults are fixtures. No live-book/model latency acceptance, production writes, Docker restart or backend progress update; WAIT-02..05 remain open.
  Timestamp: 2026-10-24

## Independent Smart Cleanup Backlog (2026-10-23)

New UI tasks 2/2 (100%); shared SCLEAN track 10/10 (100%). Backend-owned tasks are
8/8 after scripts-owned SCLEAN-10 final real-book/API/SQL/browser and isolated-image
acceptance. UI tasks are not counted again; WAIT-01 above is a separate new follow-up.
Unsupported storage stays unavailable. Source acceptance is not production deployment.
Seven final live-book flows and final reviewed-span evidence are recorded in the
[scripts acceptance](../Deep_Reflective_Reader/scripts/module-detailed-design.md#sclean-10-delivery-2026-10-24).
[Shared order](../Deep_Reflective_Reader/document_structure/module-checklist.md#independent-smart-cleanup-backlog-2026-10-23).
Timestamp: 2026-10-24

- [x] SCLEAN-08: Add the independent document Smart cleanup entry and confirmation.
  Priority: P1. Depends on: backend SCLEAN-06/07. Owner: Deep_Reader_UI.
  Acceptance: Existing action surface alongside Smart structuring, dedicated narrow controller/service and strict typed adapter. Accepted Common documents can invoke backend eligibility preflight, including before units exist; no local OCR inference. Confirm content-only impact, retain notes/drafts and distinguish unavailable/over-capacity/old-server states. One user action runs cleanup/readiness; no plan editor, arbitrary scope/provider controls, fallback reparse or manual Prepare.
  Evidence required: 1280/390 actual component/API tests for Common readiness, unavailable native/unknown/capacity/unsupported server, consent, draft protection, duplicate-click prevention, progress/outcomes and automatic reading. Backend contract claims require implemented route evidence.
  Evidence: SmartCleanupService strict runtime adapters and dedicated smart-cleanup MVC feature use the implemented preflight/execute routes only. Repairs exposes independent cleanup even without units; server preflight gates consent. Shared outcome validation/messages and reading refresh avoid duplicating enhancement or cleanup policy. Manual block editing disables cleanup; mounted Note drafts survive. Abort/generation guards, consent expiry, synchronous duplicate-submit lock and per-document opaque pending handles prevent stale updates and blind regeneration. Unknown/layout/reload recovery remains SCLEAN-09, not complete here.
  Validation: npm run check passes (existing bundle warning); test-smart-cleanup-contract.mjs rejects 31 malformed responses and verifies narrow payloads/identity/safe errors. test-smart-cleanup.mjs passes actual App/FastAPI/disposable PostgreSQL at 1280/390 with accepted Common fixtures, pre-unit layout handoff, completed/partial/no-op/skipped/provider-failed outcomes, unavailable/native/unknown/capacity/legacy gates, cancellation/expiry, notes/manual drafts, duplicate clicks, response loss/reopen/refresh blocking and late response after document switch. Zero split calls and no Prepare/reparse requests. Screenshots inspected; existing enhancement confirmation regression passes. Controlled provider and note-read fixture, not live-book/model quality acceptance; SCLEAN-10 stays open. No production document changes or Docker restart.
  Timestamp: 2026-10-24

- [x] SCLEAN-09: Add independent retry/reconciliation and reuse audit/review/Undo.
  Priority: P1. Depends on: SCLEAN-08. Owner: Deep_Reader_UI.
  Acceptance: Failed/partial automatic cleanup can start a new confirmed standalone cleanup without splitting. Same-operation lost response/refresh uses lookup, backend-directed layout continuation only; unknown/stale/superseded states never auto-regenerate. Content/review/availability reload follows authoritative revision, preserves selection and drafts, and reuses existing collapsible audit, remaining-region actions and latest Undo. No independent UI content truth or hide-all-review shortcut.
  Evidence required: Actual App/API/SQL at 1280/390 for failed-auto-then-retry, response loss, switch/refresh/restart, concurrent edits, committed layout failure, no-op/partial, review resolution/retention, Undo conflicts and stable IDs. Assert no duplicate model calls/commits and no split/OCR/profile/Prepare fallback; inspect screenshots. Final real-book evidence is counted once in SCLEAN-10.
  Evidence: Dedicated cleanup controller owns document-scoped opaque recovery handles, read-only lookup and explicit backend-directed layout resume. Failed/partial automatic cleanup opens the same standalone feature with fresh consent. Strict transport retains authoritative operation DTOs on 409/503. Lost responses, refresh/restart and unknown results cannot bypass reconciliation; terminal/current-revision refresh precedes handle removal. Existing review/audit/Undo and Note drafts are reused, not duplicated.
  Validation: npm run check passes with the existing bundle warning; test-smart-cleanup-contract.mjs rejects 36 malformed response variants. Actual App/FastAPI/disposable PostgreSQL suites test-smart-cleanup-recovery.mjs, test-smart-cleanup.mjs, test-cleanup-flow.mjs and test-cleanup-review.mjs all pass at 1280/390. Coverage includes failed/partial automatic cleanup retry, lost acknowledgement, process recreation, reading timeout/failure, layout-only resume, concurrent edit, supersession after Undo, stable IDs, draft preservation, remaining regions, collapsible Original, Ignore deletion and Undo conflicts. Counters reject duplicate model calls/commits; no hidden split/Prepare fallback. Mobile audit and desktop conflict screenshots inspected. Deterministic provider, Note-read fixture and synthetic preview image are not live-book/model/OCR-image acceptance; SCLEAN-10 remains open. No production data changes or Docker restart.
  Timestamp: 2026-10-24

## Optional OCR Cleanup Experience Backlog (2026-10-22)

Historical foundation only; its completion does not close SCLEAN-08/09 above.

CLEAN-07/08 delivered; UI 2/2 (100%). Combined track 10/10 (100%) as of 2026-10-23;
backend CLEAN-09/10 evidence is recorded once in the shared ordered track:
[backend checklist](../Deep_Reflective_Reader/document_structure/module-checklist.md#optional-ocr-cleanup-plan-backlog-2026-10-22).
No duplicate UI acceptance task is added. Historical per-item delivery evidence
below retains its original scope; final real-book browser evidence is in the design.
Timestamp: 2026-10-23

- [x] CLEAN-07: Integrate consent and nonblocking cleanup status into the enhancement flow.
  Depends on: backend CLEAN-06.
  Acceptance: Typed additive adapter with legacy compatibility; one confirm -> split in memory -> optional cleanup -> single final commit -> automatic layout/open. Never show saved success before commit; final storage failure retains the old document. Concise partial/failure/no-change messages and phase-aware retry; no raw model budgets or mandatory manual Prepare. Refresh/document switch/late responses reconcile backend operation and reading revision. Distinguish structural all-interaction deletion from cleanup generated-output invalidation with notes preserved.
  Evidence required: Desktop/mobile actual component/API tests for successful, skipped, partial, failed, stale and unknown outcomes; changed-only content reload, dirty-draft safety and no duplicate operation.
  Evidence: Typed adapters and outcome validation, bound cleanup consent, per-document session recovery locator, phase-aware resume/reconciliation, changed-only content/review refresh and Note-draft preservation. npm run check and 1280px/390px confirmation/outcomes/actual-API-disposable-SQL browser suites pass; screenshots checked. Cases include partial/skipped/failed cleanup, rollback, duplicate submit, stale/unknown/lost responses, refresh/worker restart, resume conflicts, successful reading after layout recovery and read-only retry without another LLM call. Legacy draft/document-switch regressions pass. Controlled provider/candidate and UI-only note-read fixture are not live-book or saved-note storage acceptance; CLEAN-10 remains open. No production mutation/deployment or backend progress edit.
  Timestamp: 2026-10-23

- [x] CLEAN-08: Present cleanup audit and reversible batch changes in the existing review panel.
  Depends on: CLEAN-07 and backend CLEAN-04/06.
  Acceptance: Server-owned counts, safe reasons, original text/page reference where available, collapsible details and single-level Undo. Failed/conflicting Undo preserves current content and drafts; later manual edits are not overwritten. Unresolved OCR regions remain visible with Original/Ignore flow; no new block insertion, independent UI truth or heavy task-layout payload.
  Review resolution: Remove server-confirmed fully resolved entries after final commit without redundant Ignore; warning state remains server-owned. Preserve evidence and reload restored states after Undo. Routine visibility/count policy is superseded by delivered SRMAP-04/05/06: hide only backend-proven never-in-body evidence, not all unknown mappings or manually deleted linked blocks. Historical evidence below does not validate the new policy.
  Evidence required: 1280px/390px browser checks for long/empty audit, collapse/expand, resolved/remaining queues, Undo/reload/conflict, source preservation and all-level interaction refresh. Backend CLEAN-10 owns final real-book acceptance once, not duplicate completion here.
  Evidence: CleanupAudit.tsx and existing ReadingContentActions panel consume an implemented opt-in, ten-batch reading-content projection; original quote/page/reason, remaining/undone status, pagination and latest-action Undo. Pure API/file tests pass 29 cases (five new plus 24 imported regressions); broader reading/overlay/API/preparation suites pass 237 cases, with overlapping fixtures. Actual App/FastAPI/disposable-SQL browser suite passes 1280px/390px for empty/long/collapsed history, resolved/textless/unmapped queue retention, lost-ack idempotent Undo, conflict, reload, restored content and read-only recovery. Mounted Note draft/declined discard use an explicit note-read fixture; backend tests independently prove saved-note/source preservation and generated-output invalidation. CLEAN-07 browser regression and npm run check pass; screenshots inspected. Existing valid layout stamps now advance inside explicit overlay mutations, fixing Undo then page-refresh 409 without hidden read writes. Extra real-PDF OCR review regression could not start locally (OCR/renderer unavailable); live corpus/preview acceptance remains CLEAN-10, not claimed here. No production mutation, Docker restart or backend progress edit.
  Timestamp: 2026-10-25


## OCR Review And Content Correction Queue Reference

Delivered UI work is tracked once in backend doc_loaders/module-checklist.md:
OCR-REVIEW-04 (review warnings/source preview), OCR-REVIEW-05 (block correction,
confirmed deletion/undo), OCR-REVIEW-06 (interaction guards and browser acceptance).
Implemented with Edit/Delete as block interactive actions, no manual add/insert.
Single-level undo and original-region review are verified in the production-editor
API browser harness at 1280/390px. This shared queue is not counted twice here.
See this module's detailed design for the frontend boundary and acceptance criteria.
Timestamp: 2026-10-22

## Format-Neutral Extraction Backlog (2026-10-21)

Implemented; evidence and scope are recorded per item. Ordered design: [shared plan](../Deep_Reflective_Reader/document_preparation/module-detailed-design.md#format-neutral-extraction-plan-2026-10-21).

- [x] EXTRACT-09: Present format-appropriate enhancement readiness and recovery.
  Depends on: EXTRACT-08.
  Acceptance: Use implemented backend responses; no OCR advice for TXT or verbose identity/model diagnostics; desktop/mobile ready, legacy, capacity, retry and automatic layout/open tests.
  Evidence: Reason-specific messages; TypeScript/build and 1280px/390px confirmation/recovery/no-token-details browser tests pass.
  Timestamp: 2026-10-21

## Summary And Note Delivery

Resumed standalone-action backlog, not TOC/enhancement work. Backend owns SN-01..16
in the section_tasks delivery index. UI 4/4; combined 20/20 (100%) accepted.
Legacy summary displays and the checked insight/ancestor-summary work do not close
these tasks. Enable each action only after its actual backend contract is available.

- [x] SN-17: Add typed summary/note adapters and hierarchy-aware action entry points.
  Depends on: SN-08/09; bind each action only after SN-12 or SN-13 respectively.
  Acceptance: Use the existing document/chapter/section/unit/block target identity paths and menu conventions; distinct Summary and Note actions without aliasing analysis. Read-first activation (UI-ACTIVATE-01), request cancellation, document/target switching, missing/stale/error states and late-response isolation. No heavy payloads in task-layout or fictional endpoints; note can be delivered while summary is still unavailable.
  Evidence: `SummaryNoteService.ts`, `SummaryNoteMenuItems.tsx`, App identity validation and keyed/abortable drawer. Existing five target paths and menus are reused, including an explicit section menu on merged chapter/section rows. Historical actual API browser checks passed at 1280px/390px with no generation/save on opening; missing Summary activation is superseded by UI-ACTIVATE-01.
  Timestamp: 2026-10-27
- [x] SN-18: Implement read-first Summary presentation and explicit generate/refresh.
  Depends on: SN-05/10/12/17.
  Acceptance: Dedicated scrollable drawer consistent with existing interactions; explicit Summary selection generates once after a valid missing read (UI-ACTIVATE-01). Show persisted summary, explicit retry/Refresh and pending/failure/insufficient/stale outcomes; preserve last usable content on failed refresh. All five levels, long text/mobile, document language and separate insight semantics. Reload reads persisted output rather than generating again.
  Evidence: `SummaryNoteDrawer.tsx`, actual persisted two-book responses at all five levels, failed-refresh fault injection preserves headline/body, desktop/mobile screenshots reviewed. Full-height scrollable drawer avoids reader-pane overflow; build passes.
  Timestamp: 2026-10-27
- [x] SN-19: Implement bounded Note editing with explicit save/delete and draft guards.
  Depends on: SN-06/07/09/11/13/14/15/17.
  Acceptance: Reflect the agreed single/multiple-note contract, display matching Unicode length counter/limit and retain over-limit drafts without truncation. Explicit save and delete confirmation; dirty close/target-switch protection, failed-save recovery, stale/conflict feedback and pending duplicate-submit guards. No LLM or implicit autosave; backend remains note truth. Handle confirmed structural cleanup and unchanged/title-only preservation correctly.
  Evidence: single-note editor, code-point counter, stable mutation IDs/revisions, beforeunload and close/navigation guard, explicit delete confirmation. Browser acceptance verifies failed-save draft retention, 10001 emoji rejection, cancel-delete with no request, confirmed delete, recreation and persisted reopen. Structural reset clears selected state; title-only refresh does not.
  Timestamp: 2026-10-21
- [x] SN-20: Verify summary/note end-to-end using backed-up real books.
  Depends on: SN-16/18/19.
  Acceptance: Desktop/mobile browser tests against actual APIs/storage, all target levels and both actions; isolate destructive fault/lifecycle cases except the user's explicitly authorized two-book deletion/rebuild. Verify zero-cost reads/notes, repeated summary reuse, reload, length boundaries, conflicts, failed refresh/save, dirty drafts, stale responses and lifecycle outcomes. Human-readable summary evidence comes from authorized live generation, not mocks alone.
  Evidence: `scripts/test-summary-note-live.mjs` passes both books x 1280px/390px x all five targets, real API note writes/readback and saved real-model summaries, injected refresh/save failures and deletion guards. No browser runtime errors; screenshots inspected. Backend separately passes real rename/reparse/conflict/lifecycle tests and ten actual-provider workflows. `npm run check` passes (existing bundle-size warning only). Saved private evidence: data/test-books/summary-note-20261021/browser and accept reports. Existing insight/quiz/thinking menu actions remain present.
  Timestamp: 2026-10-21

## Enhancement Reading Recovery

- [x] ENH-UX-01: Automatically prepare reading layout after confirmed enhancement commit, then reload.
  Acceptance: One consent covers enhance -> layout-only preparation -> reload; no expected 409 or extra preparation click. Preparation failure retries only reading recovery, never LLM generation. Preserve document-switch and uncertain-outcome guards.
  Evidence: `src/features/structure-enhancement/view.tsx` sequences layout-only preparation then reload, records successful preparation, and prevents overlapping document recovery. `scripts/test-enhancement-confirmation.mjs` passed at 1280px/390px for preparation/read failures, read-only retry after preparation, duplicate clicks and document switches during generation/preparation. `scripts/test-enhancement-live.mjs` through backend `accept_enhancement_browser.py` passed real API/disposable PostgreSQL at both widths with zero task-layout 409 and no manual Prepare. `npm run check` and screenshot inspection passed on 2026-10-21.
  Scope: Existing committed structural-success path only; typed three-way results and transient LLM-stage retry are still ENH-REV-03/05. No paid model call, production-document mutation or Docker restart.
  Timestamp: 2026-10-21

- [x] ENH-UX-02: Replace technical model-capacity panel with simple execution readiness.
  Acceptance: Remove model name and all token/input/output metrics from visible and accessible DOM, including the existing Model capacity panel. Show only readiness or plain-language actionable refusal; no hidden tooltip/advanced diagnostic panel. Retain server-owned capacity/consent guards and distinguish configuration/network/local-limit/too-long states.
  Evidence: `src/features/structure-enhancement/view.tsx` removes the capacity panel and shows a role=status readiness alert only with valid consent/full-source fit; `capacity.ts` supplies plain-language rejection messages. `test-enhancement-confirmation.mjs` passed at 1280px/390px with HTML/accessibility absence assertions for model/token fields, blocked zero-generation checks, expiry/legacy/network cases and screenshots. `test-enhancement-live.mjs` through isolated backend acceptance passed both widths with zero layout 409. `npm run check` and `git diff --check` passed on 2026-10-21. Backend diagnostic DTOs/guards unchanged; no paid LLM or production mutation.
  Timestamp: 2026-10-21

- [x] ENH-REV-05: Integrate conditional confirmation and three-way enhancement outcomes.
  Depends on: ENH-UX-01/02 and backend ENH-REV-01..04.
  Acceptance: Conditional full-cleanup warning before generation; no-op reports no change and preserves all state; title-only refreshes labels without clearing saved interactions/drafts/IDs; structural commit clears stale local data and automatically prepares then reads. Do not infer outcomes from HTTP success or version equality. Permit transient LLM-call retries without cost gates; unknown outcomes reconcile first. Downstream recovery must not rerun already-completed generation; normal flow never requires manual Prepare.
  Evidence: `structure-enhancement/outcome.ts`, `view.tsx`, `StructureRepairService.ts`, API DTOs and App/reader label refresh. `test-enhancement-outcomes.mjs` covers typed flags, missing/contradictory responses, unchanged/title-only saved insight and unsent draft retention, label refresh failure, transient retry, authoritative lost-response/concurrent/unknown/not-started recovery, expiry and structural reset at 1280px/390px. Existing confirmation and reader-recovery suites pass both widths; `test-enhancement-live.mjs` with isolated API/PostgreSQL proves structural preparation followed by no-op retention with zero layout 409. `npm run check` and screenshot inspection pass. The supporting backend browser harness now asserts conditional cleanup, not unconditional cleanup on HTTP 200.
  Scope: Closing the thinking drawer retains in-memory drafts so enhancement can preserve them; structural replacement/document reset clears them. No durable draft storage, paid LLM, production-document mutation or Docker restart. ENH-REV-06 integrated acceptance is now complete in the backend-owned checklist; it is not another UI task.
  Timestamp: 2026-10-21

Combined revision track: 8/8 accepted (100%); UI owns 3/3. Order:
ENH-UX-01 -> ENH-UX-02 -> backend ENH-REV-01/02/03/04 -> ENH-REV-05 ->
backend ENH-REV-06. Historical FULL completion below is not completion of this track.

## Full-Source Enhancement Follow-up

- [x] FULL-04: Display server-owned full-source capacity decisions before enhancement.
  Depends on: backend FULL-03; combined execution order FULL-01 -> FULL-02 -> FULL-03 -> FULL-04 -> FULL-05.
  Historical acceptance: Distinguish exceeded/unavailable/local-limit/network states and block rejected generation. The former model/token display requirement is superseded by ENH-UX-02; retain the guards, not the diagnostic panel. Common reading and existing data remain protected without truncation, OCR, model switching or automatic retry.
  Evidence: `src/features/structure-enhancement/capacity.ts`, `view.tsx`, `src/services/StructureRepairService.ts`, `src/types/api.ts`; `npm run check` passed; `scripts/test-enhancement-confirmation.mjs` passed at 1280px/390px with mocked API calls, missing fields/full-source guard, long names, token labels, keyboard cancel, changed policy, zero generation on rejected preflight, typed execution rejection and timeout reconciliation. No paid LLM calls or user-document writes.
  Progress: Combined FULL-01 through FULL-05 track 5/5 (100%); FULL-05 synthetic real API/PostgreSQL browser acceptance passed at 1280px/390px via `scripts/test-enhancement-live.mjs` on 2026-10-20. Backend completion is owned by document_structure, not counted as another UI task; live model quality remains unverified.
  Timestamp: 2026-10-20

## Content Read Performance (2026-10-21)

- [x] CONTENT-04: Bound section content requests and abort obsolete navigation requests.
  Acceptance: See app detailed design Content Read Recovery; evidence required before completion.
  Evidence: TaskUnitContentService uses sequential batches of 20 with AbortSignal; reader controller aborts reset/navigation/unmount and rejects late responses. Build and two-book desktop/mobile real API browser acceptance pass, including cancellation and all five block menu actions.
  Timestamp: 2026-10-21

## Purpose

This checklist tracks the first Deep_Reader_UI vertical slice.

## Completed Checklist

- [x] Synchronize Library, book search, and recent-reading entries after accepted deletion.
  Evidence: `src/features/file-management/controller.ts`; `src/features/book-search/controller.ts`; `src/App.tsx`; `npm run check` passed on 2026-10-14; source review of accepted-only invalidation and request sequence guards.
  Notes: Accepted deletion removes matching local candidates/recent entries and invalidates older requests; rejected deletion retains candidates. Browser click-through validation remains unverified because no browser connection was available.
  Timestamp: 2026-10-14

- [x] Establish Deep_Reader_UI frontend foundation.
  Evidence: `Deep_Reader_UI/package.json`; `Deep_Reader_UI/package-lock.json`; `Deep_Reader_UI/vite.config.ts`; `Deep_Reader_UI/index.html`; `Deep_Reader_UI/src/main.tsx`; `Deep_Reader_UI/src/App.tsx`.
  Notes: Created an independent React + TypeScript + MUI UI client with Vite dev/build tooling and backend proxy.

- [x] Implement document-name based Reader entry flow.
  Evidence: `Deep_Reader_UI/src/features/book-search/`; `Deep_Reader_UI/src/services/DocumentCatalogService.ts`; `Deep_Reader_UI/src/App.tsx`.
  Notes: The MUI combo box searches backend candidates and only enables load for a selected API-returned `doc_name`.

- [x] Consume and render task-layout hierarchy.
  Evidence: `Deep_Reader_UI/src/services/TaskLayoutService.ts`; `Deep_Reader_UI/src/features/hierarchy-navigation/`; API used: `POST /documents/task-layout`.
  Notes: Renders backend chapters and sections without adding heavy content to task-layout; task units are retained internally for content addressing.

- [x] Implement Chapter -> Section navigation with internal Task Unit aggregation.
  Evidence: `Deep_Reader_UI/src/features/hierarchy-navigation/`; `Deep_Reader_UI/src/styles.css`.
  Notes: Navigation follows backend hierarchy order and makes sections selectable; task units are no longer displayed as primary navigation items.

- [x] Fetch selected section content on demand with segmented=true task-unit requests.
  Evidence: `Deep_Reader_UI/src/features/reader-content/controller.ts`; `Deep_Reader_UI/src/services/TaskUnitContentService.ts`; API used: `GET /documents/{doc_name}/task-units/{task_unit_id}/content?segmented=true`.
  Notes: Content is fetched only after a section is selected; the UI requests the selected section's task units in backend order.

- [x] Render section content from aggregated content_blocks.
  Evidence: `Deep_Reader_UI/src/features/reader-content/view.tsx`; `Deep_Reader_UI/src/features/reader-content/model.ts`; `Deep_Reader_UI/src/styles.css`.
  Notes: Renders flattened `content_blocks[]` from the selected section's task units in backend response order and does not require duplicated raw content.

- [x] Add initial/loading/error/empty states.
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/shared/components/StateView.tsx`; `Deep_Reader_UI/src/styles.css`.
  Notes: Separate state handling exists for layout and content requests.

- [x] Add basic responsive and accessible interaction behavior.
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/features/book-search/view.tsx`; `Deep_Reader_UI/src/features/hierarchy-navigation/view.tsx`; `Deep_Reader_UI/src/styles.css`.
  Notes: Uses MUI semantic controls, `aria-live`, `aria-busy`, `aria-selected`, focusable section controls, and responsive layout.

- [x] Validate first Reader vertical slice.
  Evidence: `npm run check`; Vite dev server HTTP smoke validation.
  Notes: TypeScript build and Vite production build pass. Dev server smoke validates the UI entry path. Backend API behavior remains an external runtime dependency.

- [x] Refactor Reader UI to modular React, TypeScript, and MUI.
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/services/`; `Deep_Reader_UI/src/types/api.ts`; `Deep_Reader_UI/src/features/book-search/`; `Deep_Reader_UI/src/features/hierarchy-navigation/`; `Deep_Reader_UI/src/features/reader-content/`; `Deep_Reader_UI/src/shared/components/StateView.tsx`; `Deep_Reader_UI/src/theme.ts`.
  Notes: UI is split by API, type, reader state, document search, hierarchy navigation, content rendering, shared state view, and theme responsibilities.

- [x] Consume backend document list/search API for combo-box candidates.
  Evidence: `Deep_Reader_UI/src/services/DocumentCatalogService.ts`; `Deep_Reader_UI/src/types/api.ts`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/features/book-search/`.
  Notes: Opening the combo box calls `GET /documents?limit=200`; local typing filters those API-returned options. Load is disabled unless the current value is one of the API-returned options.

- [x] Aggregate section reading content from internal task units.
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/features/hierarchy-navigation/`; `Deep_Reader_UI/src/features/reader-content/`; validation: `npm run check`.
  Notes: The UI displays Chapter -> Section navigation, keeps task unit ids inside the selected section, fetches each selected section task unit with `segmented=true`, ignores stale content responses after rapid section changes, and renders the aggregated `content_blocks[]` as one section.

- [x] Add role-oriented DOM class names for Reader UI elements.
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/features/book-search/view.tsx`; `Deep_Reader_UI/src/features/hierarchy-navigation/view.tsx`; `Deep_Reader_UI/src/features/reader-content/view.tsx`; `Deep_Reader_UI/src/shared/components/StateView.tsx`; validation: `npm run check`; `git diff --check -- Deep_Reader_UI`.
  Notes: Adds stable class names for app shell, topbar, document search, structure repair controls, hierarchy navigation, section selectors, reader content, content blocks, and state views without changing backend API contracts, task-layout behavior, or visual styling.

- [x] Refactor Reader UI into feature folders with MVC-style internal structure.
  Evidence: `Deep_Reader_UI/src/features/book-search/`; `Deep_Reader_UI/src/features/hierarchy-navigation/`; `Deep_Reader_UI/src/features/reader-content/`; `Deep_Reader_UI/src/App.tsx`; validation: `npm run check`.
  Notes: Splits the top book search, lower-left hierarchy navigation, and lower-right reader content responsibilities into separate feature folders with `model.ts`, `controller.ts`, `view.tsx`, and `index.ts`. `App.tsx` remains the shell/composition layer.

- [x] Replace scattered frontend REST functions with endpoint-oriented service classes.
  Evidence: `Deep_Reader_UI/src/services/RestClient.ts`; `Deep_Reader_UI/src/services/DocumentCatalogService.ts`; `Deep_Reader_UI/src/services/TaskLayoutService.ts`; `Deep_Reader_UI/src/services/TaskUnitContentService.ts`; `Deep_Reader_UI/src/services/StructureRepairService.ts`; `Deep_Reader_UI/src/services/DocumentPreparationService.ts`; validation: `npm run check`.
  Notes: REST request construction and error handling are centralized behind service classes; the legacy `src/api/client.ts` function module is removed. No manual-structure UI endpoint is introduced.

## Needs Confirmation

None.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items.

No coding task should be considered complete unless the corresponding checklist item is updated.

## Future Checklist

### Source-Preserving Enhancement (2026-10-19)

New track: UI 1/1; combined backend/UI 6/6 (100%; backend 5/5; isolated engineering acceptance). Execution index is in
`../Deep_Reflective_Reader/document_preparation/module-detailed-design.md`.
Historical completed recovery tasks below do not complete this new requirement.

- [x] ENH-UI-01: Confirm discarding existing interaction data before LLM enhance.
  Depends on: ENH-APP-01, ENH-PREP-02 and ENH-DB-01; ENH-PREP-03 activates only supported storage with migration/profile guards. UI always honors execution_available.
  Scope: read-only preflight, backend impact inventory including book-level data and any persisted notes, cancel/default focus, explicit discard-and-enhance action, snapshot-bound consent and pending/stale/unknown-outcome states.
  Acceptance: cancellation makes no generation/deletion call; invalid source offers explicit Prepare; no keep-and-continue or automatic OCR; failure preserves data; only confirmed commit clears caches/drafts; reload failure never repeats enhancement. Verify desktop/mobile, keyboard, duplicate clicks, document switch, concurrent interactions and timeout reconciliation.
  Evidence: `features/structure-enhancement/view.tsx`, App integration, typed preflight/consent service; `scripts/test-enhancement-confirmation.mjs` and updated reader recovery tests pass at 1280px/390px with mocked APIs. Includes expired/stale consent, duplicate clicks, source/server gates, definite 502 failure, network loss, actual client timeout, late document response and reload-only recovery. Screenshots inspected; npm run check passes.
  Integration evidence: ENH-PREP-03 browser/FastAPI/disposable-PostgreSQL harness passes desktop/mobile, including explicit Prepare reading layout after committed replacement returns read-only 409. This action sets layout_only=true; no implicit source preparation or repeat enhancement. Mocked confirmation suite also covers this recovery.
  Boundary: server execution_available remains authoritative; deployment of the user's Docker instance and paid-model quality are not claimed. Unknown operations remain blocked when unchanged version cannot prove completion. Mounted-session tracking is not durable cross-tab/job-status support.
  Timestamp: 2026-10-19

### Fragmented PDF Structure / Interaction Recovery (2026-10-17)

New track: UI 1/1; combined 10/10 (100%). NATIVE-01..06, INTERACT-01..03 and READER-01 are complete under the isolated engineering acceptance scope; live LLM quality and deployment are not claimed. Backend ownership, investigation and execution
order are in `Deep_Reflective_Reader/document_structure/module-detailed-design.md`.
The completed Library/PDF-mode track below remains a separate historical scope.

- [x] READER-01: Present genuine structure, LLM fallback and interaction recovery honestly.
  Acceptance: Backend-supported implicit/no-TOC state does not produce fabricated headings; valid backend IDs preserve reading/actions/editing. Requested LLM versus effective common fallback is visible. Distinguish reasoned insufficient-content/stale/generation failures, offer explicit retry/refresh, invalidate stale local IDs after successful reparse, preserve state after failure, and never generate on passive menu opening or navigation; explicit interaction selection follows UI-ACTIVATE-01. Desktop/mobile regressions cover flat articles and real TOCs.
  Depends on: NATIVE-05 and INTERACT-03; new backend provenance must be implemented and contract-tested before UI consumes it. No UI inference becomes backend truth.
  Additional scope (corrected 2026-10-19): Pre-prepare Auto/Present/Absent TOC declaration is independent of PDF extraction mode and snapshotted per request. Offer one LLM enhance action, not separate restoration/outline actions. Show generated origins from accepted chapter/section provenance, not the legacy request intent. Common single-body success permits Open without enhancement; failed enhancement preserves usable content/readiness. Cover auto-upload preparation, reprepare and explicit reparse without draft-triggered mutations.
  Evidence: `scripts/test-reader-recovery.mjs`, `test-library-pdf-modes.mjs` and all seven Library/reader browser suites pass (mocked APIs, desktop/mobile); `npm run check` passes. Backend `scripts/test_reader_provenance.py` passes four contract/read-only tests; task-layout readiness passes eleven. Accepted replacement clears old targets/content; rejected enhancement preserves reading. Non-2xx envelopes retain reason/status; stale thinking requires explicit new-question generation and failed requests preserve drafts. No deployment or live provider/private-document acceptance is claimed; NATIVE-06 owns that remaining scope.
  Correction evidence: `test-reader-recovery.mjs` passes at 1280px/390px with exactly one LLM enhance item and no LLM intent items; existing default wire value can display backend-generated titles. Build/typecheck passes; failure preservation and accepted ID replacement remain covered.
  Timestamp: 2026-10-27

### Library Entry And Preparation Follow-up (2026-10-15)

Implementation: 6/6 UI items complete (100%). See detailed design's dated Library
entry section and PDF mode revision. Combined follow-up is 17/17 complete (100%), 0 pending
tasks: UI 6, loader 5, preparation 2, main 1, app 3. Old PDF-FIX-01 is superseded
by PDF-AUTO-01, not completed or counted twice. OCR-01..08 stays historical.
PDF-AUTO-01 and PDF-FIX-03 are complete in the owning loader checklist; the final
three-mode matrix passed 21/21 cases. This is acceptance, not production deployment.
Backend LAYOUT-01..03 evidence is in `Deep_Reflective_Reader/app/module-checklist.md`;
LIB-05 evidence below records the latest UI completion. Updated: 2026-10-17.

#### Execution Order (2026-10-15)

Execute one owning checklist item per turn. This order supersedes the earlier
LIB numeric sequence; references below are not additional tasks or completion claims.

| Order | Task | Reason |
|---|---|---|
| 1 | LIB-01 | Stop reader identity loss across entry points |
| 2 | LIB-02 | Guard content/interaction requests and recovery |
| 3 | PDF-MODE-01 (main) | Freeze validated API contract before consumers |
| 4 | PDF-MODE-02 (loader) | Default native-only fast extraction |
| 5 | PDF-FIX-02 (loader) | Correct English/multilingual quality rejection |
| 6 | PDF-MODE-03 (loader) | Explicit all-page OCR with corrected quality |
| 7 | PDF-MODE-04 (preparation) | Mode-aware provenance/reuse before integration |
| 8 | PDF-MODE-05 (preparation) | End-to-end propagation and optional-path guards |
| 9 | LAYOUT-01 (app) | Measure remaining non-OCR latency |
| 10 | LAYOUT-02 (app) | Explicit preparation versus read-only readiness |
| 11 | LAYOUT-03 (app) | Remove ordinary-read evidence reloads |
| 12 | LIB-06 | Mode selection against a verified backend contract |
| 13 | LIB-03 | Visible upload-to-prepare workflow and client cancellation |
| 14 | LIB-04 | Shared readiness gates for all Open controls |
| 15 | LIB-05 | Full browser/race workflow regression |
| 16 | PDF-AUTO-01 (loader) | Resolve observed Auto routing dependency after fast/OCR acceptance |
| 17 | PDF-FIX-03 (loader) | Final three-mode integration acceptance rerun |

PDF-MODE-01 establishes the boundary first; until PDF-MODE-05 is delivered,
unsupported mode execution must fail explicitly, never be silently ignored or
advertised as operational. LIB-06 can establish mode draft/retry handling first;
LIB-03 subsequently connects its snapshot to automatic post-upload preparation.
Those integration references are not circular prerequisites for individual slices.
The original backend checklists remain status owners; this is the cross-module
execution index. Backend evidence prompted the final two items' reorder; no UI
runtime/API behavior is changed by this index update. Timestamp: 2026-10-16

- [x] LIB-01: Separate active reader identity from document search input.
  Acceptance: Library/recent/search/route entry share committed backend doc_name plus navigation generation; Autocomplete reset/blur, absent options and the 200-item limit cannot clear it. Late layouts cannot bind to a different document. Regress Library-first open with empty options and a newly uploaded name absent from an already loaded list.
  Evidence: `src/features/book-search/view.tsx`, `controller.ts`; `src/App.tsx`; `scripts/test-library-identity.mjs`; npm run check; mocked-API Chromium regression passed for empty/stale options, Library opening, typing/clear/select, delayed A/B layout responses, reload and browser Back. Content requests retained both document name and matching body task-unit IDs; no page errors or unexpected API calls.
  Scope: Reader identity is separate from searchInput; Library/recent/search/routes share generation-guarded layout loading and clear stale content. Selected identity remains valid outside the capped search list. Open reads no longer automatically fall back to prepare on errors. Full content-error handling, readiness, auto-prepare, PDF modes and broader workflow regression remain LIB-02..06, not completed by this task.
  Timestamp: 2026-10-15

- [x] LIB-02: Guard content/interaction identity and recover from failed content reads.
  Acceptance: No fetch with empty doc_name or invalid task-unit IDs; content uses the active layout identity, not search text. Catch and expose 404/network errors, support retry/section or document switching, invalidate stale responses, and keep interactions usable for valid targets. Preserve batch IDs in request body; do not invent URL IDs or backend hierarchy.
  Evidence: `src/shared/readerIdentity.ts`; `src/features/reader-content/controller.ts`, `view.tsx`; `src/services/TaskUnitContentService.ts`, `ReadingInteractionService.ts`; `src/App.tsx`; `scripts/test-reader-guards.mjs` (42 identity-rejection cases without transport; mocked Chromium 404/network/wrong-unit/wrong-document recovery, retry, stale section/document responses and successful insight read after errors); `npm run check`.
  Scope: Bind content to accepted layout's requested doc_name; reject stale layout membership, blank/duplicate IDs and mismatched content responses. Validate interaction target against active hierarchy and loaded blocks before calls, with identity guards at service boundaries. Retry is explicit and read-only. No prepare fallback, invented IDs, backend mutation, PDF runtime change or full LIB-05 acceptance.
  Depends on: LIB-01.
  Timestamp: 2026-10-15

- [x] LIB-03: Automatically start visible preparation after accepted upload, with frontend cancellation.
  Acceptance: Exactly one explicit non-force preparation workflow for the backend-returned unique name; queued/preparing/ready/failed/cancelled states, operation token and request abort support. Cover full reader readiness via explicit prepare-task-layout, not implicit Open side effects. Cancel before submission prevents work; after submission only cancels waiting/late UI updates, with honest server-continuation semantics and readiness reconciliation before retry. No auto-open on late completion or duplicate upload conflict.
  Depends on: LIB-01; coordinate explicit read/write separation with LAYOUT-02 and mode snapshot/forwarding with LIB-06. Base structured readiness alone does not certify task-unit layout readiness. New uploads default to fast; never silently retry with auto/ocr.
  Evidence: file-management controller/model/view, `App.tsx`, `DocumentPreparationService` and `TaskLayoutService`; `scripts/test-library-auto-prepare.mjs` passed at 1280px/390px with mocked APIs and controlled timers: pre-dispatch cancellation, aborted fetch, unique-name/mode preservation, duplicate submission guard, read-only reconciliation, replacement upload/late response isolation and close/reopen. Updated PDF mode and Library identity regressions plus reader guards passed; `npm run check` passed. Pending/ready screenshots inspected.
  Scope: Accepted upload automatically queues one non-force prepare-task-layout; no prepare on rejected/conflicting upload. Post-upload Open requires validated success. Cancel invalidates the operation token and aborts waiting, not backend work. Cancelled dispatched work and lost connections require read-only readiness reconciliation; absent readiness never blindly starts another prepare. In-memory uncertainty also protects the existing-PDF prepare entry during this mounted session. No server job status/cancellation or restart persistence is claimed. Full row/recent Open gates remain LIB-04; live OCR acceptance remains PDF-FIX-03.
  Timestamp: 2026-10-16

- [x] LIB-04: Consume management readiness and gate every Library Open entry point.
  Acceptance: Adapt existing GET /documents/management status fields; use common identity/readiness guards for upload result, rows and recent shortcuts. Preparing/failed/cancelled/unknown cannot bypass Open checks. Validate layout readiness read-only for existing structured-ready rows; missing/stale layout returns to explicit preparation. Refresh readiness after success without making local state backend truth or generating tasks during catalog reads.
  Depends on: LIB-01, LIB-03 and LAYOUT-02's explicit readiness behavior.
  Evidence: file-management controller/model/view, `App.tsx`, responsive Library styles; `scripts/test-library-readiness.mjs` passed at 1280px/390px: management status gates, recent shortcuts, cancelled readiness response, missing/malformed layout, explicit TXT preparation and cancelled-upload row bypass prevention. Existing identity, PDF modes, auto-prepare and reader-guard suites passed; `npm run check` and `git diff --check` passed. Desktop/mobile screenshots inspected.
  Scope: Rows/recent/upload-result/prepare-dialog Open share eligibility and fresh read-only layout validation. Unknown/stale/failed management status is not reader readiness; local failed/cancelled work remains blocked across upload-result reset until explicit backend reconciliation/preparation. Opening never prepares. Missing layout retains Library and offers explicit preparation where raw source is available, including TXT. Modal close, refresh and conflicting operations invalidate pending Open; App navigation generation prevents late navigation. No backend API changes, live OCR acceptance or full LIB-05 race-matrix completion claimed.
  Timestamp: 2026-10-16

- [x] LIB-05: Add Library-to-reader browser and race regressions.
  Acceptance: Cold Library-first entry, stale/empty search options, prepared row, upload auto-prepare, duplicate-confirmed unique name, cancellation before/after dispatch, failed prepare, double click, close/reopen, rapid A/B navigation and delete/rename during a pending request. Assert URLs AND batch request bodies, no empty/cross-document fetches, no unhandled rejections, recoverable 404, working insight/quiz/thinking menus, and Open disabled until verified readiness. Include default-fast, explicit ocr/auto, non-PDF control omission, no-text retry, mode snapshot across upload/prepare and failed-reprepare preservation. Use mocked network for race coverage and disposable live fixtures only.
  Depends on: LIB-01..04 and LIB-06; record browser evidence separately from compile/source checks.
  Evidence: `npm run test:library` passed all six browser scripts via `scripts/test-library-suite.mjs`. New `test-library-workflow.mjs` passed at 1280px/390px: confirmed duplicate upload with unique Unicode name, auto-prepare, disabled Open/lifecycle actions while preparing, reader URL and batch identity, 404/retry, insight/quiz/thinking read menus, and confirmed rename/delete during pending readiness without late navigation. Existing identity, PDF modes, auto-prepare, readiness and reader guards passed; auto-prepare now asserts unchanged pathname after cancelled completion. `npm run check` and `git diff --check` passed. Workflow screenshots inspected at both sizes.
  Scope: Test runner stops on failure and bounds each suite to 180 seconds. Every API is intercepted; no real user uploads/deletes, OCR or LLM calls. No product runtime changes in this task. The detailed design records per-suite coverage; real PDF three-mode acceptance remains backend PDF-FIX-03, followed by deferred PDF-AUTO-01.
  Timestamp: 2026-10-16

- [x] LIB-06: Add PDF fast/OCR/auto selection before upload and explicit reprepare.
  Acceptance: PDF-only single-choice control defaults to fast, with ocr and auto alternatives; snapshot selection before upload and send it with the one visible preparation request for the returned unique doc_name. Existing PDF reprepare permits explicit mode choice; opening/reading never changes it. Lock pending selection or use a separate next-operation draft; cancellation/late responses cannot restore stale mode/readiness. No-text fast failure offers explicit ocr/auto retry without escalation. Keep old valid artifacts readable on failed rebuild; reconcile readiness with backend. No invented server progress/cancellation API.
  Depends on: PDF-MODE-01/04/05 backend contract and LIB-01; integrate with LIB-03/04. The selector is not proof the three modes already exist in the API.
  Evidence: `PdfModeControl.tsx`; file-management model/controller/view; `DocumentManagementService`, `DocumentPreparationService`, `TaskLayoutService`; `App.tsx`; `scripts/test-library-pdf-modes.mjs` passed at 1280px and 390px with intercepted APIs, including duplicate-name identity, fast no-text explicit retry, pending mode lock/close/reopen, TXT omission and failed reprepare read-only reconciliation. Existing Library identity and reader guard regressions passed; `npm run check` passed. Desktop/mobile screenshots inspected.
  Scope: Upload remains raw-only and preparation remains explicit until LIB-03. Management source fields identify existing PDFs; complete readiness/Open gates remain LIB-04. Pending operations are locked, not cancellable yet. No live OCR/LLM integration claimed by browser mocks; broader acceptance remains LIB-05/PDF-FIX-03.
  Timestamp: 2026-10-16

### Earlier Planned And Delivered Work

- [x] Plan file-management parent feature route and surface.
  Evidence: `Deep_Reader_UI/module-detailed-design.md` future direction section defines file management as the parent document library surface, keeps upload as an embedded child workflow, preserves the current document combo box as a compact reader entry, and records route/service boundary expectations for future implementation.
  Notes: This completes design governance only; no UI route, component, service, backend route, upload behavior, catalog status API, hidden prepare/reparse, or runtime mutation is implemented by this task.
  Timestamp: 2026-10-13

- [x] Define frontend document-management item model and service boundary.
  Evidence: `Deep_Reader_UI/src/types/api.ts`; `Deep_Reader_UI/src/services/DocumentManagementService.ts`; `Deep_Reader_UI/src/services/index.ts`; `Deep_Reader_UI/module-detailed-design.md`; validation: `npm run check`.
  Notes: Adds a frontend `DocumentManagementItem` / `DocumentManagementListResult` model and a narrow read-only `DocumentManagementService` that maps current `GET /documents` discovery results into backend-owned `docName`, title, and source fields only. It does not invent source hash, preparation status, structure version, upload metadata, lifecycle state, or call upload/delete/archive/prepare/reparse/task-layout routes.
  Timestamp: 2026-10-13

- [x] Implement document library loading, empty, error, and row-action UX.
  Evidence: `Deep_Reader_UI/src/features/file-management/`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/styles.css`; `Deep_Reader_UI/module-detailed-design.md`; validation: `npm run check`.
  Notes: Adds a read-only file-management dialog opened from the topbar Library action. The dialog loads document candidates through `DocumentManagementService`, renders loading/empty/error/list states, supports Refresh, and provides per-row Open actions. Open uses the read-centric task-layout route only and does not call upload, delete, archive, prepare, reparse, OCR, profile, retrieval, interaction generation, or hidden task-layout mutation.
  Timestamp: 2026-10-13

- [x] Plan upload as an embedded file-management subflow.
  Evidence: `Deep_Reader_UI/src/features/file-management/model.ts`; `Deep_Reader_UI/src/features/file-management/controller.ts`; `Deep_Reader_UI/src/features/file-management/view.tsx`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/styles.css`; `Deep_Reader_UI/module-detailed-design.md`; validation: `npm run check`.
  Notes: Upload controls now live inside the file-management dialog and expose frontend-local idle, validating, selected, failed, cancelled, plus reserved uploading/uploaded states. File selection performs advisory `.txt` / `.pdf` extension validation only. The current Upload action reports unavailable service state and does not store files, call backend upload/delete/archive routes, refresh catalog as success, prepare, reparse, OCR, profile, retrieval, task-layout, or interaction generation.
  Timestamp: 2026-10-13

- [x] Add upload service and typed error normalization after backend upload route exists.
  Evidence: `Deep_Reader_UI/src/services/DocumentUploadService.ts`; `Deep_Reader_UI/src/services/RestClient.ts`; `Deep_Reader_UI/src/types/api.ts`; `Deep_Reader_UI/src/features/file-management/model.ts`; `Deep_Reader_UI/src/features/file-management/controller.ts`; `Deep_Reader_UI/src/features/file-management/view.tsx`; `Deep_Reader_UI/src/App.tsx`; validation: `npm run check`.
  Notes: `DocumentUploadService` calls the backend multipart `/documents/upload` route with `conflict_policy=reject`, preserves typed `DocumentUploadResponse` payloads for non-2xx upload failures, and maps missing source, unsupported type, file-too-large, duplicate conflict, unsafe filename, validation failure, and storage failure into display-only UI errors. Frontend extension checks remain advisory; backend validation remains authority. Upload success refreshes the document catalog but does not auto-prepare, auto-open, OCR, profile, retrieval, task-layout, or interaction-generation work.
  Timestamp: 2026-10-13

- [x] Add duplicate upload confirmation before unique-name upload.
  Evidence: `Deep_Reader_UI/src/services/DocumentUploadService.ts`; `Deep_Reader_UI/src/features/file-management/model.ts`; `Deep_Reader_UI/src/features/file-management/controller.ts`; `Deep_Reader_UI/src/features/file-management/view.tsx`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reflective_Reader/scripts/test_document_upload_route.py`; validation: `npm run check`; `PYTHONPATH=Deep_Reflective_Reader Deep_Reflective_Reader/.venv/bin/python Deep_Reflective_Reader/scripts/test_document_upload_route.py`.
  Notes: Upload now first submits same-name files with `conflict_policy=reject`; backend `duplicate_conflict` opens a user confirmation dialog. Only explicit confirmation retries with `conflict_policy=create_distinct`, after which the UI displays the backend-owned unique `docName` / source filename and refreshes the catalog. The view defensively handles absent duplicate-conflict props during hot reload or stale render state. The UI does not silently replace files, invent unique names locally, auto-prepare, call task-layout, or generate interactions.
  Timestamp: 2026-10-14

- [x] Add explicit post-upload prepare/open/retry actions.
  Evidence: `Deep_Reader_UI/src/features/file-management/model.ts`; `Deep_Reader_UI/src/features/file-management/controller.ts`; `Deep_Reader_UI/src/features/file-management/view.tsx`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Uploaded documents now stay in the file-management dialog with explicit next actions for Prepare, Open, and Upload another / Retry upload. Upload success refreshes the catalog but does not automatically enter prepare-task-layout or open the reader; Prepare calls `/documents/prepare` only after user action, Open uses the existing read-centric managed-document open path, and Retry upload remains a separate user action with visible status feedback.
  Timestamp: 2026-10-13

- [x] Add delete document UI flow with required user confirmation.
  Evidence: `Deep_Reader_UI/src/services/DocumentLifecycleService.ts`; `Deep_Reader_UI/src/types/api.ts`; `Deep_Reader_UI/src/features/file-management/controller.ts`; `Deep_Reader_UI/src/features/file-management/view.tsx`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: File-management rows now expose an explicit Delete action that opens a confirmation dialog before calling `POST /documents/lifecycle/delete` with `action=delete` and `confirm=true`. Accepted deletion refreshes the document catalog; backend rejected/conflict/failure responses are surfaced as display messages. Delete is not triggered by row selection, Open, upload controls, task-layout, prepare, local-only state, or hidden mutation.
  Timestamp: 2026-10-13

- [x] Add rename document UI flow after backend rename route exists.
  Evidence: `Deep_Reader_UI/src/services/DocumentLifecycleService.ts`; `Deep_Reader_UI/src/types/api.ts`; `Deep_Reader_UI/src/features/file-management/controller.ts`; `Deep_Reader_UI/src/features/file-management/view.tsx`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: File-management rows now expose an explicit Rename action that opens a dialog, validates a non-empty changed `new_doc_name`, submits `POST /documents/lifecycle/rename`, displays backend validation/conflict/failure reasons, refreshes the catalog after accepted rename, and does not simulate rename through upload replace/create-distinct behavior or frontend-only identity mutation. Archive/restore remain hidden because they are not part of the current lifecycle MVP.
  Timestamp: 2026-10-13

- [x] Validate file-management/upload UI boundary.
  Evidence: `Deep_Reader_UI/src/features/file-management/controller.ts`; `Deep_Reader_UI/src/features/file-management/view.tsx`; `Deep_Reader_UI/src/features/file-management/model.ts`; `Deep_Reader_UI/src/services/DocumentManagementService.ts`; `Deep_Reader_UI/module-detailed-design.md`; validation: `npm run check`.
  Notes: Source smoke confirms catalog loading uses `DocumentManagementService` over `GET /documents`, upload file selection remains frontend-local in-memory state, unsupported file handling is advisory display state, and the current Upload action reports unavailable service state without calling upload/delete/archive/prepare/reparse/OCR/profile/retrieval/task-layout/interaction-generation routes. Conflict and successful upload states remain covered by the later backend upload-route task.
  Timestamp: 2026-10-13

- [x] Replace primary Library `source` column with reader-facing document metadata
  Evidence: `Deep_Reader_UI/src/features/file-management/view.tsx`; `Deep_Reader_UI/src/features/file-management/model.ts`; `Deep_Reader_UI/src/types/api.ts`; validation: `npm run check`.
  Notes: File-management table now uses a `Details` column instead of a primary `Source` column. The display prefers backend-provided profile summary, author/language/profile label, preparation status, or a neutral profile-unavailable fallback without inventing profile fields.
  Timestamp: 2026-10-13

- [x] Add Library advanced filter surface for large document collections
  Evidence: `Deep_Reader_UI/src/features/file-management/model.ts`; `Deep_Reader_UI/src/features/file-management/controller.ts`; `Deep_Reader_UI/src/features/file-management/view.tsx`; `Deep_Reader_UI/src/styles.css`; `Deep_Reader_UI/src/App.tsx`; validation: `npm run check`.
  Notes: The Library dialog now exposes a local Advanced filters surface with query, readiness, language, profile label, source type, and source status filters derived only from backend-provided catalog fields. Filter state remains frontend-local, shows filtered count versus total count, supports reset, and does not mutate backend catalog, task-layout, profile diagnostics, document identity, upload state, or lifecycle state.
  Timestamp: 2026-10-13

- [x] Add recent-reading shortcuts for the last three opened books
  Evidence: `Deep_Reader_UI/src/features/file-management/model.ts`; `Deep_Reader_UI/src/features/file-management/controller.ts`; `Deep_Reader_UI/src/features/file-management/view.tsx`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Library now shows up to three recent-reading shortcuts below the main document list after successful read-centric managed-document opens. Shortcuts reuse the same `onOpenDocument` / task-layout flow as row Open, remain frontend-local in-memory state, deduplicate by `docName`, keep newest first, and do not create backend reading history, mutate task-layout, prepare/reparse documents, or treat local history as backend truth.
  Timestamp: 2026-10-13

- [x] Consume selected-section task-unit content through a batch API
  Evidence: `Deep_Reader_UI/src/services/TaskUnitContentService.ts`; `Deep_Reader_UI/src/features/reader-content/controller.ts`; `Deep_Reader_UI/src/types/api.ts`; validation with `npm run check`.
  Notes: `TaskUnitContentService` exposes a batch read method, reader-content selection calls it once per section selection with ordered `task_unit_ids`, aggregation preserves backend order, and the single-task endpoint remains fallback behavior. This avoids many concurrent `GET /task-units/{id}/content?segmented=true` requests while keeping `/documents/task-layout` lightweight and preserving content-block-first rendering.
  Timestamp: 2026-10-05

- [x] Introduce React Router with document-scoped reader and TOC editor routes
  Evidence: `Deep_Reader_UI/package.json`; `Deep_Reader_UI/src/main.tsx`; `Deep_Reader_UI/src/App.tsx`; validation: `npm run check`.
  Notes: Direct cold-start entry into `toc-edit` must be guarded and must not auto-load layout.

- [x] Add hierarchy-navigation TOC edit trigger
  Evidence: `Deep_Reader_UI/src/features/hierarchy-navigation/view.tsx`; `Deep_Reader_UI/src/App.tsx`; validation: `npm run check`.
  Notes: Trigger routes to `/documents/:docName/toc-edit` and keeps hierarchy navigation as left-pane context.

- [x] Add independent `toc-editor` feature folder
  Evidence: `Deep_Reader_UI/src/features/toc-editor/model.ts`; `Deep_Reader_UI/src/features/toc-editor/controller.ts`; `Deep_Reader_UI/src/features/toc-editor/view.tsx`; `Deep_Reader_UI/src/features/toc-editor/index.ts`; validation: `npm run check`.
  Notes: The feature owns manual TOC editing state, validation, source range assignment, and commit orchestration. It must not be folded into `reader-content`.

- [x] Define integrated TOC editor surface
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/features/toc-editor/view.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Anchor range editing is part of the TOC editor itself; validation and hard-reparse notices should use MUI popup/dialog/snackbar interactions rather than permanent main-screen panels.

- [x] Support two-level manual TOC editing with chapter-only special case
  Evidence: `Deep_Reader_UI/src/features/toc-editor/model.ts`; `Deep_Reader_UI/src/features/toc-editor/controller.ts`; `Deep_Reader_UI/src/features/toc-editor/view.tsx`; validation: `npm run check`.
  Notes: Section is optional in the editing experience, but persistence must still respect backend hierarchy contracts.

- [x] Add frontend validation for manual TOC edits
  Evidence: `Deep_Reader_UI/src/features/toc-editor/model.ts`; `Deep_Reader_UI/src/features/toc-editor/controller.ts`; validation: `npm run check`.
  Notes: First implementation should use `char_range`; `page_range` remains future work until page-boundary mapping is explicit.

- [x] Add backend manual-structure validation call from TOC editor
  Evidence: `Deep_Reader_UI/src/services/ManualStructureService.ts`; `Deep_Reader_UI/src/features/toc-editor/controller.ts`; `Deep_Reader_UI/src/features/toc-editor/view.tsx`; validation: `npm run check`.
  Notes: Backend validation is authoritative; frontend validation is an early guard.

- [x] Add explicit hard-reparse commit flow from TOC editor
  Evidence: `Deep_Reader_UI/src/services/ManualStructureService.ts`; `Deep_Reader_UI/src/features/toc-editor/controller.ts`; `Deep_Reader_UI/src/features/toc-editor/view.tsx`; validation: `npm run check`.
  Notes: Commit requires MUI confirmation dialog warning that hard reparse replaces structure and generated QA, summary, quiz, and derived artifacts will not be preserved.

- [x] Split TOC editor draft source into from-scratch and edit-existing modes
  Evidence: `Deep_Reader_UI/src/features/toc-editor/model.ts`; `Deep_Reader_UI/src/features/toc-editor/controller.ts`; `Deep_Reader_UI/src/features/toc-editor/view.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: `from scratch` is the default and starts with an empty draft for missing or unrecognized TOC cases. `edit existing` explicitly seeds the draft from the loaded layout. Mode switching resets in-memory draft and validation state without mutating task-layout.

- [x] Load selected documents directly from combo-box selection
  Evidence: `Deep_Reader_UI/src/features/book-search/view.tsx`; `Deep_Reader_UI/src/App.tsx`; validation: `npm run check`.
  Notes: Removes the `.document-search-submit` button. Typing still filters loaded API options locally; selecting an API-returned document immediately triggers the layout load flow.

- [x] Add page-first anchor entry for pageable documents in TOC editor
  Evidence: `Deep_Reader_UI/src/types/api.ts`; `Deep_Reader_UI/src/features/toc-editor/model.ts`; `Deep_Reader_UI/src/features/toc-editor/controller.ts`; `Deep_Reader_UI/src/features/toc-editor/view.tsx`; validation: `npm run check`.
  Notes: The TOC editor now detects lightweight backend `anchor_evidence` from task-layout and defaults draft anchors to `page_range` when page evidence is available. The source anchor workspace exposes a Page/Char segmented control; page entry uses 1-based page numbers in the UI and submits zero-based `page_start_index` / `page_end_index` to the manual-structure backend. `char_range` remains available as fallback and explicit override, and no task-layout mutation is introduced.

- [x] Seed edit-existing TOC anchors from parsed structure evidence
  Evidence: `Deep_Reader_UI/src/features/toc-editor/model.ts`; validation: `npm run check`.
  Notes: In `edit existing` mode, chapters and sections now seed source anchors from task-layout `anchor_evidence` when backend evidence is `available`. Page-backed evidence is displayed as 1-based page numbers and submitted as zero-based `page_range`; character-backed evidence is displayed as `char_start` / `char_end`. Missing or unavailable evidence remains empty and must be supplied by the user rather than invented.

- [x] Wire TOC edit page-default refresh to explicit backend page-evidence opt-in
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/services/TaskLayoutService.ts`; `Deep_Reader_UI/src/features/toc-editor/model.ts`; validation with `npm run check`.
  Notes: This frontend integration must keep task-layout lightweight and read-only, avoid hidden prepare/OCR work during ordinary document open, avoid inventing page defaults when backend evidence is unavailable, and preserve zero-based backend `page_range` submission semantics.
  Timestamp: 2026-10-05

- [x] Move detailed UI error and validation messages into dismissible popup notifications
  Evidence: `Deep_Reader_UI/src/services/RestClient.ts`; `Deep_Reader_UI/src/shared/components/AppNotification.tsx`; `Deep_Reader_UI/src/features/toc-editor/controller.ts`; `Deep_Reader_UI/src/features/toc-editor/view.tsx`; `Deep_Reader_UI/src/App.tsx`; validation: `npm run check`.
  Notes: UI normalizes backend `detail` / `error` / `reason` / `errors[]` payloads for display, then shows detailed runtime failures and validation summaries through dismissible MUI Snackbar/Alert or Dialog interactions instead of occupying permanent TOC editor or main reader workspace. Backend API contracts remain unchanged.

- [x] Document frontend-only reading content pagination design
  Evidence: `Deep_Reader_UI/module-detailed-design.md`.
  Notes: Reading content pagination is documented as a frontend-only viewport projection derived from backend ordered `content_blocks`. Backend APIs, task-layout, hierarchy truth, task-unit semantics, profile diagnostics, and parser artifacts remain unchanged. First implementation should paginate at content-block granularity so the current group does not force `.reader-content-block-list` scrolling, except when a single task unit or non-splittable block is itself oversized; DOM measurement and resize/typography repagination remain follow-up evolution.

- [x] Implement frontend reading content pagination
  Evidence: `Deep_Reader_UI/src/features/reader-content/model.ts`; `Deep_Reader_UI/src/features/reader-content/controller.ts`; `Deep_Reader_UI/src/features/reader-content/view.tsx`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Reader content now preserves fetched task-unit groups, measures rendered group heights in the frontend, packs the largest ordered group that fits the visible `.reader-content-block-list`, exposes Previous/Next controls, resets to page one on section/content changes, repaginates on resize, and permits local scrolling only for a single oversized task unit/block page. Backend APIs and hierarchy/task-unit contracts remain unchanged.

- [x] Prefer read-only task-layout when opening existing documents
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/services/TaskLayoutService.ts`; validation with `npm run check`.
  Notes: This prevents ordinary combo-box selection from hiding expensive backend OCR/language/profile/LLM work behind a cache hit. The UI must keep backend hierarchy as source of truth and must not mutate task-layout, profile diagnostics, or parser artifacts.
  Timestamp: 2026-10-05

- [x] Contain normal app scrolling inside `.reader-layout`
  Evidence: `Deep_Reader_UI/src/styles.css`; validation: `npm run check`; `git diff --check -- Deep_Reader_UI/module-detailed-design.md Deep_Reader_UI/module-checklist.md Deep_Reader_UI/src/styles.css`; Vite dev server HTTP smoke.
  Notes: `html`, `body`, `#root`, and `.app-shell` are bounded to the viewport without document-level scrolling; `.reader-layout` owns workspace overflow; content/navigation panes size from available layout space instead of adding viewport-height children on top of reader-layout padding. This is frontend-only CSS/layout work and does not alter backend APIs, task-layout payloads, hierarchy truth, task-unit content semantics, profile diagnostics, parser artifacts, or manual TOC commit behavior.

- [x] Keep desktop `.reader-layout` fixed while `.chapter-list` and reader content scroll internally
  Evidence: `Deep_Reader_UI/src/styles.css`; validation with `npm run check`; `git diff --check -- Deep_Reader_UI/src/styles.css Deep_Reader_UI/module-checklist.md Deep_Reader_UI/module-detailed-design.md`.
  Notes: Desktop `.reader-layout` now remains a bounded non-scrolling workspace. The hierarchy pane and `.hierarchy-navigation` form a flex height chain so `.chapter-list` owns long-navigation scrolling. The content pane is height-bound with hidden overflow, while reader-content pagination and oversized block-list scrolling remain internal to the reader surface. Mobile single-column layout keeps `.reader-layout` scrollable to avoid viewport clipping. This is frontend-only CSS/layout work and does not alter backend APIs, task-layout payloads, hierarchy truth, task-unit content semantics, profile diagnostics, parser artifacts, or manual TOC commit behavior.
  Timestamp: 2026-10-04

- [x] Remove unused reader status region from the topbar
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/styles.css`; validation with `npm run check`.
  Notes: Removed the low-value `.reader-status-region` display and its derived section/unit-count wiring, then collapsed the topbar from three columns to the active document-search and repair-control columns. Detailed runtime feedback remains available through existing loading controls and popup notifications.
  Timestamp: 2026-10-04

- [x] Simplify document search input helper copy
  Evidence: `Deep_Reader_UI/src/features/book-search/view.tsx`; validation with `npm run check`.
  Notes: Removed the `.document-search-input` helper text to save topbar space and changed the input label from `Document` to `Select one Document`.
  Timestamp: 2026-10-04

- [x] Document reading interaction UI behavior design
  Evidence: `Deep_Reader_UI/module-detailed-design.md`; backend exposure audit in `Deep_Reflective_Reader/main.module-detailed-design.md`, `Deep_Reflective_Reader/api_schemas.module-detailed-design.md`, and `Deep_Reflective_Reader/app/module-detailed-design.md`.
  Notes: Defines the next UI behavior for insight, quiz, and critical-thinking features. Target rows use a vertical three-dot MUI menu with Insights, Quiz, and Critical thinking actions. Insight renders inline beneath book/chapter/section/unit targets and indents one visual level; quiz and critical-thinking use right-side drawers for complex interaction. Backend generic routes are now implemented; current UI service wiring starts with insight and still must not mutate task-layout, make UI state backend truth, or expand `/documents/task-layout`.
  Timestamp: 2026-10-09

### Reading Interaction UI Implementation Task Split

- [x] Define `ReadingInteractionTarget` frontend model.
  Evidence: `Deep_Reader_UI/src/features/reading-interactions/model.ts`; `Deep_Reader_UI/src/features/reading-interactions/index.ts`; validation: `npm run check`.
  Notes: Adds a frontend-only target model for document, chapter, section, and task-unit interaction targets, plus required identity validation for target levels. It uses backend-provided ids for chapter, section, and task-unit targeting and does not introduce target-key helpers, API calls, backend schema claims, or UI mutation behavior.
  Timestamp: 2026-10-09

- [x] Define reading-interaction target key and breadcrumb helpers.
  Evidence: `Deep_Reader_UI/src/features/reading-interactions/model.ts`; `Deep_Reader_UI/src/features/reading-interactions/index.ts`; validation: `npm run check`.
  Notes: Adds deterministic frontend cache keys and breadcrumb formatting helpers for reading-interaction targets. Keys are local render/cache identifiers derived from target level, document scope, and backend ids; they are not backend identifiers and do not introduce API calls or persistence behavior.
  Timestamp: 2026-10-09

- [x] Add document-level three-dot interaction menu entry.
  Evidence: `Deep_Reader_UI/src/features/hierarchy-navigation/view.tsx`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Adds a document-level `MoreVert` icon button near the loaded document title with Insights, Quiz, and Critical thinking menu actions. Opening the menu and selecting options do not trigger generation, refresh, prepare, task-layout mutation, or interaction API calls; selected actions currently surface a local notification until generic backend interaction routes exist.
  Timestamp: 2026-10-09

- [x] Add chapter-level three-dot interaction menu entry.
  Evidence: `Deep_Reader_UI/src/features/hierarchy-navigation/view.tsx`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Adds a chapter-level `MoreVert` icon button at the right edge of chapter rows, including merged single-section chapter rows. The menu exposes Insights, Quiz, and Critical thinking actions and passes the backend chapter object, including `chapter_id`, to the App-level placeholder handler. Selecting options does not trigger generation, refresh, prepare, task-layout mutation, or interaction API calls.
  Timestamp: 2026-10-09

- [x] Add section-level three-dot interaction menu entry.
  Evidence: `Deep_Reader_UI/src/features/hierarchy-navigation/view.tsx`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Adds a section-level `MoreVert` icon button at the right edge of section rows. The menu exposes Insights, Quiz, and Critical thinking actions and passes the backend chapter and section objects, including `chapter_id` and `section_id`, to the App-level placeholder handler. Selecting options does not trigger generation, refresh, prepare, task-layout mutation, or interaction API calls.
  Timestamp: 2026-10-09

- [x] Add task-unit-level three-dot interaction menu entry inside reader content.
  Evidence: `Deep_Reader_UI/src/features/reader-content/view.tsx`; `Deep_Reader_UI/src/features/reader-content/index.ts`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Renders task-unit-level `MoreVert` icon buttons only inside existing reader content task-unit groups. The menu exposes Insights, Quiz, and Critical thinking actions and passes the selected section plus task-unit group, including `taskUnitId`, to the App-level placeholder handler. The hierarchy navigation remains chapter/section-only, and selecting options does not trigger generation, refresh, prepare, task-layout mutation, or interaction API calls.
  Timestamp: 2026-10-09

- [x] Create `src/features/reading-interactions/` feature folder.
  Evidence: `Deep_Reader_UI/src/features/reading-interactions/model.ts`; `Deep_Reader_UI/src/features/reading-interactions/controller.ts`; `Deep_Reader_UI/src/features/reading-interactions/view.tsx`; `Deep_Reader_UI/src/features/reading-interactions/index.ts`; validation: `npm run check`.
  Notes: Establishes the standard feature folder with `model.ts`, `controller.ts`, `view.tsx`, and `index.ts`. The new controller/view files are inert scaffolding only; they do not implement menu state, inline insight state, drawer state, API calls, task-layout mutation, or backend schema assumptions.
  Timestamp: 2026-10-09

- [x] Implement reading-interaction menu controller.
  Evidence: `Deep_Reader_UI/src/features/reading-interactions/controller.ts`; `Deep_Reader_UI/src/features/reading-interactions/index.ts`; `Deep_Reader_UI/src/App.tsx`; validation: `npm run check`.
  Notes: Adds `useReadingInteractionMenuController` to own `interactionMenuTarget`, selected interaction action, deterministic `targetKey`, breadcrumb projection, request id advancement, latest-request guard, and surface derivation (`insight` to inline insight, quiz/critical-thinking to drawer). App-level document/chapter/section/task-unit handlers now build frontend targets and route selection through the controller. This remains placeholder orchestration only and does not implement inline insight state, drawer state, API calls, generation, refresh, prepare, or task-layout mutation.
  Timestamp: 2026-10-09

- [x] Implement inline insight state model.
  Evidence: `Deep_Reader_UI/src/features/reading-interactions/model.ts`; `Deep_Reader_UI/src/features/reading-interactions/index.ts`; validation: `npm run check`.
  Notes: Adds frontend-only `InteractionStatus`, `InteractionMetadataView`, `InsightViewState`, `InlineInsightStateByTarget`, and `createInitialInsightViewState` for inline insight projection state. The model tracks target key, target identity, status, content, metadata, expanded/collapsed state, and error message without rendering inline placement, calling APIs, generating content, mutating task-layout, or treating UI state as backend truth.
  Timestamp: 2026-10-09

- [x] Implement inline insight placement for document, chapter, section, and task-unit targets.
  Evidence: `Deep_Reader_UI/src/features/reading-interactions/controller.ts`; `Deep_Reader_UI/src/features/reading-interactions/view.tsx`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/features/hierarchy-navigation/view.tsx`; `Deep_Reader_UI/src/features/reader-content/view.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Selecting Insights now creates frontend-local inline insight state and renders a placeholder inline region directly below document, chapter, section, or task-unit targets with one-level visual indentation. Placement uses existing hierarchy rows and reader content task-unit groups only; it does not expand `/documents/task-layout`, persist collapse state, call APIs, generate insight content, or treat UI state as backend truth.
  Timestamp: 2026-10-09

- [x] Implement inline insight status UI.
  Evidence: `Deep_Reader_UI/src/features/reading-interactions/view.tsx`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/features/hierarchy-navigation/view.tsx`; `Deep_Reader_UI/src/features/reader-content/view.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Inline insight regions now render status labels, status-specific messages, content/error display, and explicit Generate/Refresh controls for supported states. The UI covers loading, not-generated, generating, refreshing, completed, insufficient-content, stale-target, generation-failed, validation-failed, and related interaction statuses. Generate/Refresh actions remain local placeholders that notify backend route readiness requirements and do not call APIs, trigger LLM work, mutate task-layout, or persist UI state.
  Timestamp: 2026-10-09

- [x] Build shared right-side reading-interaction drawer shell.
  Evidence: `Deep_Reader_UI/src/features/reading-interactions/view.tsx`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Adds a shared right-side MUI drawer shell for quiz and critical-thinking interactions. The shell shows the target breadcrumb, close behavior, command slot, loading/error/body empty states, and explicitly states that opening the drawer does not call generation or refresh APIs. Insight continues to use inline placement, and the drawer shell does not implement quiz item layout, critical-thinking form flow, API calls, generation, refresh, or task-layout mutation.
  Timestamp: 2026-10-09

- [x] Implement quiz drawer layout.
  Evidence: `Deep_Reader_UI/src/features/reading-interactions/model.ts`; `Deep_Reader_UI/src/features/reading-interactions/controller.ts`; `Deep_Reader_UI/src/features/reading-interactions/view.tsx`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Adds frontend-local quiz drawer display state and renders a read-first quiz drawer layout with status summary, explicit Generate quiz and Refresh quiz controls, error/metadata slots, and completed-state item summary placeholder. Opening the drawer initializes missing local state only; it does not call quiz read/generate/refresh APIs, use legacy section/chapter quiz routes, implement type-specific quiz item rendering, persist local state, or mutate task-layout.
  Timestamp: 2026-10-09

- [x] Implement quiz item rendering model.
  Evidence: `Deep_Reader_UI/src/features/reading-interactions/model.ts`; `Deep_Reader_UI/src/features/reading-interactions/view.tsx`; `Deep_Reader_UI/src/features/reading-interactions/index.ts`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Adds frontend-only quiz item view types for `short_answer`, `multiple_choice`, and `true_false`, plus drawer rendering for prompt text, multiple-choice/true-false options, answer display, and explanation display when an item payload provides them. This does not add local answer inputs, local reveal-state tracking, backend quiz API calls, legacy quiz route usage, persisted quiz results, or task-layout mutation.
  Timestamp: 2026-10-09

- [x] Implement local quiz answer practice UI.
  Evidence: `Deep_Reader_UI/src/features/reading-interactions/model.ts`; `Deep_Reader_UI/src/features/reading-interactions/view.tsx`; `Deep_Reader_UI/src/features/reading-interactions/index.ts`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Adds drawer-local quiz practice state for typed short answers, selected multiple-choice/true-false answers, and per-item answer reveal toggles. Practice answers and reveal state reset when the quiz target or item payload changes, remain frontend-only while the drawer is mounted, and are not persisted, evaluated, submitted, sent to backend routes, described as saved quiz results, or written into task-layout.
  Timestamp: 2026-10-09

- [x] Implement critical-thinking drawer question, answer, and evaluation layout.
  Evidence: `Deep_Reader_UI/src/features/reading-interactions/model.ts`; `Deep_Reader_UI/src/features/reading-interactions/controller.ts`; `Deep_Reader_UI/src/features/reading-interactions/view.tsx`; `Deep_Reader_UI/src/features/reading-interactions/index.ts`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Adds frontend-local critical-thinking session view state and drawer layout for not-generated, question-generated, answer-submitted, completed, insufficient-content, stale-target, generation-failed, validation-failed, and evaluation-failed states. The drawer can render a generated question, answer input layout, submitted answer, evaluation feedback, score, suggested refinement, metadata, and explicit Generate question / Submit answer placeholder controls. It does not persist draft answers, implement retry state, submit answers, call backend routes, create sessions, or mutate task-layout.
  Timestamp: 2026-10-09

- [x] Implement critical-thinking draft answer and retry UI state.
  Evidence: `Deep_Reader_UI/src/features/reading-interactions/view.tsx`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Historical initial drawer-local implementation. ENH-REV-05 supersedes close-to-discard: drafts now remain in App memory by target when the drawer closes, and survive unchanged/title-only enhancement. Document reset, successful answer/question replacement and confirmed structural replacement clear the relevant state. This does not persist drafts or make them hierarchy truth. Service-backed submit/retry delivery is recorded in later items.
  Timestamp: 2026-10-21

- [x] Add `ReadingInteractionService` after generic backend routes exist.
  Evidence: `Deep_Reader_UI/src/services/ReadingInteractionService.ts`; `Deep_Reader_UI/src/services/index.ts`; `Deep_Reader_UI/src/types/api.ts`; `Deep_Reader_UI/src/features/reading-interactions/model.ts`; `Deep_Reader_UI/src/features/reading-interactions/controller.ts`; `Deep_Reader_UI/src/App.tsx`; validation: `npm run check`.
  Notes: Adds the first service-backed reading-interaction vertical slice for insight read/generate/refresh using the backend generic `/documents/reading-interactions/insight/*` routes. Selecting Insights now performs read-or-generate activation (UI-ACTIVATE-01); Refresh remains explicit. The service/controller mapping uses the shared backend target/envelope/payload contract, does not call legacy quiz routes, does not mutate task-layout, and keeps inline expansion state frontend-local. Later items cover quiz and critical-thinking wiring.
  Timestamp: 2026-10-27

- [x] Add quiz API types and service methods.
  Evidence: `Deep_Reader_UI/src/types/api.ts`; `Deep_Reader_UI/src/services/ReadingInteractionService.ts`; validation: `npm run check`.
  Notes: Adds quiz artifact item/payload/interaction response types matching the backend generic quiz schema and exposes `readQuiz`, `generateQuiz`, and `refreshQuiz` through `ReadingInteractionService` using `/documents/reading-interactions/quiz/*`. This does not call legacy `/documents/section-quiz` or `/documents/chapter-quiz`, place REST calls in drawer views, map quiz responses into drawer state, or wire drawer lifecycle/actions.
  Timestamp: 2026-10-10

- [x] Map quiz backend responses into drawer view state.
  Evidence: `Deep_Reader_UI/src/features/reading-interactions/model.ts`; validation: `npm run check`.
  Notes: Adds `mapQuizInteractionResponseToQuizViewState` for generic `QuizInteractionResponse` payloads with normalized status, stable item ids, item types, string-option labels, answers, explanations, envelope metadata, generated/updated timestamps, and reason-backed error messages. Practice answers and answer reveal state remain drawer-local frontend state and are not treated as persisted backend quiz results. This does not wire drawer open/read lifecycle or explicit generate/refresh service calls.
  Timestamp: 2026-10-10

- [x] Wire quiz drawer read-first lifecycle.
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/features/reading-interactions/controller.ts`; validation: `npm run check`.
  Notes: Selecting `Quiz` opens the drawer and calls `ReadingInteractionService.readQuiz` against the generic `/documents/reading-interactions/quiz/read` route first. As of UI-ACTIVATE-01, valid `not_generated` results trigger one separate generation request within the explicit selection; failures and stale reads do not. No task-layout mutation, prepare or reparse. Stale read responses are ignored through the reading-interaction controller request guard, and local practice answers/reveal state remain drawer-local.
  Timestamp: 2026-10-27

- [x] Wire quiz explicit generate and refresh actions.
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/features/reading-interactions/controller.ts`; validation: `npm run check`.
  Notes: `Generate quiz` calls `ReadingInteractionService.generateQuiz` against `/documents/reading-interactions/quiz/generate`, and `Refresh quiz` calls `ReadingInteractionService.refreshQuiz` against `/documents/reading-interactions/quiz/refresh`. Both actions update drawer state through `mapQuizInteractionResponseToQuizViewState` and the shared request guard. Legacy `/documents/section-quiz` and `/documents/chapter-quiz` remain outside this flow. UI-ACTIVATE-01 adds generation after a valid missing read on explicit Quiz selection; retry reconciles saved state first and Refresh stays explicit. Read endpoints remain read-only.
  Timestamp: 2026-10-27

- [x] Add critical-thinking API types and service methods.
  Evidence: `Deep_Reader_UI/src/types/api.ts`; `Deep_Reader_UI/src/services/ReadingInteractionService.ts`; validation: `npm run check`.
  Notes: Adds critical-thinking session payload/evaluation/response types matching the backend public schema and exposes `readCriticalThinkingSession`, `generateCriticalThinkingQuestion`, `submitCriticalThinkingAnswer`, and `retryCriticalThinkingEvaluation` using `/documents/reading-interactions/critical-thinking/*`. Submit/retry methods carry the shared target object plus backend `session_id`; submit also carries the answer. This does not map responses into drawer state, read on drawer open, generate questions, submit answers, retry evaluations, or mutate task-layout.
  Timestamp: 2026-10-10

- [x] Map critical-thinking backend responses into drawer view state.
  Evidence: `Deep_Reader_UI/src/features/reading-interactions/model.ts`; validation: `npm run check`.
  Notes: Adds `mapCriticalThinkingSessionResponseToViewState` for generic `CriticalThinkingSessionResponse` payloads with normalized status, backend session metadata, generated/evaluated timestamps, question, backend submitted answer, evaluation feedback, score, suggested refinement, retry eligibility, and reason-backed error messages. Unsent draft answer text remains drawer-local view state and is not included in backend-derived state.
  Timestamp: 2026-10-10

- [x] Wire critical-thinking drawer read-first lifecycle.
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/features/reading-interactions/controller.ts`; validation: `npm run check`.
  Notes: Selecting `Critical thinking` opens the drawer and calls `ReadingInteractionService.readCriticalThinkingSession` against `/documents/reading-interactions/critical-thinking/read` first. UI-ACTIVATE-01 generates one question after a valid missing response on explicit selection. Existing sessions/answers remain intact; submission and evaluation retry remain explicit. No task-layout mutation, prepare or reparse. Stale read responses are ignored through the reading-interaction controller request guard.
  Timestamp: 2026-10-27

- [x] Wire critical-thinking explicit generate, submit, and retry actions.
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/features/reading-interactions/controller.ts`; `Deep_Reader_UI/src/features/reading-interactions/view.tsx`; validation: `npm run check`.
  Notes: `Generate question`, `Submit answer`, and `Retry evaluation` call their corresponding generic critical-thinking routes, update drawer state through `mapCriticalThinkingSessionResponseToViewState`, and use the shared request guard. Submit/retry include the shared target object plus backend `session_id`; retry does not regenerate the question. Draft answers remain frontend-local until submit, are cleared after successful submission/new-question generation, but retained by target on close (ENH-REV-05); UI-ACTIVATE-01 read reuse does not clear them, and are not written to task-layout or treated as backend truth.
  Timestamp: 2026-10-27

- [x] Preserve task-unit interaction menu when content grouping uses compatibility payloads.
  Evidence: `Deep_Reader_UI/src/types/api.ts`; `Deep_Reader_UI/src/features/reader-content/model.ts`; `Deep_Reader_UI/src/features/reader-content/view.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Reader content now keeps a task-unit group header for each task-unit content response and falls back from `content_blocks[]` to compatibility `content` when needed, so the task-unit `MoreVert` interaction menu remains attached to the individual unit instead of dropping to section-only interaction behavior. This is frontend render resilience only; it does not mutate task-layout, persist UI state, or create interaction artifacts outside explicit insight/quiz/critical-thinking actions.
  Timestamp: 2026-10-11

- [x] Define visible content-block interaction menu behavior.
  Evidence: `Deep_Reader_UI/module-detailed-design.md`; backend handoff in `Deep_Reflective_Reader/section_tasks/module-detailed-design.md`; backend route/schema docs in `Deep_Reflective_Reader/api_schemas.module-detailed-design.md` and `Deep_Reflective_Reader/main.module-detailed-design.md`.
  Notes: The behavior is now specified as a future UI implementation contract: `ContentBlockView` may render a per-block `MoreVert` menu that submits `target_type=content_block`, response-level `task_unit_id`, optional parent `chapter_id` / `section_id`, and `content_blocks[].block_id` as `target.content_block_id`, with optional `metadata.source_hash` as stale-target evidence. This checklist item completes design/handoff only; it does not implement the per-block menu, mutate task-layout, make content blocks hierarchy nodes, or make UI menu/drawer/draft state backend truth.
  Timestamp: 2026-10-11

### Backend-Ready Content-Block Interaction Frontend Task Split

- [x] Extend frontend reading-interaction target model for `content_block`.
  Evidence: `Deep_Reader_UI/src/features/reading-interactions/model.ts`; `Deep_Reader_UI/src/types/api.ts`; `Deep_Reader_UI/src/App.tsx`; validation: `npm run check`.
  Notes: Adds `content_block` as a frontend target level with deterministic local target keys, API target serialization, `taskUnitId`, `contentBlockId`, optional parent chapter/section ids, and optional source hash evidence. `toReadingInteractionTargetRequest` now emits backend `target_type=content_block`, parent ids, `task_unit_id`, and `content_block_id` without deriving identity from rendered text, row position, pagination page, menu state, drawer state, or draft answer state. This is model/service-boundary preparation only; it does not render content-block menus or mutate task-layout.
  Timestamp: 2026-10-11

- [x] Preserve content-block owner context through reader-content aggregation.
  Evidence: `Deep_Reader_UI/src/types/api.ts`; `Deep_Reader_UI/src/features/reader-content/model.ts`; `Deep_Reader_UI/src/features/reader-content/controller.ts`; `Deep_Reader_UI/src/features/reader-content/view.tsx`; validation: `npm run check`.
  Notes: `TaskUnitContent` now accepts response-owned optional `chapter_id` / `section_id`, and reader-content aggregation wraps every rendered block in a frontend-only `ReaderContentBlock` with `ownerContext` carrying document evidence, `taskUnitId`, optional parent ids, canonical block id via the original block, and optional `metadata.source_hash`. This preserves target construction evidence for later content-block interactions without rendering menus, mutating task-layout, persisting UI state, or adding content blocks to `/documents/task-layout`.
  Timestamp: 2026-10-11

- [x] Gate content-block menus by canonical target eligibility.
  Evidence: `Deep_Reader_UI/src/types/api.ts`; `Deep_Reader_UI/src/features/reader-content/model.ts`; `Deep_Reader_UI/src/features/reader-content/view.tsx`; validation: `npm run check`.
  Notes: Content blocks now expose typed optional `artifact_target_refs` and `interaction_availability`, and reader-content provides a reusable eligibility gate that requires response-owned task-unit/block ids before a block may present the interaction menu. Availability and target refs are consumed as confidence/status hints rather than a hard display gate, so `segmented=true` payloads that omit those optional hints can still start the read-first backend validation flow. Blocks missing task-unit or block identity remain render-only and do not silently fall back to parent task-unit targets, mutate task-layout, create artifacts, or persist UI state.
  Timestamp: 2026-10-12

- [x] Add eligible content-block three-dot interaction menu in `ContentBlockView`.
  Evidence: `Deep_Reader_UI/src/features/reader-content/view.tsx`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Eligible block rows now expose a per-row `MoreVert` menu with the same `Insights`, `Quiz`, and `Critical thinking` actions as document/chapter/section/task-unit targets. The menu is omitted for render-only or otherwise ineligible blocks, closes locally on selection, and currently reports a frontend-only notification until the next task routes content-block actions through the reading-interaction controller. Opening/selecting the menu does not generate, refresh, prepare, reparse, mutate task-layout, create artifacts, call backend routes, or persist UI state.
  Timestamp: 2026-10-12

- [x] Route content-block actions through existing reading-interaction controller surfaces.
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/features/reader-content/view.tsx`; `Deep_Reader_UI/src/features/reading-interactions/controller.ts`; validation: `npm run check`.
  Notes: Content-block menu selections now build a `content_block` `ReadingInteractionTarget` from response-owned block owner context, selected-section parent ids, `taskUnitId`, `contentBlockId`, and optional `sourceHash`, then route through the existing `selectInteractionAction` controller path. Content-block `Insights` render through the shared inline insight region under the selected block, while content-block `Quiz` and `Critical thinking` use the existing shared drawer read-first flow. UI-ACTIVATE-01 now generates missing output after explicit selection and a valid read. There is no implicit refresh, prepare, reparse, task-layout mutation, UI-state persistence or fallback to parent task-unit targets.
  Timestamp: 2026-10-27

- [x] Consume content-block `interaction_availability` for lightweight menu/status hints.
  Evidence: `Deep_Reader_UI/src/types/api.ts`; `Deep_Reader_UI/src/features/reader-content/model.ts`; `Deep_Reader_UI/src/features/reader-content/view.tsx`; `Deep_Reader_UI/src/features/reading-interactions/model.ts`; validation: `npm run check`.
  Notes: Optional content-block availability slots are now normalized into lightweight menu hints for insight, quiz, and critical-thinking. The block menu renders status labels such as not-generated, ready, insufficient-content, stale-target, generation-failed, or critical-thinking session progress as secondary text when backend projection is present. Missing availability remains non-blocking and does not require artifact payloads, source text, secondary context text, drawer state, collapse state, quiz practice answers, critical-thinking drafts, task-layout mutation, or hidden backend writes.
  Timestamp: 2026-10-12

- [x] Normalize content-block interaction error states.
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/features/reading-interactions/controller.ts`; `Deep_Reader_UI/src/features/reading-interactions/model.ts`; `Deep_Reader_UI/src/features/reading-interactions/view.tsx`; validation: `npm run check`.
  Notes: Reading-interaction response mappers and thrown request failures now normalize `insufficient_content`, `stale_target`, `generation_failed`, `evaluation_failed`, validation failures, and content-block resolver/not-found style errors into consistent inline insight and drawer states. Content-block stale/validation copy explicitly tells the user to reload or choose a valid block and does not retry with title text, visible row position, rendered text, or parent task-unit fallback. This is presentation/state normalization only; it does not change backend routes, generate artifacts, mutate task-layout, or persist UI state.
  Timestamp: 2026-10-12

- [x] Validate backend-ready content-block frontend integration.
  Evidence: `npm run check`; targeted source smoke via `rg` over `Deep_Reader_UI/src` for document/chapter/section/task-unit/content-block target construction, `toReadingInteractionTargetRequest`, content-block id serialization, availability projection consumption, and `/documents/task-layout` route usage; `git diff --check -- Deep_Reader_UI`.
  Notes: Validation confirms document, chapter, section, task-unit, and content-block targets are still constructed explicitly in `App.tsx`; content-block flows serialize `target_type=content_block`, parent ids, `task_unit_id`, `content_block_id`, and optional source evidence through the existing reading-interaction serializer; inline insight and drawer state remain frontend-local controller/view state; and `/documents/task-layout` remains a lightweight layout route with no content-block payload expansion or UI state persistence. The validation is source/build smoke only; it does not generate artifacts, mutate task-layout, prepare/reparse documents, or perform hidden backend writes.
  Timestamp: 2026-10-12

- [x] Bound long document-level inline insight scrolling.
  Evidence: `Deep_Reader_UI/src/styles.css`; `Deep_Reader_UI/module-detailed-design.md`; validation: `npm run check`; `git diff --check -- Deep_Reader_UI/src/styles.css Deep_Reader_UI/module-detailed-design.md Deep_Reader_UI/module-checklist.md`.
  Notes: Book/document inline insight now keeps long generated text inside a bounded local scroll surface in the hierarchy pane, preserving the fixed reader workspace and avoiding document-level scroll, task-layout mutation, backend API changes, artifact writes, or UI state persistence.
  Timestamp: 2026-10-12
