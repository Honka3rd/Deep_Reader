# Deep_Reader_UI Checklist

## Purpose

This checklist tracks the first Deep_Reader_UI vertical slice.

## Completed Checklist

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
  Notes: Upload now first submits same-name files with `conflict_policy=reject`; backend `duplicate_conflict` opens a user confirmation dialog. Only explicit confirmation retries with `conflict_policy=create_distinct`, after which the UI displays the backend-owned unique `docName` / source filename and refreshes the catalog. The UI does not silently replace files, invent unique names locally, auto-prepare, call task-layout, or generate interactions.
  Timestamp: 2026-10-13

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
  Notes: Adds drawer-local critical-thinking draft answer state that is preserved while the drawer remains open, resets when the active target/question/submitted answer changes, and reports dirty state to the App shell so closing the drawer warns when unsent draft text is discarded. The command area now exposes `Retry evaluation` only for `evaluation_failed` state. Draft text and retry remain frontend-only placeholders and are not persisted, submitted, evaluated, sent to backend routes, or written into task-layout.
  Timestamp: 2026-10-09

- [x] Add `ReadingInteractionService` after generic backend routes exist.
  Evidence: `Deep_Reader_UI/src/services/ReadingInteractionService.ts`; `Deep_Reader_UI/src/services/index.ts`; `Deep_Reader_UI/src/types/api.ts`; `Deep_Reader_UI/src/features/reading-interactions/model.ts`; `Deep_Reader_UI/src/features/reading-interactions/controller.ts`; `Deep_Reader_UI/src/App.tsx`; validation: `npm run check`.
  Notes: Adds the first service-backed reading-interaction vertical slice for insight read/generate/refresh using the backend generic `/documents/reading-interactions/insight/*` routes. Selecting Insights performs a read-first call; Generate and Refresh are explicit write actions. The service/controller mapping uses the shared backend target/envelope/payload contract, does not call legacy quiz routes, does not mutate task-layout, and keeps inline expansion state frontend-local. Quiz and critical-thinking drawer API wiring remain follow-up tasks.
  Timestamp: 2026-10-10

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
  Notes: Selecting `Quiz` opens the drawer and calls `ReadingInteractionService.readQuiz` against the generic `/documents/reading-interactions/quiz/read` route only. Returned `not_generated` artifacts render through the drawer state mapper without generation, refresh, task-layout mutation, prepare, or reparse. Stale read responses are ignored through the reading-interaction controller request guard, and local practice answers/reveal state remain drawer-local.
  Timestamp: 2026-10-10

- [x] Wire quiz explicit generate and refresh actions.
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/features/reading-interactions/controller.ts`; validation: `npm run check`.
  Notes: `Generate quiz` calls `ReadingInteractionService.generateQuiz` against `/documents/reading-interactions/quiz/generate`, and `Refresh quiz` calls `ReadingInteractionService.refreshQuiz` against `/documents/reading-interactions/quiz/refresh`. Both actions update drawer state through `mapQuizInteractionResponseToQuizViewState` and the shared request guard. Legacy `/documents/section-quiz` and `/documents/chapter-quiz` remain outside this flow. Generation and refresh are explicit write actions only; drawer open and read still never generate.
  Timestamp: 2026-10-10

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
  Notes: Selecting `Critical thinking` opens the drawer and calls `ReadingInteractionService.readCriticalThinkingSession` against `/documents/reading-interactions/critical-thinking/read` only. Returned `not_generated` sessions render through the drawer state mapper without generating a question, submitting an answer, retrying evaluation, mutating task-layout, prepare, or reparse. Stale read responses are ignored through the reading-interaction controller request guard.
  Timestamp: 2026-10-10

- [x] Wire critical-thinking explicit generate, submit, and retry actions.
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/features/reading-interactions/controller.ts`; `Deep_Reader_UI/src/features/reading-interactions/view.tsx`; validation: `npm run check`.
  Notes: `Generate question`, `Submit answer`, and `Retry evaluation` call their corresponding generic critical-thinking routes, update drawer state through `mapCriticalThinkingSessionResponseToViewState`, and use the shared request guard. Submit/retry include the shared target object plus backend `session_id`; retry does not regenerate the question. Draft answers remain frontend-local until submit, are cleared after submit/target close, and are not written to task-layout or treated as backend truth.
  Timestamp: 2026-10-10

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
  Notes: Content-block menu selections now build a `content_block` `ReadingInteractionTarget` from response-owned block owner context, selected-section parent ids, `taskUnitId`, `contentBlockId`, and optional `sourceHash`, then route through the existing `selectInteractionAction` controller path. Content-block `Insights` render through the shared inline insight region under the selected block, while content-block `Quiz` and `Critical thinking` use the existing shared drawer read-first flow. This does not generate, refresh, prepare, reparse, mutate task-layout, persist UI state, or fall back to parent task-unit targets.
  Timestamp: 2026-10-12

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
