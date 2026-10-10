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

- [ ] Add quiz API types and service methods.
  Evidence needed: `Deep_Reader_UI/src/types/api.ts` defines quiz target/envelope/payload types matching backend public schemas; `Deep_Reader_UI/src/services/ReadingInteractionService.ts` exposes `readQuiz`, `generateQuiz`, and `refreshQuiz` using `/documents/reading-interactions/quiz/*`.
  Notes: Do not call legacy `/documents/section-quiz` or `/documents/chapter-quiz`; do not place REST calls in drawer views.
  Timestamp: 2026-10-10

- [ ] Map quiz backend responses into drawer view state.
  Evidence needed: `Deep_Reader_UI/src/features/reading-interactions/model.ts` maps `QuizInteractionResponse` into `QuizViewState` with normalized status, item ids/types/options/answers/explanations, metadata, and error/reason fields.
  Notes: Mapping must preserve local practice answers and answer reveal state as frontend-only state and must not treat quiz practice answers as persisted backend results.
  Timestamp: 2026-10-10

- [ ] Wire quiz drawer read-first lifecycle.
  Evidence needed: selecting `Quiz` opens the drawer and calls the generic quiz read route only; missing artifacts display `not_generated` without generation, refresh, task-layout mutation, prepare, or reparse.
  Notes: Stale read responses must be ignored through the reading-interaction controller request guard.
  Timestamp: 2026-10-10

- [ ] Wire quiz explicit generate and refresh actions.
  Evidence needed: `Generate quiz` calls `/documents/reading-interactions/quiz/generate`; `Refresh quiz` calls `/documents/reading-interactions/quiz/refresh`; both update drawer state through the shared model mapper and keep legacy quiz routes out of this flow.
  Notes: Generation and refresh are explicit write actions only; drawer open and read must never generate.
  Timestamp: 2026-10-10

- [ ] Add critical-thinking API types and service methods.
  Evidence needed: `Deep_Reader_UI/src/types/api.ts` defines critical-thinking request/response/session/evaluation types matching backend public schemas; `Deep_Reader_UI/src/services/ReadingInteractionService.ts` exposes read, generate-question, submit-answer, and retry-evaluation methods using `/documents/reading-interactions/critical-thinking/*`.
  Notes: Submit/retry request types must carry the shared target object plus backend `session_id`.
  Timestamp: 2026-10-10

- [ ] Map critical-thinking backend responses into drawer view state.
  Evidence needed: `Deep_Reader_UI/src/features/reading-interactions/model.ts` maps `CriticalThinkingSessionResponse` into `CriticalThinkingViewState` with session id, question, submitted answer, evaluation feedback, score, suggested refinement, status, metadata, and retryable error state.
  Notes: Mapping must keep unsent draft text out of backend-derived state and preserve backend submitted answer separately from local draft answer.
  Timestamp: 2026-10-10

- [ ] Wire critical-thinking drawer read-first lifecycle.
  Evidence needed: selecting `Critical thinking` opens the drawer and calls the generic critical-thinking read route only; missing sessions display `not_generated` without generating a question, submitting an answer, retrying evaluation, mutating task-layout, prepare, or reparse.
  Notes: Read state must remain separate from question generation and must ignore stale responses.
  Timestamp: 2026-10-10

- [ ] Wire critical-thinking explicit generate, submit, and retry actions.
  Evidence needed: `Generate question`, `Submit answer`, and `Retry evaluation` call their corresponding critical-thinking routes, update drawer state through the shared mapper, preserve failed evaluation context, and do not regenerate the question during retry.
  Notes: Draft answers remain frontend-local until submit; submit/retry must include the shared target object plus backend `session_id`.
  Timestamp: 2026-10-10
