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

- [x] Move detailed UI error and validation messages into dismissible popup notifications
  Evidence: `Deep_Reader_UI/src/services/RestClient.ts`; `Deep_Reader_UI/src/shared/components/AppNotification.tsx`; `Deep_Reader_UI/src/features/toc-editor/controller.ts`; `Deep_Reader_UI/src/features/toc-editor/view.tsx`; `Deep_Reader_UI/src/App.tsx`; validation: `npm run check`.
  Notes: UI normalizes backend `detail` / `error` / `reason` / `errors[]` payloads for display, then shows detailed runtime failures and validation summaries through dismissible MUI Snackbar/Alert or Dialog interactions instead of occupying permanent TOC editor or main reader workspace. Backend API contracts remain unchanged.

- [x] Document frontend-only reading content pagination design
  Evidence: `Deep_Reader_UI/module-detailed-design.md`.
  Notes: Reading content pagination is documented as a frontend-only viewport projection derived from backend ordered `content_blocks`. Backend APIs, task-layout, hierarchy truth, task-unit semantics, profile diagnostics, and parser artifacts remain unchanged. First implementation should paginate at content-block granularity so the current group does not force `.reader-content-block-list` scrolling, except when a single task unit or non-splittable block is itself oversized; DOM measurement and resize/typography repagination remain follow-up evolution.

- [x] Implement frontend reading content pagination
  Evidence: `Deep_Reader_UI/src/features/reader-content/model.ts`; `Deep_Reader_UI/src/features/reader-content/controller.ts`; `Deep_Reader_UI/src/features/reader-content/view.tsx`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/styles.css`; validation: `npm run check`.
  Notes: Reader content now preserves fetched task-unit groups, measures rendered group heights in the frontend, packs the largest ordered group that fits the visible `.reader-content-block-list`, exposes Previous/Next controls, resets to page one on section/content changes, repaginates on resize, and permits local scrolling only for a single oversized task unit/block page. Backend APIs and hierarchy/task-unit contracts remain unchanged.

- [ ] Prefer read-only task-layout when opening existing documents
  Evidence needed: selecting an API-returned document first requests the existing layout through `POST /documents/task-layout`; `POST /documents/prepare-task-layout` is used only for first-time prepare, explicit retry/repair, or fallback when the read-centric route reports unavailable layout.
  Notes: This prevents ordinary combo-box selection from hiding expensive backend OCR/language/profile/LLM work behind a cache hit. The UI must keep backend hierarchy as source of truth and must not mutate task-layout, profile diagnostics, or parser artifacts.

- [x] Contain normal app scrolling inside `.reader-layout`
  Evidence: `Deep_Reader_UI/src/styles.css`; validation: `npm run check`; `git diff --check -- Deep_Reader_UI/module-detailed-design.md Deep_Reader_UI/module-checklist.md Deep_Reader_UI/src/styles.css`; Vite dev server HTTP smoke.
  Notes: `html`, `body`, `#root`, and `.app-shell` are bounded to the viewport without document-level scrolling; `.reader-layout` owns workspace overflow; content/navigation panes size from available layout space instead of adding viewport-height children on top of reader-layout padding. This is frontend-only CSS/layout work and does not alter backend APIs, task-layout payloads, hierarchy truth, task-unit content semantics, profile diagnostics, parser artifacts, or manual TOC commit behavior.
