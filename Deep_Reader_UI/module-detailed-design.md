# Deep_Reader_UI Detailed Design

## Module Purpose

`Deep_Reader_UI` is the Web UI client for Deep Reflective Reader.

The first-phase responsibilities are:

- consume task-layout responses from the backend
- render Chapter / Section navigation
- keep task units as internal section content-addressing data
- fetch selected section content on demand by requesting its task-unit content in backend order
- render segmented content blocks
- provide frontend loading, error, empty, and initial states
- provide a responsive reading layout
- keep navigation state local to the frontend runtime
- provide a document-name search control constrained to backend API candidates

## Architecture Boundary

- Backend hierarchy is the source of truth.
- `/documents/task-layout` is a lightweight projection for navigation.
- Section is the visible reading/navigation unit for this slice.
- Task units are backend content-addressing units retained inside each section.
- Task-unit content is fetched on demand from the task-unit content endpoint when a section is selected.
- The frontend prioritizes rendering `content_blocks`.
- Task units are not rendered as primary hierarchy nodes.
- Content blocks are not hierarchy nodes.
- Frontend state is not backend truth.
- The frontend must not perform hidden backend mutation.
- The frontend must not extend or reinterpret backend API contracts.
- Future artifact, annotation, and question interactions are out of scope for this phase.
- Backend schema/API changes are out of scope for this module slice.
- Manual TOC editing uses explicit validation and hard-reparse submit paths, not hidden task-layout mutation.
- Selecting an already prepared backend document should prefer the read-centric task-layout route; prepare-then-read is a fallback or explicit repair path, not the default meaning of "open document."

## First Vertical Slice

```text
Document input
  -> Task Layout
  -> Chapter
  -> Section
  -> Section Selection
  -> On-demand Task-Unit Content Fetches
  -> Aggregated Content Blocks
```

### UI State

- `docName`: controlled combo-box input value.
- `layout`: latest task-layout response for the loaded document.
- `selectedSection`: frontend-local selected section reference with retained `task_units`.
- `contentBlocks`: aggregated `content_blocks` from the selected section's task units.
- `layoutStatus`: initial / loading / success / error.
- `contentStatus`: initial / loading / success / error.
- `errorMessage`: visible failure detail for the relevant request.

### Component Responsibility

- App shell owns state, API calls, and top-level layout.
- Document search owns doc-name input, backend candidate lookup, empty-result list fallback, and API-returned option selection. Selecting an API-returned option triggers layout loading without a separate submit button.
- Hierarchy navigation renders backend chapter and section projection.
- Section buttons update local selection state while retaining internal task-unit ids.
- Reader panel fetches each task-unit content for the selected section on demand and renders aggregated `content_blocks`.
- Empty-state views handle missing document, missing hierarchy, missing task units, and missing content blocks.

### Reading Content Pagination

Reading content pagination is a frontend-only viewport projection. It exists to prevent
very long chapters or sections from overwhelming the reading surface, especially on
small screens or when a user uses larger type. It should improve reading rhythm without
changing backend hierarchy, backend task-unit semantics, or task-layout projection.

The backend should continue to provide ordered task units and ordered `content_blocks`.
It should not calculate how many task units belong on one UI page, because page capacity
depends on frontend-only runtime conditions such as reader pane width and height, font
size, line height, browser zoom, device density, and responsive layout. A UI page is not
a backend page, not a document hierarchy node, and not parser authority.

The reader-content feature owns pagination as local UI state:

- `content_blocks` remain the render/input units received from backend task-unit content.
- frontend pages are derived from the current `content_blocks` plus current viewport and
  typography conditions.
- section selection resets the current reader page to the first page for that section.
- viewport resize, responsive breakpoint changes, or future reader typography changes
  should trigger repagination.
- page state must not be persisted as backend truth and must not mutate task-layout,
  task units, content blocks, profile diagnostics, or parser artifacts.

The first implementation should use block-level pagination: `content_blocks` are the
smallest non-splittable page units. A frontend page may contain blocks from multiple task
units, and one task unit may span multiple frontend pages. If a single content block is
larger than the available reader page, the initial behavior may keep the block intact and
allow that page to overflow locally; block-internal soft splitting can be a later
enhancement after the basic page model is stable.

The page packing rule should be that the current rendered task-unit/content-block group
does not make `.reader-content-block-list` contain hidden, unreachable, or clipped content
that requires the whole reading panel to scroll. Pagination should choose the largest
ordered group that fits the visible reader page. The explicit exception is when one
individual task unit or one non-splittable content block is itself longer than the
available page. In that case the UI should render that single unit/block as the page
content and allow local scrolling for that oversized page, rather than combining it with
additional units that would make the overflow worse.

The expected evolution is:

1. start with deterministic block-level pagination using a conservative estimate or
   simple measurement strategy;
2. add DOM measurement pagination based on actual rendered block heights;
3. repaginate on reader-pane resize and future typography changes;
4. preserve stable block/task-unit identifiers so future reading progress, annotation,
   or selection features can map back to backend-provided content targets without making
   frontend page numbers backend truth.

### Scroll Containment

The Reader UI should keep document-level scrolling disabled during normal app use. The
browser `html` / `body` surface is the viewport shell, while `.reader-layout` is the
bounded reader workspace.

This matters because the top app shell combines a fixed-height viewport, a sticky
topbar, grid padding, and independently constrained navigation/content panes. If
`html`, `body`, `#root`, or `.app-shell` use only `min-height` without a bounded height
and overflow policy, the reader workspace can push the full document taller than the
viewport and create page-level scrolling. That makes pagination and TOC editor range
workspaces feel unstable because scrolling occurs outside the reader workspace.

Expected layout behavior:

- `html`, `body`, and `#root` occupy the viewport and do not become the primary scroll
  surface.
- `.app-shell` is a viewport-height flex container with `min-height: 0`.
- `.reader-layout` is the non-scrolling desktop workspace container for the hierarchy
  and content panes. It should be height-bounded by `.app-shell`, use `min-height: 0`,
  and avoid becoming the primary scroll surface during normal desktop reading.
- The hierarchy pane should constrain its own height and let `.chapter-list` own
  hierarchy overflow. The hierarchy header remains visible while the chapter/section
  list scrolls inside its parent.
- The content pane should constrain its own height and let the reader-content internal
  surface own content overflow. The right pane should not push `.reader-layout` taller
  than the viewport.
- child panes such as `.navigation-pane` and `.content-pane` should size from the
  available `.reader-layout` space rather than independently adding another
  `calc(100vh - ...)` height on top of grid padding.
- oversized single reader pages may still scroll locally inside
  `.reader-content-block-list-oversized`, but ordinary page overflow should remain
  contained by reader-content pagination/inner content surfaces rather than by
  `.reader-layout`.

Desktop target behavior:

- `.reader-layout` does not scroll.
- `.chapter-list` is the primary scroll surface for long chapter/section navigation.
- `.reader-content` or its inner content list is the primary scroll surface for
  oversized reader content, with ordinary paginated content still avoiding scroll when
  it fits the visible reader page.
- If no suitable parent element currently exists for either scroll surface, the UI may
  introduce a thin structural wrapper whose only role is layout containment. That
  wrapper must not add hierarchy semantics, backend state, task-layout mutation, or a
  second source of truth.

Responsive/mobile behavior may remain more flexible: single-column layouts may still
use local pane scrolling where needed, but should preserve the same principle that
document-level scrolling is not the normal reading surface.

This is a frontend-only viewport contract. It must not change backend hierarchy,
task-layout payloads, task-unit content APIs, profile diagnostics, parser artifacts, or
manual TOC commit semantics.

### Feature Module Structure

The Reader UI is organized by user-facing responsibility:

- `src/features/book-search/`: top document-search entry feature.
- `src/features/hierarchy-navigation/`: lower-left chapter/section hierarchy navigation feature.
- `src/features/reader-content/`: lower-right section content rendering feature.
- `src/features/toc-editor/`: right-pane manual TOC editing feature for source documents whose table of contents is missing, incorrect, or not automatically recognized.

Each feature follows a lightweight MVC-style split:

- `model.ts`: pure UI/domain helpers such as option normalization, hierarchy counts, display labels, parser-mode derivation, heading selection, and content-block aggregation.
- `controller.ts`: React state orchestration hooks and side-effect coordination for that feature.
- `view.tsx`: MUI/DOM rendering only.
- `index.ts`: public feature exports.

`App.tsx` remains the shell/composition layer. It wires feature controllers, feature views, top-level repair controls, route selection, and layout panes without owning low-level REST request construction or content aggregation logic.

### Route Structure

React Router provides document-scoped routes:

- `/documents/:docName`: default reader route. The right pane renders `reader-content`.
- `/documents/:docName/toc-edit`: manual TOC editor route. The right pane is fully replaced by `toc-editor`; the reader-content DOM is not mounted for this route.

The `toc-edit` route is not a cold-start entry point. It requires a loaded task layout in frontend state. If a user directly opens or refreshes `/documents/:docName/toc-edit` without a loaded layout, the right pane must show a guarded state such as "Load document first" and must not automatically prepare or load the layout. The user can return to `/documents/:docName` or the default entry flow.

The TOC editor entry trigger belongs in the hierarchy navigation header, near the document title and unit count. It is enabled only when the layout has loaded successfully. Activating it routes to `/documents/:docName/toc-edit` while preserving the loaded hierarchy in the left pane as context.

### TOC Editor Page Contract

The TOC editor exists to support documents where automatic table-of-contents recognition is missing, failed, or needs correction. It must not be treated as a visual-only tree editor; user edits must be validated before they can produce an explicit hard reparse.

The TOC editor has two draft source modes:

- `from scratch`: the default mode. It starts with an empty editable tree and is optimized for the common failure case where the original TOC is missing, unreadable, or not automatically recognized.
- `edit existing`: a secondary mode. It seeds the editable tree from the currently loaded task-layout hierarchy for the rarer case where an existing structure mostly works but needs correction.

Switching modes resets the in-memory editor draft and validation state. It must not mutate task-layout or backend hierarchy until the user completes frontend validation, backend validation, and explicit hard-reparse commit.

The right pane is a single integrated TOC editor, not a collection of permanent subpanels. Its main screen uses two coordinated work surfaces inside one editor:

- editable TOC tree: supports `chapter -> section` editing, including add, delete, rename, and reorder operations
- source anchor workspace: shows source context and lets the user assign anchors to the selected chapter/section item

`section` is optional in the UI editing experience. A chapter-only book is represented as the special case where every chapter contains exactly one section. For display, the UI may use the single section title as the chapter display name, matching the existing reader rendering convention. Persistence and backend submission must still preserve the backend hierarchy contract rather than introducing root `sections[]` or deeper hierarchy levels.

The TOC editor supports both `page_range` and `char_range` anchors. The anchor workspace is page-first when the loaded task-layout exposes reliable backend `anchor_evidence` with page boundaries, and remains character-range based when page evidence is unavailable or when the user explicitly switches to character anchors:

- pageable source with reliable page evidence: default to page-number entry and submit page-backed anchors through the manual-structure flow
- pageable source without reliable page evidence: show the parsed evidence gap and fall back to `char_range`
- non-pageable source: use `char_range`
- advanced/manual override: allow the user to choose `char_range` even when page entry is available

`edit existing` mode should seed each chapter/section with default anchors from the current parsed structure when that evidence exists. For pageable documents, the displayed defaults should be the parsed page numbers. For non-pageable documents, or when page evidence is unavailable, the displayed defaults should be the parsed `char_start` / `char_end` range. If the current layout does not expose reliable anchor evidence, the editor should leave the anchor empty and surface validation guidance rather than inventing positions.

`page_range` support depends on backend page-boundary mapping and manual-structure validation/commit support. The UI must not claim page anchors are authoritative when backend page evidence is missing; it must fall back to `char_range` or require explicit user override.

The TOC editor must use an independent feature folder:

- `src/features/toc-editor/model.ts`: editable TOC tree model, from-scratch draft construction, existing-layout draft construction, chapter-only normalization, anchor/range validation helpers, manual-structure request projection
- `src/features/toc-editor/controller.ts`: route guard, draft source mode, selected TOC item, dirty state, frontend validation, backend validation, and commit confirmation orchestration
- `src/features/toc-editor/view.tsx`: integrated TOC editor surface
- `src/features/toc-editor/index.ts`: public feature exports

### TOC Validation And Commit Flow

Manual TOC editing must use a three-stage flow:

1. Frontend validation:
   - enforce the two-level maximum (`chapter -> section`)
   - allow chapter-only as the one-section-per-chapter special case
   - require usable titles
   - require anchors for submitted items
   - validate `char_start < char_end` for `char_range`
   - validate page order and source page availability for future `page_range`
   - reject missing, overlapping, or out-of-order anchors where they would make projection ambiguous
2. Backend validation:
   - call the non-mutating manual-structure validation endpoint
   - display backend issues and preview evidence
   - do not persist hierarchy or mutate task-layout during validation
3. Commit / hard reparse:
   - require successful frontend and backend validation
   - use an explicit reparse commit route
   - show a MUI confirmation dialog before commit

Entering the TOC editor may show a low-disruption notice that submitting edits will trigger hard reparse. The destructive warning belongs at commit time: the confirmation dialog must clearly state that hard reparse replaces the current structure and derived QA, summaries, quiz artifacts, and similar generated outputs will not be preserved.

Validation results and hard-reparse warnings should use MUI popup/dialog/snackbar-style interactions rather than occupying permanent space in the main editor surface.

### API Boundary

Task layout:

```http
POST /documents/task-layout
Content-Type: application/json
```

```json
{
  "doc_name": "<doc_name>",
  "refresh_task_units": false,
  "task_unit_split_mode": "progressive"
}
```

Document-open policy:

- For an API-returned document candidate, the UI should treat document selection as an
  existing-layout read first.
- The preferred route for existing layout display is `POST /documents/task-layout`.
- `POST /documents/prepare-task-layout` should be reserved for first-time prepare,
  explicit repair/retry, or fallback when the read-centric layout route reports that the
  layout is missing or unavailable.
- The UI should not rely on prepare-then-read as a hidden way to repair cache,
  structured-document, language, profile, or OCR state during ordinary document open.
- This policy avoids surprising OCR/LLM-backed backend work during normal document
  selection while preserving backend hierarchy as the source of truth.

TOC edit page-default handshake:

- Ordinary reader entry must request task layout without page-evidence opt-in so normal
  document opening does not trigger backend source-evidence/OCR work.
- Entering TOC edit mode is the explicit UI intent that may request page-backed anchor
  defaults. Before rendering edit-existing page defaults, the UI should refresh the
  current layout through `POST /documents/task-layout` with
  `include_anchor_page_evidence: true`.
- The refreshed layout remains a lightweight task-layout projection. The UI consumes
  only `anchor_evidence` metadata on chapter/section nodes and must not request or store
  raw text, page text, OCR geometry, or content blocks through task-layout.
- `edit existing` mode may prefill page inputs only when backend `anchor_evidence`
  reports available `page_range` evidence. Displayed page defaults are 1-based UI page
  numbers derived from zero-based backend page indices; manual-structure submissions
  must convert them back to zero-based `page_range` indices.
- If the opt-in layout refresh fails or returns unavailable page evidence, the TOC
  editor should keep page defaults empty and fall back to `char_range` guidance rather
  than inventing page positions.
- This handshake is read-only until the user explicitly validates and commits a
  manual-structure hard reparse.

Task-unit content, issued once per selected section:

```http
POST /documents/{doc_name}/task-units/content
```

```json
{
  "task_unit_ids": ["<task_unit_id>"],
  "segmented": true,
  "include_raw_content": false
}
```

The frontend requests the selected section's task units in backend layout order with one batch content request and renders the flattened `content_blocks` in that same order. It does not request duplicated raw content by default.

Single task-unit content remains the compatibility/fallback endpoint:

```http
GET /documents/{doc_name}/task-units/{task_unit_id}/content?segmented=true
```

- The batch request preserves current render options, especially `segmented=true` and default raw-content suppression.
- The batch response preserves per-task-unit content ordering and reuses the backend `TaskUnitContentResponse` shape so reader aggregation logic remains content-block-first.
- Batch content must remain an on-demand content API path. It must not add task-unit content or `content_blocks` to `/documents/task-layout`.
- The frontend must not reinterpret batch response order as hierarchy truth, persist content as backend state, or trigger hidden prepare/reparse/profile mutation from section selection.

### Loading / Error / Empty Behavior

- Initial state shows no hierarchy until a document is loaded.
- Layout loading disables the document search control and marks the navigation area busy.
- Layout error shows the backend error detail and keeps the user on the document input.
- Empty layout shows a no-content state if the backend returns no chapters, no sections, or no task units.
- Selecting a section starts content loading for that section's task units.
- Content error is scoped to the reader panel.
- Empty content shows a no-content-blocks state when `content_blocks` is empty.

### Error Notification Policy

The UI consumes several backend error payload shapes, including FastAPI `detail`,
operation-level `error`, task `reason`, string `errors[]`, and structured validation
`errors[]` issue objects. Backend API normalization is out of scope for this UI module
slice, so the frontend should normalize these payloads at the REST/client boundary for
display only.

Error presentation should avoid consuming permanent main-screen workspace, especially in
the TOC editor where source range editing has limited space. Runtime API failures,
validation summaries, and commit failures should use dismissible MUI popup interactions
such as `Snackbar` + `Alert`. Destructive hard-reparse confirmation remains a `Dialog`.

Main panes may keep compact state placeholders when the primary workflow cannot
continue, but detailed backend or validation text should be surfaced through the
dismissible notification layer. Validation issue lists may be opened from a popup or
dialog when multiple issues need inspection, instead of being permanently rendered in
the range workspace.

## Frontend Technical Structure

- Framework: React with TypeScript.
- UI library: MUI for app shell, combo box, buttons, navigation controls, chips, loading indicators, and alerts.
- Build tooling: Vite with `@vitejs/plugin-react`.
- Dev server: Vite dev server with `/api/*` proxy to the backend target from `DEEP_READER_API_URL` or `http://localhost:8000`.
- Routing strategy: React Router with document-scoped reader and TOC editor routes as described above.
- State approach: component-local React state in `src/App.tsx`; no external state library.
- REST service organization: `src/services/RestClient.ts` owns shared request/error handling; endpoint groups are exposed through typed service classes.
- Type organization: `src/types/api.ts` mirrors the consumed backend response fields.
- Feature organization: `src/features/book-search/`, `src/features/hierarchy-navigation/`, and `src/features/reader-content/` own MVC-style model/controller/view files.
- Shared component organization: `src/shared/components/StateView.tsx` owns cross-feature loading/error/empty state rendering.
- Styling approach: MUI theme in `src/theme.ts` plus scoped layout CSS in `src/styles.css`.
- Testing approach: TypeScript validation and production build through `npm run check`, HTTP smoke through the Vite dev server, and manual browser/API validation against the backend.

### REST Service Classes

Backend REST usage is centralized behind service classes:

- `DocumentCatalogService`: `GET /documents`
- `TaskLayoutService`: `POST /documents/task-layout`; `POST /documents/prepare-task-layout`
- `TaskUnitContentService`: `POST /documents/{doc_name}/task-units/content` for selected-section batch read; `GET /documents/{doc_name}/task-units/{task_unit_id}/content?segmented=true` as single-task compatibility/fallback
- `StructureRepairService`: `POST /documents/reparse-structure` for common / LLM structure repair
- `ManualStructureService`: `POST /documents/manual-structure/validate`; `POST /documents/reparse-structure` with `parser_mode=manual_structure`
- `DocumentPreparationService`: `POST /documents/prepare`

These services preserve the existing backend API contract and keep manual-structure validation/commit explicit.

### Document Combo Box

The first slice consumes the backend document list API for document-name candidates. It still does not implement a full document library UI.

`DocumentSearch` uses MUI `Autocomplete` and consumes the backend `GET /documents` list/search API:

- opening the combo box calls `GET /documents?limit=200`
- users may type to filter the loaded API options locally
- the loaded document must be selected from API-returned options, and selection immediately starts the layout load flow
- the UI does not issue per-keystroke search requests
- successful document names are not added as local-only options

This control remains a document-name entry surface, not a document library UI. It does not create local-only selectable documents.

## Non-Goals

- document upload
- document library
- summary UI
- quiz UI
- free QA
- annotation
- text selection
- content-block highlight
- ask-about-selection
- artifact creation
- artifact persistence UI
- generic reparse UI outside the explicit TOC editor flow
- enhanced parse diagnostics UI
- authentication
- persistent reading progress

## Future Non-Goals For TOC Editor

- direct cold-start editing from `/documents/:docName/toc-edit`
- automatic layout loading from the TOC edit route guard
- hidden task-layout mutation
- treating frontend-edited TOC as backend truth before validation and explicit commit
- deeper-than-two-level hierarchy editing
- manual-structure UI support for `page_range` before page-boundary mapping is available
