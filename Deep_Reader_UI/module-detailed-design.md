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
- Document search owns doc-name input, backend candidate lookup, empty-result list fallback, API-returned option selection, and load action.
- Hierarchy navigation renders backend chapter and section projection.
- Section buttons update local selection state while retaining internal task-unit ids.
- Reader panel fetches each task-unit content for the selected section on demand and renders aggregated `content_blocks`.
- Empty-state views handle missing document, missing hierarchy, missing task units, and missing content blocks.

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

Task-unit content, issued once per selected section task unit:

```http
GET /documents/{doc_name}/task-units/{task_unit_id}/content?segmented=true
```

The frontend does not request duplicated raw content. It requests selected section task units in backend layout order and renders the flattened `content_blocks` in that same order.

### Loading / Error / Empty Behavior

- Initial state shows no hierarchy until a document is loaded.
- Layout loading disables the load button and marks the navigation area busy.
- Layout error shows the backend error detail and keeps the user on the document input.
- Empty layout shows a no-content state if the backend returns no chapters, no sections, or no task units.
- Selecting a section starts content loading for that section's task units.
- Content error is scoped to the reader panel.
- Empty content shows a no-content-blocks state when `content_blocks` is empty.

## Frontend Technical Structure

- Framework: React with TypeScript.
- UI library: MUI for app shell, combo box, buttons, navigation controls, chips, loading indicators, and alerts.
- Build tooling: Vite with `@vitejs/plugin-react`.
- Dev server: Vite dev server with `/api/*` proxy to the backend target from `DEEP_READER_API_URL` or `http://localhost:8000`.
- Routing strategy: single-page application with no client router.
- State approach: component-local React state in `src/App.tsx`; no external state library.
- API client organization: `src/api/client.ts` owns typed task-layout and task-unit content requests.
- Type organization: `src/types/api.ts` mirrors the consumed backend response fields.
- Component organization: `src/components/DocumentSearch.tsx`, `HierarchyNavigation.tsx`, `ReaderContent.tsx`, and `StateView.tsx`.
- Styling approach: MUI theme in `src/theme.ts` plus scoped layout CSS in `src/styles.css`.
- Testing approach: TypeScript validation and production build through `npm run check`, HTTP smoke through the Vite dev server, and manual browser/API validation against the backend.

### Document Combo Box

The first slice consumes the backend document list API for document-name candidates. It still does not implement a full document library UI.

`DocumentSearch` uses MUI `Autocomplete` and consumes the backend `GET /documents` list/search API:

- opening the combo box calls `GET /documents?limit=200`
- users may type to filter the loaded API options locally
- the loaded document must be selected from API-returned options
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
- reparse UI
- enhanced parse diagnostics UI
- authentication
- persistent reading progress
