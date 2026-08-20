# Deep_Reader_UI Checklist

## Purpose

This checklist tracks the first Deep_Reader_UI vertical slice.

## Completed Checklist

- [x] Establish Deep_Reader_UI frontend foundation.
  Evidence: `Deep_Reader_UI/package.json`; `Deep_Reader_UI/package-lock.json`; `Deep_Reader_UI/vite.config.ts`; `Deep_Reader_UI/index.html`; `Deep_Reader_UI/src/main.tsx`; `Deep_Reader_UI/src/App.tsx`.
  Notes: Created an independent React + TypeScript + MUI UI client with Vite dev/build tooling and backend proxy.

- [x] Implement document-name based Reader entry flow.
  Evidence: `Deep_Reader_UI/src/components/DocumentSearch.tsx`; `Deep_Reader_UI/src/App.tsx`.
  Notes: The MUI combo box searches backend candidates and only enables load for a selected API-returned `doc_name`.

- [x] Consume and render task-layout hierarchy.
  Evidence: `Deep_Reader_UI/src/api/client.ts`; `Deep_Reader_UI/src/components/HierarchyNavigation.tsx`; API used: `POST /documents/task-layout`.
  Notes: Renders backend chapters and sections without adding heavy content to task-layout; task units are retained internally for content addressing.

- [x] Implement Chapter -> Section navigation with internal Task Unit aggregation.
  Evidence: `Deep_Reader_UI/src/components/HierarchyNavigation.tsx`; `Deep_Reader_UI/src/styles.css`.
  Notes: Navigation follows backend hierarchy order and makes sections selectable; task units are no longer displayed as primary navigation items.

- [x] Fetch selected section content on demand with segmented=true task-unit requests.
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/api/client.ts`; API used: `GET /documents/{doc_name}/task-units/{task_unit_id}/content?segmented=true`.
  Notes: Content is fetched only after a section is selected; the UI requests the selected section's task units in backend order.

- [x] Render section content from aggregated content_blocks.
  Evidence: `Deep_Reader_UI/src/components/ReaderContent.tsx`; `Deep_Reader_UI/src/styles.css`.
  Notes: Renders flattened `content_blocks[]` from the selected section's task units in backend response order and does not require duplicated raw content.

- [x] Add initial/loading/error/empty states.
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/components/StateView.tsx`; `Deep_Reader_UI/src/styles.css`.
  Notes: Separate state handling exists for layout and content requests.

- [x] Add basic responsive and accessible interaction behavior.
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/components/DocumentSearch.tsx`; `Deep_Reader_UI/src/components/HierarchyNavigation.tsx`; `Deep_Reader_UI/src/styles.css`.
  Notes: Uses MUI semantic controls, `aria-live`, `aria-busy`, `aria-selected`, focusable section controls, and responsive layout.

- [x] Validate first Reader vertical slice.
  Evidence: `npm run check`; Vite dev server HTTP smoke validation.
  Notes: TypeScript build and Vite production build pass. Dev server smoke validates the UI entry path. Backend API behavior remains an external runtime dependency.

- [x] Refactor Reader UI to modular React, TypeScript, and MUI.
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/api/client.ts`; `Deep_Reader_UI/src/types/api.ts`; `Deep_Reader_UI/src/components/DocumentSearch.tsx`; `Deep_Reader_UI/src/components/HierarchyNavigation.tsx`; `Deep_Reader_UI/src/components/ReaderContent.tsx`; `Deep_Reader_UI/src/components/StateView.tsx`; `Deep_Reader_UI/src/theme.ts`.
  Notes: UI is split by API, type, reader state, document search, hierarchy navigation, content rendering, shared state view, and theme responsibilities.

- [x] Consume backend document list/search API for combo-box candidates.
  Evidence: `Deep_Reader_UI/src/api/client.ts`; `Deep_Reader_UI/src/types/api.ts`; `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/components/DocumentSearch.tsx`.
  Notes: Opening the combo box calls `GET /documents?limit=200`; local typing filters those API-returned options. Load is disabled unless the current value is one of the API-returned options.

- [x] Aggregate section reading content from internal task units.
  Evidence: `Deep_Reader_UI/src/App.tsx`; `Deep_Reader_UI/src/components/HierarchyNavigation.tsx`; `Deep_Reader_UI/src/components/ReaderContent.tsx`; validation: `npm run check`.
  Notes: The UI displays Chapter -> Section navigation, keeps task unit ids inside the selected section, fetches each selected section task unit with `segmented=true`, ignores stale content responses after rapid section changes, and renders the aggregated `content_blocks[]` as one section.

## Needs Confirmation

None.

## Future Task Policy

New future tasks for this module must be added here first as unchecked items.

No coding task should be considered complete unless the corresponding checklist item is updated.
