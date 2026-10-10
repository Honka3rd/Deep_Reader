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

### Reading Interaction UI Design

This section defines the Reader UI behavior for backend reading-interaction features:
insight, quiz, and critical-thinking session. Backend REST routes and public schemas
for generic reading interactions are now exposed for all three interaction families.
The current UI integration wires the first insight vertical slice through
`ReadingInteractionService`; quiz and critical-thinking drawer API calls remain
follow-up frontend wiring work.

Reading interactions are artifact/session interactions attached to backend hierarchy
targets. They must not become frontend hierarchy, task-layout truth, or hidden backend
mutation. The target identity must come from backend-provided ids:

- book/document target: current loaded `doc_name` / document identity
- chapter target: `chapter_id`
- section target: `section_id`
- unit target: `task_unit_id`

Optional parent ids may be carried by future requests for consistency checks, but the UI
must not target interactions by title text or by visible row position alone.

The UI implementation should be decomposed into four large design areas:

- target/action surface: document, chapter, section, and task-unit three-dot menus
- inline insight surface: compact target-local insight placement and statuses
- quiz drawer: read-first quiz rendering, local practice, and generation controls
- critical-thinking drawer: session question, answer, evaluation, and retry flow

The checklist splits these areas into 18 implementation tasks so backend API exposure
can remain narrow and each UI behavior can be validated independently.

#### Interaction Entry Points

Each readable target should expose a compact MUI icon button using the vertical
three-dot `MoreVert` affordance. The button opens a MUI menu with exactly these first
actions:

- `Insights`
- `Quiz`
- `Critical thinking`

Placement:

- book/document action: in the hierarchy/navigation header near the loaded document
  title and existing document-level controls
- chapter action: at the right edge of each chapter row
- section action: at the right edge of each section row
- unit action: at the right edge of a task-unit content group header inside the reader
  content surface

Task units remain internal content-addressing units, not primary navigation nodes. The
unit action is therefore rendered only inside reader content where task-unit grouping is
already available from the selected section content response. The hierarchy navigation
must not become a task-unit tree.

The menu should be target-aware. Opening it records a frontend-local
`interactionTarget` object derived from the current row/group. Selecting a menu item
starts a read-first interaction flow for that target; it must not immediately generate,
refresh, prepare, reparse, or mutate task-layout.

#### Insight Inline Behavior

Insight is expected to be short enough to live inline beneath the target that requested
it. When the user selects `Insights`, the UI should insert an inline insight region
directly below the target row or group and indented one visual level deeper than that
target:

- book/document insight: below the document header or first hierarchy header area,
  indented relative to document controls
- chapter insight: below the chapter row, before its section rows
- section insight: below the section row, before the next sibling section
- unit insight: below the task-unit group header/content block group inside reader
  content

The inline region should be compact, dismissible/collapsible, and status-aware. It
should support:

- loading persisted insight
- missing/not-generated state
- explicit generate
- explicit refresh
- completed insight payload
- insufficient-content state
- stale-target state
- generation-failed state

The inline insight region must not push raw source text, child artifact payloads, or
task-unit content into `/documents/task-layout`. It may show bounded metadata such as
status, generated time, target level, and artifact reference summary when backend API
schemas expose those fields.

Only one inline insight per target should be open at a time. Multiple targets may keep
their own collapsed insight summaries locally, but frontend collapse/open state is not
backend truth and must not be persisted as artifact state.

#### Quiz Drawer Behavior

Quiz interaction needs more space than inline insight because it includes question
items, answer reveal, local answer inputs, generation/refresh controls, and validation
states. Selecting `Quiz` should open a right-side MUI drawer anchored to the current
reader workspace.

The drawer should display the current target breadcrumb, for example:

```text
Book > Chapter > Section > Unit
```

The quiz drawer should use read-first behavior:

- first load persisted quiz artifact for the target through the generic quiz read route
- show `not_generated` without calling LLM
- provide an explicit `Generate quiz` action
- provide an explicit `Refresh quiz` action only when overwriting/regenerating is
  intended
- show completed quiz items with their type-specific UI
- show insufficient-content, stale-target, validation-failed, and generation-failed
  states distinctly

The drawer may support local answer practice before backend answer-submission exists.
Those local answers are UI state only. They must not be described as persisted quiz
results unless a backend endpoint explicitly supports saving or evaluating them.

Existing `/documents/section-quiz` and `/documents/chapter-quiz` routes are legacy
section/chapter quiz generation routes and should not be treated as the full
target-agnostic quiz interaction API. The new drawer should wait for generic
read/generate quiz routes before implementation, or use a clearly marked temporary
compatibility adapter if the maintainer explicitly chooses that bridge later.

#### Critical-Thinking Drawer Behavior

Critical thinking is a session-shaped interaction and should also use the right-side
drawer. It must not be rendered inline because the user needs room for prompt reading,
answer writing, evaluation feedback, and retry controls.

The first-version drawer flow is:

1. Read current/persisted critical-thinking session state for the target through the
   generic critical-thinking read route.
2. If missing, show an explicit `Generate question` action.
3. After question generation, show the single generated question and an answer input.
4. On submit, send one user answer for evaluation through the submit route.
5. If evaluation succeeds, show feedback, score, and suggested refinement.
6. If evaluation fails, preserve the submitted answer and expose `Retry evaluation`.

The UI should recognize these first-version statuses:

- `not_generated`
- `question_generated`
- `insufficient_content`
- `answer_submitted`
- `evaluation_failed`
- `completed`
- `stale_target`

Critical-thinking drawer state may cache typed answer text locally while the drawer is
open, but that draft is not backend truth. Closing the drawer should warn only if there
is unsent local answer text; it should not imply a backend session was created unless
the generation route already succeeded.

#### Shared Interaction State Model

The next UI slice should introduce a dedicated feature folder:

- `src/features/reading-interactions/model.ts`: target identity, status normalization,
  target breadcrumb helpers, inline insight placement helpers, menu action definitions,
  and drawer mode derivation
- `src/features/reading-interactions/controller.ts`: target menu state, inline insight
  state, drawer state, read-first request orchestration, explicit generate/refresh,
  critical-thinking submit/retry, and stale response suppression
- `src/features/reading-interactions/view.tsx`: target action menu, inline insight
  region, quiz drawer, critical-thinking drawer, compact status views
- `src/features/reading-interactions/index.ts`: public feature exports

REST usage is centralized in `ReadingInteractionService` rather than placed directly in
views. The first implementation covers insight read/generate/refresh and maps the
shared backend target/envelope contract into inline insight state. Drawer interactions
for quiz and critical-thinking should extend the same service/controller boundary
without using legacy section/chapter quiz routes as the generic drawer API.

#### Remaining Frontend API Wiring Plan

The frontend API wiring is intentionally split into three major steps:

1. Insight inline API wiring: completed. `ReadingInteractionService` supports
   insight read/generate/refresh, and the inline insight surface maps backend
   analysis envelopes into local view state.
2. Quiz drawer API wiring: remaining. This should be delivered as a drawer-focused
   vertical slice without touching critical-thinking flow.
3. Critical-thinking drawer API wiring: remaining. This should be delivered after or
   separate from quiz wiring because it has a session lifecycle, draft-answer state,
   submit semantics, and retry semantics.

The remaining two major steps are split into smaller implementation tasks:

Quiz drawer API wiring:

- add quiz request/response API types and `ReadingInteractionService` methods for
  `readQuiz`, `generateQuiz`, and `refreshQuiz`
- map `QuizInteractionResponse` into `QuizViewState`, including strict item ids,
  item types, options, answers, explanations, metadata, error/reason fields, and
  status normalization
- wire drawer open to read-first quiz loading while preserving local practice answers
  and reveal toggles as frontend-only state
- wire explicit `Generate quiz` and `Refresh quiz` controls to service calls, keeping
  legacy `/documents/section-quiz` and `/documents/chapter-quiz` out of the drawer API

Critical-thinking drawer API wiring:

- add critical-thinking request/response API types and `ReadingInteractionService`
  methods for read, generate-question, submit-answer, and retry-evaluation
- map `CriticalThinkingSessionResponse` into `CriticalThinkingViewState`, including
  session id, question, submitted answer, evaluation feedback, score, suggested
  refinement, status, metadata, and retryable error state
- wire drawer open to read-first critical-thinking session loading without creating a
  question session from the read path
- wire explicit generate-question, submit-answer, and retry-evaluation controls while
  keeping unsent draft answers frontend-local and sending the shared target object plus
  backend session id for submit/retry calls

Both remaining steps must keep stale response suppression in the controller, keep REST
calls out of view components, keep `/documents/task-layout` unchanged, and avoid
persisting drawer open state, local quiz practice answers, or unsent critical-thinking
draft text as backend truth.

Recommended frontend state:

- `interactionMenuTarget`: current target for the open three-dot menu
- `openInlineInsightsByTarget`: local map of target key to inline insight view state
- `interactionDrawer`: closed / quiz / critical-thinking with target key
- `interactionArtifactsByTarget`: local cache of read responses, keyed by target level
  and id
- `interactionRequestStatus`: read / generate / refresh / submit / retry status
- `draftCriticalThinkingAnswer`: local unsent answer text for the active session

All of this state is frontend-local projection/control state. It must not mutate
backend hierarchy, task-layout, profile diagnostics, parser metadata, or artifact
persistence except through explicit future interaction mutation routes.

#### Frontend Data Structures For Minimal API Exposure

The UI should define a small internal model before backend route wiring begins. This
model is a frontend contract for rendering and request orchestration; it is not a
claim that backend schemas already exist. Future backend APIs can use this model as a
guide for minimal exposure: return enough target identity, status, artifact metadata,
and interaction payload to render the feature, while keeping hierarchy ownership and
artifact persistence in the backend.

Recommended target model:

```ts
type ReadingInteractionTargetLevel =
  | "document"
  | "chapter"
  | "section"
  | "task_unit";

type ReadingInteractionKind =
  | "insight"
  | "quiz"
  | "critical_thinking";

interface ReadingInteractionTarget {
  targetLevel: ReadingInteractionTargetLevel;
  docName: string;
  documentId?: string;
  chapterId?: string;
  sectionId?: string;
  taskUnitId?: string;
  parentChapterId?: string;
  parentSectionId?: string;
  displayTitle: string;
  breadcrumb: string[];
  sourceStructureVersion?: string;
  sourceHash?: string;
}

type ReadingInteractionTargetKey = string;
```

`ReadingInteractionTargetKey` should be a deterministic frontend key derived from
`targetLevel` plus backend ids. It is only a cache/rendering key. It must not become a
backend identifier and must not be derived from mutable display title text alone.

Recommended status model:

```ts
type InteractionStatus =
  | "idle"
  | "loading"
  | "not_generated"
  | "generating"
  | "refreshing"
  | "submitting"
  | "retrying"
  | "completed"
  | "insufficient_content"
  | "stale_target"
  | "generation_failed"
  | "validation_failed"
  | "evaluation_failed";

interface InteractionMetadataView {
  artifactId?: string;
  sessionId?: string;
  generatedAt?: string;
  updatedAt?: string;
  targetLevel: ReadingInteractionTargetLevel;
  referencedArtifactSummary?: Array<{
    artifactId: string;
    artifactType: string;
    targetLevel: ReadingInteractionTargetLevel;
  }>;
}
```

Recommended view state model:

```ts
interface InsightViewState {
  targetKey: ReadingInteractionTargetKey;
  target: ReadingInteractionTarget;
  status: InteractionStatus;
  content?: string;
  metadata?: InteractionMetadataView;
  expanded: boolean;
  errorMessage?: string;
}

interface QuizItemView {
  itemId: string;
  itemType: "short_answer" | "multiple_choice" | "true_false";
  prompt: string;
  options?: string[];
  answer?: string;
  explanation?: string;
}

interface QuizViewState {
  targetKey: ReadingInteractionTargetKey;
  target: ReadingInteractionTarget;
  status: InteractionStatus;
  items: QuizItemView[];
  answersByItemId: Record<string, string>;
  revealByItemId: Record<string, boolean>;
  metadata?: InteractionMetadataView;
  errorMessage?: string;
}

interface CriticalThinkingViewState {
  targetKey: ReadingInteractionTargetKey;
  target: ReadingInteractionTarget;
  status: InteractionStatus;
  question?: string;
  draftAnswer: string;
  submittedAnswer?: string;
  evaluation?: {
    feedback: string;
    score?: number;
    suggestedRefinement?: string;
  };
  metadata?: InteractionMetadataView;
  errorMessage?: string;
}

interface InteractionDrawerState {
  kind: "quiz" | "critical_thinking" | null;
  targetKey?: ReadingInteractionTargetKey;
}
```

Recommended cache/request model:

```ts
interface InteractionArtifactCache {
  insightsByTarget: Record<ReadingInteractionTargetKey, InsightViewState>;
  quizzesByTarget: Record<ReadingInteractionTargetKey, QuizViewState>;
  criticalThinkingByTarget: Record<
    ReadingInteractionTargetKey,
    CriticalThinkingViewState
  >;
}

interface InteractionRequestState {
  targetKey: ReadingInteractionTargetKey;
  kind: ReadingInteractionKind;
  operation: "read" | "generate" | "refresh" | "submit" | "retry";
  status: "pending" | "succeeded" | "failed";
  requestId: string;
}
```

Backend API exposure should stay minimal and target-aware. The UI should need:

- a generic target envelope that echoes `targetLevel`, document identity, backend ids,
  and optional parent ids for consistency checks
- separate read, generate, refresh, submit, and retry operations so read paths never
  trigger LLM work
- normalized status and reason/error fields that map into `InteractionStatus`
- artifact/session metadata such as artifact id, session id, generated time, updated
  time, and backend-owned source version/hash when available
- bounded referenced-artifact summaries, not raw child artifact payloads
- quiz item type and prompt/option/answer/explanation fields needed for rendering
- critical-thinking question, submitted answer, evaluation feedback, optional score,
  and suggested refinement when the session reaches those states

The UI should not require backend APIs to expose raw source text, expanded child
content, full artifact dependency graphs, task-layout mutations, drawer state,
collapsed/expanded state, local quiz practice answers, or unsent critical-thinking
draft text.

#### Interaction API Boundary

Backend routes support separate read and write paths:

- read persisted interaction artifact/session
- generate or refresh insight
- generate or refresh quiz
- generate critical-thinking question session
- submit critical-thinking answer for evaluation
- retry failed critical-thinking evaluation

The UI must not call generation from a read path, and must not hide LLM/cost-bearing
work behind menu open, target selection, document load, section selection, or drawer
open. Explicit user commands such as `Generate`, `Refresh`, `Submit answer`, and `Retry
evaluation` are the only places where interaction mutation should occur.

Frontend API wiring should only use backend-implemented public contracts. The current
`src/types/api.ts` reading-interaction types mirror the implemented insight analysis
target/envelope/payload schema used by `ReadingInteractionService`; quiz and
critical-thinking type/service expansion should be added with the same evidence before
their drawer controls call backend routes.

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

The list below describes the completed first Reader slice, not the next reading
interaction design slice.

- document upload
- document library
- summary UI
- quiz/critical-thinking backend mutation from drawer open or read paths
- quiz or critical-thinking drawer API calls before their planned service/controller
  wiring tasks are implemented
- free QA
- annotation
- text selection
- content-block highlight
- ask-about-selection
- artifact creation outside explicit future interaction routes
- artifact persistence UI outside explicit future interaction routes
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
