# Frontend Development Standards

Applies to `Deep_Reader_UI`. Read with [AGENTS.md](../../AGENTS.md) and the
[UI design](../../Deep_Reader_UI/module-detailed-design.md)/[checklist](../../Deep_Reader_UI/module-checklist.md).
These are implementation requirements, not a claim that all existing code complies.

## Structure and Ownership

- Follow the existing lightweight MVC: `model.ts` owns pure transformations/types; `controller.ts` owns state and effects; `view.tsx` renders and dispatches actions; `index.ts` exposes the minimal feature interface.
- Keep `App.tsx` a composition shell. Put API calls in typed services through `RestClient`, not views; keep domain policy out of JSX and avoid oversized controllers or speculative abstractions.
- Backend hierarchy, identities, revisions and operation outcomes remain authoritative. UI state is a projection; reads must not implicitly prepare, reparse, regenerate profiles or start interactions.
- Give each state one owner. Derive values instead of duplicating them; keep transient state local and scope shared/cache state by document, target and relevant revision. Validate required IDs before sending requests.

## Notifications and DOM Identity

- Centralize transient success/warning/error feedback through the shared `AppNotification` presentation contract. Use one active notification surface for the current screen/dialog; deduplicate messages and avoid scattered alert panels or native `window.alert` calls.
- Keep only compact blocking/empty states and field-specific validation inline. Use the established confirmation dialog for destructive actions; make detailed diagnostics available on demand rather than occupying permanent workspace. Notifications must remain accessible inside modal focus boundaries.
- Give every application-controlled interactive element, stateful region and component root a meaningful, stable class, unique ID or semantic label. Repeated items use a feature class plus stable `data-*` identity; never duplicate IDs or depend on generated MUI classes, DOM position or list indices.
- Use native controls and proper labels/accessible names. ARIA describes real semantics, not debug identifiers; use `data-testid` only where semantic selectors are insufficient. Do not embed private content or tokens in DOM attributes.

## Lifecycle and Async Safety

- A component/controller owns its requests and resources. On unmount, dialog close (even when kept mounted), target change or supersession, abort obsolete requests through `AbortSignal` and clear owned timers, polling, listeners, subscriptions and object URLs.
- Cancellation is not enough: check request generation and document/target identity before applying success, failure, notifications, navigation or `finally` updates. Late responses must not overwrite current data, clear newer loading state or reopen closed UI.
- Reset transient selection, menu anchors, errors and pending state when their owner disappears. Protect unsaved drafts before closing/switching; deliberately retained drafts or recovery handles must be identity-scoped and revalidated, never mistaken for current server state.
- Aborting HTTP does not prove a backend mutation stopped or rolled back. For unknown outcomes, reconcile using the implemented operation API before retrying. Background work must have an explicit longer-lived owner, not an accidental surviving component callback.
- Keep render pure and effects cleanup-symmetric, including React Strict Mode. Start mutations from explicit user actions; prevent duplicate submission, handle rejected promises, and do not show expected cancellation as an error.

## Performance and Usability

- For long lists, evaluate pagination and windowed/virtualized rendering; lazy-load expensive content and previews. Infinite loading alone does not bound DOM growth. Preserve stable item keys, selection, focus and scroll position; measure before adding memoization.
- Fetch content on demand and deduplicate/debounce appropriate reads. Do not download whole books for navigation or eagerly mount every drawer, preview and interaction. Release unused media resources.
- Reuse existing MUI components, icons and spacing. Keep controls stable in size and long text scrollable/wrappable; dialogs and panes need bounded height and an explicit scroll container. Verify narrow screens and long-content cases.
- Dialogs need a title, initial focus, contained keyboard navigation, an accessible close action and focus restoration. Preserve dirty-input confirmation; never leave focus on an element that has been removed.
- Show concise user-facing states and actionable failures, not raw JSON, model budgets or provider internals. Distinguish empty, loading, failed, partial and unknown outcomes; neither HTTP 200 nor an aborted request implies business success.

## Verification and Safety

- Test models/controllers and actual browser workflows: rapid target switching, close/reopen, unmount during requests, out-of-order responses, duplicate clicks, failures and draft protection. Assert no stale updates, unhandled rejections or duplicate mutations.
- For UI changes, run type/build checks and relevant browser tests; inspect desktop/mobile screenshots, keyboard behavior, long-list responsiveness and scroll containment. Report skipped checks and distinguish mocked contracts from real-backend acceptance.
- Treat document/LLM output as untrusted text; sanitize permitted rich HTML and URLs. Never ship private API keys in frontend bundles or browser storage. Keep changes scoped and update governed documentation only with evidence.

## References

Lifecycle: [React useEffect](https://react.dev/reference/react/useEffect) and
[AbortController](https://developer.mozilla.org/en-US/docs/Web/API/AbortController).
Accessibility: [WAI-ARIA modal dialog pattern](https://www.w3.org/WAI/ARIA/apg/patterns/dialog-modal/).
