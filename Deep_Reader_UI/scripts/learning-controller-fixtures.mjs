// Isolated transport fixtures, shared by controller and actual-App browser acceptance.
export const selection = { mode_id: "language_learning", version: "1", language_code: "en", scheme_id: "cefr", scheme_version: "1", level_id: "A1" };
export function book(doc = "Book", mode = "language_learning") {
 const unavailable = mode === "future";
 const reasons = unavailable ? [{ code: "mode_unknown", field: "active_mode", message: "Unsupported mode" }] : [];
 return { doc_name: doc, revision: 1, workspace: { revision: 1, native_language: "zh" }, active_mode: mode,
  effective_mode: unavailable ? null : mode, source_language_override: null, source_language: "en", source_origin: "profile", active_reasons: reasons,
  modes: ["normal", "language_learning"].map(mode_id => ({ mode_id, label: mode_id, description: mode_id, settings_version: "1", eligible: true, reasons: [] })),
  saved_settings: [{ mode_id: "language_learning", version: "1", supported: true, selection: { ...selection } }] };
}
export function block(id = "b1", content = "Hello world.") {
 return { content_block_id: id, source_token: "a".repeat(64), source_length: Array.from(content).length,
  sentences: content ? [{ sentence_id: content.trim() ? `${id}:s1` : null, start: 0, end: Array.from(content).length, text: content }] : [],
  components: ["translation", "hints"].map(component => ({ component, state: "missing", saved: null, operation: null, operation_state: null })) };
}
export function save(b, component, revision = 1) {
 const view = b.components.find(c => c.component === component);
 view.saved = { component, result_revision: revision, source_language: "en", explanation_language: "zh",
  ...(component === "translation" ? { rows: b.sentences.filter(s => s.sentence_id).map(s => ({ sentence_id: s.sentence_id, translated_text: "你好世界。" })) }
   : { proficiency: { ...selection }, items: [] }) }; view.state = "ready";
 return view;
}
export function operation(b, component, state = "running", id = `${b.content_block_id}:${component}`) {
 return { handle: { doc_name: "Book", operation_id: id }, state, durable_state: state, component, content_block_id: b.content_block_id, source_token: b.source_token,
  source_current: true, settings_current: true, owns_current_slot: true, completed_batches: state === "succeeded" || state === "partial" ? 1 : 0,
  total_batches: 1, result_revision: state === "succeeded" ? 1 : null, outcome_uncertain: state === "running" || state === "unknown", failure_code: null };
}
export function attach(b, op) { const view = b.components.find(c => c.component === op.component); view.operation = op.handle; view.operation_state = op.state;
 if (!view.saved) view.state = ["running", "claimed"].includes(op.state) ? "pending" : op.state === "succeeded" ? "unavailable" : op.state; }
export function submission(command, outcome = "claimed", handle = { doc_name: command.doc_name, operation_id: command.command_id }) {
 return { doc_name: command.doc_name, command_id: command.command_id, outcome, operation: ["reuse", "no_content", "denied", "missing"].includes(outcome) ? null : handle,
  replay: false, reason: outcome === "denied" ? "capacity" : null, result_revision: ["reuse", "succeeded"].includes(outcome) ? 1 : null };
}
