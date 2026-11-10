import { LearningFailure, LearningService, sameSettings, type LearningBlock, type LearningCommand, type LearningComponent, type LearningHandle, type LearningRead, type Operation, type Submission } from "../../services/LearningService";
import type { BookPreferences } from "../../services/ReadingPreferencesService";
import { emptyPage, itemKey, phaseFor, recoveryKey, type LearningItem, type LearningPageState, type LearningSessionMemory, type PageSource } from "./model";

type Action = "initial" | "check" | "refresh" | "retry";
const terminal = new Set(["succeeded", "failed", "unknown", "stale", "missing"]);
export class LearningPageSession {
  readonly abort = new AbortController();
  private state = emptyPage();
  private queue: { key: string; action: Action; consent: boolean }[] = [];
  private reserved = new Set<string>();
  private running = 0;
  private started = false;
  constructor(readonly book: BookPreferences, readonly sources: PageSource[], private readonly memory: LearningSessionMemory,
    private readonly publish: (state: LearningPageState) => void, private readonly api = new LearningService(),
    private readonly policy = { concurrency: 2 }) {}
  get signal() { return this.abort.signal; }
  private emit() { if (!this.signal.aborted) this.publish(this.state); }
  private patch(key: string, value: Partial<LearningItem>) {
    if (this.signal.aborted || !this.state.items[key]) return;
    this.state = { ...this.state, items: { ...this.state.items, [key]: { ...this.state.items[key], ...value } } }; this.emit();
  }
  stop() { this.abort.abort(); this.queue = []; this.reserved.clear(); }
  private verify(read: LearningRead) {
    if (!sameSettings(read.settings, this.book) || read.settings.effective_mode !== "language_learning") throw new LearningFailure("read", "settings_changed");
    for (const b of read.blocks) {
      const source = this.sources.find(s => s.id === b.content_block_id), prior = this.state.blocks[b.content_block_id];
      if (!source || !b.sentences || b.sentences.map(s => s.text).join("") !== source.text || (prior && prior.source_token !== b.source_token))
        throw new LearningFailure("read", "source_changed");
    }
  }
  async start() {
    if (this.started || this.signal.aborted) return; this.started = true;
    try {
      const blocks: LearningBlock[] = [];
      for (let i = 0; i < this.sources.length; i += 50) {
        const read = await this.api.read(this.book.doc_name, this.sources.slice(i, i + 50).map(s => s.id), this.signal);
        if (this.signal.aborted) return; this.verify(read); blocks.push(...read.blocks);
      }
      if (this.signal.aborted) return;
      const items: Record<string, LearningItem> = {};
      for (const block of blocks) for (const view of block.components) items[itemKey(block.content_block_id, view.component)] = {
        blockId: block.content_block_id, component: view.component, view, phase: phaseFor(view), message: "",
      };
      this.state = { loading: false, error: "", blocks: Object.fromEntries(blocks.map(b => [b.content_block_id, b])), items }; this.emit();
      for (const key of Object.keys(items)) this.enqueue(key, "initial");
    } catch (e) { if (!this.signal.aborted) { this.state = { ...this.state, loading: false, error: e instanceof LearningFailure ? e.message : "Learning data is unavailable." }; this.emit(); } }
  }
  enqueue(key: string, action: Action, consent = false) {
    if (this.signal.aborted || !this.state.items[key] || this.reserved.has(key)) return;
    this.reserved.add(key); this.patch(key, { busy: true }); this.queue.push({ key, action, consent }); this.pump();
  }
  private pump() {
    if (this.signal.aborted) return;
    while (this.running < this.policy.concurrency && this.queue.length) {
      const work = this.queue.shift()!; this.running++;
      void this.process(work.key, work.action, work.consent).catch(e => {
        if (!this.signal.aborted) this.patch(work.key, { phase: e instanceof LearningFailure && e.kind === "unknown" ? "unknown" : "error",
          message: e instanceof LearningFailure ? e.message : "Learning data is unavailable. Check progress to read it again." });
      }).finally(() => { this.running--; this.reserved.delete(work.key); this.patch(work.key, { busy: false }); this.pump(); });
    }
  }
  private async readOne(key: string, minimumRevision?: number) {
    const item = this.state.items[key];
    const read = await this.api.read(this.book.doc_name, [item.blockId], this.signal);
    if (this.signal.aborted) return null; this.verify(read);
    const b = read.blocks[0], view = b.components.find(v => v.component === item.component)!;
    if (item.view.saved && (!view.saved || view.saved.result_revision < item.view.saved.result_revision)) throw new LearningFailure("read");
    if (minimumRevision !== undefined && (!view.saved || view.saved.result_revision < minimumRevision)) throw new LearningFailure("read");
    this.state = { ...this.state, blocks: { ...this.state.blocks, [item.blockId]: b } };
    this.patch(key, { view, phase: phaseFor(view), message: "", ...(view.saved ? { partial: undefined } : {}) }); return view;
  }
  private verifyOperation(op: Operation, block: LearningBlock, component: LearningComponent) {
    if (op.state !== "missing" && (op.component !== component || op.content_block_id !== block.content_block_id || op.source_token !== block.source_token))
      throw new LearningFailure("read", "operation_binding_mismatch");
  }
  private async partial(key: string, op: Operation) {
    if (!op.completed_batches || op.state === "succeeded" || !op.source_current || !op.settings_current) return;
    const value = await this.api.partial(op.handle, 0, this.signal);
    if (this.signal.aborted) return;
    const item = this.state.items[key], block = this.state.blocks[item.blockId];
    if (value.component !== item.component || value.source_token !== block.source_token) throw new LearningFailure("read");
    const sentences = block.sentences!, ids = sentences.filter(s => s.sentence_id).map(s => s.sentence_id), source = Array.from(sentences.map(s => s.text).join(""));
    let last = -1;
    for (const batch of value.batches) {
      if (batch.component === "translation") for (const row of batch.rows) {
        const index = ids.indexOf(row.sentence_id); if (index <= last) throw new LearningFailure("read"); last = index;
      }
      else for (const hint of batch.items) {
        const s = sentences.find(s => s.sentence_id === hint.sentence_id);
        if (!s || hint.span.start < s.start || hint.span.end > s.end || source.slice(hint.span.start, hint.span.end).join("") !== hint.source_form || hint.language_code !== this.book.source_language) throw new LearningFailure("read");
      }
    }
    this.patch(key, { partial: value });
  }
  private async observe(key: string, h: LearningHandle, reconcile: boolean) {
    const item = this.state.items[key], block = this.state.blocks[item.blockId], memoryKey = recoveryKey(this.book, block, item.component);
    this.patch(key, { awaitingCompletion: true });
    let op = await (reconcile ? this.api.operation(h, this.signal) : this.api.wait(h, this.signal));
    if (this.signal.aborted) return;
    this.verifyOperation(op, block, item.component);
    if (reconcile && !terminal.has(op.state) && op.source_current && op.settings_current && op.owns_current_slot) {
      const response = await this.api.reconcile(h, this.signal); if (this.signal.aborted) return;
      if (response.outcome === "unknown") { this.patch(key, { phase: "unknown", message: "Recovery outcome is unknown. Check progress again." }); return; }
      op = await this.api.wait(h, this.signal); if (this.signal.aborted) return;
      this.verifyOperation(op, block, item.component);
    }
    if (op.state === "missing" || !op.source_current || !op.settings_current || !op.owns_current_slot) {
      this.patch(key, { phase: "stale", awaitingCompletion: false, message: "This operation is no longer current. Read the page again; no new generation was started." }); return;
    }
    this.patch(key, { awaitingCompletion: !terminal.has(op.state) });
    if (op.state === "succeeded") {
      await this.readOne(key, op.result_revision!); if (!this.signal.aborted) this.memory.commands.delete(memoryKey); return;
    }
    const pending = !terminal.has(op.state);
    this.patch(key, { phase: op.state === "claimed" || op.state === "running" ? "pending" : op.state,
      view: { ...this.state.items[key].view, operation: h, operation_state: op.durable_state },
      message: pending ? "Still in progress. Select Check progress to wait again; generation has not been restarted."
        : op.failure_code === "invalid_output" ? "The generated response could not be read. Existing saved content is retained." : "" });
    await this.partial(key, op);
  }

  private async submitted(key: string, response: Submission) {
    const item = this.state.items[key], block = this.state.blocks[item.blockId], mk = recoveryKey(this.book, block, item.component);
    const record = this.memory.commands.get(mk);
    if (record) { record.handle = response.operation; record.uncertain = response.outcome === "unknown"; record.blocked = ["denied", "unavailable"].includes(response.outcome); }
    if (response.outcome === "reuse" || response.outcome === "succeeded") {
      this.patch(key, { awaitingCompletion: false });
      await this.readOne(key, response.result_revision!); if (!this.signal.aborted) this.memory.commands.delete(mk); return;
    }
    if (response.operation) {
      // Keep accepted work observable if the completion request itself fails.
      this.patch(key, { phase: "pending", view: { ...item.view, operation: response.operation } });
      await this.observe(key, response.operation, false); return;
    }
    this.patch(key, { phase: response.outcome === "no_content" ? "empty" : response.outcome === "unknown" ? "unknown" : "unavailable",
      awaitingCompletion: response.outcome === "unknown",
      message: response.outcome === "unknown" ? "The command outcome is unknown. Check progress before retrying." : response.outcome === "no_content" ? "" : "Generation was not accepted. No automatic retry will be made." });
  }
  private async process(key: string, action: Action, consent: boolean) {
    const item = this.state.items[key], block = this.state.blocks[item.blockId], mk = recoveryKey(this.book, block, item.component);
    let view = item.view;
    if (action !== "initial") { const updated = await this.readOne(key); if (!updated || this.signal.aborted) return; view = updated; }
    if (!block.sentences?.some(s => s.sentence_id)) { this.patch(key, { phase: "empty" }); return; }
    const recovery = this.memory.commands.get(mk);
    if (recovery?.uncertain) {
      const found = await this.api.command(this.book.doc_name, recovery.command.command_id, this.signal); if (this.signal.aborted) return;
      if (!found.found) { this.patch(key, { phase: "unknown", message: "The command has no confirmed receipt yet. Check again; it has not been resubmitted." }); return; }
      recovery.handle = found.receipt!.operation; recovery.uncertain = false;
      if (!recovery.handle) { await this.submitted(key, found.receipt!); return; }
    }
    // A current saved read proves this completed result is reusable, even if
    // its historical operation predates a settings roundtrip. Active/failed/
    // unknown refreshes and unresolved commands still require observation.
    const committed = view.state === "ready" && !!view.saved && view.operation_state === "succeeded";
    let h = committed && !recovery ? null : view.operation ?? recovery?.handle;
    if (h) {
      if (action !== "retry" && action !== "refresh") { await this.observe(key, h, action === "check"); return; }
      const op = await this.api.operation(h, this.signal); if (this.signal.aborted) return; this.verifyOperation(op, block, item.component);
      this.patch(key, { awaitingCompletion: !terminal.has(op.state) && op.source_current && op.settings_current && op.owns_current_slot });
      if (action === "refresh" && op.state === "succeeded" && op.source_current && op.settings_current && op.owns_current_slot) {
        h = null; // A confirmed completed operation permits a new explicit regeneration.
      } else {
        if (action === "refresh" || !op.source_current || !op.settings_current || !op.owns_current_slot || !["failed", "unknown", "stale"].includes(op.state)) {
          await this.observe(key, h, true); return;
        }
        if (op.outcome_uncertain && !consent) { this.patch(key, { phase: "unknown", message: "Retry may incur duplicate cost. Explicit confirmation is required." }); return; }
      }
    } else if (action === "check" || (action === "initial" && (view.state !== "missing" || !!recovery))) {
      this.patch(key, { phase: view.state === "missing" ? "unavailable" : phaseFor(view), message: recovery ? "No automatic retry will be made." : "" }); return;
    } else if (action === "initial" && view.state !== "missing") return;
    else if (action === "retry" && view.state !== "missing" && !recovery?.blocked) return;
    const command: LearningCommand = { doc_name: this.book.doc_name, content_block_id: item.blockId, component: item.component,
      command_id: crypto.randomUUID(), expected_source_token: block.source_token,
      expected_book_revision: this.book.revision, expected_workspace_revision: this.book.workspace.revision,
      intent: h && action === "retry" ? "retry" : action === "refresh" ? "regenerate" : "get_or_generate",
      ...(h && action === "retry" ? { retry_operation: h, acknowledge_possible_duplicate_cost: consent } : {}) };
    this.memory.commands.set(mk, { command, handle: null, uncertain: true, blocked: false });
    this.patch(key, { phase: "pending", awaitingCompletion: true, message: "" });
    try { const response = await this.api.generate(command, this.signal); if (!this.signal.aborted) await this.submitted(key, response); }
    catch (e) { if (!this.signal.aborted && e instanceof LearningFailure && e.kind === "rejected") {
      const record = this.memory.commands.get(mk)!; record.uncertain = false; record.blocked = true;
      this.patch(key, { awaitingCompletion: false });
    } throw e; }
  }
}
