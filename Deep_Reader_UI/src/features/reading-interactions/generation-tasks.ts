/** Mounted reader-session ownership, separate from the currently visible surface. */
export class InteractionGenerationTasks {
  private readonly documents = new Map<string, {
    epoch: number;
    tasks: Map<string, { promise: Promise<unknown>; controller: AbortController; failed: boolean }>;
  }>();

  sourceEpoch(docName: string) { return this.documents.get(docName)?.epoch ?? 0; }

  private document(docName: string) {
    let document = this.documents.get(docName);
    if (!document) {
      document = { epoch: 0, tasks: new Map() };
      this.documents.set(docName, document);
    }
    return document;
  }

  find<T>(docName: string, key: string, explicitRetry: boolean): Promise<T> | undefined {
    const tasks = this.documents.get(docName)?.tasks;
    const task = tasks?.get(key);
    if (explicitRetry && task?.failed) { tasks!.delete(key); return; }
    return task?.promise as Promise<T> | undefined;
  }

  start<T>(docName: string, key: string, generate: (signal: AbortSignal) => Promise<T>): Promise<T> {
    const existing = this.find<T>(docName, key, false);
    if (existing) return existing;
    const { tasks } = this.document(docName);
    const controller = new AbortController();
    const task = { controller, failed: false, promise: Promise.resolve().then(() => generate(controller.signal)) };
    tasks.set(key, task);
    // Observe both outcomes even after the visible observer has detached.
    void task.promise.then(() => {
      if (tasks.get(key) === task) tasks.delete(key);
    }, () => { task.failed = true; });
    return task.promise;
  }

  /** Explicit source changes invalidate this book's aggregates, not other books. */
  invalidateSources(docName: string) {
    const document = this.document(docName);
    document.epoch += 1;
    for (const task of document.tasks.values()) task.controller.abort();
    document.tasks.clear();
  }

  dispose() {
    for (const docName of this.documents.keys()) this.invalidateSources(docName);
  }
}
