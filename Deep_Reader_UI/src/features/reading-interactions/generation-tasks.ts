/** Mounted reader-session ownership, separate from the currently visible surface. */
export class InteractionGenerationTasks {
  private epoch = 0;
  get sourceEpoch() { return this.epoch; }
  private readonly tasks = new Map<string, { promise: Promise<unknown>; controller: AbortController; failed: boolean }>();

  find<T>(key: string, explicitRetry: boolean): Promise<T> | undefined {
    const task = this.tasks.get(key);
    if (explicitRetry && task?.failed) { this.tasks.delete(key); return; }
    return task?.promise as Promise<T> | undefined;
  }

  start<T>(key: string, generate: (signal: AbortSignal) => Promise<T>): Promise<T> {
    const existing = this.find<T>(key, false);
    if (existing) return existing;
    const controller = new AbortController();
    const task = { controller, failed: false, promise: Promise.resolve().then(() => generate(controller.signal)) };
    this.tasks.set(key, task);
    // Observe both outcomes even after the visible observer has detached.
    void task.promise.then(() => {
      if (this.tasks.get(key) === task) this.tasks.delete(key);
    }, () => { task.failed = true; });
    return task.promise;
  }

  /** A source/layout reset invalidates all aggregate targets, even without hashes. */
  invalidateSources() {
    this.epoch += 1;
    for (const task of this.tasks.values()) task.controller.abort();
    this.tasks.clear();
  }

  dispose() { this.invalidateSources(); }
}
