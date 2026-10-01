const API_BASE = "/api";

export class RestClient {
  constructor(private readonly apiBase: string = API_BASE) {}

  async requestJson<T>(path: string, options: RequestInit = {}): Promise<T> {
    const response = await fetch(`${this.apiBase}${path}`, {
      ...options,
      headers: {
        "content-type": "application/json",
        ...(options.headers || {}),
      },
    });
    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : await response.text();

    if (!response.ok) {
      throw new Error(this.resolveErrorDetail(payload, response.status));
    }

    return payload as T;
  }

  private resolveErrorDetail(payload: unknown, status: number): string {
    if (!payload || typeof payload !== "object") {
      return `HTTP ${status}`;
    }

    const payloadRecord = payload as Record<string, unknown>;
    for (const field of ["error", "reason", "detail"]) {
      const value = payloadRecord[field];
      if (typeof value === "string" && value.trim()) {
        return value;
      }
      if (Array.isArray(value) && value.length > 0) {
        return value.map(formatErrorItem).join("; ");
      }
    }

    if (Array.isArray(payloadRecord.errors) && payloadRecord.errors.length > 0) {
      return payloadRecord.errors.map(formatErrorItem).join("; ");
    }
    return `HTTP ${status}`;
  }
}

function formatErrorItem(item: unknown): string {
  if (typeof item === "string") {
    return item;
  }
  if (item && typeof item === "object" && "message" in item) {
    const message = (item as { message?: unknown }).message;
    if (typeof message === "string" && message.trim()) {
      return message;
    }
  }
  return JSON.stringify(item);
}
