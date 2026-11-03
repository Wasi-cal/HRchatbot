export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(path, init);
  let body: unknown = null;
  try { body = await res.json(); } catch { /* non-JSON body */ }
  if (!res.ok) {
    const d = (body as { detail?: unknown } | null)?.detail;
    const detail = typeof d === "string" ? d : d ? JSON.stringify(d) : res.statusText;
    throw new ApiError(res.status, detail || `HTTP ${res.status}`);
  }
  return body as T;
}

export interface ChatReply { type: string; text: string; title?: string | null }

export function postChat(query: string, history: { role: string; content: string }[], signal: AbortSignal) {
  return api<ChatReply>("/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ query, history }),
    signal,
  });
}

/** A human-readable reason a request failed (never a stack trace or raw JSON). */
export function friendlyError(err: unknown): string {
  if (err instanceof ApiError) {
    if (err.status === 429) return "Too many requests right now. Please wait a moment and try again.";
    if (err.status >= 500) return "Something went wrong on our side. Please try again.";
    return `The request was rejected (${err.status}). ${err.message}`;
  }
  if (typeof navigator !== "undefined" && !navigator.onLine) return "You appear to be offline. Check your connection and try again.";
  return "Couldn't reach the assistant. Please check your connection and try again.";
}

export const newId = () =>
  typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
