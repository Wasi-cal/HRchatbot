export type Role = "user" | "assistant" | "error";

export interface Message {
  id: string;
  role: Role;
  text: string;
  /** "refusal": the backend couldn't find it; "voice": a transcript line from a call; "stopped": the user cancelled. */
  kind?: "refusal" | "voice" | "stopped";
  /** For role "error" / kind "stopped": the question to re-send on Retry. */
  retryQuery?: string;
}

export interface Conversation {
  id: string;
  title: string;
  updatedAt: number;
  messages: Message[];
}

export interface PolicyDoc {
  id: string;
  title: string;
  updatedAt: string | null;
}

export const MAX_QUERY_CHARS = 2000; // ChatRequest.query max_length on the backend
export const STARTER_QUESTIONS = [
  "How much annual leave do I get?",
  "What is the remote work policy?",
  "How do I claim expenses?",
  "How do I report harassment?",
];
