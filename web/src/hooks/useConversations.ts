"use client";
import { useCallback, useEffect, useState } from "react";
import type { Conversation, Message } from "@/lib/types";

const KEY = "tessa.conversations.v2";
const MAX_CONVS = 50;

/** Fallback sidebar title (first message) until the model supplies a summary: tidy it and cut at a word boundary. */
function makeTitle(text: string): string {
  const t = text.trim().replace(/\s+/g, " ").replace(/[?!.\s]+$/, "");
  if (t.length <= 48) return t || "New conversation";
  return t.slice(0, 48).replace(/\s+\S*$/, "") + "…";
}

/** Conversations kept in this browser (there is no per-user history API). Loaded after mount to avoid a hydration mismatch. */
export function useConversations() {
  const [convs, setConvs] = useState<Conversation[]>([]);
  const [hydrated, setHydrated] = useState(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(KEY);
      if (raw) setConvs(JSON.parse(raw) as Conversation[]);
    } catch { /* storage blocked or corrupt: start empty */ }
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;  // never write the initial empty list over what is stored
    try { localStorage.setItem(KEY, JSON.stringify(convs.slice(0, MAX_CONVS))); } catch { /* quota / blocked */ }
  }, [convs, hydrated]);

  /** Append to a conversation, creating it if it doesn't exist yet. `title` (the model's summary of the whole conversation) replaces the current one when given. */
  const append = useCallback((convId: string, msg: Message, title?: string | null) => {
    setConvs((all) => {
      const existing = all.find((c) => c.id === convId);
      if (!existing) {
        const title = makeTitle(msg.text);
        return [{ id: convId, title, updatedAt: Date.now(), messages: [msg] }, ...all];
      }
      return all.map((c) => (c.id === convId
        ? { ...c, title: title?.trim() || c.title, updatedAt: Date.now(), messages: [...c.messages, msg] }
        : c));
    });
  }, []);

  const removeMessage = useCallback((convId: string, msgId: string) => {
    setConvs((all) => all.map((c) => (c.id === convId ? { ...c, messages: c.messages.filter((m) => m.id !== msgId) } : c)));
  }, []);

  const remove = useCallback((convId: string) => setConvs((all) => all.filter((c) => c.id !== convId)), []);

  return { convs, hydrated, append, removeMessage, remove };
}
