"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useConversations } from "@/hooks/useConversations";
import { useVoice } from "@/hooks/useVoice";
import { friendlyError, newId, postChat } from "@/lib/api";
import { STARTER_QUESTIONS, type Message } from "@/lib/types";
import { MenuIcon, DocIcon } from "./Icons";
import { MessageBox, type MessageBoxHandle } from "./MessageBox";
import { MessageList } from "./MessageList";
import { PoliciesPanel } from "./PoliciesPanel";
import { Sidebar } from "./Sidebar";
import { VoiceView } from "./VoiceView";

const HISTORY_TURNS = 6;  // the server caps and truncates this again; the client can edit it

export function TessaApp() {
  const { convs, hydrated, append, removeMessage, remove } = useConversations();
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [view, setView] = useState<"chat" | "voice">("chat");
  const [menuOpen, setMenuOpen] = useState(false);
  const [policiesOpen, setPoliciesOpen] = useState(false);
  const [pendingIds, setPendingIds] = useState<string[]>([]);
  const [online, setOnline] = useState(true);
  const inflight = useRef(new Map<string, AbortController>());
  const startBox = useRef<MessageBoxHandle>(null);
  const chatBox = useRef<MessageBoxHandle>(null);

  const convsRef = useRef(convs);
  convsRef.current = convs;
  const current = useMemo(() => convs.find((c) => c.id === currentId) ?? null, [convs, currentId]);
  const showStart = view === "chat" && !current;
  const busy = currentId !== null && pendingIds.includes(currentId);

  // ------------------------------------------------------------------ text chat
  const send = useCallback(async (text: string, opts: { convId?: string; retry?: boolean } = {}) => {
    const convId = opts.convId ?? currentId ?? newId();
    setCurrentId(convId);
    setView("chat");
    setMenuOpen(false);
    if (!opts.retry) append(convId, { id: newId(), role: "user", text });

    const prior = convsRef.current.find((c) => c.id === convId)?.messages ?? [];
    const history = prior
      .filter((m) => (m.role === "user" || m.role === "assistant") && m.kind !== "voice")
      .map((m) => ({ role: m.role, content: m.text }));
    if (history.length && history[history.length - 1].role === "user" && history[history.length - 1].content === text) history.pop();  // retry: the question is already there

    const ctl = new AbortController();
    inflight.current.get(convId)?.abort();
    inflight.current.set(convId, ctl);
    setPendingIds((p) => (p.includes(convId) ? p : [...p, convId]));
    try {
      const res = await postChat(text, history.slice(-HISTORY_TURNS), ctl.signal);
      append(convId, { id: newId(), role: "assistant", text: res.text, kind: res.type === "refusal_shortcut" ? "refusal" : undefined }, res.title);
    } catch (err) {
      if (ctl.signal.aborted) append(convId, { id: newId(), role: "error", kind: "stopped", text: "Response stopped.", retryQuery: text });
      else append(convId, { id: newId(), role: "error", text: friendlyError(err), retryQuery: text });
    } finally {
      if (inflight.current.get(convId) === ctl) inflight.current.delete(convId);
      setPendingIds((p) => p.filter((id) => id !== convId));
    }
  }, [currentId, append]);

  const stopGenerating = useCallback(() => { if (currentId) inflight.current.get(currentId)?.abort(); }, [currentId]);

  const retry = useCallback((msg: Message) => {
    if (!currentId || !msg.retryQuery) return;
    removeMessage(currentId, msg.id);
    void send(msg.retryQuery, { convId: currentId, retry: true });
  }, [currentId, removeMessage, send]);

  // ------------------------------------------------------------------ voice
  const voiceConv = useRef<string | null>(null);
  const voice = useVoice({
    onTranscript: (role, text) => {
      const id = voiceConv.current;
      if (id) append(id, { id: newId(), role, text, kind: "voice" });
    },
  });

  const enterVoice = useCallback(() => {
    voiceConv.current = currentId ?? newId();
    setCurrentId(voiceConv.current);
    setView("voice");
    setMenuOpen(false);
    void voice.start();
  }, [currentId, voice]);

  const leaveVoice = useCallback(() => {
    voice.stop();
    setView("chat");
  }, [voice]);

  useEffect(() => {
    if (view !== "voice") return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") leaveVoice(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [view, leaveVoice]);

  // ------------------------------------------------------------------ navigation
  const newConversation = useCallback(() => {
    if (view === "voice") leaveVoice();
    setCurrentId(null);
    setView("chat");
    setMenuOpen(false);
  }, [view, leaveVoice]);

  const openConversation = useCallback((id: string) => {
    if (view === "voice") leaveVoice();
    setCurrentId(id);
    setView("chat");
    setMenuOpen(false);
  }, [view, leaveVoice]);

  const deleteConversation = useCallback((id: string) => {
    inflight.current.get(id)?.abort();
    remove(id);
    if (id === currentId) setCurrentId(null);
  }, [currentId, remove]);

  // Move focus to the message box when the screen it lives on appears.
  useEffect(() => {
    if (view !== "chat" || !hydrated) return;
    (showStart ? startBox : chatBox).current?.focus();
  }, [view, showStart, currentId, hydrated]);

  useEffect(() => {
    const sync = () => setOnline(navigator.onLine);
    sync();
    window.addEventListener("online", sync);
    window.addEventListener("offline", sync);
    return () => { window.removeEventListener("online", sync); window.removeEventListener("offline", sync); };
  }, []);

  return (
    <div className="app">
      <a className="skip" href="#main">Skip to conversation</a>
      <Sidebar
        convs={convs}
        currentId={currentId}
        open={menuOpen}
        policiesOpen={policiesOpen}
        onNew={newConversation}
        onOpen={openConversation}
        onDelete={deleteConversation}
        onTogglePolicies={() => setPoliciesOpen((v) => !v)}
        onClose={() => setMenuOpen(false)}
      />

      <main className="main" id="main">
        <div className="topbar">
          <button type="button" className="icon-btn" onClick={() => setMenuOpen(true)} aria-label="Open menu"><MenuIcon size={18} /></button>
          <span className="topbar-title" aria-hidden="true" />
          <button type="button" className="icon-btn" onClick={() => setPoliciesOpen((v) => !v)} aria-label="Policies" aria-pressed={policiesOpen}><DocIcon size={17} /></button>
        </div>
        {!online && <div className="offline" role="status">You&apos;re offline. Messages will fail until your connection is back.</div>}

        {showStart && (
          <section className="start" aria-label="New conversation">
            <div className="start-inner">
              <h1>Ask about any company policy</h1>
              <p>Answers come straight from official company policies, always the latest version.</p>
              <MessageBox id="start-input" ref={startBox} onSend={(t) => void send(t)} onVoice={enterVoice} onStop={stopGenerating} busy={busy} />
              <div className="chips">
                {STARTER_QUESTIONS.map((q) => <button key={q} type="button" className="chip" onClick={() => void send(q)}>{q}</button>)}
              </div>
              <p className="disclosure">Tessa is an AI assistant and can make mistakes. It only answers from the policy library, so check anything important with People Ops.</p>
            </div>
          </section>
        )}

        {view === "chat" && current && (
          <section className="chat" aria-label="Conversation">
            <MessageList messages={current.messages} pending={busy} onRetry={retry} onAsk={(t) => void send(t)} />
            <div className="dock">
              <MessageBox id="chat-input" ref={chatBox} onSend={(t) => void send(t)} onVoice={enterVoice} onStop={stopGenerating} busy={busy} />
              <p className="disclosure">Tessa is an AI assistant and can make mistakes. Check anything important with People Ops.</p>
            </div>
          </section>
        )}

        {view === "voice" && (
          <VoiceView
            status={voice.status}
            error={voice.error}
            caption={voice.caption}
            sample={voice.sample}
            onToggle={() => (voice.status === "idle" ? void voice.start() : voice.stop())}
            onBack={leaveVoice}
            onEnd={leaveVoice}
          />
        )}
      </main>

      {policiesOpen && <PoliciesPanel onClose={() => setPoliciesOpen(false)} />}
    </div>
  );
}
