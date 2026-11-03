"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import type { Message } from "@/lib/types";
import { STARTER_QUESTIONS } from "@/lib/types";
import { ArrowDownIcon, CheckIcon, CopyIcon, DocIcon, RetryIcon } from "./Icons";

function Avatar() { return <span className="t-avatar" aria-hidden="true"><DocIcon size={13} stroke="#fff" strokeWidth={2.4} /></span>; }

function Pending() {
  const [slow, setSlow] = useState(false);
  useEffect(() => { const t = setTimeout(() => setSlow(true), 8000); return () => clearTimeout(t); }, []);
  return (
    <div className="row-msg assistant">
      <Avatar />
      <div className="bubble pending" role="status">
        <span className="dots" aria-hidden="true"><span>•</span><span>•</span><span>•</span></span>
        <span className="sr-only">Tessa is typing</span>
        {slow && <span className="slow">Still working on it…</span>}
      </div>
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      type="button"
      className="mini-btn"
      onClick={async () => {
        try { await navigator.clipboard.writeText(text); setDone(true); setTimeout(() => setDone(false), 1500); } catch { /* clipboard blocked */ }
      }}
      aria-label={done ? "Copied" : "Copy answer"}
      title="Copy"
    >
      {done ? <CheckIcon size={13} /> : <CopyIcon size={13} />}
      <span>{done ? "Copied" : "Copy"}</span>
    </button>
  );
}

interface Props {
  messages: Message[];
  pending: boolean;
  onRetry: (msg: Message) => void;
  onAsk: (text: string) => void;
}

/** role="log" list. Stays pinned to the bottom unless the reader has scrolled up; then offers "Jump to latest". */
export function MessageList({ messages, pending, onRetry, onAsk }: Props) {
  const box = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const [showJump, setShowJump] = useState(false);

  const toBottom = useCallback((smooth = false) => {
    const el = box.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: smooth ? "smooth" : "auto" });
  }, []);

  const onScroll = () => {
    const el = box.current;
    if (!el) return;
    pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    setShowJump(!pinned.current);
  };

  useEffect(() => { if (pinned.current) toBottom(); else setShowJump(true); }, [messages, pending, toBottom]);
  useEffect(() => { pinned.current = true; toBottom(); }, [toBottom]);

  const last = messages[messages.length - 1];
  const lastRefused = !pending && last?.kind === "refusal";

  return (
    <div className="messages-wrap">
      <div className="messages" ref={box} onScroll={onScroll} role="log" aria-live="polite" aria-relevant="additions" aria-label="Conversation" tabIndex={0}>
        {messages.map((m) => (
          <div key={m.id} className={`row-msg ${m.role}`}>
            {m.role === "assistant" && <Avatar />}
            <div className="bubble-col">
              <div className={`bubble${m.kind === "refusal" ? " refusal" : ""}${m.role === "error" ? " err" : ""}`} role={m.role === "error" ? "alert" : undefined}>
                {m.role === "assistant" ? <ReactMarkdown>{m.text}</ReactMarkdown> : m.text}
              </div>
              {m.role === "assistant" && m.kind !== "voice" && (
                <div className="bubble-actions"><CopyButton text={m.text} /></div>
              )}
              {m.role === "error" && m.retryQuery && (
                <div className="bubble-actions">
                  <button type="button" className="mini-btn strong" onClick={() => onRetry(m)}><RetryIcon size={13} /><span>Retry</span></button>
                </div>
              )}
              {m.kind === "voice" && <div className="voice-tag">Voice</div>}
            </div>
          </div>
        ))}
        {pending && <Pending />}
        {lastRefused && (
          <div className="recovery">
            <p>Try one of these, or contact People Ops for anything not covered by a policy:</p>
            <div className="chips">
              {STARTER_QUESTIONS.map((q) => <button key={q} type="button" className="chip" onClick={() => onAsk(q)}>{q}</button>)}
            </div>
          </div>
        )}
      </div>
      {showJump && (
        <button type="button" className="jump" onClick={() => { pinned.current = true; setShowJump(false); toBottom(true); }}>
          <ArrowDownIcon size={14} /> Jump to latest
        </button>
      )}
    </div>
  );
}
