"use client";
import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { MAX_QUERY_CHARS } from "@/lib/types";
import { MicIcon, SendIcon, StopIcon } from "./Icons";

export interface MessageBoxHandle { focus: () => void }

interface Props {
  onSend: (text: string) => void;
  onVoice: () => void;
  onStop: () => void;
  busy: boolean;
  id: string;
}

/** Textarea that grows to ~6 lines. Enter sends, Shift+Enter adds a line; Enter during IME composition does neither. */
export const MessageBox = forwardRef<MessageBoxHandle, Props>(function MessageBox({ onSend, onVoice, onStop, busy, id }, ref) {
  const [value, setValue] = useState("");
  const area = useRef<HTMLTextAreaElement>(null);
  useImperativeHandle(ref, () => ({ focus: () => area.current?.focus() }));

  useEffect(() => {
    const el = area.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 150)}px`;
  }, [value]);

  const submit = () => {
    const text = value.trim();
    if (!text || busy) return;
    onSend(text);
    setValue("");
  };
  const near = value.length > MAX_QUERY_CHARS * 0.85;

  return (
    <form className="msgbox" onSubmit={(e) => { e.preventDefault(); submit(); }}>
      <label htmlFor={id} className="sr-only">Ask about a company policy</label>
      <textarea
        id={id}
        ref={area}
        rows={1}
        value={value}
        maxLength={MAX_QUERY_CHARS}
        placeholder="Ask about a company policy…"
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); submit(); }
        }}
      />
      {near && <span className="counter" aria-live="polite">{value.length}/{MAX_QUERY_CHARS}</span>}
      <button type="button" className="icon-btn mic-btn" onClick={onVoice} aria-label="Start voice mode" title="Voice mode"><MicIcon /></button>
      {busy ? (
        <button type="button" className="send-btn stop" onClick={onStop} aria-label="Stop generating" title="Stop"><StopIcon size={14} /></button>
      ) : (
        <button type="submit" className="send-btn" disabled={!value.trim()} aria-label="Send message" title="Send"><SendIcon size={15} /></button>
      )}
    </form>
  );
});
