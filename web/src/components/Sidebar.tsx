"use client";
import type { Conversation } from "@/lib/types";
import { CloseIcon, DocIcon, PlusIcon, TrashIcon } from "./Icons";

function dayGroup(ts: number) {
  const days = Math.floor((new Date().setHours(0, 0, 0, 0) - new Date(ts).setHours(0, 0, 0, 0)) / 864e5);
  return days <= 0 ? "Today" : days < 7 ? "Earlier this week" : "Earlier";
}

interface Props {
  convs: Conversation[];
  currentId: string | null;
  open: boolean;  // mobile drawer
  policiesOpen: boolean;
  onNew: () => void;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
  onTogglePolicies: () => void;
  onClose: () => void;
}

export function Sidebar({ convs, currentId, open, policiesOpen, onNew, onOpen, onDelete, onTogglePolicies, onClose }: Props) {
  const sorted = [...convs].sort((a, b) => b.updatedAt - a.updatedAt);
  const rows: React.ReactNode[] = [];
  let lastGroup = "";
  for (const c of sorted) {
    const g = dayGroup(c.updatedAt);
    if (g !== lastGroup) { rows.push(<div key={`g-${g}`} className="h-group">{g}</div>); lastGroup = g; }
    rows.push(
      <div key={c.id} className={`h-item${c.id === currentId ? " active" : ""}`}>
        <button type="button" className="h-row" onClick={() => onOpen(c.id)} title={c.title} aria-current={c.id === currentId ? "page" : undefined}>{c.title}</button>
        <button
          type="button"
          className="h-del"
          aria-label={`Delete conversation: ${c.title}`}
          title="Delete"
          onClick={() => { if (window.confirm("Delete this conversation? This can't be undone.")) onDelete(c.id); }}
        ><TrashIcon size={14} /></button>
      </div>,
    );
  }

  return (
    <>
      {open && <div className="scrim" onClick={onClose} aria-hidden="true" />}
      <aside className={`sidebar${open ? " open" : ""}`} aria-label="Sidebar">
        <div className="side-head">
          <span className="logo"><DocIcon size={15} stroke="#fff" strokeWidth={2.2} /></span>
          <span className="wordmark">Tessa</span>
          <button type="button" className="icon-btn side-close" onClick={onClose} aria-label="Close menu"><CloseIcon /></button>
        </div>
        <button type="button" className="new-btn" onClick={onNew}><PlusIcon size={15} />New conversation</button>
        <nav className="history" aria-label="Conversations">
          {sorted.length === 0 ? <div className="h-empty">No conversations yet.</div> : rows}
        </nav>
        <div className="side-foot">
          <span className="avatar" aria-hidden="true">YO</span>
          <span className="who"><b>You</b><small>Employee</small></span>
          <button type="button" className={`icon-btn${policiesOpen ? " on" : ""}`} onClick={onTogglePolicies} aria-pressed={policiesOpen} title="Policies" aria-label="Policies"><DocIcon size={17} /></button>
        </div>
      </aside>
    </>
  );
}
