"use client";
import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { PolicyDoc } from "@/lib/types";
import { CloseIcon, DocIcon } from "./Icons";

interface RawDoc { id: string; title: string; updated_at: string | null }

export function PoliciesPanel({ onClose }: { onClose: () => void }) {
  const [docs, setDocs] = useState<PolicyDoc[] | null>(null);
  const [error, setError] = useState(false);

  const load = useCallback(() => {
    setError(false);
    setDocs(null);
    api<RawDoc[]>("/api/policies")
      .then((rows) => setDocs(rows.map((d) => ({ id: d.id, title: d.title, updatedAt: d.updated_at }))))
      .catch(() => setError(true));
  }, []);
  useEffect(load, [load]);

  return (
    <aside className="policies" aria-label="Policies">
      <header>
        <div><h3>Policies</h3><p>The policies Tessa answers from</p></div>
        <button type="button" className="icon-btn" onClick={onClose} aria-label="Close policies"><CloseIcon /></button>
      </header>
      <div className="policy-list">
        {error && <div className="h-empty">Couldn&apos;t load the policy list. <button type="button" className="linkish" onClick={load}>Try again</button></div>}
        {!error && docs === null && [0, 1, 2].map((i) => <div key={i} className="skeleton" />)}
        {docs?.length === 0 && <div className="h-empty">No policies have been added yet.</div>}
        {docs?.map((d) => (
          <div key={d.id} className="policy">
            <DocIcon size={17} />
            <div>
              <b>{d.title}</b>
              {d.updatedAt && <small>Updated {new Date(d.updatedAt).toLocaleDateString(undefined, { month: "short", year: "numeric" })}</small>}
            </div>
          </div>
        ))}
      </div>
    </aside>
  );
}
