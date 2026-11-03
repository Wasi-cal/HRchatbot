"use client";
import { useEffect, useRef, useState } from "react";
import { ChatIcon, PhoneIcon } from "./Icons";
import type { VoiceStatus } from "@/hooks/useVoice";

type Mode = "idle" | "listening" | "speaking";

const N = 14, CTR = (N - 1) / 2, MAXD = Math.SQRT2 * CTR, BARS = 28;
const CELLS: { r: number; c: number; dist: number; phase: number }[] = [];
for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
  const dist = Math.hypot(r - CTR, c - CTR) / MAXD;
  if (dist <= 0.74) CELLS.push({ r, c, dist, phase: (r + c) * 0.035 });
}
const FROM = [0xc7, 0xd2, 0xfe], TO = [0x4f, 0x46, 0xe5];
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const colorAt = (t: number) => `rgb(${FROM.map((v, i) => Math.round(lerp(v, TO[i], t))).join(",")})`;
const STATUS_TEXT: Record<Mode, string> = { idle: "Tap to talk", listening: "Listening…", speaking: "Tessa is speaking" };

interface Props {
  status: VoiceStatus;
  error: string | null;
  caption: string;
  sample: () => { speak: number; mic: number; agentTalking: boolean };
  onToggle: () => void;
  onBack: () => void;
  onEnd: () => void;
}

export function VoiceView({ status, error, caption, sample, onToggle, onBack, onEnd }: Props) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const bars = useRef<(HTMLElement | null)[]>([]);
  const [mode, setMode] = useState<Mode>("idle");
  const live = status === "live";
  const sampleRef = useRef(sample);
  sampleRef.current = sample;
  const liveRef = useRef(live);
  liveRef.current = live;

  useEffect(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0, level = 0, scale = 1, curMode: Mode = "idle", speakUntil = 0;
    const frame = (t: number) => {
      raf = requestAnimationFrame(frame);
      const now = t / 1000;
      const s = sampleRef.current();
      let next: Mode = "idle";
      if (liveRef.current) {
        if (s.speak > 0.008) speakUntil = performance.now() + 350;
        next = s.speak > 0.008 || (s.agentTalking && performance.now() < speakUntil) ? "speaking" : "listening";
      }
      if (next !== curMode) { curMode = next; setMode(next); }

      const raw = curMode === "speaking" ? Math.min(1, s.speak * 4) : curMode === "listening" ? Math.min(1, s.mic * 3) : 0;
      level = lerp(level, raw, raw > level ? 0.5 : 0.15);
      let target = reduce ? 1 : 1 + 0.04 * (0.5 + 0.5 * Math.sin((now * 2 * Math.PI) / 5));
      if (curMode === "listening") target = 1.12;
      else if (curMode === "speaking") target = 1.05 + level * 0.5;
      scale = lerp(scale, target, 0.2);

      const cv = canvas.current;
      if (cv) {
        const size = cv.offsetWidth, dpr = window.devicePixelRatio || 1;
        if (size && cv.width !== Math.round(size * dpr)) cv.width = cv.height = Math.round(size * dpr);
        const g = cv.getContext("2d");
        if (g && size) {
          g.setTransform(dpr, 0, 0, dpr, 0, 0);
          g.clearRect(0, 0, size, size);
          const gap = 2, cell = (size - gap * (N - 1)) / N;
          for (const p of CELLS) {
            const center = 1 - p.dist;
            let a: number;
            if (curMode === "speaking") a = 0.3 + center * 0.6 * (0.45 + level * 0.55) + level * center * 0.2;
            else if (curMode === "idle") a = reduce ? 0.3 + center * 0.6 : ((0.3 + center * 0.6) * (0.55 + 0.4 * (0.5 + 0.5 * Math.sin(((now - p.phase) * 2 * Math.PI) / 2.6)))) / 0.75;
            else a = 0.3 + center * 0.6;
            g.globalAlpha = Math.max(0, Math.min(1, a));
            g.fillStyle = colorAt(p.dist / 0.74);
            g.beginPath();
            g.roundRect(p.c * (cell + gap), p.r * (cell + gap), cell, cell, 2);
            g.fill();
          }
          g.globalAlpha = 1;
        }
        cv.style.transform = `scale(${scale.toFixed(3)})`;
      }
      bars.current.forEach((b, i) => {
        if (!b) return;
        if (curMode === "idle") { b.style.transform = ""; return; }
        const shape = 0.35 + 0.65 * Math.abs(Math.sin(now * 6 + i * 0.7)) * (1 - Math.abs(i - BARS / 2) / BARS);
        b.style.transform = `scaleY(${(0.15 + level * 1.4 * shape).toFixed(3)})`;
      });
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, []);

  const label = status === "connecting" ? "Connecting…" : STATUS_TEXT[mode];

  return (
    <section className="voice" aria-label="Voice mode">
      <div className="voice-label">Voice mode</div>
      <div className="voice-stage">
        <button type="button" className="orb-wrap" onClick={onToggle} disabled={status === "connecting"} aria-label={live ? "Stop voice session" : "Start voice session"}>
          <span className="halo" aria-hidden="true" />
          <canvas ref={canvas} className="orb" aria-hidden="true" />
        </button>
        <div className={`wave${mode === "idle" ? " idle" : ""}`} aria-hidden="true">
          {Array.from({ length: BARS }, (_, i) => <i key={i} ref={(el) => { bars.current[i] = el; }} style={{ animationDelay: `${(i * 0.045).toFixed(3)}s` }} />)}
        </div>
        <div className="v-status" role="status" aria-live="polite">{label}</div>
        <div className={`v-caption${error ? " err" : ""}`} role={error ? "alert" : undefined} aria-live={error ? undefined : "off"}>{error ?? caption}</div>
      </div>
      <div className="voice-controls">
        <button type="button" className="back-btn" onClick={onBack} aria-label="Back to chat (ends the call)" title="Back to chat"><ChatIcon size={18} /></button>
        <button type="button" className="end-btn" onClick={onEnd} aria-label="End call" title="End call"><PhoneIcon size={19} style={{ transform: "rotate(135deg)" }} /></button>
      </div>
    </section>
  );
}
