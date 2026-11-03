"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { CAPTURE_SRC, PLAYER_SRC } from "@/lib/worklets";

/* eslint-disable @typescript-eslint/no-explicit-any */
// Browser -> Deepgram Voice Agent directly (wss), authenticated with a short-lived token minted by our backend.
// There is deliberately NO fallback to a raw API key here.

export type VoiceStatus = "idle" | "connecting" | "live";

interface AgentSettings {
  audio: { input: { encoding: string; sample_rate: number }; output: { encoding: string; sample_rate: number } };
  [k: string]: unknown;
}

interface Session {
  ws: WebSocket;
  ctx: AudioContext;
  stream: MediaStream | null;
  capture: AudioWorkletNode | null;
  player: AudioWorkletNode | null;
  outAn: AnalyserNode | null;
  micAn: AnalyserNode | null;
  aligner: any;
  closing: boolean;
}

let audioCoreReady: Promise<string> | null = null;
/** Loads public/audio_core.js onto window.AudioCore and returns its source text (the worklet needs it too). */
function loadAudioCore(): Promise<string> {
  if (!audioCoreReady) {
    audioCoreReady = (async () => {
      const src = await (await fetch("/audio_core.js")).text();
      if (!(window as any).AudioCore) new Function(src).call(window);  // UMD: sets window.AudioCore when `module` is undefined
      if (!(window as any).AudioCore) throw new Error("audio core failed to load");
      return src;
    })().catch((e) => { audioCoreReady = null; throw e; });
  }
  return audioCoreReady;
}

function micErrorMessage(err: unknown): string {
  const name = (err as { name?: string })?.name;
  if (name === "NotAllowedError" || name === "SecurityError") return "Microphone access is blocked. Allow it in your browser's address bar, then tap the orb again.";
  if (name === "NotFoundError" || name === "OverconstrainedError") return "No microphone was found. Connect one and try again.";
  if (name === "NotReadableError") return "Your microphone is in use by another app. Close it and try again.";
  return "Couldn't start the microphone.";
}

const makeAnalyser = (ctx: AudioContext) => {
  const an = ctx.createAnalyser();
  an.fftSize = 512;
  an.smoothingTimeConstant = 0.6;
  return an;
};

export function useVoice(opts: { onTranscript: (role: "user" | "assistant", text: string) => void }) {
  const [status, setStatus] = useState<VoiceStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [caption, setCaption] = useState("");
  const session = useRef<Session | null>(null);
  const agentTalking = useRef(false);
  const hadError = useRef(false);
  const onTranscript = useRef(opts.onTranscript);
  useEffect(() => { onTranscript.current = opts.onTranscript; });
  const bins = useRef<Uint8Array<ArrayBuffer> | null>(null);

  const rms = (an: AnalyserNode | null) => {
    if (!an) return 0;
    if (!bins.current || bins.current.length !== an.fftSize) bins.current = new Uint8Array(an.fftSize);
    an.getByteTimeDomainData(bins.current);
    let sum = 0;
    for (let i = 0; i < bins.current.length; i++) { const v = (bins.current[i] - 128) / 128; sum += v * v; }
    return Math.sqrt(sum / bins.current.length);
  };

  /** Current audio levels (0..1-ish RMS): the agent's playback and the microphone. Called every animation frame. */
  const sample = useCallback(() => {
    const s = session.current;
    return { speak: s ? rms(s.outAn) : 0, mic: s ? rms(s.micAn) : 0, agentTalking: agentTalking.current };
  }, []);

  const teardown = useCallback((s: Session) => {
    s.closing = true;
    agentTalking.current = false;
    try { s.player?.port.postMessage({ flush: true }); } catch { /* already gone */ }
    if (s.player) { s.player.port.onmessage = null; s.player.disconnect(); }
    if (s.capture) { s.capture.port.onmessage = null; s.capture.disconnect(); }
    s.stream?.getTracks().forEach((t) => t.stop());
    if (s.ws.readyState <= WebSocket.OPEN) s.ws.close(1000);
    s.ctx.close().catch(() => {});
    if (session.current === s) session.current = null;
  }, []);

  const stop = useCallback(() => {
    if (session.current) teardown(session.current);
    setStatus("idle");
    setCaption("");
  }, [teardown]);

  const start = useCallback(async () => {
    if (session.current) return;
    setError(null);
    hadError.current = false;
    setCaption("");
    if (!navigator.mediaDevices?.getUserMedia) { setError("Voice needs a secure (https or localhost) page and a browser with microphone support."); return; }
    setStatus("connecting");

    // Create the AudioContext synchronously in the user gesture (Safari refuses contexts created after awaits).
    let ctx: AudioContext;
    try { ctx = new AudioContext({ sampleRate: 24000, latencyHint: "interactive" }); }
    catch { ctx = new AudioContext({ latencyHint: "interactive" }); }
    void ctx.resume();

    let ws: WebSocket | null = null;
    let stream: MediaStream | null = null;
    const fail = (message: string) => {
      hadError.current = true;
      setError(message);
      setStatus("idle");
      stream?.getTracks().forEach((t) => t.stop());
      session.current = null;
      ctx.close().catch(() => {});
      if (ws && ws.readyState <= WebSocket.OPEN) ws.close(1000);
    };

    try {
      const [token, settings, coreSrc] = await Promise.all([
        api<{ token: string }>("/api/voice/token", { method: "POST" }),
        api<AgentSettings>("/api/voice/settings"),
        loadAudioCore(),
      ]);
      if (settings.audio.input.encoding !== "linear16" || settings.audio.output.encoding !== "linear16") {
        return fail("The voice settings use an audio format this page doesn't support.");
      }
      const inRate = settings.audio.input.sample_rate;
      const outRate = settings.audio.output.sample_rate;

      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
      } catch (err) { return fail(micErrorMessage(err)); }

      await ctx.resume();
      const addModule = async (src: string) => {
        const url = URL.createObjectURL(new Blob([src], { type: "application/javascript" }));
        try { await ctx.audioWorklet.addModule(url); } finally { URL.revokeObjectURL(url); }
      };
      await addModule(CAPTURE_SRC);
      await addModule(coreSrc + "\n" + PLAYER_SRC);

      const player = new AudioWorkletNode(ctx, "pcm-player", { numberOfInputs: 0, outputChannelCount: [1], processorOptions: { inRate: outRate, prebufferMs: 100 } });
      const outAn = makeAnalyser(ctx);
      player.connect(outAn);
      outAn.connect(ctx.destination);

      ws = new WebSocket("wss://agent.deepgram.com/v1/agent/converse", ["bearer", token.token]);
      ws.binaryType = "arraybuffer";
      const s: Session = { ws, ctx, stream: stream, capture: null, player, outAn, micAn: null, aligner: new (window as any).AudioCore.Pcm16Aligner(), closing: false };
      session.current = s;

      ws.onmessage = (ev) => {
        if (typeof ev.data !== "string") {
          const samples = s.aligner.push(new Uint8Array(ev.data as ArrayBuffer));
          if (samples.length) s.player?.port.postMessage({ pcm: samples.buffer }, [samples.buffer]);
          return;
        }
        let msg: any;
        try { msg = JSON.parse(ev.data); } catch { return; }
        switch (msg.type) {
          case "Welcome":
            ws!.send(JSON.stringify(settings));
            break;
          case "SettingsApplied": {
            const source = ctx.createMediaStreamSource(stream!);
            s.micAn = makeAnalyser(ctx);
            source.connect(s.micAn);
            const capture = new AudioWorkletNode(ctx, "pcm-capture", { processorOptions: { targetRate: inRate } });
            capture.port.onmessage = (e) => { if (ws!.readyState === WebSocket.OPEN) ws!.send(e.data); };
            const mute = ctx.createGain();  // keep the graph pulled without playing the mic back
            mute.gain.value = 0;
            source.connect(capture).connect(mute).connect(ctx.destination);
            s.capture = capture;
            setStatus("live");
            break;
          }
          case "ConversationText":
            if (msg.content) {
              setCaption(msg.content);
              onTranscript.current(msg.role === "user" ? "user" : "assistant", msg.content);
            }
            break;
          case "UserStartedSpeaking":  // barge-in: fade out and drop whatever the agent still had queued
            s.aligner.reset();
            s.player?.port.postMessage({ flush: true });
            agentTalking.current = false;
            break;
          case "AgentStartedSpeaking":
            agentTalking.current = true;
            break;
          case "AgentAudioDone":
            agentTalking.current = false;
            s.player?.port.postMessage({ end: true });  // draining the queue is not an underrun
            break;
          case "Error":
            console.error("[voice] Deepgram error", msg);  // details for developers; employees get a plain message
            hadError.current = true;
            setError("Voice is unavailable right now. You can keep using text chat, or try again in a moment.");
            break;
          default:
            break;
        }
      };
      ws.onerror = () => { if (!s.closing && !hadError.current) setError("Couldn't connect to the voice service (the session token may have been rejected)."); };
      ws.onclose = (ev) => {
        const unexpected = !s.closing;
        teardown(s);
        setStatus("idle");
        if (unexpected && !hadError.current && ev.code !== 1000) setError(`The voice session ended unexpectedly${ev.reason ? ` (${ev.reason})` : ""}. Tap the orb to reconnect.`);
      };
    } catch (err) {
      fail(err instanceof Error && err.message && !/^Failed to fetch/.test(err.message) ? `Couldn't start voice: ${err.message}` : "Couldn't start voice. Check your connection and try again.");
    }
  }, [teardown]);

  useEffect(() => () => { if (session.current) teardown(session.current); }, [teardown]);

  return { status, error, caption, start, stop, sample, agentTalking };
}
