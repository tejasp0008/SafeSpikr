// src/components/LivePredict.tsx
import React, { useEffect, useRef, useState } from "react";
import { createPredictWS } from "../api/wsPredict";
import api from "../api/axios";

async function sendCanvasFrame(ws: WebSocket, canvas: HTMLCanvasElement) {
  return new Promise<void>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) return reject(new Error("toBlob failed"));
      try {
        ws.send(blob);
        resolve();
      } catch (e) {
        reject(e);
      }
    }, "image/jpeg", 0.8);
  });
}

export default function LivePredict() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [messages, setMessages] = useState<any[]>([]);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>("idle");
  const [wsOpen, setWsOpen] = useState(false);
  const intervalRef = useRef<number | null>(null);

  // start/stop streaming and ws when selectedDeviceId changes
  useEffect(() => {
    let stream: MediaStream | null = null;

    async function start() {
      setError(null);
      setStatus("requesting-permission");
      try {
        const constraints: MediaStreamConstraints = {
          video: selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : { facingMode: "user" },
        };
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        setStatus("streaming");
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          try { await videoRef.current.play(); } catch { /* ignore autoplay policy */ }
        }

        // enumerate devices and prefer physical camera automatically
        const devs = await navigator.mediaDevices.enumerateDevices();
        const videoInputs = devs.filter(d => d.kind === "videoinput");
        setDevices(videoInputs);

        // Auto-select preferred camera if none selected yet:
        if (!selectedDeviceId) {
          const preferred = videoInputs.find(d => !/obs|virtual/i.test(d.label || ""));
          setSelectedDeviceId(preferred ? preferred.deviceId : (videoInputs[0]?.deviceId ?? null));
        }
      } catch (err: any) {
        console.error("getUserMedia failed:", err);
        setError(String(err?.name || err?.message || err));
        setStatus("error");
        return;
      }

      // open WS
      try {
        wsRef.current = createPredictWS((msg) => setMessages(m => [msg, ...m]));
        wsRef.current.onopen = () => setWsOpen(true);
        wsRef.current.onclose = () => setWsOpen(false);
        wsRef.current.onerror = (e) => {
          console.error("WS error", e);
          setError("WebSocket error");
        };
      } catch (e: any) {
        console.error("WS open failed:", e);
        setError("WebSocket open failed: " + String(e));
      }

      // continuous send
      intervalRef.current = window.setInterval(async () => {
        const ws = wsRef.current;
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN || !video || !canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        try { await sendCanvasFrame(ws, canvas); } catch (e) { console.error("send frame failed", e); }
      }, 500); // 500ms, tune as needed
    }

    start();

    return () => {
      if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
      try { wsRef.current?.close(); } catch {}
      const tracks = (videoRef.current?.srcObject as MediaStream | null)?.getTracks() || [];
      tracks.forEach(t => t.stop());
      stream = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDeviceId]);

  // Helper: send one test frame to /predict (HTTP)
  async function sendTestFrame() {
    setError(null);
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return setError("Video or canvas not ready");
    const ctx = canvas.getContext("2d");
    if (!ctx) return setError("Canvas context missing");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    try {
      const blob: Blob | null = await new Promise(resolve => canvas.toBlob(resolve as any, "image/jpeg", 0.9));
      if (!blob) throw new Error("toBlob returned null");
      const form = new FormData();
      form.append("frame", blob, "frame.jpg");
      const res = await api.post("/predict", form, { headers: { "Content-Type": "multipart/form-data" } });
      setMessages(m => [{ type: "http-predict", ts: Date.now(), data: res.data }, ...m]);
    } catch (e: any) {
      console.error("HTTP predict failed", e);
      setError("HTTP predict failed: " + (e?.message || e));
    }
  }

  return (
    <div>
      <h3>Live Predict (Webcam)</h3>

      <div style={{ marginBottom: 8 }}>
        <label style={{ marginRight: 8 }}>
          Camera:
          <select
            value={selectedDeviceId ?? ""}
            onChange={(e) => setSelectedDeviceId(e.target.value || null)}
            style={{ marginLeft: 8 }}
          >
            <option value="">Default</option>
            {devices.map(d => <option key={d.deviceId} value={d.deviceId}>{d.label || d.deviceId}</option>)}
          </select>
        </label>

        <button onClick={sendTestFrame} style={{ marginLeft: 8 }}>Send test frame (HTTP /predict)</button>
        <span style={{ marginLeft: 12 }}>
          WS: <strong style={{ color: wsOpen ? "green" : "gray" }}>{wsOpen ? "open" : "closed"}</strong>
        </span>
      </div>

      <div>
        <video ref={videoRef} width={320} height={240} style={{ border: "1px solid #ccc" }} autoPlay muted playsInline />
        <canvas ref={canvasRef} width={128} height={128} style={{ display: "none" }} />
      </div>

      <div style={{ marginTop: 8 }}>
        <strong>Status:</strong> {status} {error && <span style={{ color: "crimson" }}> — {error}</span>}
      </div>

      <div style={{ marginTop: 12 }}>
        <h4>Realtime predictions (latest)</h4>
        <div style={{ maxHeight: 260, overflow: "auto", background: "#fbfbfb", padding: 8 }}>
          {messages.length === 0 ? <div>No messages yet</div> : messages.slice(0, 20).map((m, i) => (
            <div key={i} style={{ marginBottom: 8, padding: 6, borderBottom: "1px solid #eee" }}>
              <small>{m.type ?? "ws"} @{m.ts ? new Date(m.ts).toLocaleTimeString() : ""}</small>
              <pre style={{ margin: 0 }}>{JSON.stringify(m.data ?? m, null, 2)}</pre>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
