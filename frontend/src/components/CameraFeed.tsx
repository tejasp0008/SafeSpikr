import React, { useEffect, useRef, useState } from "react";
import { Play, Pause, RotateCcw, Video, Wifi } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import api from "@/api/axios";
import { toast } from "sonner";

// safe toast fallback
const safeToast = {
  success: (s: string) => console.log("TOAST success:", s),
  error: (s: string) => console.error("TOAST error:", s),
  info: (s: string) => console.log("TOAST info:", s),
};
const usedToast = (typeof toast !== "undefined" && (toast as any).toast) ? (toast as any).toast : safeToast;

export function CameraFeed() {
  // camera state
  const [isPlaying, setIsPlaying] = useState(false);
  const [status, setStatus] = useState<"idle" | "requesting" | "streaming" | "error">("idle");
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // websocket state
  const wsRef = useRef<WebSocket | null>(null);
  const sendIntervalRef = useRef<number | null>(null);
  const [wsConnected, setWsConnected] = useState(false);

  // predictions
  const [rawPrediction, setRawPrediction] = useState<any | null>(null);
  const [personalizedPrediction, setPersonalizedPrediction] = useState<any | null>(null);

  // session user id (set after calibration)
  const [sessionUserId, setSessionUserId] = useState<number | null>(null);

  // ---------------- Camera lifecycle ----------------
  async function startStream() {
    setError(null);
    setStatus("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch((err) => console.warn("video.play() rejected:", err));
      }
      setStatus("streaming");
      usedToast.info?.("Camera started");
    } catch (err: any) {
      setError(String(err?.name || err?.message || err));
      setStatus("error");
      setIsPlaying(false);
      usedToast.error?.("Camera start failed: " + (err?.message || err));
    }
  }

  function stopStream() {
    try {
      const s = streamRef.current;
      if (s) s.getTracks().forEach((t) => t.stop());
    } catch {}
    streamRef.current = null;
    if (videoRef.current) {
      videoRef.current.pause();
      // @ts-ignore
      videoRef.current.srcObject = null;
    }
    setStatus("idle");
    usedToast.info?.("Camera stopped");
  }

  useEffect(() => {
    if (isPlaying) startStream();
    else stopStream();
    return () => stopStream();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying]);

  // ---------------- Capture helpers ----------------
  async function captureFrameAsBlob(): Promise<Blob | null> {
    const v = videoRef.current;
    if (!v) return null;
    const canvas = document.createElement("canvas");
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
    return await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.8));
  }

  async function captureFrameArrayBuffer(): Promise<ArrayBuffer | null> {
    const blob = await captureFrameAsBlob();
    if (!blob) return null;
    return await blob.arrayBuffer();
  }

  // ---------------- Calibration ----------------
  async function handleCalibrate() {
    if (status !== "streaming") {
      usedToast.error?.("Start the feed first to calibrate");
      return;
    }
    const name = window.prompt("Enter name for calibration:");
    if (!name) return;

    setStatus("requesting");
    try {
      const frames: Blob[] = [];
      for (let i = 0; i < 5; i++) {
        const b = await captureFrameAsBlob();
        if (b) frames.push(b);
        await new Promise((r) => setTimeout(r, 250));
      }
      if (frames.length === 0) throw new Error("No frames captured");

      const form = new FormData();
      form.append("name", name);
      frames.forEach((f, i) => form.append("frames", f, `frame_${i}.jpg`));

      const res = await api.post("/register_user", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      usedToast.success?.(`User ${res.data?.name} registered (id=${res.data?.id})`);
      const uid = Number(res.data.id);
      setSessionUserId(uid);

      // immediately inform WS if open
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
  wsRef.current.send(JSON.stringify({ user_id: uid }));
  console.log("Sent user_id after calibration:", uid);
} else {
  console.warn("WS not open, will retry sending user_id shortly");
  setTimeout(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ user_id: uid }));
      console.log("Retried user_id send:", uid);
    }
  }, 1000);
}

    } catch (e: any) {
      usedToast.error?.("Calibration failed: " + (e?.message || e));
    } finally {
      setStatus(isPlaying ? "streaming" : "idle");
    }
  }

  // Resend user_id to WS whenever it changes
  useEffect(() => {
    if (sessionUserId !== null && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ user_id: sessionUserId }));
      console.log("Resent user_id to WS:", sessionUserId);
    }
  }, [sessionUserId]);

  // ---------------- WebSocket realtime prediction ----------------
  useEffect(() => {
    if (status === "streaming") {
      const scheme = location.protocol === "https:" ? "wss" : "ws";
      const WS_URL = `${scheme}://${location.hostname}:8000/ws/predict`;
      const ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        console.log("WS open");
        setWsConnected(true);
        if (sessionUserId !== null) {
          ws.send(JSON.stringify({ user_id: sessionUserId }));
          console.log("Sent user_id on WS open:", sessionUserId);
        }
        // start sending frames periodically
        sendIntervalRef.current = window.setInterval(async () => {
          if (ws.readyState !== WebSocket.OPEN) return;
          const ab = await captureFrameArrayBuffer();
          if (ab) ws.send(ab);
        }, 600);
      };

      ws.onmessage = (ev) => {
        try {
          const text = typeof ev.data === "string" ? ev.data : new TextDecoder().decode(ev.data);
          const data = JSON.parse(text);
          if (data.raw_prediction) setRawPrediction(data.raw_prediction);
          if (data.personalized) setPersonalizedPrediction(data.personalized);
          if (data.info === "user_id_set") console.log("WS ack user_id:", data.user_id);
        } catch (e) {
          console.warn("WS message parse failed", e);
        }
      };

      ws.onclose = () => {
        setWsConnected(false);
        if (sendIntervalRef.current) {
          clearInterval(sendIntervalRef.current);
          sendIntervalRef.current = null;
        }
      };

      ws.onerror = (err) => console.error("WS error", err);

      return () => {
        if (sendIntervalRef.current) clearInterval(sendIntervalRef.current);
        try { ws.close(); } catch {}
        wsRef.current = null;
        setWsConnected(false);
      };
    } else {
      if (sendIntervalRef.current) clearInterval(sendIntervalRef.current);
      if (wsRef.current) { try { wsRef.current.close(); } catch {} wsRef.current = null; }
      setWsConnected(false);
    }
  }, [status]);

  // ---------------- JSX ----------------
  return (
    <Card className="futuristic-card neural-glow hover-float animate-fade-in">
      <CardHeader className="pb-4">
        <CardTitle className="flex items-center justify-between font-display">
          <div className="flex items-center space-x-3">
            <Video className="h-6 w-6 text-accent" />
            <span className="text-gradient text-xl">Live Camera Feed</span>
          </div>
          <Badge variant="secondary" className="animate-glow">
            <Wifi className="h-3 w-3 mr-1" />
            {wsConnected ? "CONNECTED" : "DISCONNECTED"}
          </Badge>
        </CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        <div className="relative aspect-video rounded-2xl overflow-hidden border border-white/10 bg-black">
          {status !== "idle" ? (
            <video ref={videoRef} autoPlay muted playsInline style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          ) : (
            <div className="flex items-center justify-center h-full text-gray-400">Neural Vision Ready</div>
          )}
        </div>

        {/* Controls */}
        <div className="flex items-center justify-center space-x-4">
          <Button variant="outline" size="lg" onClick={() => setIsPlaying((p) => !p)}>
            {isPlaying ? <Pause className="h-5 w-5 mr-2" /> : <Play className="h-5 w-5 mr-2" />}
            {isPlaying ? "Stop Feed" : "Start Feed"}
          </Button>
          <Button variant="outline" size="lg" onClick={handleCalibrate}>
            <RotateCcw className="h-5 w-5 mr-2" /> Calibrate
          </Button>
        </div>

        {/* Predictions */}
        <div className="mt-2 grid grid-cols-2 gap-4">
          <div className="p-3 bg-white/5 rounded-lg">
            <div className="text-sm font-semibold">Neural Prediction</div>
            <div className="text-lg">
              {rawPrediction ? `${rawPrediction.label} (${(rawPrediction.confidence * 100).toFixed(0)}%)` : "—"}
            </div>
          </div>
          <div className="p-3 bg-white/5 rounded-lg">
            <div className="text-sm font-semibold">Personalized</div>
            <div className="text-lg">
              {personalizedPrediction
                ? `${personalizedPrediction.label} (${(personalizedPrediction.confidence * 100).toFixed(0)}%)`
                : "—"}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

export default CameraFeed;
