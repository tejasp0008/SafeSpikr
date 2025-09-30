// src/components/LivePredict.tsx
import React, { useEffect, useRef, useState } from "react";
import { createPredictWS } from "../api/wsPredict";

async function sendCanvasFrame(ws: WebSocket, canvas: HTMLCanvasElement) {
  return new Promise<void>((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) return reject(new Error("toBlob failed"));
      ws.send(blob);
      resolve();
    }, "image/jpeg", 0.8);
  });
}

export default function LivePredict() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [messages, setMessages] = useState<any[]>([]);

  useEffect(() => {
    let interval: number | undefined;
    navigator.mediaDevices.getUserMedia({ video: true })
      .then((stream) => {
        if (!videoRef.current) return;
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      })
      .catch(console.error);

    wsRef.current = createPredictWS((msg) => setMessages((m) => [msg, ...m]));

    interval = window.setInterval(async () => {
      const ws = wsRef.current;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN || !video || !canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      try {
        await sendCanvasFrame(ws, canvas);
      } catch (e) {
        console.error("send frame failed", e);
      }
    }, 500);

    return () => {
      if (interval) clearInterval(interval);
      wsRef.current?.close();
      const tracks = (videoRef.current?.srcObject as MediaStream | null)?.getTracks() || [];
      tracks.forEach((t) => t.stop());
    };
  }, []);

  return (
    <div>
      <video ref={videoRef} width={320} height={240} />
      {/* canvas sized to model's expected size (128x128) */}
      <canvas ref={canvasRef} width={128} height={128} style={{ display: "none" }} />
      <h4>Realtime predictions</h4>
      <pre style={{ maxHeight: 300, overflow: "auto" }}>{messages.slice(0, 10).map((m, i) => <div key={i}>{JSON.stringify(m, null, 2)}</div>)}</pre>
    </div>
  );
}
