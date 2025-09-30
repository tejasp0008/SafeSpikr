// src/api/wsPredict.ts
export function buildWsUrl(): string {
  const root = (import.meta.env.VITE_API_URL as string) || "http://localhost:8000";
  // convert http(s) -> ws(s)
  return root.replace(/^http/, "ws") + "/ws/predict";
}

export function createPredictWS(onMessage: (data: any) => void): WebSocket {
  const ws = new WebSocket(buildWsUrl());

  ws.onopen = () => console.log("WS /ws/predict opened");
  ws.onmessage = (evt) => {
    try {
      const parsed = JSON.parse(evt.data);
      onMessage(parsed);
    } catch (e) {
      onMessage(evt.data);
    }
  };
  ws.onclose = () => console.log("WS closed");
  ws.onerror = (e) => console.error("WS error", e);

  return ws;
}
