// src/pages/Live.tsx
import React from "react";
import { Link } from "react-router-dom";
import LivePredict from "@/components/LivePredict";

export default function LivePage() {
  return (
    <div style={{ padding: 20 }}>
      <nav style={{ marginBottom: 16 }}>
        <Link to="/">Home</Link> · <Link to="/register">Register</Link> · <Link to="/users">Users</Link> · <Link to="/predict">Single Predict</Link>
      </nav>
      <h1>Live Predict (Webcam)</h1>
      <p>Allow camera access and the client will send JPEG frames over WebSocket to `/ws/predict`.</p>
      <LivePredict />
    </div>
  );
}
