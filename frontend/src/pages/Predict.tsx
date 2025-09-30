// src/pages/Predict.tsx
import React from "react";
import { Link } from "react-router-dom";
import QuickPredict from "@/components/QuickPredict";

export default function PredictPage() {
  return (
    <div style={{ padding: 20 }}>
      <nav style={{ marginBottom: 16 }}>
        <Link to="/">Home</Link> · <Link to="/register">Register</Link> · <Link to="/users">Users</Link> · <Link to="/live">Live</Link>
      </nav>
      <h1>Single Image Predict</h1>
      <p>Upload a single image to `/predict` endpoint.</p>
      <QuickPredict />
    </div>
  );
}
