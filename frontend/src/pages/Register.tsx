// src/pages/Register.tsx
import React from "react";
import { Link } from "react-router-dom";
import RegisterUser from "@/components/RegisterUser";

export default function RegisterPage() {
  return (
    <div style={{ padding: 20 }}>
      <nav style={{ marginBottom: 16 }}>
        <Link to="/">Home</Link> · <Link to="/users">Users</Link> · <Link to="/predict">Single Predict</Link> · <Link to="/live">Live</Link>
      </nav>
      <h1>Register User</h1>
      <p>Capture a few face frames and register a user. Backend will compute baseline.</p>
      <RegisterUser />
    </div>
  );
}
