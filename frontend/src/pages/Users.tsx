// src/pages/Users.tsx
import React from "react";
import { Link } from "react-router-dom";
import UsersList from "@/components/UsersList";

export default function UsersPage() {
  return (
    <div style={{ padding: 20 }}>
      <nav style={{ marginBottom: 16 }}>
        <Link to="/">Home</Link> · <Link to="/register">Register</Link> · <Link to="/predict">Single Predict</Link> · <Link to="/live">Live</Link>
      </nav>
      <h1>Users</h1>
      <UsersList />
    </div>
  );
}
