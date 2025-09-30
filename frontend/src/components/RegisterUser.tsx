// src/components/RegisterUser.tsx
import React, { useState } from "react";
import { registerUser } from "../api/registerUser";

export default function RegisterUser() {
  const [name, setName] = useState("");
  const [files, setFiles] = useState<FileList | null>(null);
  const [resp, setResp] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!name || !files) return alert("Name and frames required");
    setLoading(true);
    try {
      const data = await registerUser(name, files);
      setResp(data);
    } catch (err) {
      alert("Register error: " + (err as any).message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <form onSubmit={submit}>
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Name" />
        <input type="file" accept="image/*" multiple onChange={(e) => setFiles(e.target.files)} />
        <button type="submit" disabled={loading}>{loading ? "Registering..." : "Register"}</button>
      </form>
      {resp && <pre>{JSON.stringify(resp, null, 2)}</pre>}
    </div>
  );
}
