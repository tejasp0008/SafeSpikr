// src/components/UsersList.tsx
import React, { useEffect, useState } from "react";
import { listUsers, getHistorySummary } from "../api/users";

export default function UsersList() {
  const [users, setUsers] = useState<{ id: number; name: string }[]>([]);
  const [summary, setSummary] = useState<any | null>(null);

  useEffect(() => { listUsers().then(setUsers).catch(console.error); }, []);

  async function showSummary(id: number) {
    const s = await getHistorySummary(id);
    setSummary(s);
  }

  return (
    <div>
      <h3>Users</h3>
      <ul>
        {users.map((u) => (
          <li key={u.id}>
            {u.name} <button onClick={() => showSummary(u.id)}>Summary</button>
          </li>
        ))}
      </ul>
      {summary && <pre>{JSON.stringify(summary, null, 2)}</pre>}
    </div>
  );
}
