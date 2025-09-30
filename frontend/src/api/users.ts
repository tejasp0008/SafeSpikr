// src/api/users.ts
import api from "./axios";

export type User = { id: number; name: string };

export const listUsers = async (): Promise<User[]> => {
  const res = await api.get<User[]>("/users");
  return res.data;
};

export const getHistory = async (userId: number, n = 50) => {
  const res = await api.get(`/history/${userId}?n=${n}`);
  return res.data;
};

export const getHistorySummary = async (userId: number, n = 200) => {
  const res = await api.get(`/history/${userId}/summary?n=${n}`);
  return res.data;
};
