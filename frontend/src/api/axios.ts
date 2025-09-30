// src/api/axios.ts
import axios from "axios";

const API_ROOT = (import.meta.env.VITE_API_URL as string) || "http://localhost:8000";

const api = axios.create({
  baseURL: API_ROOT,
  withCredentials: false, // set true if you use cookies/sessions
  headers: {
    Accept: "application/json",
  },
});

export default api;
