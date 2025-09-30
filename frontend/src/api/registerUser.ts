// src/api/registerUser.ts
import api from "./axios";

export type RegisterUserResp = { ok: boolean; id?: number; name?: string; baseline_p_drowsy?: number };

export async function registerUser(name: string, files: FileList | File[]): Promise<RegisterUserResp> {
  const form = new FormData();
  form.append("name", name);

  if (files instanceof FileList) {
    for (let i = 0; i < files.length; i++) form.append("frames", files[i]);
  } else {
    files.forEach((f) => form.append("frames", f));
  }

  const res = await api.post<RegisterUserResp>("/register_user", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}
