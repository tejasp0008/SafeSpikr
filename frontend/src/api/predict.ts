// src/api/predict.ts
import api from "./axios";

export type PredictResult = {
  raw_prediction: any;
  personalized: any | null;
  user: any | null;
};

export async function predictImage(file: File, userId?: number): Promise<PredictResult> {
  const form = new FormData();
  form.append("frame", file);
  if (userId !== undefined && userId !== null) form.append("user_id", String(userId));

  const res = await api.post<PredictResult>("/predict", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}
