import type { HealthResponse, PredictionResponse } from "./types";

// In dev: empty string → uses Next.js proxy (/api/predict → localhost:8000/predict)
// In prod: set NEXT_PUBLIC_API_URL to the Modal endpoint URL (e.g. https://xxx.modal.run)
const API_URL = process.env.NEXT_PUBLIC_API_URL;
const PREFIX = API_URL ? API_URL : "/api";

export async function predict(
  file: File,
  mode: "fast" | "full",
  signal?: AbortSignal
): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${PREFIX}/predict?mode=${mode}`, {
    method: "POST",
    body: form,
    signal,
  });

  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail ?? `Server error (${res.status})`);
  }

  return res.json();
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch(`${PREFIX}/health`);
  if (!res.ok) {
    throw new Error("Server unavailable");
  }
  return res.json();
}
