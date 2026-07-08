import type { HealthResponse, PredictionResponse } from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

export async function predict(
  file: File,
  mode: "fast" | "full",
  signal?: AbortSignal
): Promise<PredictionResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE_URL}/api/predict?mode=${mode}`, {
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
  const res = await fetch(`${BASE_URL}/api/health`);
  if (!res.ok) {
    throw new Error("Server unavailable");
  }
  return res.json();
}
