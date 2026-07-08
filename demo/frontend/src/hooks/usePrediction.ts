"use client";

import { useCallback, useRef, useState } from "react";
import type { PredictionResponse } from "@/lib/types";
import { predict } from "@/lib/api";

type Status = "idle" | "loading" | "success" | "error";

export function usePrediction() {
  const [status, setStatus] = useState<Status>("idle");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const run = useCallback(async (file: File, mode: "fast" | "full") => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setStatus("loading");
    setResult(null);
    setError(null);

    try {
      const response = await predict(file, mode, controller.signal);
      if (!controller.signal.aborted) {
        setResult(response);
        setStatus("success");
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      setStatus("error");
    }
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setStatus("idle");
    setResult(null);
    setError(null);
  }, []);

  return { status, result, error, run, reset } as const;
}
