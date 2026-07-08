"use client";

import { useCallback, useRef, useState } from "react";

const MAX_SIZE = 10 * 1024 * 1024;
const ACCEPTED_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/bmp",
]);

interface ImageUploaderProps {
  readonly onFileSelected: (file: File) => void;
  readonly disabled: boolean;
}

export function ImageUploader({ onFileSelected, disabled }: ImageUploaderProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateAndSet = useCallback(
    (file: File) => {
      setError(null);
      if (!ACCEPTED_TYPES.has(file.type)) {
        setError("Unsupported file type. Use JPEG, PNG, WebP, or BMP.");
        return;
      }
      if (file.size > MAX_SIZE) {
        setError(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max: 10 MB.`);
        return;
      }
      const url = URL.createObjectURL(file);
      setPreview((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return url;
      });
      onFileSelected(file);
    },
    [onFileSelected]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) validateAndSet(file);
    },
    [validateAndSet]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) validateAndSet(file);
    },
    [validateAndSet]
  );

  return (
    <div className="space-y-3">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`relative flex min-h-[200px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed transition-colors ${
          dragOver
            ? "border-blue-500 bg-blue-500/10"
            : "border-zinc-700 bg-zinc-900 hover:border-zinc-500"
        } ${disabled ? "pointer-events-none opacity-50" : ""}`}
      >
        {preview ? (
          <img
            src={preview}
            alt="Preview"
            className="max-h-[300px] rounded-lg object-contain"
          />
        ) : (
          <div className="p-8 text-center">
            <svg
              className="mx-auto mb-3 h-10 w-10 text-zinc-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M12 16v-8m0 0l-3 3m3-3l3 3M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <p className="text-sm text-zinc-400">
              Drop an image here or click to select
            </p>
            <p className="mt-1 text-xs text-zinc-600">
              JPEG, PNG, WebP, BMP up to 10 MB
            </p>
          </div>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp"
          onChange={handleChange}
          className="hidden"
        />
      </div>
      {error && (
        <p className="text-sm text-red-400">{error}</p>
      )}
    </div>
  );
}
