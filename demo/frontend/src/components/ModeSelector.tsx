"use client";

interface ModeSelectorProps {
  readonly mode: "fast" | "full";
  readonly onModeChange: (mode: "fast" | "full") => void;
  readonly d3qeAvailable: boolean;
  readonly disabled: boolean;
}

export function ModeSelector({
  mode,
  onModeChange,
  d3qeAvailable,
  disabled,
}: ModeSelectorProps) {
  return (
    <div className="flex gap-2">
      <button
        type="button"
        onClick={() => onModeChange("fast")}
        disabled={disabled}
        className={`flex-1 rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
          mode === "fast"
            ? "bg-blue-600 text-white"
            : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
        } disabled:opacity-50`}
      >
        <span className="block font-semibold">Fast</span>
        <span className="block text-xs opacity-75">CLIP + Spectral</span>
      </button>
      <button
        type="button"
        onClick={() => onModeChange("full")}
        disabled={disabled || !d3qeAvailable}
        title={
          d3qeAvailable
            ? "Includes D3QE detector (~10s slower)"
            : "D3QE model not loaded on server"
        }
        className={`flex-1 rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
          mode === "full"
            ? "bg-blue-600 text-white"
            : "bg-zinc-800 text-zinc-400 hover:bg-zinc-700"
        } disabled:opacity-50 disabled:cursor-not-allowed`}
      >
        <span className="block font-semibold">Full</span>
        <span className="block text-xs opacity-75">
          + D3QE {!d3qeAvailable && "(unavailable)"}
        </span>
      </button>
    </div>
  );
}
