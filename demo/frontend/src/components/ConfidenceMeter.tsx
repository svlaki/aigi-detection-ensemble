interface ConfidenceMeterProps {
  readonly confidence: number;
}

export function ConfidenceMeter({ confidence }: ConfidenceMeterProps) {
  const pct = Math.round(confidence * 100);

  return (
    <div className="space-y-2">
      <p className="text-xs font-medium uppercase tracking-wider text-zinc-500">
        Ensemble Confidence — P(AI-Generated)
      </p>
      <div className="flex items-center justify-between text-sm">
        <span className="text-zinc-400">Likely Real</span>
        <span className="font-mono font-bold text-white">{pct}%</span>
        <span className="text-zinc-400">AI-Generated</span>
      </div>
      <div className="relative h-3 overflow-hidden rounded-full bg-zinc-800">
        <div
          className="absolute inset-0 rounded-full"
          style={{
            background:
              "linear-gradient(to right, #22c55e, #eab308 50%, #ef4444)",
          }}
        />
        <div
          className="absolute top-0 h-full rounded-full bg-zinc-800"
          style={{
            left: `${pct}%`,
            width: `${100 - pct}%`,
            transition: "left 0.6s ease-out, width 0.6s ease-out",
          }}
        />
        <div
          className="absolute top-0 h-full w-0.5 bg-white shadow-sm"
          style={{
            left: `${pct}%`,
            transition: "left 0.6s ease-out",
          }}
        />
      </div>
    </div>
  );
}
