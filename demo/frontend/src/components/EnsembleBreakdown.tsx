import type { EnsembleResult } from "@/lib/types";

const METHOD_COLORS: Record<string, string> = {
  mean_prob: "#06b6d4",
  majority_vote: "#8b5cf6",
  combiner_logreg: "#f43f5e",
  combiner_mlp: "#f97316",
};

const METHOD_DESCRIPTIONS: Record<string, string> = {
  mean_prob: "Average of calibrated member probabilities",
  majority_vote: "Fraction of members predicting AI-generated",
  combiner_logreg: "Trained LogReg on calibrated features (9-dim)",
  combiner_mlp: "Trained MLP on calibrated features (9-dim)",
};

interface EnsembleBreakdownProps {
  readonly methods: readonly EnsembleResult[];
  readonly primaryMethod: string;
}

export function EnsembleBreakdown({
  methods,
  primaryMethod,
}: EnsembleBreakdownProps) {
  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-zinc-300">
        Ensemble Strategies
      </h3>
      {methods.map((m) => {
        const pct = Math.round(m.confidence * 100);
        const color = METHOD_COLORS[m.name] ?? "#6b7280";
        const description = METHOD_DESCRIPTIONS[m.name] ?? "";
        const isPrimary = m.name === primaryMethod;
        const isAI = m.verdict === "AI-Generated";

        return (
          <div key={m.name} className="space-y-1">
            <div className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-2 font-medium text-zinc-300">
                {m.label}
                {isPrimary && (
                  <span className="rounded bg-zinc-700 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
                    primary
                  </span>
                )}
              </span>
              <span className="flex items-center gap-2">
                <span
                  className={`text-xs font-medium ${
                    isAI ? "text-red-400" : "text-green-400"
                  }`}
                >
                  {m.verdict}
                </span>
                <span className="font-mono text-zinc-400">{pct}%</span>
              </span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-zinc-800">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${pct}%`,
                  backgroundColor: color,
                }}
              />
            </div>
            <p className="text-xs text-zinc-500">{description}</p>
          </div>
        );
      })}
    </div>
  );
}
