import type { PredictionResponse } from "@/lib/types";
import { ConfidenceMeter } from "./ConfidenceMeter";
import { EnsembleBreakdown } from "./EnsembleBreakdown";
import { MemberBreakdown } from "./MemberBreakdown";

interface ResultsPanelProps {
  readonly result: PredictionResponse;
}

export function ResultsPanel({ result }: ResultsPanelProps) {
  const isAI = result.verdict === "AI-Generated";
  const primaryMethod = "combiner_logreg";

  return (
    <div className="space-y-6 rounded-xl border border-zinc-800 bg-zinc-900/50 p-6">
      {/* Verdict */}
      <div className="text-center">
        <div
          className={`inline-block rounded-full px-4 py-1 text-sm font-semibold ${
            isAI
              ? "bg-red-500/20 text-red-400"
              : "bg-green-500/20 text-green-400"
          }`}
        >
          {result.verdict}
        </div>
      </div>

      {/* Confidence — primary method */}
      <ConfidenceMeter confidence={result.confidence} />

      {/* Ensemble strategies */}
      <EnsembleBreakdown
        methods={result.ensemble_methods}
        primaryMethod={primaryMethod}
      />

      {/* Per-member scores */}
      <MemberBreakdown members={result.members} />

      {/* Footer info */}
      <div className="flex items-center justify-between border-t border-zinc-800 pt-3 text-xs text-zinc-500">
        <span>Full (3 members)</span>
        <span>{result.processing_time_ms.toFixed(0)} ms</span>
      </div>
    </div>
  );
}
