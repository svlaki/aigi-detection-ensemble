import type { MemberResult } from "@/lib/types";

const MEMBER_LABELS: Record<string, { label: string; color: string }> = {
  M1_CLIP: { label: "CLIP (Visual)", color: "#3b82f6" },
  M2_Spectral: { label: "Spectral (FFT)", color: "#a855f7" },
  M3_D3QE: { label: "D3QE (Codebook)", color: "#f59e0b" },
};

interface MemberBreakdownProps {
  readonly members: readonly MemberResult[];
}

export function MemberBreakdown({ members }: MemberBreakdownProps) {
  return (
    <div className="space-y-4">
      <h3 className="text-sm font-semibold text-zinc-300">
        Per-Member Scores
      </h3>
      {members.map((m) => {
        const meta = MEMBER_LABELS[m.name] ?? {
          label: m.name,
          color: "#6b7280",
        };
        const rawPct = Math.round(m.calibrated_score * 100);
        const pct = rawPct;

        return (
          <div key={m.name} className="space-y-1">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium text-zinc-300">{meta.label}</span>
              <span className="font-mono text-zinc-400">{pct}%</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-zinc-800">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${Math.max(pct, pct > 0 ? 2 : 0)}%`,
                  backgroundColor: meta.color,
                }}
              />
            </div>
            <p className="text-xs text-zinc-500">{m.description}</p>
          </div>
        );
      })}
    </div>
  );
}
