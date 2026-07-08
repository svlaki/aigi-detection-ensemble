import { NEGATIVE_CONTROLS } from "@/lib/resultsData";

function ControlCard({
  title,
  realValue,
  baselineValue,
  baselineLabel,
  unit = "AUROC",
}: {
  readonly title: string;
  readonly realValue: number;
  readonly baselineValue: number;
  readonly baselineLabel: string;
  readonly unit?: string;
}) {
  const realPct = Math.round(realValue * 100);
  const basePct = Math.round(baselineValue * 100);

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/50 p-4 space-y-3">
      <h4 className="text-sm font-medium text-zinc-300">{title}</h4>
      <div className="space-y-2">
        <div>
          <div className="flex justify-between text-xs text-zinc-400 mb-1">
            <span>Real model</span>
            <span className="font-mono">{(realValue * 100).toFixed(1)}% {unit}</span>
          </div>
          <div className="h-2 rounded-full bg-zinc-800">
            <div
              className="h-full rounded-full bg-green-500 transition-all"
              style={{ width: `${realPct}%` }}
            />
          </div>
        </div>
        <div>
          <div className="flex justify-between text-xs text-zinc-400 mb-1">
            <span>{baselineLabel}</span>
            <span className="font-mono">{(baselineValue * 100).toFixed(1)}% {unit}</span>
          </div>
          <div className="h-2 rounded-full bg-zinc-800">
            <div
              className="h-full rounded-full bg-red-500/60 transition-all"
              style={{ width: `${basePct}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

export function NegativeControlsPanel() {
  const ctrl = NEGATIVE_CONTROLS;

  return (
    <div className="grid gap-4 sm:grid-cols-3">
      <ControlCard
        title="A: Label Permutation"
        realValue={ctrl.label_permutation.real_auroc}
        baselineValue={ctrl.label_permutation.shuffled_mean}
        baselineLabel={`Shuffled (${"\u00B1"}${(ctrl.label_permutation.shuffled_std * 100).toFixed(0)}%)`}
      />
      <ControlCard
        title="B: Random Features"
        realValue={ctrl.label_permutation.real_auroc}
        baselineValue={ctrl.random_features.auroc}
        baselineLabel="Random input"
      />
      <ControlCard
        title="C: Member Shuffle"
        realValue={ctrl.member_shuffle.m1_real}
        baselineValue={ctrl.member_shuffle.m1_shuffled}
        baselineLabel="Shuffled labels"
      />
    </div>
  );
}
