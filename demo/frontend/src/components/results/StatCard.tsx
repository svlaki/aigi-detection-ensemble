interface StatCardProps {
  readonly value: string;
  readonly label: string;
  readonly sublabel?: string;
  readonly color?: string;
}

export function StatCard({ value, label, sublabel, color = "#3b82f6" }: StatCardProps) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4 text-center">
      <div className="font-mono text-2xl font-bold" style={{ color }}>
        {value}
      </div>
      <div className="mt-1 text-sm font-medium text-zinc-300">{label}</div>
      {sublabel && (
        <div className="mt-0.5 text-xs text-zinc-500">{sublabel}</div>
      )}
    </div>
  );
}
