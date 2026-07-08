"use client";

import { useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { COMBINER_INDIST, COMBINER_OOD } from "@/lib/resultsData";

const METRIC_COLORS = {
  auroc: "#3b82f6",
  acc: "#22c55e",
  real_acc: "#a855f7",
  fake_acc: "#f59e0b",
} as const;

const tooltipStyle = {
  backgroundColor: "#18181b",
  border: "1px solid #3f3f46",
  borderRadius: "8px",
  color: "#e4e4e7",
  fontSize: "12px",
};

export function CombinerComparisonChart() {
  const [split, setSplit] = useState<"ood" | "indist">("ood");
  const data = split === "ood" ? COMBINER_OOD : COMBINER_INDIST;

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => setSplit("ood")}
          className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
            split === "ood"
              ? "bg-blue-600 text-white"
              : "bg-zinc-800 text-zinc-400 hover:text-zinc-300"
          }`}
        >
          OOD (Held-out)
        </button>
        <button
          type="button"
          onClick={() => setSplit("indist")}
          className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
            split === "indist"
              ? "bg-blue-600 text-white"
              : "bg-zinc-800 text-zinc-400 hover:text-zinc-300"
          }`}
        >
          In-Distribution
        </button>
      </div>

      <ResponsiveContainer width="100%" height={320}>
        <BarChart data={[...data]} barCategoryGap="16%" barGap={2}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            dataKey="label"
            tick={{ fill: "#a1a1aa", fontSize: 11 }}
            stroke="#52525b"
            interval={0}
            angle={-30}
            textAnchor="end"
            height={60}
          />
          <YAxis
            domain={[0, 1]}
            tick={{ fill: "#a1a1aa", fontSize: 11 }}
            stroke="#52525b"
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`}
          />
          <Legend
            wrapperStyle={{ fontSize: "11px", color: "#a1a1aa" }}
          />
          <Bar dataKey="auroc" name="AUROC" fill={METRIC_COLORS.auroc} radius={[2, 2, 0, 0]} />
          <Bar dataKey="acc" name="Accuracy" fill={METRIC_COLORS.acc} radius={[2, 2, 0, 0]} />
          <Bar dataKey="real_acc" name="Real Acc" fill={METRIC_COLORS.real_acc} radius={[2, 2, 0, 0]} />
          <Bar dataKey="fake_acc" name="Fake Acc" fill={METRIC_COLORS.fake_acc} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>

      {split === "ood" && (
        <p className="text-xs text-zinc-500">
          The combiner dramatically improves class balance: best single member (M3) has 38% fake accuracy,
          while the learned combiner achieves 62% — a +24pp improvement.
        </p>
      )}
    </div>
  );
}
