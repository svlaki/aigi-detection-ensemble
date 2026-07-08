"use client";

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

const tooltipStyle = {
  backgroundColor: "#18181b",
  border: "1px solid #3f3f46",
  borderRadius: "8px",
  color: "#e4e4e7",
  fontSize: "12px",
};

export function GeneralizationGapChart() {
  const data = COMBINER_INDIST.map((indist) => {
    const ood = COMBINER_OOD.find((o) => o.method === indist.method);
    return {
      label: indist.label,
      indist_acc: indist.acc,
      ood_acc: ood?.acc ?? 0,
      gap: indist.acc - (ood?.acc ?? 0),
    };
  });

  return (
    <div className="space-y-3">
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} barCategoryGap="20%" barGap={4}>
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
          <Legend wrapperStyle={{ fontSize: "11px", color: "#a1a1aa" }} />
          <Bar dataKey="indist_acc" name="In-Distribution" fill="#22c55e" radius={[2, 2, 0, 0]} />
          <Bar dataKey="ood_acc" name="OOD (Held-out)" fill="#ef4444" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-zinc-500">
        All methods drop ~18% accuracy from in-distribution to OOD evaluation, reflecting the domain shift
        challenge when encountering unseen generators.
      </p>
    </div>
  );
}
