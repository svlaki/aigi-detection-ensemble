"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { ROBUSTNESS_DATA } from "@/lib/resultsData";

const tooltipStyle = {
  backgroundColor: "#18181b",
  border: "1px solid #3f3f46",
  borderRadius: "8px",
  color: "#e4e4e7",
  fontSize: "12px",
};

const SERIES = [
  { key: "M1_cal", name: "M1 CLIP", color: "#3b82f6" },
  { key: "M2_cal", name: "M2 Spectral", color: "#a855f7" },
  { key: "M3_cal", name: "M3 D3QE", color: "#f59e0b" },
  { key: "combiner_logreg", name: "Combiner LR", color: "#ef4444" },
  { key: "mean_prob", name: "Mean Prob", color: "#22c55e" },
] as const;

export function RobustnessLineChart() {
  return (
    <div className="space-y-3">
      <h4 className="text-sm font-medium text-zinc-400">
        AUROC Under Perturbation
      </h4>
      <ResponsiveContainer width="100%" height={340}>
        <LineChart data={[...ROBUSTNESS_DATA]} margin={{ bottom: 60 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            dataKey="label"
            tick={{ fill: "#a1a1aa", fontSize: 10 }}
            stroke="#52525b"
            interval={0}
            angle={-40}
            textAnchor="end"
            height={80}
          />
          <YAxis
            domain={[0.4, 0.8]}
            tick={{ fill: "#a1a1aa", fontSize: 11 }}
            stroke="#52525b"
            tickFormatter={(v: number) => v.toFixed(2)}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(value) => Number(value).toFixed(4)}
          />
          <Legend wrapperStyle={{ fontSize: "11px", color: "#a1a1aa" }} />
          <ReferenceLine y={0.5} stroke="#52525b" strokeDasharray="6 3" />
          {SERIES.map((s) => (
            <Line
              key={s.key}
              type="monotone"
              dataKey={s.key}
              name={s.name}
              stroke={s.color}
              strokeWidth={2}
              dot={{ r: 3, fill: s.color }}
              activeDot={{ r: 5 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
