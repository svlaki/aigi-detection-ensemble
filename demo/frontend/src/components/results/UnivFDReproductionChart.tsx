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
import { UNIVFD_REPRODUCTION, UNIVFD_MEAN_AP } from "@/lib/resultsData";

const tooltipStyle = {
  backgroundColor: "#18181b",
  border: "1px solid #3f3f46",
  borderRadius: "8px",
  color: "#e4e4e7",
  fontSize: "12px",
};

export function UnivFDReproductionChart() {
  const data = UNIVFD_REPRODUCTION.map((row) => ({
    label: row.label,
    published: row.published_ap,
    reproduced: row.reproduced_ap,
  }));

  return (
    <div className="space-y-3">
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} layout="vertical" barCategoryGap="16%" barGap={2}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" horizontal={false} />
          <XAxis
            type="number"
            domain={[80, 100]}
            tick={{ fill: "#a1a1aa", fontSize: 11 }}
            stroke="#52525b"
            tickFormatter={(v: number) => `${v}%`}
          />
          <YAxis
            type="category"
            dataKey="label"
            tick={{ fill: "#a1a1aa", fontSize: 11 }}
            stroke="#52525b"
            width={100}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(value) => `${Number(value).toFixed(2)}%`}
          />
          <Legend wrapperStyle={{ fontSize: "11px", color: "#a1a1aa" }} />
          <Bar dataKey="published" name="Published (Ojha et al.)" fill="#6b7280" radius={[0, 2, 2, 0]} />
          <Bar dataKey="reproduced" name="Reproduced (Ours)" fill="#22c55e" radius={[0, 2, 2, 0]} />
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs text-zinc-500">
        Mean AP: Published {UNIVFD_MEAN_AP.published.toFixed(1)}% vs. Reproduced {UNIVFD_MEAN_AP.reproduced.toFixed(1)}%.
        All per-domain deltas &lt; 1.1% — faithful reproduction of the UnivFD baseline.
      </p>
    </div>
  );
}
