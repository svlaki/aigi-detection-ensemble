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
import { LORA_ABLATION, LORA_COMBINER_FOLD } from "@/lib/resultsData";

const tooltipStyle = {
  backgroundColor: "#18181b",
  border: "1px solid #3f3f46",
  borderRadius: "8px",
  color: "#e4e4e7",
  fontSize: "12px",
};

export function LoraAblationChart() {
  const memberData = LORA_ABLATION.map((row) => ({
    label: row.label,
    m1a_auroc: row.m1a_auroc,
    m1b_auroc: row.m1b_auroc,
    delta: `+${((row.m1b_auroc - row.m1a_auroc) * 100).toFixed(1)}%`,
  }));

  const combinerData = [
    {
      label: "Full Eval",
      m1a: LORA_COMBINER_FOLD.find((r) => r.config === "combiner_M1a" && r.subset === "eval")?.auroc ?? 0,
      m1b: LORA_COMBINER_FOLD.find((r) => r.config === "combiner_M1b" && r.subset === "eval")?.auroc ?? 0,
    },
    {
      label: "Modern Subset",
      m1a: LORA_COMBINER_FOLD.find((r) => r.config === "combiner_M1a" && r.subset === "eval_modern")?.auroc ?? 0,
      m1b: LORA_COMBINER_FOLD.find((r) => r.config === "combiner_M1b" && r.subset === "eval_modern")?.auroc ?? 0,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Member-level */}
      <div>
        <h4 className="mb-2 text-sm font-medium text-zinc-400">
          M1 CLIP Probe: Frozen vs. LoRA Fine-tuned
        </h4>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={memberData} barCategoryGap="24%" barGap={4}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis
              dataKey="label"
              tick={{ fill: "#a1a1aa", fontSize: 11 }}
              stroke="#52525b"
            />
            <YAxis
              domain={[0.6, 1]}
              tick={{ fill: "#a1a1aa", fontSize: 11 }}
              stroke="#52525b"
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`}
            />
            <Legend wrapperStyle={{ fontSize: "11px", color: "#a1a1aa" }} />
            <Bar dataKey="m1a_auroc" name="M1a (Frozen)" fill="#6b7280" radius={[2, 2, 0, 0]} />
            <Bar dataKey="m1b_auroc" name="M1b (LoRA)" fill="#3b82f6" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Combiner-level */}
      <div>
        <h4 className="mb-2 text-sm font-medium text-zinc-400">
          Ensemble Combiner: With M1a vs. M1b
        </h4>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={combinerData} barCategoryGap="30%" barGap={4}>
            <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
            <XAxis
              dataKey="label"
              tick={{ fill: "#a1a1aa", fontSize: 11 }}
              stroke="#52525b"
            />
            <YAxis
              domain={[0.6, 0.85]}
              tick={{ fill: "#a1a1aa", fontSize: 11 }}
              stroke="#52525b"
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`}
            />
            <Legend wrapperStyle={{ fontSize: "11px", color: "#a1a1aa" }} />
            <Bar dataKey="m1a" name="Combiner + M1a" fill="#6b7280" radius={[2, 2, 0, 0]} />
            <Bar dataKey="m1b" name="Combiner + M1b" fill="#f59e0b" radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-zinc-500">
          LoRA fine-tuning lifts the full ensemble combiner by +5.4% AUROC on full eval
          and +7.2% on the modern generator subset.
        </p>
      </div>
    </div>
  );
}
