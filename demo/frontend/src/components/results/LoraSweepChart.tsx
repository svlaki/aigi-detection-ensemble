"use client";

import { useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { LORA_SWEEP } from "@/lib/resultsData";

const tooltipStyle = {
  backgroundColor: "#18181b",
  border: "1px solid #3f3f46",
  borderRadius: "8px",
  color: "#e4e4e7",
  fontSize: "12px",
};

type SplitKey = "cf" | "eval" | "modern_test";

const SPLIT_OPTIONS: readonly { readonly key: SplitKey; readonly label: string }[] = [
  { key: "cf", label: "Community Forensics" },
  { key: "eval", label: "Full Eval (OOD)" },
  { key: "modern_test", label: "Modern Test" },
] as const;

export function LoraSweepChart() {
  const [activeSplit, setActiveSplit] = useState<SplitKey>("cf");

  const ranks = [4, 8, 16, 32];
  const data = ranks.map((rank) => {
    const ep5 = LORA_SWEEP.find((r) => r.rank === rank && r.epochs === 5 && r.split === activeSplit);
    const ep10 = LORA_SWEEP.find((r) => r.rank === rank && r.epochs === 10 && r.split === activeSplit);
    return {
      rank,
      ep5_auroc: ep5?.auroc ?? 0,
      ep10_auroc: ep10?.auroc ?? 0,
      ep5_acc: ep5?.acc ?? 0,
      ep10_acc: ep10?.acc ?? 0,
    };
  });

  return (
    <div className="space-y-3">
      <div className="flex gap-2">
        {SPLIT_OPTIONS.map((opt) => (
          <button
            key={opt.key}
            type="button"
            onClick={() => setActiveSplit(opt.key)}
            className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
              activeSplit === opt.key
                ? "bg-blue-600 text-white"
                : "bg-zinc-800 text-zinc-400 hover:text-zinc-300"
            }`}
          >
            {opt.label}
          </button>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
          <XAxis
            dataKey="rank"
            tick={{ fill: "#a1a1aa", fontSize: 11 }}
            stroke="#52525b"
            label={{ value: "LoRA Rank", position: "insideBottom", offset: -4, fill: "#71717a", fontSize: 11 }}
          />
          <YAxis
            tick={{ fill: "#a1a1aa", fontSize: 11 }}
            stroke="#52525b"
            tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            domain={["auto", "auto"]}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`}
          />
          <Legend wrapperStyle={{ fontSize: "11px", color: "#a1a1aa" }} />
          <Line
            type="monotone"
            dataKey="ep5_auroc"
            name="AUROC (5 epochs)"
            stroke="#3b82f6"
            strokeWidth={2}
            dot={{ fill: "#3b82f6", r: 4 }}
          />
          <Line
            type="monotone"
            dataKey="ep10_auroc"
            name="AUROC (10 epochs)"
            stroke="#f59e0b"
            strokeWidth={2}
            dot={{ fill: "#f59e0b", r: 4 }}
          />
          <Line
            type="monotone"
            dataKey="ep5_acc"
            name="Accuracy (5 epochs)"
            stroke="#3b82f6"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={{ fill: "#3b82f6", r: 3 }}
          />
          <Line
            type="monotone"
            dataKey="ep10_acc"
            name="Accuracy (10 epochs)"
            stroke="#f59e0b"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={{ fill: "#f59e0b", r: 3 }}
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="text-xs text-zinc-500">
        Best CF performance at rank=16-32, 5 epochs. Higher epochs show signs of overfitting on cross-generator data.
      </p>
    </div>
  );
}
