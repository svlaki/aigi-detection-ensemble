"use client";

import { useState } from "react";
import {
  ROBUSTNESS_DATA,
  ROBUSTNESS_HEATMAP_METHODS,
  ROBUSTNESS_METHOD_LABELS,
} from "@/lib/resultsData";

function deltaColor(delta: number): string {
  // Green for positive (improvement), red for negative (degradation)
  if (delta >= 0) {
    const t = Math.min(delta / 0.15, 1);
    const r = Math.round(34 + (34 - 34) * t);
    const g = Math.round(197 + (197 - 197) * t);
    const b = Math.round(94 + (94 - 94) * t);
    return `rgba(34, 197, 94, ${0.15 + t * 0.6})`;
  }
  const t = Math.min(Math.abs(delta) / 0.15, 1);
  return `rgba(239, 68, 68, ${0.15 + t * 0.6})`;
}

function textColor(delta: number): string {
  return Math.abs(delta) > 0.08 ? "#e4e4e7" : "#a1a1aa";
}

export function RobustnessHeatmap() {
  const [hovered, setHovered] = useState<{ row: number; col: number } | null>(
    null
  );

  // Skip "clean" for the heatmap — show only perturbation deltas
  const perturbedRows = ROBUSTNESS_DATA.filter(
    (r) => r.perturbation !== "clean"
  );
  const cleanRow = ROBUSTNESS_DATA[0];

  const methods = [...ROBUSTNESS_HEATMAP_METHODS];
  const pertLabels = perturbedRows.map((r) => r.label);

  const cellW = 72;
  const cellH = 32;
  const labelW = 96;
  const topPad = 28;
  const svgWidth = labelW + methods.length * cellW;
  const svgHeight = topPad + perturbedRows.length * cellH;

  return (
    <div className="space-y-3">
      <h4 className="text-sm font-medium text-zinc-400">
        AUROC Degradation From Clean Baseline
      </h4>
      <div className="overflow-x-auto">
        <svg
          width="100%"
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="max-w-full"
        >
          {/* Column headers */}
          {methods.map((m, j) => (
            <text
              key={`col-${m}`}
              x={labelW + j * cellW + cellW / 2}
              y={18}
              textAnchor="middle"
              className="fill-zinc-400 text-[10px]"
            >
              {ROBUSTNESS_METHOD_LABELS[m]}
            </text>
          ))}

          {/* Rows */}
          {perturbedRows.map((row, i) => (
            <g key={row.perturbation}>
              {/* Row label */}
              <text
                x={labelW - 6}
                y={topPad + i * cellH + cellH / 2 + 4}
                textAnchor="end"
                className="fill-zinc-400 text-[10px]"
              >
                {pertLabels[i]}
              </text>

              {/* Cells */}
              {methods.map((m, j) => {
                const cleanVal =
                  cleanRow[m as keyof typeof cleanRow] as number;
                const pertVal = row[m as keyof typeof row] as number;
                const delta = pertVal - cleanVal;
                const isHov = hovered?.row === i && hovered?.col === j;

                return (
                  <g
                    key={`${row.perturbation}-${m}`}
                    onMouseEnter={() => setHovered({ row: i, col: j })}
                    onMouseLeave={() => setHovered(null)}
                  >
                    <rect
                      x={labelW + j * cellW + 1}
                      y={topPad + i * cellH + 1}
                      width={cellW - 2}
                      height={cellH - 2}
                      rx={3}
                      fill={deltaColor(delta)}
                      stroke={isHov ? "#e4e4e7" : "none"}
                      strokeWidth={isHov ? 1.5 : 0}
                    />
                    <text
                      x={labelW + j * cellW + cellW / 2}
                      y={topPad + i * cellH + cellH / 2 + 4}
                      textAnchor="middle"
                      fill={textColor(delta)}
                      className="text-[10px] font-mono"
                    >
                      {delta >= 0 ? "+" : ""}
                      {delta.toFixed(3)}
                    </text>
                  </g>
                );
              })}
            </g>
          ))}
        </svg>
      </div>
      <div className="flex items-center justify-center gap-4 text-[10px] text-zinc-500">
        <span className="flex items-center gap-1">
          <span
            className="inline-block h-3 w-3 rounded"
            style={{ backgroundColor: "rgba(239, 68, 68, 0.6)" }}
          />
          Degradation
        </span>
        <span className="flex items-center gap-1">
          <span
            className="inline-block h-3 w-3 rounded"
            style={{ backgroundColor: "rgba(34, 197, 94, 0.6)" }}
          />
          Improvement
        </span>
      </div>
    </div>
  );
}
