"use client";

import { useState } from "react";

interface CorrelationHeatmapProps {
  readonly matrix: readonly (readonly number[])[];
  readonly labels: readonly string[];
  readonly title: string;
  readonly lowColor?: string;
  readonly highColor?: string;
  readonly showDiagonal?: boolean;
}

function interpolateColor(value: number, low: string, high: string): string {
  const t = Math.max(0, Math.min(1, value));
  const lr = parseInt(low.slice(1, 3), 16);
  const lg = parseInt(low.slice(3, 5), 16);
  const lb = parseInt(low.slice(5, 7), 16);
  const hr = parseInt(high.slice(1, 3), 16);
  const hg = parseInt(high.slice(3, 5), 16);
  const hb = parseInt(high.slice(5, 7), 16);
  const r = Math.round(lr + (hr - lr) * t);
  const g = Math.round(lg + (hg - lg) * t);
  const b = Math.round(lb + (hb - lb) * t);
  return `rgb(${r},${g},${b})`;
}

export function CorrelationHeatmap({
  matrix,
  labels,
  title,
  lowColor = "#27272a",
  highColor = "#3b82f6",
  showDiagonal = false,
}: CorrelationHeatmapProps) {
  const [tooltip, setTooltip] = useState<{ row: number; col: number } | null>(null);
  const n = labels.length;
  const cellSize = n <= 4 ? 64 : 48;
  const labelWidth = n <= 4 ? 56 : 72;
  const svgWidth = labelWidth + n * cellSize;
  const svgHeight = 24 + n * cellSize;

  // Find max off-diagonal for scaling
  let maxVal = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j && matrix[i][j] > maxVal) maxVal = matrix[i][j];
    }
  }
  if (maxVal === 0) maxVal = 1;

  return (
    <div className="space-y-2">
      <h4 className="text-sm font-medium text-zinc-400">{title}</h4>
      <svg
        width="100%"
        viewBox={`0 0 ${svgWidth} ${svgHeight}`}
        className="max-w-full"
      >
        {/* Column labels */}
        {labels.map((label, j) => (
          <text
            key={`col-${j}`}
            x={labelWidth + j * cellSize + cellSize / 2}
            y={16}
            textAnchor="middle"
            className="fill-zinc-400 text-[11px]"
          >
            {label}
          </text>
        ))}

        {/* Rows */}
        {matrix.map((row, i) => (
          <g key={`row-${i}`}>
            {/* Row label */}
            <text
              x={labelWidth - 8}
              y={24 + i * cellSize + cellSize / 2 + 4}
              textAnchor="end"
              className="fill-zinc-400 text-[11px]"
            >
              {labels[i]}
            </text>

            {/* Cells */}
            {row.map((value, j) => {
              const isDiag = i === j;
              const normalized = isDiag ? 1 : value / maxVal;
              const bgColor = isDiag && !showDiagonal
                ? "#18181b"
                : interpolateColor(normalized, lowColor, highColor);
              const textColor = normalized > 0.6 || isDiag ? "#e4e4e7" : "#a1a1aa";
              const isHovered = tooltip?.row === i && tooltip?.col === j;

              return (
                <g
                  key={`cell-${i}-${j}`}
                  onMouseEnter={() => setTooltip({ row: i, col: j })}
                  onMouseLeave={() => setTooltip(null)}
                >
                  <rect
                    x={labelWidth + j * cellSize + 1}
                    y={24 + i * cellSize + 1}
                    width={cellSize - 2}
                    height={cellSize - 2}
                    rx={4}
                    fill={bgColor}
                    stroke={isHovered ? "#e4e4e7" : "none"}
                    strokeWidth={isHovered ? 1.5 : 0}
                  />
                  <text
                    x={labelWidth + j * cellSize + cellSize / 2}
                    y={24 + i * cellSize + cellSize / 2 + 4}
                    textAnchor="middle"
                    fill={textColor}
                    className="text-[11px] font-mono"
                  >
                    {isDiag && !showDiagonal ? "" : value.toFixed(3)}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
      </svg>
    </div>
  );
}
