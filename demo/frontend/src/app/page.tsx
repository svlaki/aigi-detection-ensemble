"use client";

import { useCallback, useState } from "react";
import { ImageUploader } from "@/components/ImageUploader";
import { ResultsPanel } from "@/components/ResultsPanel";
import { usePrediction } from "@/hooks/usePrediction";
import { SectionWrapper } from "@/components/results/SectionWrapper";
import { StatCard } from "@/components/results/StatCard";
import { CorrelationHeatmap } from "@/components/results/CorrelationHeatmap";
import { CombinerComparisonChart } from "@/components/results/CombinerComparisonChart";
import { GeneralizationGapChart } from "@/components/results/GeneralizationGapChart";
import { LoraAblationChart } from "@/components/results/LoraAblationChart";
import { LoraSweepChart } from "@/components/results/LoraSweepChart";
import { NegativeControlsPanel } from "@/components/results/NegativeControlsPanel";
import { UnivFDReproductionChart } from "@/components/results/UnivFDReproductionChart";
import { RobustnessLineChart } from "@/components/results/RobustnessChart";
import { RobustnessHeatmap } from "@/components/results/RobustnessHeatmap";
import {
  ERROR_CORR_MATRIX,
  LOGIT_CORR_MATRIX,
  MEMBER_LABELS,
  CROSS_GEN_LABELS,
  CROSS_GEN_MATRIX,
} from "@/lib/resultsData";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const { status, result, error, run, reset } = usePrediction();

  const handleFileSelected = useCallback(
    (f: File) => {
      setFile(f);
      reset();
    },
    [reset]
  );

  const handleAnalyze = useCallback(() => {
    if (file) run(file);
  }, [file, run]);

  return (
    <main className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-8 px-4 py-12">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-3xl font-bold tracking-tight text-white">
          AI Image Detector
        </h1>
        <p className="mt-2 text-sm text-zinc-400">
          Ensemble of decorrelated detectors: CLIP visual features, FFT spectral
          analysis, and D3QE codebook residuals
        </p>
      </div>

      {/* Main content */}
      <div className="grid gap-8 md:grid-cols-2">
        {/* Left: Upload + Controls */}
        <div className="space-y-4">
          <ImageUploader
            onFileSelected={handleFileSelected}
            disabled={status === "loading"}
          />

          <button
            type="button"
            onClick={handleAnalyze}
            disabled={!file || status === "loading"}
            className="w-full rounded-lg bg-blue-600 px-4 py-3 font-semibold text-white transition-colors hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {status === "loading" ? (
              <span className="flex items-center justify-center gap-2">
                <svg
                  className="h-4 w-4 animate-spin"
                  viewBox="0 0 24 24"
                  fill="none"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  />
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                  />
                </svg>
                Analyzing...
              </span>
            ) : (
              "Analyze Image"
            )}
          </button>
        </div>

        {/* Right: Results */}
        <div>
          {result && <ResultsPanel result={result} />}

          {error && (
            <div className="rounded-xl border border-red-500/30 bg-red-500/10 p-6 text-center">
              <p className="text-sm text-red-400">{error}</p>
              <button
                type="button"
                onClick={reset}
                className="mt-3 text-xs text-red-300 underline hover:text-red-200"
              >
                Dismiss
              </button>
            </div>
          )}

          {status === "idle" && !result && (
            <div className="flex h-full min-h-[200px] items-center justify-center rounded-xl border border-zinc-800 bg-zinc-900/50">
              <p className="text-sm text-zinc-500">
                Upload an image to get started
              </p>
            </div>
          )}
        </div>
      </div>

      {/* How it works */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/30 p-6">
        <h2 className="mb-3 text-sm font-semibold text-zinc-300">
          How It Works
        </h2>
        <div className="grid gap-4 text-xs text-zinc-400 sm:grid-cols-3">
          <div>
            <span className="font-medium text-blue-400">M1: CLIP Visual</span>
            <p className="mt-1">
              Frozen ViT-L/14 embeddings detect semantic artifacts in
              AI-generated content that differ from real photography.
            </p>
          </div>
          <div>
            <span className="font-medium text-purple-400">
              M2: Spectral FFT
            </span>
            <p className="mt-1">
              Frequency-domain analysis catches upsampling periodicities and
              anomalous high-frequency falloff left by generators.
            </p>
          </div>
          <div>
            <span className="font-medium text-amber-400">
              M3: D3QE Codebook
            </span>
            <p className="mt-1">
              VQ-VAE codebook residuals fused with CLIP detect autoregressive
              generation artifacts.
            </p>
          </div>
        </div>
      </div>

      {/* Divider */}
      <div className="border-t border-zinc-800 pt-4">
        <h2 className="text-center text-2xl font-bold text-white">
          Experiment Results
        </h2>
        <p className="mt-1 text-center text-sm text-zinc-400">
          Evaluation on 25,046 images across 22 generators with generator-disjoint train/eval splits
        </p>
      </div>

      {/* Hero Stats */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatCard value="0.907" label="AUROC" sublabel="LoRA-enhanced CLIP (M1b)" color="#3b82f6" />
        <StatCard value="0.037" label="Error Correlation" sublabel="Near-zero = decorrelated" color="#22c55e" />
        <StatCard value="99.2%" label="Oracle Accuracy" sublabel="Any member correct" color="#a855f7" />
        <StatCard value="+5.4%" label="AUROC Lift" sublabel="LoRA on combiner" color="#f59e0b" />
      </div>

      {/* Decorrelation */}
      <SectionWrapper
        id="decorrelation"
        title="Why Ensemble Works"
        description="Near-zero error correlation between members means they fail on different images — the key prerequisite for ensemble benefit."
      >
        <div className="grid gap-6 sm:grid-cols-2">
          <CorrelationHeatmap
            matrix={ERROR_CORR_MATRIX}
            labels={[...MEMBER_LABELS]}
            title="Error Correlation (Eval Fakes)"
            highColor="#ef4444"
          />
          <CorrelationHeatmap
            matrix={LOGIT_CORR_MATRIX}
            labels={[...MEMBER_LABELS]}
            title="Logit Correlation (All Eval)"
          />
        </div>
        <div className="rounded-lg border border-zinc-800 bg-zinc-900/30 p-3 text-center text-sm text-zinc-400">
          Oracle accuracy: <span className="font-mono font-bold text-purple-400">99.2%</span> — if any single member
          gets it right, the ensemble could too. Actual combiner captures ~68% of this headroom.
        </div>
      </SectionWrapper>

      {/* Combiner Comparison */}
      <SectionWrapper
        id="combiner"
        title="Ensemble vs. Individual Members"
        description="The learned combiner improves class balance — fake detection accuracy jumps from 38% (best member) to 62%."
      >
        <CombinerComparisonChart />
      </SectionWrapper>

      {/* Generalization Gap */}
      <SectionWrapper
        id="generalization"
        title="In-Distribution vs. OOD Generalization"
        description="Performance on training generators vs. held-out unseen generators reveals the domain shift challenge."
      >
        <GeneralizationGapChart />
      </SectionWrapper>

      {/* Cross-Generator Heatmap */}
      <SectionWrapper
        id="cross-generator"
        title="Cross-Generator Transfer"
        description="6x6 AUROC matrix: train on one generator, test on another. Strong diagonal (0.998) with solid cross-generator transfer (0.845 mean)."
      >
        <CorrelationHeatmap
          matrix={CROSS_GEN_MATRIX}
          labels={[...CROSS_GEN_LABELS]}
          title="AUROC: Train Generator (row) vs. Test Generator (column)"
          lowColor="#7f1d1d"
          highColor="#22c55e"
          showDiagonal
        />
      </SectionWrapper>

      {/* Robustness */}
      <SectionWrapper
        id="robustness"
        title="Robustness Under Perturbation"
        description="AUROC across 14 image perturbations (JPEG, blur, noise, resize, social media pipeline). M3 D3QE is the most stable; M1 CLIP degrades most under compression and noise."
      >
        <RobustnessLineChart />
        <RobustnessHeatmap />
        <p className="text-xs text-zinc-500">
          Social media pipeline (resize + JPEG + blur) is the worst case for M1 (−0.138 AUROC),
          while M3 stays within ±0.05 for most perturbations. Mild blur actually improves several methods
          by introducing the frequency artifacts M2 looks for.
        </p>
      </SectionWrapper>

      {/* LoRA */}
      <SectionWrapper
        id="lora"
        title="LoRA Fine-Tuning Impact"
        description="LoRA adapter on CLIP ViT-L/14 improves detection of modern generators (Flux, SD3.5, Midjourney) without catastrophic forgetting."
      >
        <LoraAblationChart />
      </SectionWrapper>

      {/* LoRA Sweep */}
      <SectionWrapper
        id="lora-sweep"
        title="LoRA Hyperparameter Sweep"
        description="Rank and epoch sweep reveals the sweet spot — higher rank helps, but more epochs risk overfitting on cross-generator evaluation."
      >
        <LoraSweepChart />
      </SectionWrapper>

      {/* Negative Controls */}
      <SectionWrapper
        id="controls"
        title="Negative Controls"
        description="Sanity checks confirm the combiner learns real signal, not artifacts. All controls collapse toward chance (50%)."
      >
        <NegativeControlsPanel />
      </SectionWrapper>

      {/* UnivFD Reproduction */}
      <SectionWrapper
        id="univfd"
        title="UnivFD Baseline Reproduction"
        description="Per-domain average precision matches the published UnivFD results (Ojha et al., CVPR 2023) within 1.1%, validating our M1 CLIP probe implementation."
      >
        <UnivFDReproductionChart />
      </SectionWrapper>
    </main>
  );
}
