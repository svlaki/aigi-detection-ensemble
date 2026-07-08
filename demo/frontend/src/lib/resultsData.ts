/**
 * Static experiment results data, copied from results/ CSV/JSON files.
 * No API calls needed — this data is fixed research output.
 */

// --- Phase 5: Combiner comparison (phase5_results.csv) ---

export interface CombinerRow {
  readonly method: string;
  readonly label: string;
  readonly auroc: number;
  readonly acc: number;
  readonly real_acc: number;
  readonly fake_acc: number;
}

export const COMBINER_INDIST: readonly CombinerRow[] = [
  { method: "M1_cal", label: "M1 CLIP", auroc: 0.882, acc: 0.814, real_acc: 0.834, fake_acc: 0.794 },
  { method: "M2_cal", label: "M2 Spectral", auroc: 0.584, acc: 0.562, real_acc: 0.578, fake_acc: 0.545 },
  { method: "M3_cal", label: "M3 D3QE", auroc: 0.854, acc: 0.770, real_acc: 0.742, fake_acc: 0.798 },
  { method: "mean_prob", label: "Mean Prob", auroc: 0.927, acc: 0.850, real_acc: 0.861, fake_acc: 0.838 },
  { method: "majority_vote", label: "Majority Vote", auroc: 0.863, acc: 0.816, real_acc: 0.814, fake_acc: 0.818 },
  { method: "combiner_logreg", label: "Combiner LR", auroc: 0.938, acc: 0.864, real_acc: 0.888, fake_acc: 0.840 },
  { method: "combiner_mlp", label: "Combiner MLP", auroc: 0.939, acc: 0.867, real_acc: 0.892, fake_acc: 0.841 },
] as const;

export const COMBINER_OOD: readonly CombinerRow[] = [
  { method: "M1_cal", label: "M1 CLIP", auroc: 0.757, acc: 0.622, real_acc: 0.412, fake_acc: 0.898 },
  { method: "M2_cal", label: "M2 Spectral", auroc: 0.488, acc: 0.493, real_acc: 0.581, fake_acc: 0.378 },
  { method: "M3_cal", label: "M3 D3QE", auroc: 0.642, acc: 0.633, real_acc: 0.826, fake_acc: 0.381 },
  { method: "mean_prob", label: "Mean Prob", auroc: 0.738, acc: 0.670, real_acc: 0.740, fake_acc: 0.579 },
  { method: "majority_vote", label: "Majority Vote", auroc: 0.647, acc: 0.612, real_acc: 0.660, fake_acc: 0.549 },
  { method: "combiner_logreg", label: "Combiner LR", auroc: 0.724, acc: 0.681, real_acc: 0.731, fake_acc: 0.615 },
  { method: "combiner_mlp", label: "Combiner MLP", auroc: 0.720, acc: 0.679, real_acc: 0.728, fake_acc: 0.614 },
] as const;

// --- Decorrelation (decorrelation_*.csv, decorrelation_summary.json) ---

export const MEMBER_LABELS = ["M1", "M2", "M3"] as const;

export const ERROR_CORR_MATRIX: readonly (readonly number[])[] = [
  [1.0, 0.007, 0.049],
  [0.007, 1.0, 0.054],
  [0.049, 0.054, 1.0],
] as const;

export const LOGIT_CORR_MATRIX: readonly (readonly number[])[] = [
  [1.0, 0.067, 0.154],
  [0.067, 1.0, 0.031],
  [0.154, 0.031, 1.0],
] as const;

export const DECORRELATION_SUMMARY = {
  eval_n: 3339,
  eval_fakes_n: 1687,
  member_acc: { M1: 0.543, M2: 0.501, M3: 0.573 },
  oracle_any_correct: 0.992,
  mean_err_corr_evalfakes: 0.037,
} as const;

// --- Cross-generator AUROC (benchmark_genimage_xgen.csv) ---

export const CROSS_GEN_LABELS = ["ADM", "BigGAN", "GLIDE", "Midjourney", "Wukong", "VQ-DM"] as const;

export const CROSS_GEN_MATRIX: readonly (readonly number[])[] = [
  [0.999, 0.915, 0.925, 0.928, 0.884, 0.749],
  [0.834, 1.000, 0.974, 0.613, 0.733, 0.976],
  [0.802, 0.929, 0.997, 0.824, 0.917, 0.868],
  [0.840, 0.590, 0.894, 1.000, 0.946, 0.694],
  [0.820, 0.796, 0.947, 0.928, 1.000, 0.796],
  [0.787, 0.996, 0.948, 0.703, 0.785, 0.993],
] as const;

// --- LoRA ablation (phase6_metrics.json) ---

export interface LoraResult {
  readonly split: string;
  readonly label: string;
  readonly m1a_auroc: number;
  readonly m1a_acc: number;
  readonly m1b_auroc: number;
  readonly m1b_acc: number;
}

export const LORA_ABLATION: readonly LoraResult[] = [
  { split: "modern_test", label: "Modern Test", m1a_auroc: 0.995, m1a_acc: 0.966, m1b_auroc: 0.9999, m1b_acc: 0.989 },
  { split: "cf", label: "Community Forensics", m1a_auroc: 0.744, m1a_acc: 0.654, m1b_auroc: 0.767, m1b_acc: 0.666 },
  { split: "eval", label: "Full Eval (OOD)", m1a_auroc: 0.886, m1a_acc: 0.794, m1b_auroc: 0.907, m1b_acc: 0.811 },
] as const;

// --- LoRA combiner fold (phase6_combiner_fold.csv) ---

export const LORA_COMBINER_FOLD = [
  { config: "combiner_M1a", subset: "eval", auroc: 0.733, acc: 0.674 },
  { config: "combiner_M1a", subset: "eval_modern", auroc: 0.716, acc: 0.668 },
  { config: "combiner_M1b", subset: "eval", auroc: 0.787, acc: 0.718 },
  { config: "combiner_M1b", subset: "eval_modern", auroc: 0.789, acc: 0.723 },
] as const;

// --- LoRA sweep (lora_sweep.csv, M1b rows only) ---

export interface LoraSweepRow {
  readonly rank: number;
  readonly epochs: number;
  readonly split: string;
  readonly auroc: number;
  readonly acc: number;
}

export const LORA_SWEEP: readonly LoraSweepRow[] = [
  { rank: 4, epochs: 5, split: "modern_test", auroc: 1.000, acc: 0.997 },
  { rank: 4, epochs: 5, split: "cf", auroc: 0.747, acc: 0.685 },
  { rank: 4, epochs: 5, split: "eval", auroc: 0.903, acc: 0.825 },
  { rank: 4, epochs: 10, split: "modern_test", auroc: 0.999, acc: 0.995 },
  { rank: 4, epochs: 10, split: "cf", auroc: 0.770, acc: 0.680 },
  { rank: 4, epochs: 10, split: "eval", auroc: 0.910, acc: 0.822 },
  { rank: 8, epochs: 5, split: "modern_test", auroc: 1.000, acc: 0.995 },
  { rank: 8, epochs: 5, split: "cf", auroc: 0.710, acc: 0.675 },
  { rank: 8, epochs: 5, split: "eval", auroc: 0.887, acc: 0.819 },
  { rank: 8, epochs: 10, split: "modern_test", auroc: 1.000, acc: 0.997 },
  { rank: 8, epochs: 10, split: "cf", auroc: 0.745, acc: 0.665 },
  { rank: 8, epochs: 10, split: "eval", auroc: 0.901, acc: 0.814 },
  { rank: 16, epochs: 5, split: "modern_test", auroc: 1.000, acc: 0.994 },
  { rank: 16, epochs: 5, split: "cf", auroc: 0.773, acc: 0.690 },
  { rank: 16, epochs: 5, split: "eval", auroc: 0.910, acc: 0.827 },
  { rank: 16, epochs: 10, split: "modern_test", auroc: 1.000, acc: 0.997 },
  { rank: 16, epochs: 10, split: "cf", auroc: 0.757, acc: 0.691 },
  { rank: 16, epochs: 10, split: "eval", auroc: 0.906, acc: 0.828 },
  { rank: 32, epochs: 5, split: "modern_test", auroc: 1.000, acc: 0.997 },
  { rank: 32, epochs: 5, split: "cf", auroc: 0.758, acc: 0.710 },
  { rank: 32, epochs: 5, split: "eval", auroc: 0.905, acc: 0.839 },
  { rank: 32, epochs: 10, split: "modern_test", auroc: 1.000, acc: 0.997 },
  { rank: 32, epochs: 10, split: "cf", auroc: 0.735, acc: 0.690 },
  { rank: 32, epochs: 10, split: "eval", auroc: 0.895, acc: 0.827 },
] as const;

// --- Negative controls (negative_controls.json) ---

export const NEGATIVE_CONTROLS = {
  label_permutation: { real_auroc: 0.733, shuffled_mean: 0.475, shuffled_std: 0.133 },
  random_features: { auroc: 0.490 },
  member_shuffle: {
    m1_real: 0.753, m1_shuffled: 0.559,
    m2_real: 0.517, m2_shuffled: 0.490,
  },
} as const;

// --- UnivFD reproduction (univfd_reproduction.json) ---

export interface UnivFDRow {
  readonly domain: string;
  readonly label: string;
  readonly published_ap: number;
  readonly reproduced_ap: number;
  readonly delta: number;
}

export const UNIVFD_REPRODUCTION: readonly UnivFDRow[] = [
  { domain: "guided", label: "Guided Diffusion", published_ap: 87.77, reproduced_ap: 88.27, delta: 0.50 },
  { domain: "ldm_200", label: "LDM-200", published_ap: 99.14, reproduced_ap: 99.40, delta: 0.26 },
  { domain: "ldm_200_cfg", label: "LDM-200 CFG", published_ap: 92.15, reproduced_ap: 93.22, delta: 1.07 },
  { domain: "ldm_100", label: "LDM-100", published_ap: 99.17, reproduced_ap: 99.35, delta: 0.18 },
  { domain: "glide_100_27", label: "GLIDE 100/27", published_ap: 94.74, reproduced_ap: 95.79, delta: 1.05 },
  { domain: "glide_50_27", label: "GLIDE 50/27", published_ap: 95.34, reproduced_ap: 96.03, delta: 0.69 },
  { domain: "glide_100_10", label: "GLIDE 100/10", published_ap: 94.57, reproduced_ap: 95.48, delta: 0.91 },
  { domain: "dalle", label: "DALL-E", published_ap: 97.15, reproduced_ap: 97.73, delta: 0.58 },
] as const;

export const UNIVFD_MEAN_AP = {
  published: 95.0,
  reproduced: 95.66,
} as const;
