export interface MemberResult {
  readonly name: string;
  readonly description: string;
  readonly raw_score: number;
  readonly calibrated_score: number;
}

export interface EnsembleResult {
  readonly name: string;
  readonly label: string;
  readonly confidence: number;
  readonly verdict: string;
}

export interface PredictionResponse {
  readonly verdict: string;
  readonly confidence: number;
  readonly mode: string;
  readonly members: readonly MemberResult[];
  readonly ensemble_methods: readonly EnsembleResult[];
  readonly processing_time_ms: number;
}

