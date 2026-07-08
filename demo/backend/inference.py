"""Model registry and single-image prediction pipeline.

Loads all trained models once at startup, exposes a predict() method that
mirrors the exact pipeline from scripts/build_member_outputs.py and
scripts/combiner.py.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from PIL import Image

from src import config, embeddings, spectral
from src.normalize import normalize_pil

from demo.backend.config import (
    LOAD_D3QE,
    REQUIRED_MODELS,
    D3QE_MODELS,
    NORM_JPEG_QUALITY,
)
from demo.backend.models import EnsembleResult, MemberResult, PredictionResponse


def _sigmoid(z: float) -> float:
    z_clipped = max(-50.0, min(50.0, z))
    return 1.0 / (1.0 + np.exp(-z_clipped))


def _check_models(paths: list[Path]) -> list[Path]:
    return [p for p in paths if not p.exists()]


class ModelRegistry:
    """Singleton model container. Created once at FastAPI startup."""

    def __init__(self) -> None:
        missing = _check_models(REQUIRED_MODELS)
        if missing:
            names = [p.name for p in missing]
            raise FileNotFoundError(
                f"Required model files missing: {names}. "
                "Re-run the training scripts (see README Phase 3/5)."
            )

        device = config.get_device()
        self._device = device

        self._clip_model, self._clip_preprocess, _ = embeddings.load_clip(device)
        self._m1_probe = joblib.load(config.MODELS_DIR / "m1a_clip_linear.joblib")
        self._m2_probe = joblib.load(config.MODELS_DIR / "m2_spectral.joblib")
        self._calib_m1 = joblib.load(config.MODELS_DIR / "calib_m1.joblib")
        self._calib_m2 = joblib.load(config.MODELS_DIR / "calib_m2.joblib")
        self._combiner = joblib.load(config.MODELS_DIR / "combiner_logreg.joblib")
        self._combiner_mlp = joblib.load(config.MODELS_DIR / "combiner_mlp.joblib")

        self._d3qe_model = None
        self._calib_m3 = None
        if LOAD_D3QE:
            d3qe_missing = _check_models(D3QE_MODELS)
            if d3qe_missing:
                names = [p.name for p in d3qe_missing]
                raise FileNotFoundError(
                    f"D3QE model files missing: {names}. "
                    "Download D3QE pretrained weights (see notes/data_sources.md)."
                )
            from src import d3qe
            self._d3qe_model = d3qe.load_model(device)
            self._calib_m3 = joblib.load(config.MODELS_DIR / "calib_m3.joblib")

    @property
    def device(self) -> str:
        return self._device

    @property
    def d3qe_available(self) -> bool:
        return self._d3qe_model is not None

    def predict(
        self, image: Image.Image, mode: Literal["fast", "full"] = "fast"
    ) -> PredictionResponse:
        start = time.perf_counter()

        normalized = normalize_pil(image)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            normalized.save(tmp_path, format="JPEG", quality=NORM_JPEG_QUALITY)

        try:
            members = []

            # M1: CLIP embedding -> linear probe
            clip_emb = embeddings.embed_paths(
                [tmp_path], self._clip_model, self._clip_preprocess, self._device
            )
            p1_raw = float(self._m1_probe.predict_proba(clip_emb)[:, 1][0])
            logit1 = float(self._m1_probe.decision_function(clip_emb)[0])
            p1_cal = float(self._calib_m1.predict_proba([[logit1]])[:, 1][0])
            z1 = float(self._calib_m1.decision_function([[logit1]])[0])
            members.append(MemberResult(
                name="M1_CLIP",
                description="Semantic visual features via CLIP ViT-L/14",
                raw_score=round(p1_raw, 4),
                calibrated_score=round(p1_cal, 4),
            ))

            # M2: Spectral FFT features -> linear probe
            spec_feat = spectral.features_from_image(tmp_path).reshape(1, -1)
            p2_raw = float(self._m2_probe.predict_proba(spec_feat)[:, 1][0])
            logit2 = float(self._m2_probe.decision_function(spec_feat)[0])
            p2_cal = float(self._calib_m2.predict_proba([[logit2]])[:, 1][0])
            z2 = float(self._calib_m2.decision_function([[logit2]])[0])
            members.append(MemberResult(
                name="M2_Spectral",
                description="Frequency-domain artifacts via FFT power spectrum",
                raw_score=round(p2_raw, 4),
                calibrated_score=round(p2_cal, 4),
            ))

            # M3: D3QE (optional, full mode only)
            use_d3qe = mode == "full" and self._d3qe_model is not None
            if use_d3qe:
                from src import d3qe
                logit3 = float(
                    d3qe.logits_for_paths([tmp_path], self._d3qe_model, self._device)[0]
                )
                p3_raw = _sigmoid(logit3)
                p3_cal = float(self._calib_m3.predict_proba([[logit3]])[:, 1][0])
                z3 = float(self._calib_m3.decision_function([[logit3]])[0])
                print(f"[D3QE debug] logit3={logit3:.4f}  p3_raw={p3_raw:.4f}  p3_cal={p3_cal:.4f}  z3={z3:.4f}")
                members.append(MemberResult(
                    name="M3_D3QE",
                    description="VQ-VAE codebook residuals (autoregressive detector)",
                    raw_score=round(p3_raw, 4),
                    calibrated_score=round(p3_cal, 4),
                ))

            # Combine — compute all applicable ensemble strategies
            def _verdict(p: float) -> str:
                return "AI-Generated" if p >= 0.5 else "Likely Real"

            ensemble_methods: list[EnsembleResult] = []

            if use_d3qe:
                cal_probs = [p1_cal, p2_cal, p3_cal]
                votes = [p1_cal >= 0.5, p2_cal >= 0.5, p3_cal >= 0.5]
                feature_vec = np.array([[
                    p1_cal, p2_cal, p3_cal,
                    z1, z2, z3,
                    abs(p1_cal - p2_cal),
                    abs(p1_cal - p3_cal),
                    abs(p2_cal - p3_cal),
                ]])

                p_mean = float(np.mean(cal_probs))
                p_vote = float(sum(votes)) / 3.0
                p_logreg = float(self._combiner.predict_proba(feature_vec)[:, 1][0])
                p_mlp = float(self._combiner_mlp.predict_proba(feature_vec)[:, 1][0])

                ensemble_methods = [
                    EnsembleResult(name="mean_prob", label="Mean Probability",
                                   confidence=round(p_mean, 4), verdict=_verdict(p_mean)),
                    EnsembleResult(name="majority_vote", label="Majority Vote",
                                   confidence=round(p_vote, 4), verdict=_verdict(p_vote)),
                    EnsembleResult(name="combiner_logreg", label="Learned Combiner (LogReg)",
                                   confidence=round(p_logreg, 4), verdict=_verdict(p_logreg)),
                    EnsembleResult(name="combiner_mlp", label="Learned Combiner (MLP)",
                                   confidence=round(p_mlp, 4), verdict=_verdict(p_mlp)),
                ]
                # Primary: combiner_logreg
                confidence = p_logreg
            else:
                p_mean = (p1_cal + p2_cal) / 2.0
                votes = [p1_cal >= 0.5, p2_cal >= 0.5]
                p_vote = float(sum(votes)) / 2.0

                ensemble_methods = [
                    EnsembleResult(name="mean_prob", label="Mean Probability",
                                   confidence=round(p_mean, 4), verdict=_verdict(p_mean)),
                    EnsembleResult(name="majority_vote", label="Majority Vote",
                                   confidence=round(p_vote, 4), verdict=_verdict(p_vote)),
                ]
                # Primary: mean_prob
                confidence = p_mean

            verdict = _verdict(confidence)
            elapsed_ms = (time.perf_counter() - start) * 1000

            return PredictionResponse(
                verdict=verdict,
                confidence=round(confidence, 4),
                mode=mode if use_d3qe or mode == "fast" else "fast",
                members=members,
                ensemble_methods=ensemble_methods,
                processing_time_ms=round(elapsed_ms, 1),
            )
        finally:
            tmp_path.unlink(missing_ok=True)
