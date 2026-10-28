#!/usr/bin/env python3
"""
    Shared helpers for the handling of configuration files.
"""
from functools import lru_cache
import math
import os
import ROOT
from dataclasses import dataclass
HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = os.path.normpath(os.path.join(HERE, "..", "headers", "corr2pc.h"))
@lru_cache(maxsize=1)


def _load_corr2pc():
    """Compile corr2pc.h once, on first use (not at import time)."""

    if not os.path.isfile(HEADER):
        raise FileNotFoundError(f"Correlation header not found: {HEADER}")

    if not ROOT.gInterpreter.Declare(f'#include "{HEADER}"'):
        raise RuntimeError(f"Failed to compile {HEADER}")

    if not hasattr(ROOT, "corr2pc"):
        raise RuntimeError("corr2pc namespace not available after loading header")

    return ROOT.corr2pc


def _get_axis_indices(cfg, REQUIRED_AXES):
    try:
        keep_axes = cfg["preprocess"]["keep_axes"]
    except KeyError as exc:
        raise RuntimeError(f"Missing config key: preprocess.keep_axes ({exc})") from exc

    missing = [a for a in REQUIRED_AXES if a not in keep_axes]

    if missing:
        raise RuntimeError(
            f"Correlation step requires retained axes {list(REQUIRED_AXES)}; "
            f"missing {missing}"
        )

    return {name: keep_axes.index(name) for name in REQUIRED_AXES}


def _get_mass_bins(corr_cfg):
    mass_bins = corr_cfg.get("mass_bins")

    if mass_bins is None or len(mass_bins) < 2:
        raise RuntimeError("correlation.mass_bins must contain at least two edges")

    edges = [float(x) for x in mass_bins]

    if any(hi <= lo for lo, hi in zip(edges, edges[1:])):
        raise RuntimeError(
            f"correlation.mass_bins must be strictly increasing, got {edges}"
        )
    return edges


def _make_settings(cfg, mass_bins, axes, DEFAULT_NORM_EPSILON=1e-6):
    corr_ns = _load_corr2pc()
    corr_cfg = cfg.get("correlation", {})
    settings = corr_ns.Settings()
    settings.mixingAxis = axes["pool"]
    settings.massAxis = axes["mass"]
    settings.etaAxis = axes["delta_eta"]
    settings.phiAxis = axes["delta_phi"]
    settings.normEpsilon = float(corr_cfg.get("norm_epsilon", DEFAULT_NORM_EPSILON))
    settings.massBins.clear()
    for edge in mass_bins:
        settings.massBins.push_back(edge)

    return settings
@dataclass(frozen=True)


class FourierConfig:
    eta_ranges: tuple          # ((low, high), ...)
    harmonics: tuple           # fixed by the fit implementation
    fit_range: tuple | None    # (low, high) or None
    mass_bins: tuple           # edges
    baseline_limits: tuple = (0.0, None)
    lm_scale_limits: tuple = (0.0, None)
    save_pdf: bool = False
    pt_assoc_bins: tuple | None = None
    pt_trigger_bins: tuple | None = None


def _read_config(cfg, *, harmonics=(2, 3)):
    """Read and validate the configuration once, up front."""
    fcfg = cfg.get("fourier", {})
    # Δη ranges
    raw_ranges = fcfg.get("delta_eta_ranges")

    if not raw_ranges:
        raise RuntimeError(
            "fourier.delta_eta_ranges must contain at least one interval"
        )
    eta_ranges = []
    for interval in raw_ranges:
        if len(interval) != 2:
            raise RuntimeError(
                "Each fourier.delta_eta_ranges entry must be [low, high]"
            )
        low, high = float(interval[0]), float(interval[1])

        if not high > low:
            raise RuntimeError(f"Invalid Δη interval [{low}, {high}]")
        eta_ranges.append((low, high))

    ordered = sorted(eta_ranges)
    for (_, prev_high), (next_low, _) in zip(ordered, ordered[1:]):
        if next_low < prev_high:
            raise RuntimeError(
                "fourier.delta_eta_ranges intervals must be disjoint "
                "(overlapping bins would be counted twice)"
            )

    # Harmonics are selected by the fit implementation, never by YAML.
    harmonics = tuple(harmonics)
    # Fit range
    fit_range = fcfg.get("fit_range")

    if fit_range is not None:
        if len(fit_range) != 2:
            raise RuntimeError("fourier.fit_range must be [low, high]")
        fit_range = (float(fit_range[0]), float(fit_range[1]))

        if not fit_range[1] > fit_range[0]:
            raise RuntimeError(f"Invalid Fourier fit range {list(fit_range)}")

    # Mass bins
    mass_bins = cfg.get("correlation", {}).get("mass_bins")

    if mass_bins is None or len(mass_bins) < 2:
        raise RuntimeError(
            "correlation.mass_bins must contain at least two edges "
            "to build the Fourier-vs-mass summary histograms"
        )

    def amplitude_limits(key):
        limits = fcfg.get(key, [0.0, None])

        if len(limits) != 2:
            raise ValueError(f"fourier.{key} must be [low, high], with null for an automatic upper bound")
        low = float(limits[0])
        high = float(limits[1]) if limits[1] is not None else None

        if not math.isfinite(low) or low < 0 or (high is not None and (not math.isfinite(high) or high <= low)):
            raise ValueError(f"fourier.{key} requires 0 <= low < high")
        return low, high

    # PDF output
    save_pdf = bool(fcfg.get("save_pdf", False))
    pt_trigger_bins = cfg.get(
        "bins", {}
    ).get("pt_cand")

    if save_pdf:
        if pt_trigger_bins is None or len(pt_trigger_bins) < 2:
            raise RuntimeError(
                "bins.pt_trigger_bins must contain at least two edges "
                "when fourier.save_pdf is enabled"
            )

        pt_trigger_bins = tuple(
            float(x) for x in pt_trigger_bins
        )

    return FourierConfig(
        eta_ranges=tuple(eta_ranges),
        harmonics=harmonics,
        fit_range=fit_range,
        mass_bins=tuple(float(x) for x in mass_bins),
        baseline_limits=amplitude_limits("baseline_limits"),
        lm_scale_limits=amplitude_limits("lm_scale_limits"),
        save_pdf=save_pdf,
        pt_assoc_bins=read_pt_assoc_bins(cfg) if save_pdf else None,
        pt_trigger_bins=pt_trigger_bins,
    )


def read_pt_assoc_bins(cfg):
    """Optional pt_assoc bin edges (correlation / preprocess section or top level)."""
    edges = None
    for section in (cfg.get("correlation", {}), cfg.get("preprocess", {}), cfg):
        if isinstance(section, dict) and section.get("pt_assoc_bins") is not None:
            edges = section["pt_assoc_bins"]
            break

    if edges is None:
        return None

    edges = tuple(float(x) for x in edges)

    if len(edges) < 2 or any(hi <= lo for lo, hi in zip(edges, edges[1:])):
        raise RuntimeError(
            "pt_assoc_bins must be strictly increasing with at least two edges"
        )
    return edges
