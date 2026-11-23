#!/usr/bin/env python3
"""Project and fit two-particle correlations in low/high multiplicity (LM/HM).

Input:
    <output_dir>/correlations/mult_*_pt_*.root
    with structure  pt_assoc_<i>/mass_<j>/{hCorr, hSE, [mass_range]}
Output:
    <output_dir>/fourier/mult_*_pt_*.root
    CMS mode uses <output_dir>/fourier/cms/ instead, including its pdf/ folder.
Output structure:
    pt_assoc_<i>/
        hV2DeltaVsMass, hInvariantMass, hFitStatusVsMass,
        hA<n>VsMass and, for HM templates, LM/flow diagnostic histograms
        invariant_mass_definition
        mass_<j>/
            hDeltaPhi          projection of hCorr over the selected Δη ranges
            fFourier           full fit function
            fBaseline, hBaseline
            fHarmonic<n>, hHarmonic<n>   individual a_n cos(n Δφ) terms
            hFitParameters     a0, a_n, then derived V_nΔ
            hFourierCovariance coefficient covariance, including LMScale
            fMacroFit          native fit in (c, V_nΔ, alpha) coordinates
            fit_info, delta_eta_ranges
            mass_range         (copied when available)
The Δη projection can combine one or more disjoint intervals [low, high).
Fit model (bounded c, Vn and alpha; saved as Fourier coefficients):
    LM: c * (1 + 2 Σ_{n=1,...,4} V_nΔ cos(n Δφ)).
    HM: c * (1 + 2 Σ_{n=2,3,4} V_nΔ cos(n Δφ)) + alpha * LMShape(Δφ),
        where LMShape = 1 + 2 Σ V_nΔ_LM cos(n Δφ).
    LM V_nΔ are fixed to the fit values in the corresponding LM mass bin.
    Fits use the macro's bounds and hold the LM template fixed. Saved coefficients
    are a0=c, a_n=2*c*V_nΔ, LMScale=alpha; a0 is the flow baseline.
    Projection averages selected Δη bin centres, using inverse-variance weights
    by default or uniform weights when configured.
    Harmonics are fixed in this module. fourier.is_lm selects the LM pass;
    otherwise the HM model is used, with a template when lm_file is provided.
    Optional LM-template uncertainty is propagated by varying the LM harmonics
    and refitting HM; it is added in quadrature to the reported flow V2 error.
Optional PDF output (config):
    fourier.save_pdf          : bool, default False
    correlation.mass_bins    : list of edges, required for mass summaries
One multi-page PDF is written per file to <output_dir>/fourier/pdf/:
one page per pt_assoc bin, one pad per invariant-mass bin showing the Δφ
projection, the total fit, the baseline a0 and each harmonic as
a0 + a_n cos(n Δφ), together with V_nΔ and χ²/ndf. The last pad of each page
is a legend explaining the curves and the model.
Summary PDFs collect V2Δ versus mass for each multiplicity group.
For Dmeson: hh, fit the inclusive mass_0 map without mass bins or an
invariant-mass spectrum. Per-fit outputs and PDFs are retained; mass summaries
are omitted.
Internal associated-pT directory identifiers are omitted from plot labels.

fourier.nonflow_method: cms prepares raw n=1..4 coefficients in both classes
using geometric eta-density projections. No CMS subtraction is performed here.
Run simultaneous fits for HM and LM, then apply the supplied scale with
src/cms_nonflow.py. The default method remains template.
"""

from __future__ import annotations

import glob
import math
import os
import re
import traceback
from array import array
from dataclasses import fields
from types import SimpleNamespace

import ROOT

try:
    from utils import config_utils as cu
    from utils import plot_utils as pu
    from utils import sparse_utils as su
except ImportError:  # project root not on sys.path: assume utils/ sits next to this folder
    import sys

    sys.path.insert(
        0,
        os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")),
    )
    from utils import config_utils as cu
    from utils import plot_utils as pu
    from utils import sparse_utils as su


EDGE_TOLERANCE = 1.0e-7
# The LM order also defines the coefficient and covariance indices on disk.
LM_HARMONICS = (1, 2, 3, 4, 5, 6)
HM_HARMONICS = (2, 3, 4, 5, 6)


def _project_phi_weighted(hist, eta_ranges, method="inverse_variance"):
    """Average selected Δη bins at each Δφ using the requested weights.

    The input axes are X=Δη and Y=Δφ. Selection uses bin centres in
    half-open intervals; bins with invalid errors do not contribute in either
    mode. Uniform averaging propagates independent bin errors.
    """
    if method not in ("inverse_variance", "uniform"):
        raise ValueError(f"Unknown delta-eta projection method: {method}")
    phi_axis = hist.GetYaxis()
    edges = array(
        "d", [phi_axis.GetBinLowEdge(i) for i in range(1, phi_axis.GetNbins() + 2)]
    )
    out = ROOT.TH1D("hDeltaPhi", "hDeltaPhi", phi_axis.GetNbins(), edges)
    out.SetDirectory(0)
    out.Sumw2()
    eta_axis = hist.GetXaxis()
    selected = [
        i
        for i in range(1, hist.GetNbinsX() + 1)
        if any(low <= eta_axis.GetBinCenter(i) < high for low, high in eta_ranges)
    ]

    if not selected:
        raise RuntimeError("No Δη bin centres inside fourier.delta_eta_ranges")
    for j in range(1, phi_axis.GetNbins() + 1):
        sum_weights = weighted_sum = 0.0
        for i in selected:
            error = hist.GetBinError(i, j)

            if error <= 0 or not math.isfinite(error):
                continue
            weight = 1 / error**2 if method == "inverse_variance" else 1.0
            sum_weights += weight
            weighted_sum += weight * hist.GetBinContent(i, j)

        if sum_weights > 0:
            out.SetBinContent(j, weighted_sum / sum_weights)
            if method == "inverse_variance":
                out.SetBinError(j, math.sqrt(1 / sum_weights))
            else:
                variance = sum(
                    hist.GetBinError(i, j)**2 for i in selected
                    if math.isfinite(hist.GetBinError(i, j)) and hist.GetBinError(i, j) > 0
                )
                out.SetBinError(j, math.sqrt(variance) / sum_weights)
    return out


def _mass_lm_template(lm_mass, label):
    """Use the matching LM mass bin's fitted Vn, with template c fixed to 1."""
    info = lm_mass.Get("fit_info")
    pars = lm_mass.Get("hFitParameters")
    source = lm_mass.Get("fFourier")

    if (
        not info or "status=0; covariance_status=3;" not in info.GetTitle()
        or not pars or not source
    ):
        raise RuntimeError(
            f"{label}: invalid LM fit: "
            f"{info.GetTitle() if info else 'missing fit_info'}; "
            "rerun the LM Fourier pass after checking amplitude limits"
        )

    # The HM caller checks that both passes use the same projection method.
    if "projection=" not in info.GetTitle():
        raise RuntimeError(f"{label}: LM reference has no projection metadata")
    template = _build_fourier_function(LM_HARMONICS, source.GetXmin(), source.GetXmax())
    template.SetParameter(0, 1.0)
    for i, n in enumerate(LM_HARMONICS, start=1):
        for j in range(1, pars.GetNbinsX() + 1):
            if pars.GetXaxis().GetBinLabel(j) == f"V{n}Delta":
                value, error = pars.GetBinContent(j), pars.GetBinError(j)

                if not math.isfinite(value) or not math.isfinite(error) or error <= 0:
                    raise RuntimeError(f"{label}: invalid LM V{n}Delta value or error")
                template.SetParameter(i, 2 * value)
                template.SetParError(i, 2 * error)
                break
        else:
            raise RuntimeError(f"{label}: missing LM V{n}Delta; rerun the LM Fourier pass")

    # Only harmonics absent from the HM flow model distinguish alpha from c.
    if all(
        abs(template.GetParameter(LM_HARMONICS.index(n) + 1)) < 1e-12
        for n in LM_HARMONICS if n not in HM_HARMONICS
    ):
        raise RuntimeError(f"{label}: zero LM harmonics 1, 4 cannot determine alpha")
    return template


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------
def _lm_shape_formula(lm_func):
    """Freeze the unit-baseline LM shape into the HM fit expression."""
    terms = "+".join(
        f"({lm_func.GetParameter(i):.17g})*cos({n}*x)"
        for i, n in enumerate(LM_HARMONICS, start=1)
    )
    return f"(1+{terms})"


def _build_fourier_function(harmonics, xmin, xmax, lm_func=None):
    """Build the saved coefficient model, optionally adding a fixed LM shape.

    C(Δφ) = a0 + Σ a_n cos(n Δφ) + LMScale * LMShape(Δφ).
    The final term is present only when a template is supplied.
    """
    formula = "[0]" + "".join(
        f"+[{ipar}]*cos({n}*x)" for ipar, n in enumerate(harmonics, start=1)
    )

    if lm_func is not None:
        formula += f"+[{len(harmonics) + 1}]*{_lm_shape_formula(lm_func)}"
    func = su._make_tf1("fFourier", formula, xmin, xmax)
    func.SetParName(0, "a0")
    for ipar, n in enumerate(harmonics, start=1):
        func.SetParName(ipar, f"a{n}")

    if lm_func is not None:
        func.SetParName(len(harmonics) + 1, "LMScale")
    return func


def _initial_baseline(hist):
    """Seed c from finite bin contents, falling back to a nonzero value."""
    values = [
        hist.GetBinContent(ibin)
        for ibin in range(1, hist.GetNbinsX() + 1)
    ]
    values = [v for v in values if math.isfinite(v)]

    if not values:
        return 1.0
    mean = sum(values) / len(values)
    return mean if mean != 0 else 1.0


def _fit_fourier(
    hist, harmonics, fit_range, lm_func=None,
    baseline_limits=(0.0, None), lm_scale_limits=(0.0, None),
):
    """Fit bounded (c, Vn, alpha), then export (a0, an, LMScale).

    The returned covariance is transformed into coefficient coordinates.
    LM shape parameters are fixed within this fit; the caller can propagate
    their uncertainty separately with repeated template fits.
    Returns (function, fit result, status, covariance status).
    """
    axis = hist.GetXaxis()
    xmin, xmax = fit_range if fit_range else (axis.GetXmin(), axis.GetXmax())
    formula = (
        "[0]*(1"
        + "".join(
            f"+2*[{i}]*cos({n}*x)" for i, n in enumerate(harmonics, start=1)
        )
        + ")"
    )

    if lm_func is not None:
        formula += f"+[{len(harmonics) + 1}]*{_lm_shape_formula(lm_func)}"
    native = su._make_tf1("fMacroFit", formula, xmin, xmax)
    baseline = _initial_baseline(hist)
    automatic_upper = max(
        1.0,
        10 * max(abs(hist.GetBinContent(i)) for i in range(1, hist.GetNbinsX() + 1)),
    )

    def set_amplitude(ipar, limits, initial):
        # Keep seeds just inside the bounds so MINUIT can vary the parameter.
        low, high = limits
        high = max(automatic_upper, low + 1.0) if high is None else high
        native.SetParLimits(ipar, low, high)
        margin = (high - low) * 1e-6
        native.SetParameter(ipar, min(high - margin, max(low + margin, initial)))

    native.SetParName(0, "c")
    set_amplitude(0, baseline_limits, baseline)
    for ipar, n in enumerate(harmonics, start=1):
        native.SetParName(ipar, f"V{n}Delta")
        limit = 1.0 if n == 1 else 0.1
        native.SetParLimits(ipar, -limit, limit)
        native.SetParameter(ipar, 0.0)

    if lm_func is not None:
        native.SetParName(len(harmonics) + 1, "alpha")
        set_amplitude(
            len(harmonics) + 1, lm_scale_limits, min(1.0, max(0.001, baseline * 0.1))
        )
    # E: MINOS, R: function range, M: IMPROVE, B: supplied bounds,
    # S: return fit result, Q: quiet, N: do not attach the function.
    result = hist.Fit(native, "QERMBSN")
    native_fit = result.Get()

    if not native_fit:
        raise RuntimeError(f"Fit of {hist.GetName()} returned no result")
    macro_status = native_fit.Status()
    fit_options = "QERMBSN"
    error_method = "minos"
    # Retry the same bounded model without optional IMPROVE on any failure.
    if macro_status != 0:
        fit_options = "QERBSN"
        result = hist.Fit(native, "QERBSN")
        native_fit = result.Get()

        if not native_fit:
            raise RuntimeError(f"Retry of {hist.GetName()} returned no result")
    minos_status = native_fit.Status()
    # MINOS may fail at a limit even with a valid minimum and accurate covariance.
    # Refit without MINOS, report symmetric errors and retain the earlier statuses.
    if minos_status != 0 and native_fit.IsValid() and native_fit.CovMatrixStatus() == 3:
        fit_options = "QRBSN"
        error_method = "hesse"
        result = hist.Fit(native, fit_options)
        native_fit = result.Get()

        if not native_fit:
            raise RuntimeError(f"Covariance-error retry of {hist.GetName()} returned no result")
    # Preserve the pipeline's coefficient representation: a0=c, an=2*c*Vn.
    func = _build_fourier_function(harmonics, xmin, xmax, lm_func)
    c = native.GetParameter(0)
    func.SetParameter(0, c)
    for i in range(1, len(harmonics) + 1):
        func.SetParameter(i, 2 * c * native.GetParameter(i))

    if lm_func is not None:
        func.SetParameter(len(harmonics) + 1, native.GetParameter(len(harmonics) + 1))
    func.SetChisquare(native.GetChisquare())
    func.SetNDF(native.GetNDF())
    fit = _CoefficientFitResult(native_fit, native, harmonics)
    fit.macro_status = macro_status
    fit.minos_status = minos_status
    fit.error_method = error_method
    fit.fit_options = fit_options
    for i in range(func.GetNpar()):
        func.SetParError(i, math.sqrt(max(0.0, fit.CovMatrix(i, i))))
    func.SetParError(0, native.GetParError(0))

    if lm_func is not None:
        func.SetParError(len(harmonics) + 1, native.GetParError(len(harmonics) + 1))
    return func, fit, native_fit.Status(), native_fit.CovMatrixStatus()


class _CoefficientFitResult:
    """Transform native (c, Vn, alpha) covariance into saved coefficients."""

    def __init__(self, fit, native, harmonics):
        self.native = native
        self.fit_valid = bool(fit.IsValid())
        self.covariance_status = int(fit.CovMatrixStatus())
        size = native.GetNpar()
        jac = [[float(i == j) for j in range(size)] for i in range(size)]
        for i in range(1, len(harmonics) + 1):
            jac[i][0] = 2 * native.GetParameter(i)
            jac[i][i] = 2 * native.GetParameter(0)
        # Cov(coefficients) = J Cov(native parameters) J^T, including c-Vn
        # correlations. Alpha is unchanged by the coordinate transformation.
        self.covariance = [
            [
                sum(
                    jac[i][a] * jac[j][b] * fit.CovMatrix(a, b)
                    for a in range(size) for b in range(size)
                )
                for j in range(size)
            ]
            for i in range(size)
        ]

    def IsValid(self):
        """Expose the native fit validity through the ROOT result interface."""
        return self.fit_valid

    def CovMatrixStatus(self):
        """Expose the native covariance quality (3 means fully accurate)."""
        return self.covariance_status

    def CovMatrix(self, i, j):
        """Match the ROOT fit-result accessor used by error propagation."""
        return self.covariance[i][j]


def _derived_vn(func, fit, ipar):
    """Return V_nΔ and its native error, or propagate a_n / (2 a0).

    Native fits retain MINOS errors; other fit results use the coefficient
    covariance, including the correlation between a0 and a_n.
    """

    if isinstance(fit, _CoefficientFitResult):
        # Retain the native MINOS error instead of reconstructing it from
        # the symmetric coefficient covariance.
        return fit.native.GetParameter(ipar), fit.native.GetParError(ipar)
    a0 = func.GetParameter(0)
    an = func.GetParameter(ipar)

    if a0 == 0:
        return float("nan"), float("nan")
    d_da0 = -an / (2.0 * a0 * a0)
    d_dan = 1.0 / (2.0 * a0)
    variance = (
        d_da0 * d_da0 * fit.CovMatrix(0, 0)
        + d_dan * d_dan * fit.CovMatrix(ipar, ipar)
        + 2.0 * d_da0 * d_dan * fit.CovMatrix(0, ipar)
    )
    return an / (2.0 * a0), math.sqrt(max(0.0, variance))


# --------------------------------------------------------------------------
# Derived objects
# --------------------------------------------------------------------------
def _make_parameter_hist(func, fit, harmonics):
    """Store a0, a_n, derived V_nΔ and optional LMScale in labelled bins."""
    n = len(harmonics)
    has_scale = func.GetNpar() > n + 1
    hist = ROOT.TH1D(
        "hFitParameters",
        "Fourier fit parameters;;value",
        1 + 2 * n + int(has_scale),
        0.5,
        1.5 + 2 * n + int(has_scale),
    )
    hist.SetDirectory(0)

    def fill(ibin, label, value, error):
        hist.GetXaxis().SetBinLabel(ibin, label)
        hist.SetBinContent(ibin, value)
        hist.SetBinError(ibin, error)

    fill(1, "a0", func.GetParameter(0), func.GetParError(0))
    for i, harmonic in enumerate(harmonics):
        ipar = i + 1
        fill(
            1 + ipar,
            f"a{harmonic}",
            func.GetParameter(ipar),
            func.GetParError(ipar),
        )
        value, error = _derived_vn(func, fit, ipar)
        fill(1 + n + ipar, f"V{harmonic}Delta", value, error)

    if has_scale:
        fill(2 + 2 * n, "LMScale", func.GetParameter(n + 1), func.GetParError(n + 1))
    return hist


def _make_components(hist, func, harmonics):
    """Baseline and individual a_n cos(n Δφ) terms, as TF1 and as TH1.

    The histograms are evaluated at the bin centres of the data histogram.
    Returns (baseline_func, baseline_hist, [(n, func, hist), ...]).
    """
    xmin, xmax = func.GetXmin(), func.GetXmax()
    a0 = func.GetParameter(0)
    baseline_func = su._make_tf1("fBaseline", "[0]", xmin, xmax)
    baseline_func.SetParameter(0, a0)
    baseline_hist = hist.Clone("hBaseline")
    baseline_hist.SetDirectory(0)
    baseline_hist.Reset("ICESM")
    components = []
    for i, harmonic in enumerate(harmonics):
        an = func.GetParameter(i + 1)
        component_func = su._make_tf1(
            f"fHarmonic{harmonic}", f"[0]*cos({harmonic}*x)", xmin, xmax
        )
        component_func.SetParameter(0, an)
        component_hist = hist.Clone(f"hHarmonic{harmonic}")
        component_hist.SetDirectory(0)
        component_hist.Reset("ICESM")
        components.append((harmonic, an, component_func, component_hist))
    axis = hist.GetXaxis()
    for ibin in range(1, hist.GetNbinsX() + 1):
        x = axis.GetBinCenter(ibin)
        baseline_hist.SetBinContent(ibin, a0)
        for harmonic, an, _, component_hist in components:
            component_hist.SetBinContent(ibin, an * math.cos(harmonic * x))
    return (
        baseline_func,
        baseline_hist,
        [(h, f, hh) for h, _, f, hh in components],
    )


def _lm_correction(func, fit, lm_func, harmonics, label):
    """Read flow coefficients and the scale fitted directly in the HM model."""
    lm_a1 = lm_func.GetParameter(1)

    if abs(func.GetParameter(0)) < 1e-12:
        raise RuntimeError(f"{label}: HM a0 is zero")
    lm_a1_error = lm_func.GetParError(1)
    significance = abs(lm_a1) / lm_a1_error if lm_a1_error > 0 else math.inf

    if significance < 2.0:
        print(
            f"[fourier] WARNING: {label}: LM a1 significance only "
            f"{significance:.2f} sigma; template scale may be unstable"
        )
    scale_index = len(harmonics) + 1
    return {
        "scale": (func.GetParameter(scale_index), func.GetParError(scale_index)),
        "a1_significance": significance,
        "coeff": {
            n: (func.GetParameter(i), func.GetParError(i))
            for i, n in enumerate(harmonics, start=1)
        },
        "vn": {n: _derived_vn(func, fit, i) for i, n in enumerate(harmonics, start=1)},
    }


def _raw_template_vn(func, fit, lm_func, harmonics, harmonic=2):
    """Return (an + alpha*an_LM)/(2*(c+alpha)) and its propagated error.

    The covariance includes correlations with alpha; the LM shape is fixed.
    """
    i = harmonics.index(harmonic) + 1
    scale_index = len(harmonics) + 1
    lm_coefficient = lm_func.GetParameter(LM_HARMONICS.index(harmonic) + 1)
    a0 = func.GetParameter(0) + func.GetParameter(scale_index)

    if a0 == 0:
        return float("nan"), float("nan")
    raw = func.GetParameter(i) + func.GetParameter(scale_index) * lm_coefficient
    # Differentiate with respect to the saved coefficients (a0, an, LMScale).
    gradient = {
        0: -raw / (2 * a0**2),
        i: 1 / (2 * a0),
        scale_index: lm_coefficient / (2 * a0) - raw / (2 * a0**2),
    }
    variance = sum(
        gi * gj * fit.CovMatrix(i, j)
        for i, gi in gradient.items() for j, gj in gradient.items()
    )
    return raw / (2 * a0), math.sqrt(max(0.0, variance))


def _lm_vn_covariance(lm_mass, lm_func, label):
    """Transform the saved LM coefficient covariance to V1Δ through V4Δ.

    The saved matrix uses (a0, a1, ..., a4), while template variations use
    VnΔ = an / (2*a0). Both the baseline error and its correlations matter.
    """
    cov = lm_mass.Get("hFourierCovariance")

    if not cov or cov.GetNbinsX() != 5 or cov.GetNbinsY() != 5:
        raise RuntimeError(f"{label}: missing 5x5 LM coefficient covariance")
    labels = ["a0", "a1", "a2", "a3", "a4"]

    if any(cov.GetXaxis().GetBinLabel(i + 1) != name for i, name in enumerate(labels)):
        raise RuntimeError(f"{label}: unexpected LM covariance axis labels")
    pars = lm_mass.Get("hFitParameters")
    coefficients = [pars.GetBinContent(i) for i in range(1, 6)]

    if not math.isfinite(coefficients[0]) or abs(coefficients[0]) < 1.0e-15:
        raise RuntimeError(f"{label}: invalid LM baseline")
    jacobian = [[0.0] * 5 for _ in range(4)]
    for k in range(4):
        jacobian[k][0] = -coefficients[k + 1] / (2 * coefficients[0] ** 2)
        jacobian[k][k + 1] = 1 / (2 * coefficients[0])
    # Cov(Vn) = J Cov(a) J^T retains correlations between LM harmonics.
    matrix = [
        [
            sum(
                jacobian[i][u] * cov.GetBinContent(u + 1, v + 1) * jacobian[j][v]
                for u in range(5) for v in range(5)
            )
            for j in range(4)
        ]
        for i in range(4)
    ]
    for k in range(4):

        if not math.isfinite(matrix[k][k]) or matrix[k][k] <= 0:
            raise RuntimeError(f"{label}: invalid LM V{k + 1} covariance")
    return matrix


def _propagate_lm_template_error(hphi, config, lm_mass, lm_func, label):
    """Propagate LM Vn covariance to HM V2 using central-difference refits."""
    cov = _lm_vn_covariance(lm_mass, lm_func, label)
    derivatives = []
    for k in range(4):
        sigma = math.sqrt(cov[k][k])
        values = []
        for sign in (-1, +1):
            varied = lm_func.Clone(f"lm_variation_{k}_{sign}")
            # Shape coefficients are 2*Vn: +/-2*sigma shifts Vn by +/-sigma.
            varied.SetParameter(k + 1, lm_func.GetParameter(k + 1) + sign * 2 * sigma)
            func, fit, status, cov_status = _fit_fourier(
                hphi, config.harmonics, config.fit_range, lm_func=varied,
                baseline_limits=config.baseline_limits, lm_scale_limits=config.lm_scale_limits,
            )

            if status != 0 or cov_status != 3 or not fit.IsValid():
                raise RuntimeError(f"{label}: LM variation n={k + 1} sign={sign} failed")
            v2, _ = _derived_vn(func, fit, config.harmonics.index(2) + 1)

            if not math.isfinite(v2):
                raise RuntimeError(f"{label}: nonfinite LM variation n={k + 1}")
            values.append(v2)
        derivatives.append((values[1] - values[0]) / (2 * sigma))
    # Contract the response to each LM Vn with their full covariance matrix.
    variance = sum(
        derivatives[i] * cov[i][j] * derivatives[j]
        for i in range(4) for j in range(4)
    )

    if not math.isfinite(variance) or variance < -1e-10:
        raise RuntimeError(f"{label}: invalid propagated LM variance {variance}")
    return math.sqrt(max(0.0, variance))


def _write_covariance(out_mass, fit, harmonics, has_scale=False):
    """Save coefficient covariance in hFitParameters order, excluding Vn bins."""
    names = ["a0"] + [f"a{n}" for n in harmonics] + (["LMScale"] if has_scale else [])
    size = len(names)
    hist = ROOT.TH2D(
        "hFourierCovariance", "Fourier coefficient covariance;parameter;parameter",
        size, 0.5, size + 0.5, size, 0.5, size + 0.5,
    )
    hist.SetDirectory(0)
    for i in range(size):
        hist.GetXaxis().SetBinLabel(i + 1, names[i])
        hist.GetYaxis().SetBinLabel(i + 1, names[i])
        for j in range(size):
            hist.SetBinContent(i + 1, j + 1, fit.CovMatrix(i, j))
    su._write(out_mass, hist)


def _integral_and_error_2d(hist):
    """Integral over regular bins (no overflow), uncorrelated errors."""
    total = 0.0
    variance = 0.0
    for ix in range(1, hist.GetNbinsX() + 1):
        for iy in range(1, hist.GetNbinsY() + 1):
            total += hist.GetBinContent(ix, iy)
            variance += hist.GetBinError(ix, iy) ** 2
    return total, math.sqrt(variance)


def _make_mass_summary_hist(name, title, mass_bins):
    """Create a detached summary histogram with the configured mass edges."""
    edges = array("d", mass_bins)
    hist = ROOT.TH1D(name, title, len(edges) - 1, edges)
    hist.SetDirectory(0)
    return hist


# --------------------------------------------------------------------------
# Per-file processing
# --------------------------------------------------------------------------
def _validated_lm_reference(lm_mass, hphi, config, label):
    """Require the same harmonics, Δη selection and Δφ range in both passes."""
    lm_func = lm_mass.Get("fFourier")

    if not lm_func or lm_func.GetNpar() != len(LM_HARMONICS) + 1:
        required = ", ".join(["a0"] + [f"a{n}" for n in LM_HARMONICS])
        raise RuntimeError(
            f"{label}: LM reference must contain {required}; rerun the LM Fourier pass"
        )
    expected = ["a0"] + [f"a{n}" for n in LM_HARMONICS]
    actual = [str(lm_func.GetParName(i)) for i in range(lm_func.GetNpar())]
    if actual != expected:
        raise RuntimeError(
            f"{label}: LM harmonic mismatch: expected {expected}, found {actual}. "
            "fourier.lm_file must point to a raw LM Fourier fit produced with "
            "fourier.is_lm: true, not an HM template fit."
        )

    if not all(math.isfinite(lm_func.GetParameter(i)) for i in range(lm_func.GetNpar())):
        raise RuntimeError(f"{label}: LM template has invalid coefficients")
    eta_text = "; ".join(
        f"[{low:.17g}, {high:.17g})" for low, high in config.eta_ranges
    )
    lm_eta = lm_mass.Get("delta_eta_ranges")

    if not lm_eta or lm_eta.GetTitle() != eta_text:
        raise RuntimeError(f"{label}: LM and HM delta_eta_ranges differ")
    axis = hphi.GetXaxis()
    xmin, xmax = config.fit_range or (axis.GetXmin(), axis.GetXmax())

    if (
        abs(lm_func.GetXmin() - xmin) > EDGE_TOLERANCE
        or abs(lm_func.GetXmax() - xmax) > EDGE_TOLERANCE
    ):
        raise RuntimeError(f"{label}: LM and HM Fourier fit ranges differ")
    info = lm_mass.Get("fit_info")

    if not info or f"projection={config.projection_method};" not in info.GetTitle():
        raise RuntimeError(f"{label}: LM/HM projection methods differ")
    if not info or "status=0; covariance_status=3;" not in info.GetTitle():
        raise RuntimeError(f"{label}: invalid LM fit; rerun the LM Fourier pass")
    return lm_func


def _validated_lm_template(lm_mass, hphi, config, label):
    _validated_lm_reference(lm_mass, hphi, config, label)
    return _mass_lm_template(lm_mass, label)


def _write_lm_components(out_mass, func, lm_func, harmonics, corrected):
    """Save the flow and scaled LM terms whose sum is the full HM function."""
    xmin, xmax = func.GetXmin(), func.GetXmax()
    f_flow = _build_fourier_function(harmonics, xmin, xmax)
    f_flow.SetName("fFlow")
    f_flow.SetParameter(0, func.GetParameter(0))
    f_template = _build_fourier_function(LM_HARMONICS, xmin, xmax)
    f_template.SetName("fLMTemplate")
    scale = corrected["scale"][0]
    f_template.SetParameter(0, scale)
    for i, n in enumerate(harmonics, start=1):
        f_flow.SetParameter(i, corrected["coeff"][n][0])
    for i, n in enumerate(LM_HARMONICS, start=1):
        f_template.SetParameter(i, scale * lm_func.GetParameter(i))
    su._write(out_mass, f_flow)
    su._write(out_mass, f_template)

    values = [("LMScale", *corrected["scale"])]
    values += [(f"a{n}Flow", *corrected["coeff"][n]) for n in harmonics]
    values += [(f"V{n}DeltaFlow", *corrected["vn"][n]) for n in harmonics]
    size = len(values)
    h_lm = ROOT.TH1D(
        "hLMCorrectionParameters", "LM scale and corrected coefficients;;value",
        size, 0.5, size + 0.5,
    )
    h_lm.SetDirectory(0)
    for i, (name, value, error) in enumerate(values, start=1):
        h_lm.GetXaxis().SetBinLabel(i, name)
        h_lm.SetBinContent(i, value)
        h_lm.SetBinError(i, error)
    su._write(out_mass, h_lm)


def _write_scalar(directory, name, value, error):
    hist = ROOT.TH1D(name, name, 1, 0, 1)
    hist.SetDirectory(0)
    hist.SetBinContent(1, value)
    hist.SetBinError(1, error)
    su._write(directory, hist)


def _process_mass_dir(in_mass, out_mass, config, label, lm_mass=None):
    """Fit one associated-pT/mass bin and write its objects.

    Returns (V2, V2 error, fit status, yield, yield error, function, fit,
    template results, template status). Template status is 0 on success,
    1 for an invalid HM fit, or 2 when a template reliability check fails.
    """
    hcorr = in_mass.Get("hCorr")
    hse = in_mass.Get("hSE")

    if not hcorr:
        raise RuntimeError(f"Missing hCorr in {label}")

    if not hse:
        raise RuntimeError(f"Missing hSE in {label}")
    mass_yield, mass_yield_error = _integral_and_error_2d(hse)
    cms = getattr(config, "nonflow_method", "template") == "cms"
    if cms:
        try:
            from . import correlations
        except ImportError:
            import correlations
        hphi = correlations._project_eta_density(hcorr, config.eta_ranges)
    else:
        hphi = _project_phi_weighted(hcorr, config.eta_ranges, config.projection_method)
    # CMS operates on independent raw Fourier fits, never a template fit.
    result = _process_phi_hist(hphi, out_mass, config, label, None if cms else lm_mass)
    for name in ("hJetYield", "jet_yield_definition", "hJetShortRange", "hJetLongRange"):
        su._copy_if_present(in_mass, out_mass, name)
    su._copy_if_present(in_mass, out_mass, "mass_range")
    su._copy_if_present(in_mass, out_mass, "correlation_selection")
    v2, error, status, func, fit, corrected, correction_status = result
    if cms:
        su._write(out_mass, ROOT.TNamed("nonflow_method", "cms"))
        assoc = (2 * math.pi * func.GetParameter(0),
                 2 * math.pi * math.sqrt(max(0.0, fit.CovMatrix(0, 0))))
        _write_scalar(out_mass, "hNassoc", *assoc)
    return (v2, error, status, mass_yield, mass_yield_error,
            func, fit, corrected, correction_status)


def _process_phi_hist(hphi, out_mass, config, label, lm_mass=None):
    """Fit a Δφ histogram and save the model, covariance and diagnostics."""
    lm_func = None

    # A bad LM reference aborts the file before any HM result is accepted.
    if lm_mass is not None:
        lm_func = _validated_lm_template(lm_mass, hphi, config, label)
        su._write(out_mass, lm_func, "fLMShape")
    func, fit, status, cov_status = _fit_fourier(
        hphi, config.harmonics, config.fit_range, lm_func=lm_func,
        baseline_limits=config.baseline_limits, lm_scale_limits=config.lm_scale_limits,
    )

    if status != 0 or cov_status != 3:
        print(
            f"[fourier] WARNING: {label}: fit status={status}, "
            f"covariance status={cov_status} (3 = full accurate matrix)"
        )

    if fit.error_method == "hesse":
        print(
            f"[fourier] WARNING: {label}: MINOS status={fit.minos_status}; "
            f"using symmetric covariance errors (fit status={status}, "
            f"covariance status={cov_status})"
        )
    corrected = None
    correction_status = 0

    if lm_mass is not None:
        if (
            status != 0 or cov_status != 3
            or not all(math.isfinite(func.GetParameter(i)) for i in range(func.GetNpar()))
        ):
            correction_status = 1
            print(f"[fourier] WARNING: {label}: HM fit invalid; skipping LM correction")
        else:
            try:
                corrected = _lm_correction(func, fit, lm_func, config.harmonics, label)
                # Check the saved n=1 and n=4 template significances. The scale
                # is identified by harmonics outside the chosen HM flow model.
                s1 = abs(lm_func.GetParameter(1)) / max(lm_func.GetParError(1), 1e-30)
                s4 = abs(lm_func.GetParameter(4)) / max(lm_func.GetParError(4), 1e-30)

                if max(s1, s4) < config.min_lm_shape_significance:
                    raise RuntimeError(
                        f"LM scale weakly identified: n1={s1:.2f}σ, n4={s4:.2f}σ"
                    )
                # Flag a scale sitting on a bound as unreliable.
                scale_idx = len(config.harmonics) + 1
                low, high = config.lm_scale_limits
                scale = func.GetParameter(scale_idx)

                if low is not None and scale <= low + 1e-5 * max(1.0, abs(scale)):
                    raise RuntimeError("LM scale pinned to its lower bound")
                if high is not None and scale >= high - 1e-5 * max(1.0, abs(scale)):
                    raise RuntimeError("LM scale pinned to its upper bound")
                if config.lm_template_error:
                    # Keep the fixed-template fit covariance unchanged; augment
                    # only the flow V2 uncertainty reported in the summaries.
                    template_error = _propagate_lm_template_error(
                        hphi, config, lm_mass, lm_func, label
                    )
                    v2, stat_error = corrected["vn"][2]
                    corrected["vn"][2] = (v2, math.hypot(stat_error, template_error))
                    corrected["lm_template_error"] = template_error
                    error_hist = ROOT.TH1D(
                        "hLMTemplateV2Error", "LM template contribution to V2 error",
                        1, 0.0, 1.0,
                    )
                    error_hist.SetDirectory(0)
                    error_hist.SetBinContent(1, template_error)
                    su._write(out_mass, error_hist)
            except (RuntimeError, ZeroDivisionError, OverflowError, ValueError) as exc:
                correction_status = 2
                print(f"[fourier] WARNING: {label}: LM correction skipped: {exc}")

        if corrected is not None:
            _write_lm_components(out_mass, func, lm_func, config.harmonics, corrected)
    hpars = _make_parameter_hist(func, fit, config.harmonics)
    _write_covariance(out_mass, fit, config.harmonics, has_scale=lm_mass is not None)
    v2, v2_error = _derived_vn(func, fit, config.harmonics.index(2) + 1)

    if corrected is not None:
        v2, v2_error = corrected["vn"][2]
    baseline_func, baseline_hist, components = _make_components(
        hphi, func, config.harmonics
    )
    su._write(out_mass, hphi, "hDeltaPhi")
    su._write(out_mass, func, "fFourier")
    su._write(out_mass, fit.native, "fMacroFit")
    su._write(out_mass, baseline_func, "fBaseline")
    su._write(out_mass, baseline_hist, "hBaseline")
    for harmonic, component_func, component_hist in components:
        su._write(out_mass, component_func, f"fHarmonic{harmonic}")
        su._write(out_mass, component_hist, f"hHarmonic{harmonic}")
    su._write(out_mass, hpars, "hFitParameters")
    model = "a0+sum(an*cos(n*dphi))"

    if lm_mass is not None:
        model += " + LMScale*(1+sum(2*VnLM_mass_bin*cos(n*dphi)))"
    fit_text = (
        f"status={status}; "
        f"covariance_status={cov_status}; "
        f"macro_initial_status={fit.macro_status}; "
        f"minos_status={fit.minos_status}; "
        f"error_method={fit.error_method}; fit_options={fit.fit_options}; "
        f"chi2={func.GetChisquare():.17g}; "
        f"ndf={func.GetNDF()}; "
        f"harmonics={','.join(str(n) for n in config.harmonics)}; "
        f"model={model}; "
        f"covariance=fit_fixed_template; projection={config.projection_method}; "
        "template_baseline=1; a0=flow_baseline; "
        "VnDelta=an/(2*a0)"
    )
    su._write(out_mass, ROOT.TNamed("fit_info", fit_text), "fit_info")

    if lm_mass is not None:
        su._write(out_mass, ROOT.TNamed("lm_correction_status", str(correction_status)))
    eta_text = "; ".join(
        f"[{low:.17g}, {high:.17g})" for low, high in config.eta_ranges
    )
    su._write(out_mass, ROOT.TNamed("delta_eta_ranges", eta_text), "delta_eta_ranges")
    return (
        v2, v2_error, status, func, fit, corrected, correction_status,
    )


def _process_pt_dir(in_pt, out_pt, config, label, lm_pt=None):
    """Fit all mass bins for one associated-pT interval and write summaries."""
    # subdirectories already sorts the numeric suffix, including mass_10.
    mass_dirs = pu.subdirectories(in_pt, "mass_")

    if not mass_dirs:
        raise RuntimeError(f"No mass_* directories found in {label}")
    if getattr(config, "is_hh", False):
        if mass_dirs != ["mass_0"]:
            raise RuntimeError(f"{label}: hh requires one inclusive mass_0 map, found {mass_dirs}")
        lm_mass = lm_pt.GetDirectory("mass_0") if lm_pt is not None else None
        if lm_pt is not None:
            if (pu.subdirectories(lm_pt, "mass_") != ["mass_0"] or not lm_mass
                    or not lm_mass.Get("correlation_selection")):
                raise RuntimeError(f"{label}: LM reference must contain one inclusive hh fit")
        out_mass = su._mkdir(out_pt, "mass_0")
        _process_mass_dir(in_pt.GetDirectory("mass_0"), out_mass, config,
                          f"{label}/mass_0", lm_mass=lm_mass)
        su._write(out_mass, ROOT.TNamed("correlation_selection", "Inclusive hh"))
        return 1
    expected = len(config.mass_bins) - 1

    if mass_dirs != [f"mass_{i}" for i in range(expected)]:
        raise RuntimeError(f"{label}: expected mass_0..mass_{expected - 1}, found {mass_dirs}")
    h_v2 = _make_mass_summary_hist(
        "hV2DeltaVsMass",
        "V_{2#Delta} vs invariant mass;invariant mass;V_{2#Delta}",
        config.mass_bins,
    )
    h_mass = in_pt.Get("hInvariantMass")

    if not h_mass:
        raise RuntimeError(f"{label}: missing inherited hInvariantMass; rerun correlations")
    h_status = _make_mass_summary_hist(
        "hFitStatusVsMass",
        "Fourier fit status vs invariant mass;invariant mass;fit status",
        config.mass_bins,
    )
    parameter_hists = {
        f"hA{n}VsMass": _make_mass_summary_hist(
            f"hA{n}VsMass", f"a_{{{n}}};invariant mass;a_{{{n}}}", config.mass_bins
        )
        for n in (0, *config.harmonics)
    }

    if lm_pt is not None:
        reference = lm_pt.Get("hV2DeltaVsMass")

        if not reference:
            raise RuntimeError(f"{label}: LM reference is missing hV2DeltaVsMass")
        ref_axis = reference.GetXaxis()

        if (
            ref_axis.GetNbins() != expected
            or any(
                abs(ref_axis.GetBinLowEdge(i + 1) - edge) > EDGE_TOLERANCE
                for i, edge in enumerate(config.mass_bins[:-1])
            )
            or abs(ref_axis.GetBinUpEdge(expected) - config.mass_bins[-1]) > EDGE_TOLERANCE
        ):
            raise RuntimeError(f"{label}: LM mass binning does not match HM")
        # Keep the diagnostic definitions together; all use identical mass edges.
        summaries = {
            "hV2DeltaRawVsMass": "Uncorrected V2Delta;invariant mass;V2Delta",
            "hLMScaleVsMass": "LM template scale;invariant mass;LM scale",
            "hLMA1SignificanceVsMass": "LM a1 significance;invariant mass;significance",
            "hLMCorrectionStatusVsMass": (
                "0=valid, 1=HM failed, 2=LM normalization failed, "
                "3=LM fit failed;invariant mass;status"
            ),
        }
        for n in LM_HARMONICS:
            summaries[f"hA{n}LMVsMass"] = f"LM a{n};invariant mass;a{n}"
        for n in config.harmonics:
            summaries[f"hV{n}DeltaFlowVsMass"] = (
                f"V{n}Delta after LM subtraction;invariant mass;V{n}Delta"
            )
            summaries[f"hA{n}FlowVsMass"] = f"a{n} after LM subtraction;invariant mass;a{n}"
        parameter_hists.update({
            name: _make_mass_summary_hist(name, title, config.mass_bins)
            for name, title in summaries.items()
        })
    for ibin, mass_name in enumerate(mass_dirs, start=1):
        lm_mass = lm_pt.GetDirectory(mass_name) if lm_pt is not None else None

        if lm_pt is not None and not lm_mass:
            raise RuntimeError(f"{label}: missing LM {mass_name}")
        result = _process_mass_dir(
            in_pt.GetDirectory(mass_name),
            su._mkdir(out_pt, mass_name),
            config,
            f"{label}/{mass_name}",
            lm_mass=lm_mass,
        )
        v2, v2_error, status, _, _, func, fit, corrected, correction_status = result

        # Mark failed results explicitly so summary plots cannot show them as zero.
        valid = (
            status == 0
            and fit.IsValid()
            and fit.CovMatrixStatus() == 3
            and math.isfinite(v2)
            and math.isfinite(v2_error)
            and v2_error >= 0
        )

        if (lm_pt is None and valid) or (corrected is not None and valid):
            h_v2.SetBinContent(ibin, v2)
            h_v2.SetBinError(ibin, v2_error)
        else:
            h_v2.SetBinContent(ibin, float("nan"))

        if lm_pt is not None:
            parameter_hists["hLMCorrectionStatusVsMass"].SetBinContent(ibin, correction_status)
        h_status.SetBinContent(ibin, status)
        for i, n in enumerate((0, *config.harmonics)):
            h = parameter_hists[f"hA{n}VsMass"]
            h.SetBinContent(ibin, func.GetParameter(i))
            h.SetBinError(ibin, func.GetParError(i))

        if lm_pt is not None and status == 0:
            lm_template = out_pt.GetDirectory(mass_name).Get("fLMShape")
            raw, raw_error = _raw_template_vn(func, fit, lm_template, config.harmonics)
            parameter_hists["hV2DeltaRawVsMass"].SetBinContent(ibin, raw)
            parameter_hists["hV2DeltaRawVsMass"].SetBinError(ibin, raw_error)

        if corrected is not None:
            parameter_hists["hLMScaleVsMass"].SetBinContent(ibin, corrected["scale"][0])
            parameter_hists["hLMScaleVsMass"].SetBinError(ibin, corrected["scale"][1])
            parameter_hists["hLMA1SignificanceVsMass"].SetBinContent(
                ibin, corrected["a1_significance"]
            )
            lm_par = out_pt.GetDirectory(mass_name).Get("fLMShape")
            for i, n in enumerate(LM_HARMONICS, start=1):
                parameter_hists[f"hA{n}LMVsMass"].SetBinContent(ibin, lm_par.GetParameter(i))
                parameter_hists[f"hA{n}LMVsMass"].SetBinError(ibin, lm_par.GetParError(i))
            for n in config.harmonics:
                flow, flow_err = corrected["vn"].get(n, (0.0, 0.0))
                parameter_hists[f"hV{n}DeltaFlowVsMass"].SetBinContent(ibin, flow)
                parameter_hists[f"hV{n}DeltaFlowVsMass"].SetBinError(ibin, flow_err)
                h = parameter_hists[f"hA{n}FlowVsMass"]
                h.SetBinContent(ibin, corrected["coeff"][n][0])
                h.SetBinError(ibin, corrected["coeff"][n][1])
    su._write(out_pt, h_v2)
    su._write(out_pt, h_mass)
    su._write(out_pt, h_status)
    for hist in parameter_hists.values():
        su._write(out_pt, hist)
    su._copy_if_present(in_pt, out_pt, "invariant_mass_definition")
    su._copy_if_present(in_pt, out_pt, "hJetYieldVsMass")
    return len(mass_dirs)


def _process_file(input_file, output_file, config, lm_file=None, pt_metadata=None):
    """Process one candidate-pT file, removing partial output on failure."""
    if getattr(config, "nonflow_method", "template") == "cms":
        lm_file = None
    fin = ROOT.TFile.Open(input_file, "READ")

    if not fin or fin.IsZombie():
        raise RuntimeError(f"Cannot open {input_file}")
    flm = ROOT.TFile.Open(lm_file, "READ") if lm_file else None

    if lm_file and (not flm or flm.IsZombie()):
        fin.Close()
        raise RuntimeError(f"Cannot open LM reference {lm_file}")
    fout = ROOT.TFile.Open(output_file, "RECREATE")

    if not fout or fout.IsZombie():
        fin.Close()

        if flm:
            flm.Close()
        raise RuntimeError(f"Cannot create {output_file}")
    nfits = 0
    success = False
    try:
        state = "raw" if getattr(config, "nonflow_method", "template") == "cms" else "template"
        su._write(fout, ROOT.TNamed("nonflow_input_state", state))
        if pt_metadata is not None:
            for key, edges in pt_metadata.items():
                su._write(fout, ROOT.TNamed(key, ",".join(f"{x:.17g}" for x in edges)))

        if flm:
            for key, edges in (pt_metadata or {}).items():
                ref = flm.Get(key)

                if not ref:
                    raise RuntimeError(
                        f"LM reference missing {key}; rerun LM with updated Fourier code"
                    )
                ref_edges = [float(x) for x in ref.GetTitle().split(",")]

                if len(ref_edges) != len(edges) or any(
                    abs(a - b) > EDGE_TOLERANCE for a, b in zip(ref_edges, edges)
                ):
                    raise RuntimeError(f"LM/HM {key} mismatch in {input_file}")
            su._write(fout, ROOT.TNamed("lm_reference_file", os.path.abspath(lm_file)))
        pt_dirs = pu.subdirectories(fin, "pt_assoc_")

        if not pt_dirs:
            raise RuntimeError(f"No pt_assoc_* directories found in {input_file}")
        for pt_index, pt_name in enumerate(pt_dirs):
            lm_pt = flm.GetDirectory(pt_name) if flm else None

            if flm and not lm_pt:
                raise RuntimeError(f"Missing {pt_name} in LM reference {lm_file}")
            print(f"[fourier]   Associated-pT bin {pt_index}")
            nfits += _process_pt_dir(
                fin.GetDirectory(pt_name),
                su._mkdir(fout, pt_name),
                config,
                f"{os.path.basename(input_file)}:{pt_name}",
                lm_pt=lm_pt,
            )
        fout.Write()
        success = True
    finally:
        fout.Close()
        fin.Close()

        if flm:
            flm.Close()

        if not success and os.path.exists(output_file):
            os.remove(output_file)  # never leave a half-written output behind
    return nfits


# --------------------------------------------------------------------------
# Configuration and entry point
# --------------------------------------------------------------------------


def _read_fourier_config(cfg, *, is_lm, mass_summary=True):
    """Combine shared fit settings with Fourier-specific runtime options."""
    fcfg = cfg.get("fourier", {})
    is_hh = str(cfg.get("Dmeson", "")).lower() == "hh"
    # The raw-signal DeltaPhi stage has its own CMS implementation. These
    # options apply only to this mass-binned correlation-map pipeline.
    method = fcfg.get("nonflow_method", "template") if mass_summary else "template"
    if method not in ("template", "cms"):
        raise ValueError("fourier.nonflow_method must be template or cms")
    # _read_config returns a frozen dataclass. Keep its fields unchanged, but
    # extend it with runtime-only Fourier options in a mutable namespace.
    fit_cfg = cfg
    if is_hh:
        fit_cfg = {**cfg, "correlation": {**cfg.get("correlation", {}), "mass_bins": None}}
    parsed = cu._read_config(
        fit_cfg, harmonics=LM_HARMONICS if is_lm or method == "cms" else HM_HARMONICS,
        mass_summary=mass_summary and not is_hh,
    )
    config = SimpleNamespace(
        **{field.name: getattr(parsed, field.name) for field in fields(parsed)}
    )
    config.nonflow_method = method
    config.is_hh = is_hh
    if is_hh:
        config.mass_bins = None
    config.projection_method = fcfg.get("projection_method", "inverse_variance")

    if config.projection_method not in ("inverse_variance", "uniform"):
        raise ValueError("fourier.projection_method must be inverse_variance or uniform")
    if method == "cms":
        if "projection_method" in fcfg and fcfg["projection_method"] != "uniform":
            raise ValueError("CMS requires geometric eta averaging; use fourier.projection_method: uniform")
        config.projection_method = "eta_density"
    config.lm_template_error = fcfg.get("lm_template_error", True)

    if not isinstance(config.lm_template_error, bool):
        raise ValueError("fourier.lm_template_error must be a boolean")
    config.min_lm_shape_significance = float(fcfg.get("min_lm_shape_significance", 2.0))

    if (
        not math.isfinite(config.min_lm_shape_significance)
        or config.min_lm_shape_significance < 0
    ):
        raise ValueError("fourier.min_lm_shape_significance must be finite and nonnegative")
    return config


def run_fourier(cfg):
    """Run the LM or HM pass and optionally render per-file and summary PDFs."""
    base_dir = cfg["output_dir"]
    correlation_dir = os.path.join(base_dir, "correlations")
    fourier_dir = os.path.join(base_dir, "fourier")
    fcfg = cfg.get("fourier", {})
    is_lm = fcfg.get("is_lm", False)

    if not isinstance(is_lm, bool):
        raise ValueError("fourier.is_lm must be a boolean")
    lm_path = fcfg.get("lm_file")

    if is_lm and lm_path:
        raise ValueError("An LM fit cannot also use an LM reference")
    config = _read_fourier_config(cfg, is_lm=is_lm)
    if config.nonflow_method == "cms":
        fourier_dir = os.path.join(fourier_dir, "cms")
        if lm_path:
            print("[fourier] CMS raw fits ignore fourier.lm_file; supply the LM simultaneous-fit file to cms_nonflow")
        lm_path = None
    pdf_dir = os.path.join(fourier_dir, pu.PDF_SUBDIR)
    input_files = sorted(glob.glob(os.path.join(correlation_dir, "mult_*_pt_*.root")))
    print(f"\n{'=' * 70}\n FOURIER\n{'=' * 70}")
    print(f"[fourier] Input : {correlation_dir}")
    print(f"[fourier] Found : {len(input_files)} file(s)")

    if not input_files:
        raise RuntimeError(f"No correlation files found in {correlation_dir}")
    os.makedirs(fourier_dir, exist_ok=True)
    print(
        "[fourier] Δη ranges: "
        + ", ".join(f"[{low:g}, {high:g})" for low, high in config.eta_ranges)
    )
    print("[fourier] Harmonics: " + ", ".join(str(n) for n in config.harmonics))
    print(f"[fourier] Nonflow method: {config.nonflow_method}")

    if config.save_pdf:
        print(f"[fourier] PDF output: {pdf_dir}")

    if lm_path and os.path.isfile(lm_path) and len(input_files) != 1:
        raise ValueError(
            "A single LM ROOT file can only be used with one HM input; "
            "provide the LM fourier directory"
        )
    nfiles = nfits = npdfs = 0
    for input_file in input_files:
        name = os.path.basename(input_file)
        output_file = os.path.join(fourier_dir, name)
        print(f"[fourier] {name}")
        lm_file = None

        if lm_path:
            lm_file = os.path.join(lm_path, name) if os.path.isdir(lm_path) else lm_path

            if os.path.abspath(lm_file) == os.path.abspath(output_file):
                raise ValueError("LM reference and HM output would be the same file")

            if not os.path.isfile(lm_file):
                raise FileNotFoundError(f"Missing LM template file: {lm_file}")
        match = re.fullmatch(r"mult_(\d+)_pt_(\d+)\.root", name)

        if not match:
            raise RuntimeError(f"Invalid Fourier input name: {name}")
        ip = int(match.group(2))
        pt_cand = cfg["bins"]["pt_cand"]
        pt_assoc = cfg["bins"]["pt_assoc"]

        if ip + 1 >= len(pt_cand):
            raise ValueError(f"Invalid candidate pT index in {name}")
        metadata = {"pt_cand_range": pt_cand[ip:ip + 2], "pt_assoc_edges": pt_assoc}
        nfits += _process_file(
            input_file, output_file, config, lm_file=lm_file, pt_metadata=metadata
        )
        nfiles += 1

        if config.save_pdf:
            pdf_path = os.path.join(pdf_dir, os.path.splitext(name)[0] + ".pdf")
            try:
                pt_cand_bins = cfg["bins"]["pt_cand"]
                pu._save_fit_pdf(output_file, pdf_path, config, pt_cand_bins=pt_cand_bins)
                npdfs += 1
            except Exception as exc:  # the ROOT output is already safe on disk
                print(f"[fourier] WARNING: PDF for {name} failed: {exc}")

    if config.save_pdf and not config.is_hh:
        groups = {}
        for input_file in input_files:
            name = os.path.basename(input_file)
            match = re.fullmatch(r"mult_(\d+)_pt_\d+\.root", name)

            if not match:
                raise RuntimeError(f"Invalid Fourier input name: {name}")
            groups.setdefault(int(match.group(1)), []).append(os.path.join(fourier_dir, name))

        for mult_index, files in sorted(groups.items()):
            summary_path = os.path.join(pdf_dir, f"v2delta_vs_mass_summary_mult_{mult_index}.pdf")
            try:
                pu.save_v2_mass_summary_pdf(files, summary_path, cfg["bins"]["pt_cand"])
                npdfs += 1
                print(f"[fourier] V2Delta(mass) summary: {summary_path}")
            except Exception as exc:
                print(
                    f"[fourier] WARNING: V2Delta(mass) summary "
                    f"for mult_{mult_index} failed: {exc}"
                )
                traceback.print_exc()

    print(f"\n[fourier] Processed {nfiles} file(s), performed {nfits} fit(s)")

    if config.save_pdf:
        print(f"[fourier] Wrote {npdfs} PDF file(s)")
    print(f"[fourier] Output: {fourier_dir}")
    return fourier_dir


