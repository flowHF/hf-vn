#!/usr/bin/env python3
"""Build corrected two-particle correlation maps.

Input:

    \<output_dir>/preprocessed/mult_*_pt_*.root

    (with SE/pt_assoc_\<i>/hPairs, the preprocessed same-event THnSparse)

Output:

    \<output_dir>/correlations/mult_*_pt_*.root

        pt_assoc_\<i>/mass_\<j>/{hCorr, hSE, ...}   written by corr2pc::processFile

        pt_assoc_\<i>/hInvariantMass               raw SE pair counts, native mass binning

        pt_assoc_\<i>/invariant_mass_definition

        pt_assoc_\<i>/hJetYieldVsMass             optional near-side SR-LR yields
        pt_assoc_\<i>/mass_\<j>/{hJetYield, hJetShortRange, hJetLongRange,
                                      jet_yield_definition}

hInvariantMass is built from the native mass axis of hPairs. The coarser

correlation.mass_bins are used only for the correlation maps.

For Dmeson: hh, mass/BDT axes are omitted and a single inclusive map is
stored under mass_0 for compatibility. No invariant-mass spectrum is written.

correlation.jet_yield.enabled stores jet yields from hCorr, geometrically
normalized by selected eta width and phi-bin width. short_eta_ranges,
long_eta_ranges and nearside_halfwidth select the measurement. It is enabled
automatically when fourier.nonflow_method is cms unless explicitly disabled.

Optional PDF output (config):

    correlation.save_pdf          : bool, default False

    correlation.pdf_draw_option   : ROOT draw option, default "COLZ"

                                    (3D options such as "SURF1" / "LEGO2" work too)

    correlation.pdf_palette_alpha : opacity of the cold-to-warm palette, default 0.5

    correlation.pt_assoc_bins     : optional pt_assoc edges, used for pad labels

One multi-page PDF is written per input file to \<output_dir>/correlations/pdf/:

one page per pt_assoc bin, one pad per invariant-mass bin showing hCorr.

"""
import glob
import json
import math
import os
import traceback
from array import array
import ROOT
try:
    from utils import plot_utils as pu
    from utils import config_utils as cu
except ImportError:  # project root not on sys.path: assume utils/ sits next to this folder
    import sys
    sys.path.insert(0, os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    ))
    from utils import plot_utils as pu
    from utils import config_utils as cu
REQUIRED_AXES = ("pool", "mass", "delta_eta", "delta_phi")
EDGE_TOLERANCE = 1.0e-7

def _read_jet_yield_config(cfg):
    """Read the optional SR-LR measurement, also enabled by the CMS pass."""
    settings = cfg.get("correlation", {}).get("jet_yield", {})
    cms = cfg.get("fourier", {}).get("nonflow_method", "template") == "cms"
    enabled = settings.get("enabled", cms)
    if not isinstance(enabled, bool):
        raise ValueError("correlation.jet_yield.enabled must be a boolean")
    if cms and not enabled:
        raise ValueError("CMS subtraction requires correlation.jet_yield.enabled")
    if not enabled:
        return None

    def ranges(key, default):
        raw = settings.get(key, default)
        if not raw:
            raise ValueError(f"correlation.jet_yield.{key} requires eta intervals")
        result = []
        for pair in raw:
            if len(pair) != 2:
                raise ValueError(f"correlation.jet_yield.{key} entries must be [low, high]")
            low, high = map(float, pair)
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                raise ValueError(f"Invalid correlation.jet_yield.{key} interval")
            result.append((low, high))
        result.sort()
        if any(hi > lo for (_, hi), (lo, _) in zip(result, result[1:])):
            raise ValueError(f"correlation.jet_yield.{key} intervals overlap")
        return tuple(result)

    short = ranges("short_eta_ranges", [[-1.0, 1.0]])
    long = ranges("long_eta_ranges", cfg.get("fourier", {}).get("delta_eta_ranges"))
    if any(max(a, c) < min(b, d) for a, b in short for c, d in long):
        raise ValueError("Jet-yield short- and long-range eta intervals must be disjoint")
    halfwidth = float(settings.get("nearside_halfwidth", math.pi / 2))
    if not math.isfinite(halfwidth) or not 0 < halfwidth <= math.pi:
        raise ValueError("correlation.jet_yield.nearside_halfwidth must be in (0, pi]")
    return dict(short_eta_ranges=short, long_eta_ranges=long,
                nearside_halfwidth=halfwidth)


def _eta_bins(hist, eta_ranges):
    """Select regular bins by centre, rejecting missing or clipped intervals."""
    axis = hist.GetXaxis()
    selected = set()
    for low, high in eta_ranges:
        if low < axis.GetXmin() - EDGE_TOLERANCE or high > axis.GetXmax() + EDGE_TOLERANCE:
            raise RuntimeError(f"Eta interval [{low}, {high}) exceeds hCorr acceptance")
        bins = [i for i in range(1, axis.GetNbins() + 1)
                if low <= axis.GetBinCenter(i) < high]
        if not bins:
            raise RuntimeError(f"No hCorr eta bins in [{low}, {high})")
        selected.update(bins)
    return sorted(selected)


def _project_eta_density(hist, eta_ranges, name="hDeltaPhi"):
    """Sum corrected pair counts / selected eta width / phi-bin width.

    Use geometric normalization, including zero-error bins, rather than
    statistical weights. This gives SR, LR and the CMS Fourier fit identical
    pair-density units. Errors treat the hCorr bins as independent.
    """
    selected = _eta_bins(hist, eta_ranges)
    eta_width = sum(hist.GetXaxis().GetBinWidth(i) for i in selected)
    axis = hist.GetYaxis()
    edges = array("d", [axis.GetBinLowEdge(i) for i in range(1, axis.GetNbins() + 2)])
    out = ROOT.TH1D(name, ";#Delta#varphi (rad);Corrected pairs / #Delta#eta / rad",
                    axis.GetNbins(), edges)
    out.SetDirectory(0)
    for j in range(1, axis.GetNbins() + 1):
        values = [hist.GetBinContent(i, j) for i in selected]
        errors = [hist.GetBinError(i, j) for i in selected]
        if not all(math.isfinite(x) for x in values + errors):
            raise RuntimeError(f"Nonfinite hCorr content or error at phi bin {j}")
        norm = eta_width * axis.GetBinWidth(j)
        out.SetBinContent(j, math.fsum(values) / norm)
        out.SetBinError(j, math.sqrt(math.fsum(e * e for e in errors)) / norm)
    return out


def _jet_yield_definition(hist, settings):
    """Persist requested selections and actual binning for HM/LM checks."""
    definition = dict(settings, normalization="hCorr counts / eta width / phi width",
                      eta_selection="bin centres; independent hCorr bin errors")
    for key in ("short_eta_ranges", "long_eta_ranges"):
        axis = hist.GetXaxis()
        definition[key + "_bins"] = [
            [axis.GetBinLowEdge(i), axis.GetBinUpEdge(i)]
            for i in _eta_bins(hist, settings[key])
        ]
    definition["phi_edges"] = [hist.GetYaxis().GetBinLowEdge(i)
                               for i in range(1, hist.GetNbinsY() + 2)]
    return json.dumps(definition, sort_keys=True)


def _measure_jet_yield(hist, settings):
    """Return SR/LR densities and their periodic near-side excess integral."""
    axis = hist.GetYaxis()
    span = axis.GetXmax() - axis.GetXmin()
    # Input axes can retain single-precision pi endpoints.
    if not math.isclose(span, 2 * math.pi, rel_tol=EDGE_TOLERANCE, abs_tol=EDGE_TOLERANCE):
        raise RuntimeError(
            "Jet yield requires one full 2*pi period of hCorr; "
            f"got [{axis.GetXmin():.12g}, {axis.GetXmax():.12g}] (span {span:.12g})"
        )
    short = _project_eta_density(hist, settings["short_eta_ranges"], "hJetShortRange")
    long = _project_eta_density(hist, settings["long_eta_ranges"], "hJetLongRange")
    if set(_eta_bins(hist, settings["short_eta_ranges"])) & set(_eta_bins(hist, settings["long_eta_ranges"])):
        raise RuntimeError("Jet-yield SR and LR selections share hCorr eta bins")
    halfwidth = settings["nearside_halfwidth"]
    total = variance = 0.0
    for i in range(1, axis.GetNbins() + 1):
        low, high = axis.GetBinLowEdge(i), axis.GetBinUpEdge(i)
        first = math.floor((low - halfwidth) / (2 * math.pi))
        last = math.ceil((high + halfwidth) / (2 * math.pi))
        width = sum(max(0.0, min(high, halfwidth + k * 2 * math.pi)
                        - max(low, -halfwidth + k * 2 * math.pi))
                    for k in range(first, last + 1))
        total += width * (short.GetBinContent(i) - long.GetBinContent(i))
        variance += width**2 * (short.GetBinError(i)**2 + long.GetBinError(i)**2)
    return short, long, total, math.sqrt(variance)


def _store_jet_yields(output_file, settings, mass_bins):
    """Add one jet measurement per corrected mass map and a mass summary."""
    target = ROOT.TFile.Open(str(output_file), "UPDATE")
    if not target or target.IsZombie():
        raise RuntimeError(f"Cannot update {output_file}")
    try:
        for pt_name in pu.subdirectories(target, "pt_assoc_"):
            pt = target.GetDirectory(pt_name)
            summary = None
            if mass_bins is not None:
                summary = ROOT.TH1D("hJetYieldVsMass", ";invariant mass;Jet yield",
                                    len(mass_bins) - 1, array("d", mass_bins))
                summary.SetDirectory(0)
            for i in range(len(mass_bins) - 1 if mass_bins is not None else 1):
                directory = pt.GetDirectory(f"mass_{i}")
                hist = directory.Get("hCorr") if directory else None
                if not hist:
                    raise RuntimeError(f"Missing {pt_name}/mass_{i}/hCorr in {output_file}")
                short, long, value, error = _measure_jet_yield(hist, settings)
                jet = ROOT.TH1D("hJetYield", "Near-side SR-LR jet yield", 1, 0, 1)
                jet.SetDirectory(0)
                jet.SetBinContent(1, value)
                jet.SetBinError(1, error)
                definition = ROOT.TNamed("jet_yield_definition", _jet_yield_definition(hist, settings))
                for obj in (short, long, jet, definition):
                    directory.WriteTObject(obj, obj.GetName(), "Overwrite")
                if summary is not None:
                    summary.SetBinContent(i + 1, value)
                    summary.SetBinError(i + 1, error)
            if summary is not None:
                pt.WriteTObject(summary, summary.GetName(), "Overwrite")
    finally:
        target.Close()


def _store_invariant_mass(input_file, output_file, axes):
    """Project each associated-pT bin's SE pairs onto their native mass axis."""
    source = ROOT.TFile.Open(str(input_file), "READ")

    if not source or source.IsZombie():
        raise RuntimeError(f"Cannot open pair file: {input_file}")
    target = ROOT.TFile.Open(str(output_file), "UPDATE")

    if not target or target.IsZombie():
        source.Close()
        raise RuntimeError(f"Cannot update {output_file}")
    try:
        for pt_name in pu.subdirectories(target, "pt_assoc_"):
            pair_path = f"SE/{pt_name}/hPairs"
            sparse = source.Get(pair_path)

            if not sparse or not sparse.InheritsFrom("THnSparse"):
                raise RuntimeError(f"Missing THnSparse {pair_path} in {input_file}")
            # Integrate all retained axes over regular bins, including any BDT
            # axes. The preprocessed pair selection already defines this sample.
            for index in range(sparse.GetNdimensions()):
                axis = sparse.GetAxis(index)
                axis.SetRange(1, axis.GetNbins())
            mass_hist = sparse.Projection(axes["mass"], "E")
            mass_hist.SetDirectory(0)
            mass_hist.SetName("hInvariantMass")
            mass_hist.SetTitle("SE pairs;invariant mass;Pair counts")
            directory = target.GetDirectory(pt_name)
            directory.WriteTObject(mass_hist, "hInvariantMass", "Overwrite")
            definition = ROOT.TNamed(
                "invariant_mass_definition",
                f"Raw SE pair counts from {input_file}:{pair_path}; native mass binning; "
                "all other axes integrated over regular bins; no per-trigger normalization",
            )
            directory.WriteTObject(definition, "invariant_mass_definition", "Overwrite")
    finally:
        target.Close()
        source.Close()

def run_correlations(cfg):
    base_dir = cfg["output_dir"]
    preprocess_dir = os.path.join(base_dir, "preprocessed")
    correlation_dir = os.path.join(base_dir, "correlations")
    pdf_dir = os.path.join(correlation_dir, pu.PDF_SUBDIR)
    input_files = sorted(glob.glob(os.path.join(preprocess_dir, "mult_*_pt_*.root")))
    print(f"\n{'=' * 70}\n CORRELATIONS\n{'=' * 70}")
    print(f"[correlations] Input : {preprocess_dir}")
    print(f"[correlations] Found : {len(input_files)} file(s)")

    if not input_files:
        raise RuntimeError(f"No preprocessed files found in {preprocess_dir}")
    corr_cfg = cfg.get("correlation", {})
    is_hh = str(cfg.get("Dmeson", "")).lower() == "hh"
    if is_hh:
        # Match preprocess.read_config's reduced axis order, including configs
        # that still list the D-meson mass and BDT axes.
        keep = cfg.get("preprocess", {}).get("keep_axes", cfg.get("keep_axes", REQUIRED_AXES))
        keep = [name for name in keep
                if name not in ("mass", "bdt") and not name.startswith("bdt_")]
        axis_cfg = {"preprocess": {"keep_axes": keep}}
        axes = cu._get_axis_indices(axis_cfg, ("pool", "delta_eta", "delta_phi"))
        axes["mass"] = -1
        mass_bins = None
    else:
        axes = cu._get_axis_indices(cfg, REQUIRED_AXES)
        mass_bins = cu._get_mass_bins(corr_cfg)
    jet_settings = _read_jet_yield_config(cfg)
    settings = cu._make_settings(cfg, mass_bins or [], axes, 0.001)
    save_pdf = bool(corr_cfg.get("save_pdf", False))
    draw_option = str(corr_cfg.get("pdf_draw_option", 'COLZ'))
    palette_alpha = float(corr_cfg.get("pdf_palette_alpha", pu.DEFAULT_PALETTE_ALPHA))
    pt_cand_bins = cfg.get("bins", {}).get("pt_cand") if save_pdf else None

    if save_pdf:
        if not pt_cand_bins or len(pt_cand_bins) < 2:
            raise RuntimeError("Missing bins.pt_cand for PDF candidate-pT labels")
        pt_cand_bins = tuple(float(x) for x in pt_cand_bins)

        if any(hi <= lo for lo, hi in zip(pt_cand_bins, pt_cand_bins[1:])):
            raise RuntimeError("bins.pt_cand must be strictly increasing")
        print(f"[correlations] PDF output: {pdf_dir} (draw option '{draw_option}')")
    os.makedirs(correlation_dir, exist_ok=True)
    print("[correlations] Inclusive hh maps (no mass axis)" if is_hh else
          "[correlations] Mass bins: " + ", ".join(f"{x:g}" for x in mass_bins))

    corr_ns = cu._load_corr2pc() # load the C++ shared library and get the corr2pc namespace
    nfiles = nmaps = npdfs = 0
    for input_file in input_files:
        name = os.path.basename(input_file)
        output_file = os.path.join(correlation_dir, name)
        print(f"[correlations] {name}")
        try:
            maps = corr_ns.processFile(input_file, output_file, settings)
        except Exception as exc:
            # cppyy exposes the original C++ diagnostic through what().
            detail = exc.what() if callable(getattr(exc, "what", None)) else str(exc)
            raise RuntimeError(
                f"Correlation processing failed for {input_file}: {detail}"
            ) from exc

        if maps is None or maps < 0:
            raise RuntimeError(f"processFile failed for {input_file} (returned {maps})")
        if not is_hh:
            _store_invariant_mass(input_file, output_file, axes)
        if jet_settings is not None:
            _store_jet_yields(output_file, jet_settings, mass_bins)
        nfiles += 1
        nmaps += maps

        if save_pdf:
            pdf_path = os.path.join(pdf_dir, os.path.splitext(name)[0] + ".pdf")
            eta_min, eta_max = cfg["fourier"]["delta_eta_ranges"][0][0], cfg["fourier"]["delta_eta_ranges"][1][1]
            try:
                for histogram_name, suffix in (
                    ("hCorr", ""),
                    ("hSE", "_SE"),
                    ("hME", "_ME"),
                ):
                    output_pdf = os.path.splitext(pdf_path)[0] + suffix + ".pdf"
                    pu._save_maps_pdf(
                        output_file, output_pdf, mass_bins, draw_option,
                        palette_alpha, eta_min, eta_max,
                        histogram_name=histogram_name,
                        pt_cand_bins=pt_cand_bins,
                    )
                npdfs += 1
                print(f"[correlations] Wrote PDFs for {name}")
            except Exception as exc:
                print(f"[correlations] WARNING: PDF for {name} failed: {exc}")
                traceback.print_exc()
    print(
        f"\n[correlations] Processed {nfiles} file(s), "
        f"wrote {nmaps} mass-map triplets"
    )

    if save_pdf:
        print(f"[correlations] Wrote {npdfs} PDF file(s)")
    print(f"[correlations] Output: {correlation_dir}")
    return correlation_dir
