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

hInvariantMass is built from the native mass axis of hPairs. The coarser

correlation.mass_bins are used only for the correlation maps.

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

def _store_invariant_mass(mass_file, output_file):
    """Copy candidate-level mass spectrum from preprocessed D-candidate file to correlation output file."""
    source = ROOT.TFile.Open(str(mass_file), "READ")

    if not source or source.IsZombie():
        raise RuntimeError(f"Cannot open D-candidate mass file: {mass_file}")
    target = ROOT.TFile.Open(str(output_file), "UPDATE")

    if not target or target.IsZombie():
        source.Close()
        raise RuntimeError(f"Cannot update {output_file}")
    try:
        mass_hist = source.Get("hInvariantMass")

        if not mass_hist:
            raise RuntimeError(f"Missing hInvariantMass in {mass_file}")
        for pt_name in pu.subdirectories(target, "pt_assoc_"):
            directory = target.GetDirectory(pt_name)
            directory.WriteTObject(mass_hist, "hInvariantMass", "Overwrite")
            definition = ROOT.TNamed("invariant_mass_definition", f"Candidate-level histogram from {mass_file}")
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
    axes = cu._get_axis_indices(cfg, REQUIRED_AXES)
    mass_bins = cu._get_mass_bins(corr_cfg)
    settings = cu._make_settings(cfg, mass_bins, axes, 0.001)
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
    print("[correlations] Mass bins: " + ", ".join(f"{x:g}" for x in mass_bins))

    corr_ns = cu._load_corr2pc() # load the C++ shared library and get the corr2pc namespace
    nfiles = nmaps = npdfs = 0
    for input_file in input_files:
        name = os.path.basename(input_file)
        output_file = os.path.join(correlation_dir, name)
        print(f"[correlations] {name}")
        maps = corr_ns.processFile(input_file, output_file, settings) # call the C++ function to process the file and generate correlation maps

        if maps is None or maps < 0:
            raise RuntimeError(f"processFile failed for {input_file} (returned {maps})")
        mass_file = os.path.join(base_dir, "preprocessed_dmeson", name)

        if not os.path.isfile(mass_file):
            raise FileNotFoundError(f"Missing preprocessed D-candidate file: {mass_file}")
        _store_invariant_mass(mass_file, output_file) # copy the candidate-level mass spectrum from the preprocessed D-candidate only file to the correlation output file. Not using the D-h pairs.
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
