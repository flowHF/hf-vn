#!/usr/bin/env python3
"""
Supports two extraction modes:
  - DeltaPhiBinning (PairYieldsVsPhi.root):
    ONE histogram per (ptCand, ptHad).  Single integrated mass bin.
  - MassBinning (CorrelationsResults.root):
    One mass sub-bin per (ptCand, mass) — different pt bins may have
    different mass edge arrays.

Input YAML config:
  - Dmeson, outdir, suffix        : identify D meson species & extraction output
  - ptBinsCand, ptBinsHad, invMassBins : binning definitions
  - task_LM (optional)            : low-multiplicity template config
  - fitConfig (optional)          : fit parameters (FitFunction, FixBaseline, …)

Output:
  {outdir}/CorrelExtract_{suffix}/CorrelationFitResults/
    Output_CorrelationFitting_Root/CorrPhi{DMeson}_FinalPlots.root
    outputPathOutput_CorrelationFitting_png/*.png

Usage:
  python3 FitCorrel.py config_CorrAnalysis_v2_010_negDeta.yaml
"""

import argparse
import math
import os
import sys
from array import array
from ctypes import c_int, c_double

import ROOT
import yaml

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

# ===================================================================
# Compile DhCorrelationFitter C++ class
# ===================================================================
ROOT.gSystem.AddIncludePath("-I/home/wuct/Software/miniforge3/envs/alice/include")
_fitter_cxx = os.path.join(os.path.dirname(__file__), "DhCorrelationFitter.cxx")
ROOT.gSystem.CompileMacro(_fitter_cxx, "kO")
from ROOT import DhCorrelationFitter

# ===================================================================
# ROOT style helpers
# ===================================================================
def set_canvas_style():
    style = ROOT.gStyle
    style.SetOptStat(0)
    style.SetPadLeftMargin(0.2)
    style.SetPadRightMargin(0.005)
    style.SetPadBottomMargin(0.2)
    style.SetFrameLineWidth(2)
    style.SetLineWidth(2)
    style.SetCanvasDefH(1126)
    style.SetCanvasDefW(1840)


def set_th1_style(histo, title, x_title, y_title,
                  marker_style=ROOT.kFullCircle,
                  marker_color=ROOT.kRed + 1,
                  marker_size=1.4,
                  line_color=ROOT.kRed + 1,
                  line_width=3):
    histo.SetTitle(title)
    histo.GetXaxis().SetTitle(x_title)
    histo.GetYaxis().SetTitle(y_title)
    histo.SetMarkerStyle(marker_style)
    histo.SetMarkerColor(marker_color)
    histo.SetMarkerSize(marker_size)
    histo.SetLineColor(line_color)
    histo.SetLineWidth(line_width)
    histo.GetXaxis().SetTitleOffset(0.8)
    histo.GetYaxis().SetTitleOffset(1.3)
    histo.GetXaxis().SetTitleSize(0.045)
    histo.GetYaxis().SetTitleSize(0.045)
    histo.GetXaxis().SetLabelSize(0.045)
    histo.GetYaxis().SetLabelSize(0.045)


# ===================================================================
# Directory name helpers
# ===================================================================
def pt_cand_dir_name(pt_min, pt_max):
    low = int(round(pt_min * 10))
    high = int(round(pt_max * 10))
    return f"PtCandBin_{low}_{high}"


def pt_had_dir_name(pt_min, pt_max):
    low = int(round(pt_min * 10))
    high = int(round(pt_max * 10))
    return f"PtHadBin_{low}_{high}"


# ===================================================================
# Helper: ensure per-bin parameter values
# ===================================================================
def ensure_per_bin(value, n_bins, name, default):
    if value is None:
        return [default] * n_bins
    if isinstance(value, list):
        if len(value) < n_bins:
            raise ValueError(f"{name} must provide at least {n_bins} entries, got {len(value)}")
        return value[:n_bins]
    return [value] * n_bins


# ===================================================================
# Main entry point
# ===================================================================
def fit_correl(cfg_path):
    set_canvas_style()

    # ---- Load config --------------------------------------------------
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    suffix = config["suffix"]
    outdir = config["outdir"]

    # ---- D meson species ----------------------------------------------
    Dmeson = config.get("Dmeson", "Dzero")
    species_map = {
        "D0": (0, "D0", "D^{0}"),
        "Dzero": (0, "D0", "D^{0}"),
        "Dplus": (1, "Dplus", "D^{+}"),
        "Ds": (2, "Ds", "D_{s}^{+}"),
    }
    if Dmeson not in species_map:
        print(f"[ERROR] Unknown D meson: {Dmeson}")
        sys.exit(1)
    dmeson_species, dmeson_name, dmeson_label = species_map[Dmeson]

    # ---- Paths --------------------------------------------------------
    extract_dir = os.path.join(outdir, f"CorrelExtract_{suffix}")

    pair_yields_path = os.path.join(extract_dir, "AssociatedPairsYields",
                                    "PairYieldsVsPhi.root")
    correlations_path = os.path.join(extract_dir, "CorrelationsResults",
                                     "CorrelationsResults.root")

    use_pairs_file = os.path.exists(pair_yields_path)
    input_file_path = pair_yields_path if use_pairs_file else correlations_path

    if not os.path.exists(input_file_path):
        print(f"[ERROR] Input file not found")
        print(f"  Checked: {pair_yields_path}")
        print(f"  Checked: {correlations_path}")
        sys.exit(1)

    output_path = os.path.join(extract_dir, "CorrelationFitResults")
    out_png_dir = os.path.join(output_path, "outputPathOutput_CorrelationFitting_png")
    out_root_dir = os.path.join(output_path, "Output_CorrelationFitting_Root")
    os.makedirs(out_png_dir, exist_ok=True)
    os.makedirs(out_root_dir, exist_ok=True)

    # Directories for fit parameters and covariance matrices
    out_par_dir = os.path.join(output_path, "FitParameters")
    os.makedirs(out_par_dir, exist_ok=True)

    print(f"[INFO] Input file:  {input_file_path}")
    print(f"[INFO] Output PNG:  {out_png_dir}")
    print(f"[INFO] Output ROOT: {out_root_dir}")

    # ---- Binning ------------------------------------------------------
    pt_bins_cand = [float(x) for x in config["ptBinsCand"]]
    pt_bins_had = [float(x) for x in config["ptBinsHad"]]
    inv_mass_bins_raw = config["invMassBins"]

    n_pt_cand = len(pt_bins_cand) - 1
    n_pt_had = len(pt_bins_had) - 1

    # Build mass edges per pt-candidate bin (different pt bins may have
    # different mass edge arrays)
    mass_edges_per_pt = []
    for i_pt in range(n_pt_cand):
        if i_pt < len(inv_mass_bins_raw):
            edges = [float(x) for x in inv_mass_bins_raw[i_pt]]
        else:
            edges = [float(x) for x in inv_mass_bins_raw[-1]]
        mass_edges_per_pt.append(edges)

    if use_pairs_file:
        # ---- DeltaPhiBinning ----------------
        # PairYieldsVsPhi.root has ONE histogram per (ptCand, ptHad).
        # Single integrated mass bin (matching C++ nBinsInvMass=1).
        # All ptCand fits fill one output histogram per (ptHad).
        mass_min = mass_edges_per_pt[0][0]
        mass_max = mass_edges_per_pt[0][-1]
        n_mass_total = 1
        mass_combos = [(0, mass_min, mass_max)]  # dummy entry for the loop
        print(f"[INFO] Mass range: [{mass_min}, {mass_max}]  "
              f"(single integrated bin — DeltaPhiBinning)")
    else:
        # ---- MassBinning ----------------------------
        # CorrelationsResults.root has one directory per mass sub-bin.
        mass_combos = []
        for i_pt in range(n_pt_cand):
            sub_edges = mass_edges_per_pt[i_pt]
            for i_mass_local in range(len(sub_edges) - 1):
                mass_combos.append((i_pt, i_mass_local,
                                    sub_edges[i_mass_local],
                                    sub_edges[i_mass_local + 1]))
        n_mass_total = len(mass_combos)
        print(f"[INFO] MassBinning: {n_mass_total} mass sub-bins "
              f"across {n_pt_cand} pt cand bins")

    # ---- LM template --------------------------------------------------
    method = config.get("method", "DeltaPhiBinning")
    task_lm = config.get("task_LM", {})
    lm_template_path = None
    if task_lm and task_lm.get("do", False):
        lm_outdir = task_lm.get("outdir", "")
        if lm_outdir:
            if method == "MassBinning":
                lm_template_path = os.path.join(
                    lm_outdir, f"CorrelExtract_{suffix}",
                    "CorrelationsResults", "CorrelationsResults.root",
                )
            else:
                lm_template_path = os.path.join(
                    lm_outdir, f"CorrelExtract_{suffix}",
                    "AssociatedPairsYields", "PairYieldsVsPhi.root",
                )
            if not os.path.exists(lm_template_path):
                print(f"[WARNING] LM template not found: {lm_template_path}")
                lm_template_path = None
            else:
                print(f"[INFO] LM template: {lm_template_path}")

    # ---- Fit config ---------------------------------------------------
    fit_config = config.get("fitConfig", {})
    fit_functions_raw = fit_config.get("FitFunction", config.get("FitFunction"))
    if fit_functions_raw is None:
        fit_functions = [8] * n_pt_cand
    else:
        fit_functions = ensure_per_bin(fit_functions_raw, n_pt_cand,
                                       "FitFunction", 8)
    fit_functions = [int(f) for f in fit_functions]

    # If no LM template, force all to type 8
    if lm_template_path is None:
        for i in range(n_pt_cand):
            fit_functions[i] = DhCorrelationFitter.kV2DeltaModulationLowMult
        print("[INFO] No LM template — all fit functions forced to type 8")

    fix_baseline = int(fit_config.get("FixBaseline", config.get("FixBaseline", 0)))
    fix_mean = int(fit_config.get("FixMean", config.get("FixMean", 0)))
    n_baseline_points = int(fit_config.get("nBaselinePoints",
                                         config.get("nBaselinePoints", 0)))
    points_for_baseline = fit_config.get("binsForBaseline",
                                       config.get("binsForBaseline", []))
    if n_baseline_points and len(points_for_baseline) != n_baseline_points:
        print("[ERROR] 'binsForBaseline' length must match 'nBaselinePoints'")
        sys.exit(1)
    points_for_baseline = [int(x) for x in points_for_baseline]

    with_ped_lm = fit_config.get("WithPedLM", False)
    fix_lm_factor = fit_config.get("FixLMFactor", False)
    temp_func = int(fit_config.get("tempFunc", 0))

    par_vals = [float(x) for x in fit_config.get("parVals", config.get("parVals", []))]
    par_low = [float(x) for x in fit_config.get("parLowBounds", config.get("parLowBounds", []))]
    par_up = [float(x) for x in fit_config.get("parUpperBounds", config.get("parUpperBounds", []))]
    if par_vals and not (len(par_vals) == len(par_low) == len(par_up)):
        print("[ERROR] parVals, parLowBounds, parUpperBounds must have same length")
        sys.exit(1)

    is_reflected = config.get("IsReflected", config.get("IsRiflected", False))
    shift_base_up = config.get("ShiftBaseUp", False)
    shift_base_down = config.get("ShiftBaseDown", False)
    draw_systematics = config.get("DrawSystematics", False)
    same_systematics = config.get("SameSystematics", False)

    # ---- Print config summary -----------------------------------------
    print("===========================")
    print("Input variables from config")
    for i, func_val in enumerate(fit_functions):
        print(f"  iPt = {i + 1}  FitFunction = {func_val}")
    print(f"  FixBaseline = {fix_baseline}")
    print(f"  FixMean     = {fix_mean}")
    print("===========================\n")

    # ---- Open input file -----------------------------------------------
    in_file = ROOT.TFile.Open(input_file_path)
    if not in_file or in_file.IsZombie():
        print(f"[ERROR] Could not open {input_file_path}")
        sys.exit(1)

    in_file_lm = None
    if lm_template_path:
        in_file_lm = ROOT.TFile.Open(lm_template_path)
        if not in_file_lm or in_file_lm.IsZombie():
            print(f"[WARNING] Could not open LM template: {lm_template_path}")
            in_file_lm = None

    # ---- Fit range -------------------------
    f_min = -0.5 * math.pi
    f_max = 1.5 * math.pi

    # ---- Prepare output histograms (indexed by [i_pt_had][i_mass_local])
    pt_cand_edges_arr = array("d", pt_bins_cand)

    # For MassBinning: use local mass index; for DeltaPhiBinning: 1 mass bin
    if not use_pairs_file:
        # MassBinning: n_mass_local = mass sub-bins per ptCand
        n_mass_local = max(len(me) - 1 for me in mass_edges_per_pt)
    else:
        n_mass_local = 1

    h_v2_delta = [[None] * n_mass_local for _ in range(n_pt_had)]
    h_lm_factor = [None] * n_pt_had

    for i_pt_had in range(n_pt_had):
        for i_mass_local in range(n_mass_local):
            tag = f"PtBinAssoc{i_pt_had + 1}_InvMassBin{i_mass_local + 1}"
            h = ROOT.TH1D(f"hv2Delta_{tag}", "", n_pt_cand, pt_cand_edges_arr)
            h.SetDirectory(ROOT.nullptr)
            h_v2_delta[i_pt_had][i_mass_local] = h

        # LM Factor: single 1D histogram per ptHad (no mass binning)
        h_lm = ROOT.TH1D(f"hLMFactor_PtBinAssoc{i_pt_had + 1}", "", n_pt_cand, pt_cand_edges_arr)
        h_lm.SetDirectory(ROOT.nullptr)
        h_lm_factor[i_pt_had] = h_lm

    # Storage for covariance matrix 2D histograms
    # cov_global_histos → appended to canvas ROOT file (CorrPhiD0_*.root)
    # cov_lm_histos     → appended to hLMtemplate ROOT file
    cov_global_histos = {}
    cov_lm_histos = {}
    # Storage for parameter TH1D histograms (same target as covariance)
    par_global_histos = {}
    par_lm_histos = {}
    # Storage for the LM template fit chi2/ndf (1-bin TH1D, hLMtemplate ROOT file)
    chi2lm_histos = {}

    # # merge LM for mass binning case (same LM for all mass bins of a given ptHad)
    # merged_lm = {}
    # if not use_pairs_file and in_file_lm:
    #     for i_pt_cand in range(n_pt_cand):
    #         pt_cand_min = pt_bins_cand[i_pt_cand]
    #         pt_cand_max = pt_bins_cand[i_pt_cand + 1]
    #         for i_pt_had, (pt_had_min, pt_had_max) in enumerate(
    #                 zip(pt_bins_had[:-1], pt_bins_had[1:])):
    #             pc_dir = pt_cand_dir_name(pt_cand_min, pt_cand_max)
    #             ph_dir = pt_had_dir_name(pt_had_min, pt_had_max)
    #             key = f"{pc_dir}/{ph_dir}"

    #             h_merged = None
    #             for i_mass_global, mass_combo in enumerate(mass_combos):
    #                 mass_i_pt, mass_i_local, mass_min, mass_max = mass_combo
    #                 if mass_i_pt != i_pt_cand:
    #                     continue
    #                 im_dir = (f"InvMassBin_{int(mass_min * 1000):.0f}_"
    #                           f"{int(mass_max * 1000):.0f}")
    #                 hist_path = f"{pc_dir}/{ph_dir}/{im_dir}/hCorrectedCorrel"
    #                 h_tmp = in_file_lm.Get(hist_path)
    #                 if h_tmp:
    #                     h_tmp.SetDirectory(ROOT.nullptr)
    #                     if h_merged is None:
    #                         h_merged = ROOT.TH1D(h_tmp)
    #                         h_merged.SetDirectory(ROOT.nullptr)
    #                     else:
    #                         h_merged.Add(h_tmp)
    #             merged_lm[key] = h_merged
    #             if h_merged:
    #                 print(f"    LM merged for {key}: {h_merged.GetNbinsX()} bins")

    # ============== MAIN FIT LOOP ==============
    # For use_pairs_file (DeltaPhiBinning): iterate over ptCand internally
    # For !use_pairs_file (MassBinning): each mass_combos entry carries
    #   its own (i_pt_cand, mass_min, mass_max).
    for i_mass_global, mass_combo in enumerate(mass_combos):
        if len(mass_combo) == 4:
            mass_i_pt, mass_i_local, mass_min, mass_max = mass_combo
        else:
            mass_i_pt, mass_min, mass_max = mass_combo
            mass_i_local = 0
        # Determine which ptCand bins to process in this mass combo
        if use_pairs_file:
            cand_indices = list(range(n_pt_cand))
            mass_range_label = f"[{mass_min}, {mass_max}]"
        else:
            cand_indices = [mass_i_pt]
            mass_range_label = (f"[{mass_min}, {mass_max}]  "
                                f"PtCand[{pt_bins_cand[mass_i_pt]}, {pt_bins_cand[mass_i_pt+1]}]")

        for i_pt_cand in cand_indices:
            pt_cand_min = pt_bins_cand[i_pt_cand]
            pt_cand_max = pt_bins_cand[i_pt_cand + 1]
            print(f"\\n[INFO] Mass[{mass_i_local + 1}/{n_mass_local}] "
                  f"PtCand[{pt_cand_min}, {pt_cand_max}]  {mass_range_label}")

            for i_pt_had, (pt_had_min, pt_had_max) in enumerate(
                    zip(pt_bins_had[:-1], pt_bins_had[1:])):
                print(f"[INFO] PtHad: {pt_had_min} - {pt_had_max}")

                # ---- Build histogram path --------------------------------
                pc_dir = pt_cand_dir_name(pt_cand_min, pt_cand_max)
                ph_dir = pt_had_dir_name(pt_had_min, pt_had_max)

                if use_pairs_file:
                    hist_path = f"{pc_dir}/{ph_dir}/hPairsYields_vs_DeltaPhi"
                else:
                    im_dir = (f"InvMassBin_{int(mass_min * 1000):.0f}_"
                              f"{int(mass_max * 1000):.0f}")
                    hist_path = f"{pc_dir}/{ph_dir}/{im_dir}/hCorrectedCorrel"

                h_corr = in_file.Get(hist_path)
                if not h_corr and not use_pairs_file:
                    base_dir = f"{pc_dir}/{ph_dir}"
                    dir_obj = in_file.Get(base_dir)
                    if dir_obj and dir_obj.IsFolder():
                        for sk in dir_obj.GetListOfKeys():
                            sub_path = f"{base_dir}/{sk.GetName()}/hCorrectedCorrel"
                            h_corr = in_file.Get(sub_path)
                            if h_corr:
                                print(f"    Found: {sub_path}")
                                break
                if not h_corr:
                    print(f"    [WARNING] Histogram not found: {hist_path}, skip")
                    continue
                print(f"    Histogram: {hist_path}")

                # ---- LM template -----------------------------------------
                h_corr_lm = None
                if in_file_lm:
                    h_corr_lm = in_file_lm.Get(hist_path)
                    if not h_corr_lm:
                        alt_path = (f"{pc_dir}/{ph_dir}/hPairsYields_vs_DeltaPhi")
                        h_corr_lm = in_file_lm.Get(alt_path)
                        if h_corr_lm:
                            print(f"    LM template from: {alt_path}")
                        else:
                            print(f"    [WARNING] LM template not found: {alt_path}")
                    # if not use_pairs_file:
                    #     # MassBinning: use pre-merged LM template
                    #     key = f"{pc_dir}/{ph_dir}"
                    #     h_corr_lm = merged_lm.get(key)
                    #     if h_corr_lm:
                    #         print(f"    LM template: pre-merged from {key}")
                    #     else:
                    #         print(f"    [WARNING] Merged LM not found for {key}")
                    # else:
                    #     # DeltaPhiBinning: read LM template
                    #     h_corr_lm = in_file_lm.Get(hist_path)
                    #     if not h_corr_lm:
                    #         alt_path = (f"{pc_dir}/{ph_dir}/hPairsYields_vs_DeltaPhi")
                    #         h_corr_lm = in_file_lm.Get(alt_path)
                    #         if h_corr_lm:
                    #             print(f"    LM template from: {alt_path}")
                    #         else:
                    #             print(f"    [WARNING] LM template not found: {alt_path}")
                use_template = h_corr_lm is not None

                # ---- Read ry_trigger --------------------------------------
                ry_val = 1.0
                ry_err = 0.0
                h_ry = in_file.Get(f"{pc_dir}/{ph_dir}/ry_trigger")
                if h_ry:
                    n_ry_bins = h_ry.GetNbinsX()
                    if n_ry_bins >= mass_i_local + 1:
                        ry_val = h_ry.GetBinContent(mass_i_local + 1)
                        ry_err = h_ry.GetBinError(mass_i_local + 1)
                        print(f"    ry_trigger[mass{mass_i_local + 1}] = {ry_val:.0f} +/- {ry_err:.0f}")
                    else:
                        ry_val = h_ry.GetBinContent(1)
                        ry_err = h_ry.GetBinError(1)
                        print(f"    ry_trigger (single bin) = {ry_val:.0f} +/- {ry_err:.0f}")
                lm_ry_val = 1.0
                lm_ry_err = 0.0
                h_pairs_lm = 0.0
                if in_file_lm:
                    h_lm_ry = in_file_lm.Get(f"{pc_dir}/{ph_dir}/ry_trigger")
                    if h_lm_ry:
                        n_ry_bins = h_lm_ry.GetNbinsX()
                        if n_ry_bins >= mass_i_local + 1:
                            # Per-mass-bin ry_trigger
                            lm_ry_val = h_lm_ry.GetBinContent(mass_i_local + 1)
                            lm_ry_err = h_lm_ry.GetBinError(mass_i_local + 1)
                            lm_ry_total = h_lm_ry.Integral()
                            if lm_ry_total > 0:
                                h_pairs_lm = lm_ry_val / lm_ry_total
                            print(f"    LM ry_trigger[mass{mass_i_local + 1}] = {lm_ry_val:.0f} +/- {lm_ry_err:.0f}  "
                                  f"(total={lm_ry_total:.0f}, ratio={h_pairs_lm:.6f})")
                        else:
                            # Single-bin ry_trigger
                            lm_ry_val = h_lm_ry.GetBinContent(1)
                            lm_ry_err = h_lm_ry.GetBinError(1)
                            print(f"    LM ry_trigger (single bin) = {lm_ry_val:.0f} +/- {lm_ry_err:.0f}")

                # ---- Convert to TH1F for the fitter ---------------------
                h_fit = ROOT.TH1F(
                    f"hFit_{h_corr.GetName()}", h_corr.GetTitle(),
                    h_corr.GetNbinsX(),
                    h_corr.GetXaxis().GetXmin(),
                    h_corr.GetXaxis().GetXmax())
                for i_bin in range(1, h_corr.GetNbinsX() + 1):
                    h_fit.SetBinContent(i_bin, h_corr.GetBinContent(i_bin))
                    h_fit.SetBinError(i_bin, h_corr.GetBinError(i_bin))
                ROOT.SetOwnership(h_fit, False)
                set_th1_style(h_fit, "",
                                "#Delta#phi [rad]",
                                "#frac{dN^{assoc}}{d#Delta#phi} [rad^{-1}]")

                # ---- Create DhCorrelationFitter -------------------------
                corr_fitter = DhCorrelationFitter(h_fit, f_min, f_max)
                ROOT.SetOwnership(corr_fitter, True)

                corr_fitter.SetHistoIsReflected(False)

                if use_template:
                    corr_fitter.SetWithPedLM(with_ped_lm)
                    corr_fitter.SetFixLMFactor(fix_lm_factor)
                    corr_fitter.SetTempFunc(temp_func)
                    corr_fitter.SetFixBaseline(fix_baseline)
                    corr_fitter.SetBaselineUpOrDown(shift_base_up, shift_base_down)
                    if n_baseline_points > 0:
                        baseline_arr = (c_int * n_baseline_points)(*points_for_baseline)
                        corr_fitter.SetPointsForBaseline(n_baseline_points, baseline_arr)
                else:
                    corr_fitter.SetFixBaseline(0)
                    corr_fitter.SetBaselineUpOrDown(False, False)

                corr_fitter.SetReflectedCorrHisto(not is_reflected)
                corr_fitter.SetFixMean(fix_mean)
                corr_fitter.SetPtRanges(pt_cand_min, pt_cand_max,
                                        pt_had_min, pt_had_max)

                # Pass ry_trigger for raw-yield-scaled F initial value & limits
                corr_fitter.SetRyTrigger(ry_val, ry_err)
                if use_template:
                    corr_fitter.SetLMRyTrigger(lm_ry_val, lm_ry_err)
                if h_corr_lm:
                    # Clone the merged LM template so each mass bin gets its own copy
                    h_lm_clone = ROOT.TH1D(h_corr_lm)
                    h_lm_clone.SetDirectory(ROOT.nullptr)
                    ROOT.SetOwnership(h_lm_clone, False)
                    corr_fitter.SetLMTemplate(h_lm_clone)
                    if not use_pairs_file and h_pairs_lm > 0:
                        corr_fitter.SetLMPairs(h_pairs_lm)
                        print(f"      SetLMPairs = {h_pairs_lm:.6f}")

                if par_vals:
                    npars = len(par_vals)
                    vals = (c_double * npars)(*par_vals)
                    lows = (c_double * npars)(*par_low)
                    ups = (c_double * npars)(*par_up)
                    corr_fitter.SetExternalValsAndBounds(npars, vals, lows, ups)

                # ---- Fit -------------------------------------------------
                func_type = DhCorrelationFitter.FunctionType(fit_functions[i_pt_cand])
                corr_fitter.SetFuncType(func_type)

                mass_idx_label = mass_i_local + 1

                # create canvas FIRST then Fitting()
                # draws fit + split-term components onto it.
                canvas_name = (f"CanvasCorrPhi_PtBinCand{i_pt_cand + 1}_"
                               f"PtBinAssoc{i_pt_had + 1}_InvMassBin{mass_idx_label}")
                canvas_title = (f"CorrPhi{dmeson_name}_PtBinCand{i_pt_cand + 1}_"
                                f"PtBinAssoc{i_pt_had + 1}_InvMassBin{mass_idx_label}")
                canvas = ROOT.TCanvas(canvas_name, canvas_title, 1840, 1126)
                canvas.SetBottomMargin(0.08)
                canvas.SetLeftMargin(0.12)
                canvas.SetRightMargin(0.02)
                canvas.SetTopMargin(0.1)
                ROOT.SetOwnership(canvas, False)
                canvas.SetTickx()
                canvas.SetTicky()
                canvas.cd()

                # Fitting(drawSplitTerm=kTRUE, useExternalPars=kTRUE)
                corr_fitter.Fitting(True, True)
                # canvas.Update()
                h_fit.GetYaxis().SetRangeUser(h_fit.GetMinimum()*0.96, h_fit.GetMaximum() * 1.05)
                set_th1_style(h_fit, f"",
                                "#Delta#phi [rad]",
                                "#frac{dN^{assoc}}{d#Delta#phi} [rad^{-1}]")
                canvas.Update()

                # ---- Get v2_delta result ---------------------------------
                v2_val = corr_fitter.Getv2Delta()
                v2_err = corr_fitter.Getv2DeltaError()
                bin_idx = i_pt_cand + 1

                print(f"      fit: v2_delta={v2_val:.6f} +/- {v2_err:.6f}")

                # ---- Fill output histogram -------------------------------
                mass_idx = mass_i_local if not use_pairs_file else i_mass_global
                h_v2_delta[i_pt_had][mass_idx].SetBinContent(bin_idx, v2_val)
                h_v2_delta[i_pt_had][mass_idx].SetBinError(bin_idx, v2_err)

                # ---- Fill LM Factor histogram (par 0) ---------------------
                if use_template and mass_i_local == 0:
                    lm_val = corr_fitter.GetLMFactor()
                    lm_err = corr_fitter.GetLMFactorError()
                    print(f"      LM Factor = {lm_val:.6f} +/- {lm_err:.6f}")
                    h_lm_factor[i_pt_had].SetBinContent(bin_idx, lm_val)
                    h_lm_factor[i_pt_had].SetBinError(bin_idx, lm_err)

                # ============================================================
                # ---- Save global fit parameters and covariance ------------
                # ============================================================
                f_fit = corr_fitter.GetFitFunction()
                if f_fit:
                    npar = corr_fitter.GetNFitParams()

                    # ---- TH1D for global fit parameter values ----
                    h_par_g_name = (f"hParValues_PtCand{i_pt_cand + 1}_"
                                    f"PtHad{i_pt_had + 1}_"
                                    f"InvMassBin{mass_idx_label}")
                    h_par_g = ROOT.TH1D(h_par_g_name,
                                        f"Global Fit Parameters (type={fit_functions[i_pt_cand]}, "
                                        f"#chi^{{2}}/NDF={f_fit.GetChisquare():.1f}/{f_fit.GetNDF()});"
                                        f"Parameter;Value",
                                        npar, -0.5, npar - 0.5)
                    h_par_g.SetDirectory(ROOT.nullptr)
                    h_par_g.SetStats(0)
                    for ip in range(npar):
                        h_par_g.GetXaxis().SetBinLabel(ip + 1, f_fit.GetParName(ip))
                        h_par_g.SetBinContent(ip + 1, f_fit.GetParameter(ip))
                        h_par_g.SetBinError(ip + 1, f_fit.GetParError(ip))
                    par_global_histos[(i_pt_had, mass_idx_label, i_pt_cand)] = h_par_g

                    # Draw global par PNG
                    c_par_g = ROOT.TCanvas(f"cParValues_PtCand{i_pt_cand + 1}_PtHad{i_pt_had + 1}_InvMassBin{mass_idx_label}",
                                           "", 1200, 800)
                    c_par_g.SetLeftMargin(0.2); c_par_g.SetRightMargin(0.05); c_par_g.SetBottomMargin(0.25)
                    c_par_g.cd(); ROOT.SetOwnership(c_par_g, False)
                    h_par_g.GetXaxis().SetLabelSize(0.04)
                    h_par_g.Draw("E TEXT")
                    c_par_g.Update()
                    c_par_g.SaveAs(os.path.join(out_png_dir, f"{h_par_g_name}.png"))
                    c_par_g.Close(); del c_par_g
                    
                    # ---- save chi2/NDF value to a histogram ----
                    h_chi2_name = (f"hChi2NDF_PtCand{i_pt_cand + 1}_"
                                   f"PtHad{i_pt_had + 1}_"
                                   f"InvMassBin{mass_idx_label}")
                    h_chi2 = ROOT.TH1D(h_chi2_name,
                                       f"Chi2/NDF (type={fit_functions[i_pt_cand]});"
                                       f"Chi2/NDF;Value",
                                       1, 0.5, 1.5)
                    h_chi2.SetDirectory(ROOT.nullptr)
                    h_chi2.SetBinContent(1, f_fit.GetChisquare() / f_fit.GetNDF())
                    h_chi2.SetBinError(1, 0.0)
                    par_global_histos[(i_pt_had, mass_idx_label, i_pt_cand, "chi2")] = h_chi2

                    # ---- Global fit covariance TH2D ----
                    cov_g_name = (f"hCovMatrix_PtCand{i_pt_cand + 1}_"
                                  f"PtHad{i_pt_had + 1}_"
                                  f"InvMassBin{mass_idx_label}")
                    h_cov_g = ROOT.TH2D(cov_g_name,
                                        f"Global Fit Cov Matrix;Parameter;Parameter",
                                        npar, -0.5, npar - 0.5,
                                        npar, -0.5, npar - 0.5)
                    h_cov_g.SetDirectory(ROOT.nullptr)

                    for ip in range(npar):
                        h_cov_g.GetXaxis().SetBinLabel(ip + 1, f_fit.GetParName(ip))
                        h_cov_g.GetYaxis().SetBinLabel(ip + 1, f_fit.GetParName(ip))
                        for jp in range(npar):
                            cov_ij = corr_fitter.GetCovMatrixElement(ip, jp)
                            h_cov_g.SetBinContent(ip + 1, jp + 1, cov_ij)

                    cov_global_histos[(i_pt_had, mass_idx_label, i_pt_cand)] = h_cov_g

                    # Draw global cov PNG
                    c_cov_g = ROOT.TCanvas(f"cCovMatrix_PtCand{i_pt_cand + 1}_PtHad{i_pt_had + 1}_InvMassBin{mass_idx_label}",
                                           "", 1200, 1000)
                    c_cov_g.SetLeftMargin(0.2); c_cov_g.SetRightMargin(0.15); c_cov_g.SetBottomMargin(0.2)
                    c_cov_g.cd(); ROOT.SetOwnership(c_cov_g, False)
                    h_cov_g.SetStats(0)
                    h_cov_g.GetXaxis().SetLabelSize(0.035); h_cov_g.GetYaxis().SetLabelSize(0.035)
                    h_cov_g.Draw("COLZ TEXT"); h_cov_g.SetMarkerSize(0.8)
                    c_cov_g.Update()
                    c_cov_g.SaveAs(os.path.join(out_png_dir, f"{cov_g_name}.png"))
                    c_cov_g.Close(); del c_cov_g

                    print(f"      Saved global fit parameters TH1D: {h_par_g_name}")

                # ============================================================
                # ---- Save LM template fit parameters and covariance -------
                # ============================================================
                f_lm = corr_fitter.GetLMTemplateFunc()
                lm_func_clean = None
                if f_lm:
                    npar_lm = corr_fitter.GetLMTemplateNFitParams()

                    # ---- Clean copy of the LM fit function (for the output ROOT file) ----
                    # The TF1 that gets drawn on the cLMTemplate canvas is corrupted on
                    # read-back (TFormula substitutes parameter names into its expression
                    # when the function is painted -> parameter names/index mapping break).
                    # Keep a copy of the *undrawn* function here and write it as a top-level
                    # key into the hLMtemplate ROOT file, so downstream code can safely
                    # reload it.  Do NOT rebuild it from GetExpFormula(): that string is
                    # already name-substituted and loses the parameter mapping.
                    lm_func_clean = ROOT.TF1(f_lm)   # copy: formula + values + errors
                    lm_func_clean.SetName("fLMTemplateFunc")
                    lm_func_clean.SetRange(f_lm.GetXmin(), f_lm.GetXmax())

                    # ---- TH1D for LM template fit parameter values ----
                    h_par_lm_name = (f"hLMParValues_PtCand{i_pt_cand + 1}_"
                                     f"PtHad{i_pt_had + 1}_"
                                     f"InvMassBin{mass_idx_label}")
                    h_par_lm = ROOT.TH1D(h_par_lm_name,
                                         f"LM Template Fit Parameters (tempFunc={corr_fitter.GetTempFunc()});"
                                         f"Parameter;Value",
                                         npar_lm, -0.5, npar_lm - 0.5)
                    h_par_lm.SetDirectory(ROOT.nullptr)
                    h_par_lm.SetStats(0)
                    for ip in range(npar_lm):
                        h_par_lm.GetXaxis().SetBinLabel(ip + 1, f_lm.GetParName(ip))
                        h_par_lm.SetBinContent(ip + 1, f_lm.GetParameter(ip))
                        h_par_lm.SetBinError(ip + 1, f_lm.GetParError(ip))
                    par_lm_histos[(i_pt_had, mass_idx_label, i_pt_cand)] = h_par_lm

                    # Draw LM par PNG
                    c_par_lm = ROOT.TCanvas(f"cLMParValues_PtCand{i_pt_cand + 1}_PtHad{i_pt_had + 1}_InvMassBin{mass_idx_label}",
                                            "", 1200, 800)
                    c_par_lm.SetLeftMargin(0.2); c_par_lm.SetRightMargin(0.05); c_par_lm.SetBottomMargin(0.25)
                    c_par_lm.cd(); ROOT.SetOwnership(c_par_lm, False)
                    h_par_lm.GetXaxis().SetLabelSize(0.04)
                    h_par_lm.Draw("E TEXT")
                    c_par_lm.Update()
                    c_par_lm.SaveAs(os.path.join(out_png_dir, f"{h_par_lm_name}.png"))
                    c_par_lm.Close(); del c_par_lm

                    # ---- LM template fit covariance TH2D ----
                    cov_lm_name = (f"hLMCovMatrix_PtCand{i_pt_cand + 1}_"
                                   f"PtHad{i_pt_had + 1}_"
                                   f"InvMassBin{mass_idx_label}")
                    h_cov_lm = ROOT.TH2D(cov_lm_name,
                                         f"LM Template Cov Matrix;Parameter;Parameter",
                                         npar_lm, -0.5, npar_lm - 0.5,
                                         npar_lm, -0.5, npar_lm - 0.5)
                    h_cov_lm.SetDirectory(ROOT.nullptr)

                    for ip in range(npar_lm):
                        h_cov_lm.GetXaxis().SetBinLabel(ip + 1, f_lm.GetParName(ip))
                        h_cov_lm.GetYaxis().SetBinLabel(ip + 1, f_lm.GetParName(ip))
                        for jp in range(npar_lm):
                            cov_ij = corr_fitter.GetLMTemplateCovMatrixElement(ip, jp)
                            h_cov_lm.SetBinContent(ip + 1, jp + 1, cov_ij)

                    cov_lm_histos[(i_pt_had, mass_idx_label, i_pt_cand)] = h_cov_lm

                    # ---- LM template fit chi2/ndf (1-bin TH1D) ----
                    # -1 when the template was never fitted (external-params path, or the
                    # raw/spline shapes tempFunc 0/1/2) -> GetNDF() == 0
                    ndf_lm = f_lm.GetNDF()
                    chi2ndf_lm = f_lm.GetChisquare() / ndf_lm if ndf_lm > 0 else -1.0
                    h_chi2_lm_name = (f"hLMChi2NDF_PtCand{i_pt_cand + 1}_"
                                      f"PtHad{i_pt_had + 1}_"
                                      f"InvMassBin{mass_idx_label}")
                    h_chi2_lm = ROOT.TH1D(h_chi2_lm_name,
                                          f"LM Template Fit chi2/ndf (tempFunc={corr_fitter.GetTempFunc()}, "
                                          f"ndf={ndf_lm});chi2/ndf;",
                                          1, 0, 1)
                    h_chi2_lm.SetDirectory(ROOT.nullptr)
                    h_chi2_lm.SetStats(0)
                    h_chi2_lm.SetBinContent(1, chi2ndf_lm)
                    chi2lm_histos[(i_pt_had, mass_idx_label, i_pt_cand)] = h_chi2_lm
                    print(f"      Saved LM template fit chi2/ndf TH1D: {h_chi2_lm_name} "
                          f"= {chi2ndf_lm:.4f} (chi2={f_lm.GetChisquare():.4f}, ndf={ndf_lm})")

                    # Draw LM cov PNG
                    c_cov_lm = ROOT.TCanvas(f"cLMCovMatrix_PtCand{i_pt_cand + 1}_PtHad{i_pt_had + 1}_InvMassBin{mass_idx_label}",
                                            "", 1200, 1000)
                    c_cov_lm.SetLeftMargin(0.2); c_cov_lm.SetRightMargin(0.15); c_cov_lm.SetBottomMargin(0.2)
                    c_cov_lm.cd(); ROOT.SetOwnership(c_cov_lm, False)
                    h_cov_lm.SetStats(0)
                    h_cov_lm.GetXaxis().SetLabelSize(0.035); h_cov_lm.GetYaxis().SetLabelSize(0.035)
                    h_cov_lm.Draw("COLZ TEXT"); h_cov_lm.SetMarkerSize(0.8)
                    c_cov_lm.Update()
                    c_cov_lm.SaveAs(os.path.join(out_png_dir, f"{cov_lm_name}.png"))
                    c_cov_lm.Close(); del c_cov_lm

                    print(f"      Saved LM fit parameters TH1D: {h_par_lm_name}")

                # ---- Finish drawing the canvas ---------------------------
                set_th1_style(h_corr, "",
                              "#Delta#phi [rad]",
                              "#frac{dN^{assoc}}{d#Delta#phi} [rad^{-1}]")
                h_corr.SetStats(0)
                h_corr.SetMinimum(0)
                h_corr.Draw("same")

                pt_text = ROOT.TPaveText(0.15, 0.9, 0.85, 0.95, "NDC")
                pt_text.SetFillStyle(0)
                pt_text.SetBorderSize(0)
                pt_text.AddText(
                    0., 0.8,
                    f"{pt_cand_min:.1f} < p_{{T}}^{{{dmeson_label}}} < "
                    f"{pt_cand_max:.1f} GeV/c, "
                    f"{pt_had_min:.1f} < p_{{T}}^{{assoc}} < {pt_had_max:.1f} GeV/c, "
                    f"Mass[{mass_min:.3f}, {mass_max:.3f}] GeV/c^{{2}}")
                pt_text.SetTextAlign(22)
                pt_text.Draw("same")

                png_path = os.path.join(
                    out_png_dir,
                    f"CorrPhi{dmeson_name}_PtBinCand{i_pt_cand + 1}_"
                    f"PtBinAssoc{i_pt_had + 1}_InvMassBin{mass_idx_label}.png")
                root_canvas_path = os.path.join(
                    out_root_dir,
                    f"CorrPhi{dmeson_name}_PtBinCand{i_pt_cand + 1}_"
                    f"PtBinAssoc{i_pt_had + 1}_InvMassBin{mass_idx_label}.root")
                canvas.SaveAs(png_path)
                canvas.SaveAs(root_canvas_path)
                canvas.Close()

                # ---- Append global fit cov TH2D + par TH1D + chi2/NDF TH1D to canvas ROOT file ----
                cov_g_key = (i_pt_had, mass_idx_label, i_pt_cand)
                chi2_g_key = (i_pt_had, mass_idx_label, i_pt_cand, "chi2")
                if (cov_g_key in cov_global_histos or cov_g_key in par_global_histos
                        or chi2_g_key in par_global_histos):
                    f_cov_update = ROOT.TFile(root_canvas_path, "UPDATE")
                    f_cov_update.cd()
                    if cov_g_key in par_global_histos:
                        par_global_histos[cov_g_key].Write()
                    if chi2_g_key in par_global_histos:
                        par_global_histos[chi2_g_key].Write()
                    if cov_g_key in cov_global_histos:
                        cov_global_histos[cov_g_key].Write()
                    f_cov_update.Close()
                    del f_cov_update

                # ---- Draw hLM_template (LM data + template function + baseline) ----
                if use_template and mass_i_local == 0:
                    try:
                        lm_out = corr_fitter.GetLMOutput()
                        lm_func = corr_fitter.GetLMTemplateFunc()
                        if lm_out:
                            lm_out.SetDirectory(ROOT.nullptr)
                            bl_val = corr_fitter.GetPedestal()
                            bl_line = ROOT.TF1("fBaseLine", "[0]", 0, 2*math.pi)
                            bl_line.SetParameter(0, bl_val)
                            bl_line.SetLineColor(ROOT.kGray + 2)
                            bl_line.SetLineStyle(3)
                            bl_line.SetLineWidth(3)

                            c_lm = ROOT.TCanvas(
                                f"cLMTemplate_PtBinCand{i_pt_cand + 1}_"
                                f"PtBinAssoc{i_pt_had + 1}",
                                "LM Template Comparison", 1600, 1200)
                            c_lm.SetLeftMargin(0.15)
                            c_lm.SetRightMargin(0.05)
                            c_lm.SetBottomMargin(0.12)
                            c_lm.SetTopMargin(0.05)
                            c_lm.cd()
                            lm_out.SetTitle("")
                            lm_out.GetXaxis().SetTitle("#Delta#phi [rad]")
                            lm_out.GetYaxis().SetTitle("#frac{dN^{assoc}}{d#Delta#phi} [rad^{-1}]")
                            lm_out.SetStats(0)
                            lm_out.Draw("E")
                            if lm_func:
                                lm_func.SetLineColor(ROOT.kBlue)
                                lm_func.SetLineWidth(3)
                                lm_func.SetLineStyle(2)
                                lm_func.DrawClone("Same")
                            bl_line.Draw("Same")

                            leg_lm = ROOT.TLegend(0.65, 0.70, 0.92, 0.88)
                            leg_lm.SetFillStyle(0)
                            leg_lm.SetBorderSize(0)
                            leg_lm.SetTextSize(0.035)
                            leg_lm.AddEntry(lm_out, "LM Data (raw)", "lep")
                            if lm_func:
                                leg_lm.AddEntry(lm_func, "Template Function", "l")
                            # Draw sub-component functions from fLMOutput's function list
                            for i_f in range(lm_out.GetListOfFunctions().GetSize()):
                                f_obj = lm_out.GetListOfFunctions().At(i_f)
                                if f_obj:
                                    f_name = str(f_obj.GetName())
                                    if "Near" in f_name:
                                        f_obj.DrawClone("Same")
                                        leg_lm.AddEntry(f_obj, "Near-Side", "l")
                                    elif "Away" in f_name:
                                        f_obj.DrawClone("Same")
                                        leg_lm.AddEntry(f_obj, "Away-Side", "l")
                            leg_lm.AddEntry(bl_line, "Baseline", "l")
                            leg_lm.Draw()

                            pt_text_lm = ROOT.TPaveText(0.20, 0.90, 0.85, 0.96, "NDC")
                            pt_text_lm.SetFillStyle(0)
                            pt_text_lm.SetBorderSize(0)
                            pt_text_lm.AddText(
                                f"LM Template: {pt_cand_min:.1f} < #it{{p}}_{{T}}^{{{dmeson_label}}} < {pt_cand_max:.1f} GeV/c, "
                                f"{pt_had_min:.1f} < #it{{p}}_{{T}}^{{assoc}} < {pt_had_max:.1f} GeV/c, "
                                f"  tempFunc={corr_fitter.GetTempFunc()}")
                            pt_text_lm.SetTextAlign(22)
                            pt_text_lm.Draw("Same")
                            c_lm.Update()

                            lm_png_name = (f"hLMtemplate_PtBinCand{i_pt_cand + 1}_"
                                          f"PtBinAssoc{i_pt_had + 1}")
                            c_lm.SaveAs(os.path.join(out_png_dir, f"{lm_png_name}.png"))
                            lm_root_path = os.path.join(
                                out_root_dir, f"{lm_png_name}.root")
                            c_lm.SaveAs(lm_root_path)

                            # ---- Append LM fit cov TH2D + par TH1D + chi2/ndf to hLMtemplate ROOT file ----
                            cov_lm_key = (i_pt_had, mass_idx_label, i_pt_cand)
                            if (cov_lm_key in cov_lm_histos or cov_lm_key in par_lm_histos
                                    or cov_lm_key in chi2lm_histos):
                                f_lm_update = ROOT.TFile(lm_root_path, "UPDATE")
                                f_lm_update.cd()
                                if cov_lm_key in par_lm_histos:
                                    par_lm_histos[cov_lm_key].Write()
                                if cov_lm_key in cov_lm_histos:
                                    cov_lm_histos[cov_lm_key].Write()
                                if cov_lm_key in chi2lm_histos:
                                    chi2lm_histos[cov_lm_key].Write()
                                f_lm_update.Close()
                                del f_lm_update

                            # ---- Append the clean LM fit function (top-level TF1 key) ----
                            # Reloadable via TFile::Get("fLMTemplateFunc"): parameters keep
                            # their names/values/errors (unlike the one stored in the canvas).
                            if lm_func_clean is not None:
                                f_lm_func_update = ROOT.TFile(lm_root_path, "UPDATE")
                                f_lm_func_update.cd()
                                lm_func_clean.Write()
                                f_lm_func_update.Close()
                                del f_lm_func_update
                                print(f"      Saved clean LM fit function TF1: "
                                      f"{lm_func_clean.GetName()} (npar={lm_func_clean.GetNpar()})")

                            c_lm.Close()
                            ROOT.SetOwnership(c_lm, False)
                    except Exception as e:
                        print(f"[WARNING] hLM_template drawing failed: {e}")

                del corr_fitter, h_fit, canvas

    # ---- Close input files ---------------------------------------------
    in_file.Close()
    if in_file_lm:
        in_file_lm.Close()

    # ============== SAVE FINAL PLOTS ==============
    final_plot_path = os.path.join(out_root_dir,
                                   f"CorrPhi{dmeson_name}_FinalPlots.root")
    print(f"\n[INFO] Saving final plots to: {final_plot_path}")

    out_file = ROOT.TFile(final_plot_path, "RECREATE")
    out_file.cd()

    n_mass_out = n_mass_local
    for i_mass_out in range(n_mass_out):
        for i_pt_had in range(n_pt_had):
            h = h_v2_delta[i_pt_had][i_mass_out]
            h.SetDirectory(out_file)
            out_file.cd()
            h.Write()
    # Write LM Factor histograms (flat — one per ptHad, no mass binning)
    for i_pt_had in range(n_pt_had):
        h_lm = h_lm_factor[i_pt_had]
        h_lm.SetDirectory(out_file)
        out_file.cd()
        h_lm.Write()

    # Note: LM covariance TH2D histograms are saved in canvas ROOT files
    # (e.g., CorrPhiD0_PtBinCand1_PtBinAssoc1_InvMassBin1.root)
    # — no longer stored in FinalPlots.root to avoid breaking downstream scripts.

    # todo: to be removed
    for i_mass_out in range(n_mass_out):
        for i_pt_had in range(n_pt_had):
            src = h_v2_delta[i_pt_had][i_mass_out]
            for name_template in [
                "hBaselin_PtBinAssoc{}_InvMassBin{}",
                "hNSYield_PtBinAssoc{}_InvMassBin{}",
                "hNSSigma_PtBinAssoc{}_InvMassBin{}",
                "hASYield_PtBinAssoc{}_InvMassBin{}",
                "hASSigma_PtBinAssoc{}_InvMassBin{}",
                "hBeta_PtBinAssoc{}_InvMassBin{}",
                "hNSYieldBinCount_PtBinAssoc{}_InvMassBin{}",
                "hASYieldBinCount_PtBinAssoc{}_InvMassBin{}",
            ]:
                h_name = name_template.format(i_pt_had + 1, i_mass_out + 1)
                h_pl = ROOT.TH1D(h_name, "", n_pt_cand, pt_cand_edges_arr)
                for i_bin in range(1, n_pt_cand + 1):
                    h_pl.SetBinContent(i_bin, src.GetBinContent(i_bin))
                    h_pl.SetBinError(i_bin, src.GetBinError(i_bin))
                h_pl.SetDirectory(out_file)
                out_file.cd()
                h_pl.Write()
                del h_pl

    out_file.Close()
    print("[INFO] Done!")
    print(f"[INFO] Final plots saved to: {final_plot_path}")


# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit azimuthal correlations (rewrite of FitCorrel.C)")
    parser.add_argument("config",
                        nargs="?",
                        default="config_CorrAnalysis_v2_010_negDeta.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    fit_correl(args.config)
