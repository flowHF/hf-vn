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
  {outdir}/maps_fits/CorrPhi{DMeson}.root
  {outdir}/maps_fits/CorrPhi{DMeson}_PtCand_XX_YY.pdf

Usage:
  python3 fit_correl.py config_CorrAnalysis_v2_010_negDeta.yaml
"""

import argparse
import math
import os
import sys
from array import array
from ctypes import c_int, c_double
import pathlib as PATH

import ROOT
from ROOT import TFile, TH1D, TH1F, TPaveText, TCanvas, TLegend, TF1, gROOT, gSystem, gStyle
import yaml

gROOT.SetBatch(True)
gErrorIgnoreLevel = ROOT.kWarning

# ===================================================================
# Compile DhCorrelationFitter C++ class
# ===================================================================
gSystem.AddIncludePath("-I/home/mdicosta/local/include")
gSystem.AddDynamicPath("/home/mdicosta/local/lib")
gSystem.Load("libyaml-cpp")
# gSystem.AddIncludePath("-I/home/wuct/Software/miniforge3/envs/alice/include")
_fitter_cxx = os.path.join(os.path.dirname(__file__), "DhCorrelationFitter.cxx")
gSystem.CompileMacro(_fitter_cxx, "kO")
from ROOT import DhCorrelationFitter

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '../../', 'utils'))
from utils import logger, make_dir_root_file

# ===================================================================
# ROOT style helpers
# ===================================================================
def set_canvas_style():
    style = gStyle
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


# Directory name helpers
def pt_dir_name(base, pt_min, pt_max):
    low = int(round(pt_min * 10))
    high = int(round(pt_max * 10))
    return f"{base}_{low}_{high}"


def get_mass_distribution(mass_vs_pt, pt_min, pt_max):
    """Project the 2D MassVsPt histogram for a given pt range."""
    yaxis = mass_vs_pt.GetYaxis()
    bin_min = yaxis.FindBin(pt_min * 1.0001)
    bin_max = yaxis.FindBin(pt_max * 0.9999)
    yaxis.SetRange(bin_min, bin_max)
    proj = mass_vs_pt.ProjectionX(f"mass_proj_{pt_min:.1f}_{pt_max:.1f}")
    proj.SetDirectory(0)
    return proj


def build_mass_v2(config):
    """Build v2-vs-mass histograms from InvMassVsPt.root + CorrPhiD0.root."""

    outdir = PATH.Path(config["outdir"])
    v2_h = float(config.get("v2DeltaHH", 0.07))

    ry_trig_path = outdir / "trigger_yields" / "InvMassVsPt.root"
    correl_path = outdir / "maps_fits" / f"CorrPhi{config['Dmeson']}.root"
    if not ry_trig_path.exists():
        logger(f"{ry_trig_path} not found", level="FATAL")
        sys.exit(1)
    if not correl_path.exists():
        logger(f"{correl_path} not found", level="FATAL")
        sys.exit(1)

    pt_bins_cand = [float(ptBinCand) for ptBinCand in config["ptBinsCand"]]
    pt_bins_had  = [float(ptBinHad) for ptBinHad in config["ptBinsHad"]]

    print(f"ry_trig_path: {ry_trig_path}")
    file_mass = TFile.Open(str(ry_trig_path))
    h_mass_vs_pt = file_mass.Get("hMassVsPt")

    out_mass_v2_dir = outdir / "vn_vs_mass"
    os.makedirs(str(out_mass_v2_dir), exist_ok=True)

    print(f"correl_path: {correl_path}")
    file_v2 = TFile.Open(str(correl_path), "READ")
    for pt_had_min, pt_had_max in zip(pt_bins_had[:-1], pt_bins_had[1:]):
        pt_had_str = f"PtAssoc{int(pt_had_min*10):02d}to{int(pt_had_max*10):02d}"
        h_masses, h_v2_vs_masses = [], []
        for pt_cand_min, pt_cand_max in zip(pt_bins_cand[:-1], pt_bins_cand[1:]):
            pt_cand_str = f"PtCand_{int(pt_cand_min*10):02d}_{int(pt_cand_max*10):02d}"

            tmp = h_mass_vs_pt.Clone(f"hMass_{pt_cand_str}_{pt_had_str}")
            h_masses.append(get_mass_distribution(tmp, pt_cand_min, pt_cand_max))
            print(f"Retrieved: {pt_cand_str}/hV2DeltaVsMass")
            h_v2_vs_mass = file_v2.Get(f"{pt_cand_str}/hV2DeltaVsMass")
            h_v2_vs_mass.SetDirectory(0)
            h_v2_vs_mass.Scale(1.0 / v2_h)
            h_v2_vs_masses.append(h_v2_vs_mass)

        out_path = out_mass_v2_dir / f"InvMassVsV2_{pt_had_str}.root"
        out_file = TFile.Open(str(out_path), "RECREATE")
        for i_pt in range(len(pt_bins_cand)-1):
            pd = f"pt_{int(pt_bins_cand[i_pt]*10):.0f}_{int(pt_bins_cand[i_pt+1]*10):.0f}"
            make_dir_root_file(pd, out_file)
            out_file.cd(pd)
            h_masses[i_pt].Write("hMassData")
            h_v2_vs_masses[i_pt].Write("hVnVsMassData")
        out_file.Close()

    file_mass.Close()
    file_v2.Close()
    logger(f"[INFO] All MassVsV2 files in {out_mass_v2_dir}", level="INFO")


# ===================================================================
# Main entry point
# ===================================================================
def fit_correl(cfg_path):
    set_canvas_style()

    # ---- Load config --------------------------------------------------
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # ---- D meson species ----------------------------------------------
    Dmeson = config.get("Dmeson", "Dzero")
    species_map = {
        "D0": (0, "D0", "D^{0}"),
        "Dzero": (0, "D0", "D^{0}"),
        "Dplus": (1, "Dplus", "D^{+}"),
        "Ds": (2, "Ds", "D_{s}^{+}"),
    }
    if Dmeson not in species_map:
        logger(f"Unknown D meson: {Dmeson}", level="FATAL")
        sys.exit(1)
    _, dmeson_name, dmeson_label = species_map[Dmeson]

    # ---- Paths --------------------------------------------------------
    suffix = config["suffix"]
    outdir = config["outdir"]
    pair_yields_path = os.path.join(outdir, "maps", "AssociatedPairsYields",
                                    "PairYieldsVsPhi.root")
    correlations_path = os.path.join(outdir, "maps", "CorrelationsResults.root")

    # ---- Open input file -----------------------------------------------
    use_pairs_file = os.path.exists(pair_yields_path)
    input_file_path = pair_yields_path if use_pairs_file else correlations_path
    if not os.path.exists(input_file_path):
        logger(f"Input file not found: {input_file_path}", level="FATAL")
        sys.exit(1)
    in_file = TFile.Open(input_file_path)

    out_dir = os.path.join(outdir, "maps_fits")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Binning ------------------------------------------------------
    pt_bins_cand = [float(x) for x in config["ptBinsCand"]]
    pt_bins_had = [float(x) for x in config["ptBinsHad"]]
    n_pt_cand = len(pt_bins_cand) - 1
    n_pt_had = len(pt_bins_had) - 1
    vn_vs_mass_bins = config["invMassBins"]
    n_vn_vs_mass_bins = 1 if use_pairs_file else max(len(mass_bin_pt) - 1 for mass_bin_pt in vn_vs_mass_bins)

    # ---- LM template --------------------------------------------------
    method = config.get("method", "DeltaPhiBinning")
    task_lm = config.get("task_LM", {})
    do_lm = task_lm.get("do", False)
    lm_template_path, in_file_lm = None, None
    if do_lm:
        out_file_path_lm = f"{outdir}/maps_fits/LMTemplates{config['Dmeson']}.root"
        out_file_lm = TFile.Open(out_file_path_lm, "RECREATE")
        if method == "MassBinning":
            lm_template_path = os.path.join(
                config['outdir'], "low_mult", "maps", "CorrelationsResults.root",
            )
        else:
            lm_template_path = os.path.join(
                config['outdir'], "low_mult", f"CorrelExtract_{suffix}",
                 "AssociatedPairsYields", "PairYieldsVsPhi.root",
            )
        if os.path.exists(lm_template_path):
            logger(f"Taking LM template from {lm_template_path}", level="INFO")
            in_file_lm = TFile.Open(lm_template_path)
            if not in_file_lm or in_file_lm.IsZombie():
                logger(f"Could not open LM template: {lm_template_path}", level="FATAL")
                in_file_lm = None
        else:
            logger(f"LM template not found: {lm_template_path}", level="FATAL")
            lm_template_path = None

    # ---- Fit config ---------------------------------------------------
    fit_config = config.get("fitConfig", {})
    if isinstance(fit_config["FitFunction"], list):
        fit_functions = [int(f) for f in fit_config["FitFunction"]]
    else:
        fit_functions = [int(fit_config["FitFunction"]) for _ in range(n_pt_cand)]

    # If no LM template, force all to type 8
    if lm_template_path is None:
        logger(f"lm_template_path is None, forcing all fit functions to type 8", level="WARNING")
        for i_func in range(len(fit_functions)):
            fit_functions[i_func] = DhCorrelationFitter.kV2DeltaModulationLowMult
        logger("No LM template — all fit functions forced to type 8", level="WARNING")

    par_vals = [float(x) for x in fit_config.get("parVals", [])]
    par_low  = [float(x) for x in fit_config.get("parLowBounds", [])]
    par_up   = [float(x) for x in fit_config.get("parUpperBounds", [])]
    if par_vals and not (len(par_vals) == len(par_low) == len(par_up)):
        logger(f"parVals, parLowBounds, parUpperBounds must have same length", level="FATAL")
        sys.exit(1)

    # ---- Prepare output histograms (indexed by [i_pt_had][i_mass_local])
    pt_cand_edges_arr = array("d", pt_bins_cand)

    h_v2_delta = [
        TH1D(f"hV2DeltaVsMassPtBinCand_{int(pt_bins_cand[i_pt_cand]*10)}to{int(pt_bins_cand[i_pt_cand + 1]*10)}", "", len(vn_vs_mass_bins[i_pt_cand]) - 1, 
             array("d", vn_vs_mass_bins[i_pt_cand])) for i_pt_cand in range(n_pt_cand)
    ]
    for h in h_v2_delta:
        h.SetDirectory(ROOT.nullptr)

    h_lm_factor = [None] * n_pt_had
    for i_pt_had in range(n_pt_had):
        # LM Factor: single 1D histogram per ptHad (no mass binning)
        h_lm = TH1D(f"hLMFactor_PtBinAssoc{i_pt_had + 1}", "", n_pt_cand, pt_cand_edges_arr)
        h_lm.SetDirectory(ROOT.nullptr)
        h_lm_factor[i_pt_had] = h_lm

    mass_bins_cfgs = []
    if use_pairs_file:
        mass_bins_cfgs.append({
            "cand_indices": range(n_pt_cand),
            "mass_bin": 0,
            "mass_min": vn_vs_mass_bins[0][0],
            "mass_max": vn_vs_mass_bins[0][-1],
        })
    else:
        for i_pt in range(n_pt_cand):
            sub_edges = vn_vs_mass_bins[i_pt]

            for i_mass in range(len(sub_edges) - 1):
                mass_bins_cfgs.append({
                    "cand_indices": [i_pt],
                    "mass_bin": i_mass,
                    "mass_min": sub_edges[i_mass],
                    "mass_max": sub_edges[i_mass + 1],
                })
    n_mass_total = len(mass_bins_cfgs)
    logger(f"MassBinning: {n_mass_total} mass sub-bins "
           f"across {n_pt_cand} pt cand bins", level="INFO")

    # ============== MAIN FIT LOOP ==============
    # For use_pairs_file (DeltaPhiBinning): iterate over ptCand internally
    # For !use_pairs_file (MassBinning): each mass_bins_cfgs entry carries
    #   its own (i_pt_cand, mass_min, mass_max).
    final_plot_path = os.path.join(out_dir, f"CorrPhi{dmeson_name}.root")
    corr_fits_file = TFile.Open(final_plot_path, "RECREATE")
    for i_mass_bin, mass_bin_cfg in enumerate(mass_bins_cfgs):
        mass_min = mass_bin_cfg["mass_min"]
        mass_max = mass_bin_cfg["mass_max"]
        i_vn_vs_mass_bin = mass_bin_cfg["mass_bin"]
        logger(f"Processing mass combo {i_mass_bin + 1}/{n_mass_total}: {mass_bin_cfg}", level="INFO")

        for i_pt_cand in mass_bin_cfg["cand_indices"]:

            # Determine which ptCand bins to process in this mass combo
            mass_range_label = f"[{mass_min}, {mass_max}]" if use_pairs_file else \
                               f"[{mass_min}, {mass_max}] PtCand[{pt_bins_cand[i_pt_cand]}, {pt_bins_cand[i_pt_cand+1]}]"
            print(f"\n=== Mass bin {i_mass_bin + 1}/{n_mass_total} (Mass range: {mass_range_label}) - Processing ptCand bin {i_pt_cand + 1}/{n_pt_cand} ===")
            pt_cand_min = pt_bins_cand[i_pt_cand]
            pt_cand_max = pt_bins_cand[i_pt_cand + 1]
            logger(f"Mass[{i_vn_vs_mass_bin + 1}/{n_vn_vs_mass_bins}] "
                   f"PtCand[{pt_cand_min}, {pt_cand_max}]  {mass_range_label}", level="INFO")

            for i_pt_had, (pt_had_min, pt_had_max) in enumerate(zip(pt_bins_had[:-1], pt_bins_had[1:])):

                # ---- Build histogram path --------------------------------
                pc_dir = pt_dir_name("PtCand", pt_cand_min, pt_cand_max)
                ph_dir = pt_dir_name("PtHad", pt_had_min, pt_had_max)

                # ---- Retrieve correlation histogram ----------------------
                h_corr, correl_hist_path = None, None
                if use_pairs_file:
                    correl_hist_path = f"{pc_dir}/{ph_dir}/hPairsYields_vs_DeltaPhi"
                else:
                    im_dir = (f"InvMassBin_{int(mass_min * 1000):.0f}_{int(mass_max * 1000):.0f}")
                    correl_hist_path = f"{pc_dir}/{ph_dir}/{im_dir}/hCorrectedCorrel"
                h_corr = in_file.Get(correl_hist_path)
                if h_corr is None:
                    logger("Histogram not found!", level="ERROR")
                    continue
                logger(f"Got {correl_hist_path} histogram", level="INFO")

                # ---- Create DhCorrelationFitter -------------------------
                h_fit = TH1F(f"hFit_{h_corr.GetName()}", h_corr.GetTitle(),
                             h_corr.GetNbinsX(), h_corr.GetXaxis().GetXmin(), h_corr.GetXaxis().GetXmax())
                for i_bin in range(1, h_corr.GetNbinsX() + 1):
                    h_fit.SetBinContent(i_bin, h_corr.GetBinContent(i_bin))
                    h_fit.SetBinError(i_bin, h_corr.GetBinError(i_bin))
                ROOT.SetOwnership(h_fit, False)
                set_th1_style(h_fit, "", "#Delta#phi [rad]", "#frac{dN^{assoc}}{d#Delta#phi} [rad^{-1}]")
                # Last two arguments are the fit range in DeltaPhi
                corr_fitter = DhCorrelationFitter(h_fit, -0.5 * math.pi, 1.5 * math.pi)
                corr_fitter.SetVerbosity(0)  # 0=silent, 1=normal
                ROOT.SetOwnership(corr_fitter, True)
                corr_fitter.SetFixMean(int(fit_config.get("FixMean", 0)))
                corr_fitter.SetPtRanges(pt_cand_min, pt_cand_max, pt_had_min, pt_had_max)

                # ---- Set Raw Yield trigger --------------------------------------
                ry_val = 1.0
                ry_err = 0.0
                h_ry = in_file.Get(f"{pc_dir}/{ph_dir}/ry_trigger")
                if h_ry:
                    n_ry_bins = h_ry.GetNbinsX()
                    if n_ry_bins >= i_vn_vs_mass_bin + 1:
                        ry_val = h_ry.GetBinContent(i_vn_vs_mass_bin + 1)
                        ry_err = h_ry.GetBinError(i_vn_vs_mass_bin + 1)
                        logger(f"    ry_trigger[mass{i_vn_vs_mass_bin + 1}] = {ry_val:.0f} +/- {ry_err:.0f}", level="INFO")
                    else:
                        ry_val = h_ry.GetBinContent(1)
                        ry_err = h_ry.GetBinError(1)
                        logger(f"    ry_trigger (single bin) = {ry_val:.0f} +/- {ry_err:.0f}", level="INFO")

                # Pass ry_trigger for raw-yield-scaled F initial value & limits
                corr_fitter.SetRyTrigger(ry_val, ry_err)
                corr_fitter.SetFixBaseline(0)
                corr_fitter.SetBaselineUpOrDown(False, False)

                # ---- LM template -----------------------------------------
                if do_lm:
                    h_corr_lm = None
                    lm_ry_val = 1.0
                    lm_ry_err = 0.0
                    h_pairs_lm = 0.0

                    # Get LM template histogram and trigger normalization
                    h_corr_lm = in_file_lm.Get(correl_hist_path)
                    if not h_corr_lm:
                        logger(f"LM template histogram not found at {correl_hist_path} in {lm_template_path}", level="FATAL")
                        h_corr_lm = None
                    h_lm_ry = in_file_lm.Get(f"{pc_dir}/{ph_dir}/ry_trigger")
                    if not h_lm_ry:
                        logger(f"LM template ry_trigger histogram not found at {pc_dir}/{ph_dir}/ry_trigger in {lm_template_path}", level="WARNING")
                        h_lm_ry = None

                    n_ry_bins = h_lm_ry.GetNbinsX()
                    if n_ry_bins >= i_vn_vs_mass_bin + 1:
                        # Per-mass-bin ry_trigger
                        lm_ry_val = h_lm_ry.GetBinContent(i_vn_vs_mass_bin + 1)
                        lm_ry_err = h_lm_ry.GetBinError(i_vn_vs_mass_bin + 1)
                        lm_ry_total = h_lm_ry.Integral()
                        if lm_ry_total > 0:
                            h_pairs_lm = lm_ry_val / lm_ry_total
                        logger(f"    LM ry_trigger[mass{i_vn_vs_mass_bin + 1}] = {lm_ry_val:.0f} +/- {lm_ry_err:.0f}  "
                            f"(total={lm_ry_total:.0f}, ratio={h_pairs_lm:.6f})", level="INFO")
                    else:
                        # Single-bin ry_trigger
                        lm_ry_val = h_lm_ry.GetBinContent(1)
                        lm_ry_err = h_lm_ry.GetBinError(1)
                        logger(f"    LM ry_trigger (single bin) = {lm_ry_val:.0f} +/- {lm_ry_err:.0f}", level="INFO")

                    # Anchor the LM template
                    corr_fitter.SetWithPedLM(fit_config.get("WithPedLM", False))
                    corr_fitter.SetFixLMFactor(fit_config.get("FixLMFactor", False))
                    corr_fitter.SetTempFunc(int(fit_config.get("tempFunc", 0)))

                    corr_fitter.SetFixBaseline(int(fit_config.get("FixBaseline", 0)))
                    corr_fitter.SetBaselineUpOrDown(config.get("ShiftBaseUp", False), config.get("ShiftBaseDown", False))
                    n_baseline_points = int(fit_config.get("nBaselinePoints", 0))
                    points_for_baseline = [int(x) for x in fit_config.get("binsForBaseline", [])]
                    if n_baseline_points > 0:
                        if len(points_for_baseline) != n_baseline_points:
                            logger(f"'binsForBaseline' length must match 'nBaselinePoints'", level="FATAL")
                            sys.exit(1)
                        baseline_arr = (c_int * n_baseline_points)(*points_for_baseline)
                        corr_fitter.SetPointsForBaseline(n_baseline_points, baseline_arr)

                    corr_fitter.SetLMRyTrigger(lm_ry_val, lm_ry_err)
                    # Clone the merged LM template so each mass bin gets its own copy
                    h_lm_clone = TH1D(h_corr_lm)
                    h_lm_clone.SetDirectory(ROOT.nullptr)
                    ROOT.SetOwnership(h_lm_clone, False)
                    corr_fitter.SetLMTemplate(h_lm_clone)
                    if not use_pairs_file and h_pairs_lm > 0:
                        corr_fitter.SetLMPairs(h_pairs_lm)
                        logger(f"      SetLMPairs = {h_pairs_lm:.6f}", level="INFO")

                # ---- Reflections -----------------------------------------
                corr_fitter.SetHistoIsReflected(False)
                corr_fitter.SetReflectedCorrHisto(config.get("IsReflected", False))

                if par_vals:
                    npars = len(par_vals)
                    vals = (c_double * npars)(*par_vals)
                    lows = (c_double * npars)(*par_low)
                    ups = (c_double * npars)(*par_up)
                    corr_fitter.SetExternalValsAndBounds(npars, vals, lows, ups)

                # ---- Fit -------------------------------------------------
                func_type = DhCorrelationFitter.FunctionType(fit_functions[i_pt_cand])
                corr_fitter.SetFuncType(func_type)

                # create canvas FIRST then Fitting() draws fit + split-term components onto it.
                canvas_name = (f"CanvasCorrPhi_{pc_dir}_PtBinAssoc{i_pt_had + 1}_MassBin_{mass_min}_{mass_max}")
                canvas_title = (f"CorrPhi{dmeson_name}_{pc_dir}_PtBinAssoc{i_pt_had + 1}_MassBin_{mass_min}_{mass_max}")
                cDeltaPhiFits = TCanvas(canvas_name, canvas_title, 1840, 1126)
                cDeltaPhiFits.SetBottomMargin(0.08)
                cDeltaPhiFits.SetLeftMargin(0.12)
                cDeltaPhiFits.SetRightMargin(0.02)
                cDeltaPhiFits.SetTopMargin(0.1)
                ROOT.SetOwnership(cDeltaPhiFits, False)
                cDeltaPhiFits.SetTickx()
                cDeltaPhiFits.SetTicky()
                cDeltaPhiFits.cd()

                # ---- Print config summary -----------------------------------------
                logger("===========================")
                logger("Input variables from config")
                for i, func_val in enumerate(fit_functions):
                    logger(f"  iPt = {i + 1}  FitFunction = {func_val}")
                logger(f"  FixBaseline = {int(fit_config.get("FixBaseline", 0))}")
                logger(f"  FixMean     = {int(fit_config.get("FixMean", 0))}")
                logger("===========================\n")

                # Fitting(drawSplitTerm=kTRUE, useExternalPars=kTRUE)
                corr_fitter.Fitting(True, True)

                # cDeltaPhiFits.Update()
                h_fit.GetYaxis().SetRangeUser(h_fit.GetMinimum()*0.96, h_fit.GetMaximum() * 1.05)
                set_th1_style(h_fit, f"", "#Delta#phi [rad]",
                              "#frac{dN^{assoc}}{d#Delta#phi} [rad^{-1}]")
                cDeltaPhiFits.Update()

                # ---- Get v2_delta result ---------------------------------
                v2_val = corr_fitter.Getv2Delta()
                v2_err = corr_fitter.Getv2DeltaError()
                bin_idx = i_pt_cand + 1

                logger(f"      fit: v2_delta={v2_val:.6f} +/- {v2_err:.6f}", level="INFO")

                # ---- Fill output histogram -------------------------------
                h_v2_delta[i_pt_cand].SetBinContent(i_vn_vs_mass_bin + 1, v2_val)
                h_v2_delta[i_pt_cand].SetBinError(i_vn_vs_mass_bin + 1, v2_err)

                # ---- Fill LM Factor histogram (par 0) ---------------------
                if h_corr_lm is not None and i_vn_vs_mass_bin == 0:
                    lm_val = corr_fitter.GetLMFactor()
                    lm_err = corr_fitter.GetLMFactorError()
                    logger(f"      LM Factor = {lm_val:.6f} +/- {lm_err:.6f}", level="INFO")
                    h_lm_factor[i_pt_had].SetBinContent(bin_idx, lm_val)
                    h_lm_factor[i_pt_had].SetBinError(bin_idx, lm_err)

                # ---- Finish drawing the canvas ---------------------------
                set_th1_style(h_corr, "", "#Delta#phi [rad]", "#frac{dN^{assoc}}{d#Delta#phi} [rad^{-1}]")
                h_corr.SetStats(0)
                h_corr.SetMinimum(0)
                h_corr.Draw("same")

                pt_text = TPaveText(0.15, 0.9, 0.85, 0.95, "NDC")
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

                if use_pairs_file:
                    cDeltaPhiFits.SaveAs(f"CorrPhi{dmeson_name}_AllPtBins_MassBin1.pdf")
                else:
                    canva_path = os.path.join(out_dir, f"CorrPhi{dmeson_name}_{pc_dir}")
                    if i_vn_vs_mass_bin == 0:
                        suffix_pdf = '('
                    elif i_vn_vs_mass_bin == n_vn_vs_mass_bins - 1:
                        suffix_pdf = ')'
                    else:
                        suffix_pdf = ''
                    if n_pt_cand == 1:
                        cDeltaPhiFits.SaveAs(f'{canva_path}.pdf')
                    else:
                        cDeltaPhiFits.SaveAs(f'{canva_path}.pdf{suffix_pdf}')

                make_dir_root_file(f"{pc_dir}", corr_fits_file)
                corr_fits_file.cd(f"{pc_dir}")
                cDeltaPhiFits.Write()
                cDeltaPhiFits.Close()

                # ---- Draw hLM_template (LM data + template function + baseline) ----
                if do_lm and i_vn_vs_mass_bin == 0:
                    lm_out  = corr_fitter.GetLMOutput()
                    lm_func = corr_fitter.GetLMTemplateFunc()
                    bl_val  = corr_fitter.GetPedestal()

                    lm_out.SetDirectory(ROOT.nullptr)
                    bl_line = TF1("fBaseLine", "[0]", -0.5*math.pi, 1.5*math.pi)
                    bl_line.SetParameter(0, bl_val)
                    bl_line.SetLineColor(ROOT.kGray + 2)
                    bl_line.SetLineStyle(3)
                    bl_line.SetLineWidth(3)

                    c_lm = TCanvas(f"cLMTemplate_{pc_dir}_PtBinAssoc{i_pt_had + 1}",
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

                    leg_lm = TLegend(0.65, 0.70, 0.92, 0.88)
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

                    pt_text_lm = TPaveText(0.20, 0.90, 0.85, 0.96, "NDC")
                    pt_text_lm.SetFillStyle(0)
                    pt_text_lm.SetBorderSize(0)
                    pt_text_lm.AddText(
                        f"LM Template: {pt_cand_min:.1f} < #it{{p}}_{{T}}^{{{dmeson_label}}} < {pt_cand_max:.1f} GeV/c, "
                        f"{pt_had_min:.1f} < #it{{p}}_{{T}}^{{assoc}} < {pt_had_max:.1f} GeV/c, "
                        f"  tempFunc={corr_fitter.GetTempFunc()}")
                    pt_text_lm.SetTextAlign(22)
                    pt_text_lm.Draw("Same")
                    c_lm.Update()

                    lm_png_name = (f"hLMtemplate_{pc_dir}_PtBinAssoc{i_pt_had + 1}")
                    c_lm.SaveAs(os.path.join(out_dir, f"{lm_png_name}.pdf"))
                    out_file_lm.cd()
                    c_lm.Write()
                    c_lm.Close()
                    ROOT.SetOwnership(c_lm, False)

            del corr_fitter, h_fit

    # ---- Close input files ---------------------------------------------
    in_file.Close()
    if in_file_lm:
        in_file_lm.Close()

    corr_fits_file.cd()

    for i_pt_cand in range(n_pt_cand):
        corr_fits_file.cd(f"PtCand_{int(pt_bins_cand[i_pt_cand] * 10)}_{int(pt_bins_cand[i_pt_cand + 1] * 10)}")
        h_v2_delta[i_pt_cand].Write("hV2DeltaVsMass")

    # Write LM Factor histograms (flat — one per ptHad, no mass binning)
    for i_pt_had in range(n_pt_had):
        h_lm = h_lm_factor[i_pt_had]
        h_lm.SetDirectory(corr_fits_file)
        corr_fits_file.cd()
        h_lm.Write()

    corr_fits_file.Close()
    out_file_lm.Close()
    logger(f"Final plots saved to: {final_plot_path}", level="INFO")

    # Generate v2 vs Mass files for each pt_had bin
    if method == "MassBinning":
        build_mass_v2(config)

# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fitter for azimuthal correlations")
    parser.add_argument("config", nargs="?", default="config.yml", help="Path to YAML config file")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        logger(f"Config file not found: {args.config}", level="FATAL")
        sys.exit(1)

    fit_correl(args.config)
