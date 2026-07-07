#!/usr/bin/env python3
"""
construct_v2_mass.py — Build v2-vs-mass histograms from FitCorrel output.
Reads:  InvMassVsPt.root + CorrPhiD0.root
Writes: MassVsV2/InvMassVsV2_PtAssoc*.root  (hMassData + hVnVsMassData per pt)

Also provides extract_ry_trigger subcommand to inject ry_trigger into
CorrelationsResults.root before fit_correl.py runs.

Usage:
  # Step 1 (before FitCorrel): inject ry_trigger
  python3 construct_v2_mass.py extract-ry-trigger config.yaml

  # Step 2 (after FitCorrel): build v2-vs-mass histograms
  python3 construct_v2_mass.py build-mass-v2 config.yaml
"""

import argparse
import os
import sys
import numpy as np
import yaml
import pathlib as PATH
import ROOT
from ROOT import TFile, gDirectory, TH1D, TH1F, TObject, TDirectoryFile

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '../../', 'utils'))
from utils import check_dir, logger, make_dir_root_file
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "flareflyfitter"))
from raw_yield_fitter import RawYieldFitter

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning


def get_v2_vs_mass_bins(inv_mass_bins, pt_bins_cand):

    # Build per-pt-cand mass bin edges (matching FitCorrel logic)
    mass_edges_per_pt = []
    for i_pt in range(len(pt_bins_cand) - 1):
        if i_pt < len(inv_mass_bins):
            edges = [float(x) for x in inv_mass_bins[i_pt]]
        else:
            logger(f"invMassBins has only {len(inv_mass_bins)}, but {len(pt_bins_cand) - 1} pt cand bins — using last entry for remaining bins", level="WARNING")
            edges = [float(x) for x in inv_mass_bins[-1]]
        mass_edges_per_pt.append(edges)

    return mass_edges_per_pt


def extract_ry_trigger(config):
    """Fit mass projections from InvMassVsPt.root → inject ry_trigger into
    CorrelationsResults.root (MUST run BEFORE fit_correl.py)."""

    if config["method"] != "MassBinning":
        logger(f"[INFO] method='{config['method']}' — skip.", level="INFO")
        return

    # Retrieve trigger mass vs pt histo
    suffix = config["suffix"]
    outdir = PATH.Path(config["outdir"])
    inv_mass_path = outdir / "trigger_yields" / "InvMassVsPt.root"
    if not inv_mass_path.exists():
        logger(f"{inv_mass_path} not found", level="FATAL")
        sys.exit(1)

    corr_results_path = (outdir / "maps"/ "CorrelationsResults.root")
    if not corr_results_path.exists():
        logger(f"{corr_results_path} not found", level="FATAL")
        sys.exit(1)

    pt_bins_cand = [float(ptBinCand) for ptBinCand in config["ptBinsCand"]]
    pt_bins_had  = [float(ptBinHad) for ptBinHad in config["ptBinsHad"]]
    fit_config = config.get("fitConfig", {})
    Dmeson = config["Dmeson"]

    file_mass = ROOT.TFile.Open(str(inv_mass_path))
    h_mass_vs_pt = file_mass.Get("hMassVsPt")
    mass_edges_per_pt = get_v2_vs_mass_bins(config["invMassBins"], pt_bins_cand)

    # Fit mass projections for each pt bin, extract raw yields and per-mass-bin counts
    out_file = TFile.Open(str(corr_results_path), "UPDATE")
    for i_pt_cand, (pt_cand_min, pt_cand_max, mass_edges) in enumerate(zip(pt_bins_cand[:-1], pt_bins_cand[1:], mass_edges_per_pt)):
        y1 = h_mass_vs_pt.GetYaxis().FindBin(pt_cand_min*1.0001)
        y2 = h_mass_vs_pt.GetYaxis().FindBin(pt_cand_max*0.9999)
        h_mass_temp = h_mass_vs_pt.ProjectionX(f"_ryproj_pc{i_pt_cand}", y1, y2)
        h_mass_temp.SetDirectory(0)
        total_count = h_mass_temp.Integral()

        mass_fit_range = fit_config["MassFitRanges"][i_pt_cand]
        sgn_func = (fit_config["SgnFunc"][i_pt_cand] if isinstance(fit_config.get("SgnFunc"), list)
                    else fit_config["SgnFunc"])
        bkg_func = (fit_config["BkgFunc"][i_pt_cand] if isinstance(fit_config.get("BkgFunc"), list)
                    else fit_config["BkgFunc"])
        rebin = (fit_config["Rebin"][i_pt_cand] if isinstance(fit_config.get("Rebin"), list)
                 else fit_config.get("Rebin", 1))

        fitter = RawYieldFitter(Dmeson, pt_cand_min, pt_cand_max,
                                f"pt_{int(pt_cand_min*10)}_{int(pt_cand_max*10)}",
                                fit_config.get("minimizer", "flarefly"))
        fitter.set_fit_range(mass_fit_range[0], mass_fit_range[1])
        fitter.add_sgn_func(sgn_func, "sgn", Dmeson)
        fitter.add_bkg_func(bkg_func, "Comb_bkg")
        fitter.set_name(f"rytrigger_pc{i_pt_cand}")
        fitter.set_rebin(rebin)
        fitter.set_data_to_fit_hist(h_mass_temp)
        fitter.setup()
        fitter.fit()
        outfile_qa_fit = TFile.Open(str(outdir / "trigger_yields" / f"ry_trigger_fit_pc{i_pt_cand}.root"), "RECREATE")
        fitter.plot_fit(logy=False, path=str(outdir / "trigger_yields" / f"ry_trigger_fit_pc{i_pt_cand}.pdf"), out_file=outfile_qa_fit, show_extra_info=False)
        outfile_qa_fit.Close()
        fit_info, _, _, _, _ = fitter.get_fit_info()

        # Compute per-mass-bin counts from the mass projection
        mass_bin_counts = []
        n_mass_bins = len(mass_edges) - 1
        if n_mass_bins > 0:
            hry = TH1D("ry_trigger", "", n_mass_bins, np.array(mass_edges, dtype="d"))
            for i_ml in range(n_mass_bins):
                bin_low = h_mass_temp.GetXaxis().FindBin(mass_edges[i_ml] * 1.0001)
                bin_high = h_mass_temp.GetXaxis().FindBin(mass_edges[i_ml + 1] * 0.9999)
                counts = int(h_mass_temp.Integral(bin_low, bin_high))
                hry.SetBinContent(i_ml + 1, counts)
        else:
            hry = TH1D("ry_trigger", "", 1, 0., 1.)
            hry.SetBinContent(1, total_count)
            mass_bin_counts.append(counts)

        hry.SetDirectory(0)
        for pt_had_min, pt_had_max in zip(pt_bins_had[:-1], pt_bins_had[1:]):
            pt_had_str = f"PtHad_{int(pt_had_min*10):.0f}_{int(pt_had_max*10):.0f}"
            dir_pt = f"PtCand_{int(pt_cand_min*10):.0f}_{int(pt_cand_max*10):.0f}/{pt_had_str}"
            out_file.cd(dir_pt)
            hry.Write("ry_trigger", TObject.kOverwrite)
        logger(f"  ptCand [{pt_cand_min:.1f}, {pt_cand_max:.1f}]: ry_trigger = {fit_info['sgn']['ry']:.0f} ± {fit_info['sgn']['ry_unc']:.0f}, total_count = {total_count:.0f}", level="INFO")
        logger(f"  mass bins: {n_mass_bins}, bin counts: {mass_bin_counts}", level="INFO")

    file_mass.Close()
    out_file.Close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MassBinning post-processing")
    parser.add_argument("mode", choices=["extract-ry-trigger", "build-mass-v2"], help="Processing mode")
    parser.add_argument("config", help="Path to config file")

    args = parser.parse_args()
    if not os.path.exists(args.config):
        logger(f"Config not found: {args.config}", level="FATAL")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    extract_ry_trigger(config)
