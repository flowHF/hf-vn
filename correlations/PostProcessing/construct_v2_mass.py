#!/usr/bin/env python3
"""
construct_v2_mass.py — Build v2-vs-mass histograms from FitCorrel output.
Reads:  InvMassVsPt.root + CorrPhiD0_FinalPlots.root
Writes: MassVsV2/InvMassVsV2_PtAssoc*.root  (hMassData + hVnVsMassData per pt)

Also provides extract_ry_trigger subcommand to inject ry_trigger into
CorrelationsResults.root before FitCorrel.py runs.

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

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '../../', 'utils'))
from utils import check_dir, logger

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning


def get_mass_distribution(mass_vs_pt, pt_min, pt_max):
    """Project the 2D MassVsPt histogram for a given pt range."""
    yaxis = mass_vs_pt.GetYaxis()
    bin_min = yaxis.FindBin(pt_min * 1.0001)
    bin_max = yaxis.FindBin(pt_max * 0.9999)
    yaxis.SetRange(bin_min, bin_max)
    proj = mass_vs_pt.ProjectionX(f"mass_proj_{pt_min:.1f}_{pt_max:.1f}")
    proj.SetDirectory(0)
    return proj


def build_mass_v2(cfg_path):
    """Build v2-vs-mass histograms from InvMassVsPt.root + CorrPhiD0_FinalPlots.root."""
    # Load config and check method
    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    method = config.get("method")
    suffix = config["suffix"]
    outdir = PATH.Path(config["outdir"])
    meson = config.get("Dmeson", "Dzero")
    v2_hh = float(config.get("v2HH", 0.07))
    if v2_hh <= 0:
        logger(f"Invalid v2HH: {v2_hh}", level="FATAL")
    elif isinstance(v2_hh, float):
        logger(f"Using v2HH = {v2_hh} for scaling v2 values", level="WARNING")
    elif config.get("v2HH") is not None:
        logger(f"v2HH should be a single float value, but got {config.get('v2HH')} — using default {v2_hh}", level="WARNING")
    
    if method != "MassBinning":
        logger(f"[method='{method}' — skip (not MassBinning).")
        return

    extract_dir = outdir / f"CorrelExtract_{suffix}"

    inv_mass_path = outdir / "InvMass" / "InvMassVsPt.root"
    final_plots_path = (extract_dir / "CorrelationFitResults"
                        / "Output_CorrelationFitting_Root"
                        / f"CorrPhi{meson}_FinalPlots.root")
    
    if not inv_mass_path.exists(): logger(f"{inv_mass_path} not found", level="FATAL")
    if not final_plots_path.exists(): logger(f"{final_plots_path} not found", level="FATAL")

    pt_bins_cand = [float(ptBinCand) for ptBinCand in config["ptBinsCand"]]
    pt_bins_had  = [float(ptBinHad) for ptBinHad in config["ptBinsHad"]]
    inv_mass_bins_raw = config["invMassBins"]
    n_pt_cand= len(pt_bins_cand) - 1

    # Build per-pt-cand mass bin edges (matching FitCorrel logic)
    mass_edges_per_pt = []
    for i_pt in range(n_pt_cand):
        if i_pt < len(inv_mass_bins_raw):
            edges = [float(x) for x in inv_mass_bins_raw[i_pt]]
        else:
            logger(f"invMassBins has only {len(inv_mass_bins_raw)}, but {n_pt_cand} pt cand bins — using last entry for remaining bins", level="WARNING")
            edges = [float(x) for x in inv_mass_bins_raw[-1]]
        mass_edges_per_pt.append(edges)

    file_mass = ROOT.TFile.Open(str(inv_mass_path))
    h_mass_vs_pt = file_mass.Get("hMassVsPt")
    file_v2 = ROOT.TFile.Open(str(final_plots_path))
    h_v2_deltas = {obj.GetName(): file_v2.Get(obj.GetName()) for obj in file_v2.GetListOfKeys()}
    for h in h_v2_deltas.values():
        h.SetDirectory(0)
    logger(f"Loaded {len(h_v2_deltas)} histos from FinalPlots")

    out_mass_v2_dir = extract_dir / "CorrelationFitResults" / "MassVsV2"
    os.makedirs(str(out_mass_v2_dir), exist_ok=True)

    for i_pt_had, (pt_had_min, pt_had_max) in enumerate(zip(pt_bins_had[:-1], pt_bins_had[1:])):
        pt_had_str = f"PtAssoc{int(pt_had_min*10):02d}to{int(pt_had_max*10):02d}"
        h_masses, h_mass_vs_v2s = [], []
        for i_pt_cand, (pt_cand_min, pt_cand_max) in enumerate(zip(pt_bins_cand[:-1], pt_bins_cand[1:])):
            pt_cand_str = f"PtCand{int(pt_cand_min*10):02d}to{int(pt_cand_max*10):02d}"

            # Use this pt cand bin's mass edges
            mass_edges = mass_edges_per_pt[i_pt_cand]
            n_mass_bins = len(mass_edges) - 1

            tmp = h_mass_vs_pt.Clone(f"tMass_{pt_cand_str}_{pt_had_str}")
            h_masses.append(get_mass_distribution(tmp, pt_cand_min, pt_cand_max))

            h_mass_vs_v2 = ROOT.TH1F(f"v2Mass_{pt_cand_str}_{pt_had_str}", "", n_mass_bins,
                           np.array(mass_edges, dtype="d"))
            h_mass_vs_v2.SetTitle("Mass vs V2; Mass (GeV/c^{2}); V2")
            for i_ml in range(n_mass_bins):
                hn = f"hv2Delta_PtBinAssoc{i_pt_had+1}_InvMassBin{i_ml+1}"
                hv = h_v2_deltas.get(hn)
                if hv:
                    # hv2Delta is a 1D histogram with n_pt_cand bins (one per pt cand).
                    # For MassBinning, only the bin matching i_pt_cand is filled.
                    h_mass_vs_v2.SetBinContent(i_ml+1, hv.GetBinContent(i_pt_cand+1))
                    h_mass_vs_v2.SetBinError(i_ml+1, hv.GetBinError(i_pt_cand+1))
                else:
                    print(f"  [WARNING] {hn} not found in FinalPlots — skipping mass bin {i_ml+1} for ptCand {i_pt_cand}")
            h_mass_vs_v2.Scale(1.0 / v2_hh)
            h_mass_vs_v2s.append(h_mass_vs_v2)

        out_path = out_mass_v2_dir / f"InvMassVsV2_{pt_had_str}.root"
        out_file = ROOT.TFile.Open(str(out_path), "RECREATE")
        for i_pt in range(n_pt_cand):
            pd = f"pt_{int(pt_bins_cand[i_pt]*10):.0f}_{int(pt_bins_cand[i_pt+1]*10):.0f}"
            out_file.mkdir(pd)
            out_file.cd(pd)
            h_masses[i_pt].Write("hMassData"); h_mass_vs_v2s[i_pt].Write("hVnVsMassData")
        out_file.Close()
        print(f"[INFO] Saved {out_path}")

    file_mass.Close()
    file_v2.Close()
    print(f"[INFO] All MassVsV2 files in {out_mass_v2_dir}")


def extract_ry_trigger(cfg_path):
    """Fit mass projections from InvMassVsPt.root → inject ry_trigger into
    CorrelationsResults.root (MUST run BEFORE FitCorrel.py)."""
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..", "..", "flareflyfitter"))
    from raw_yield_fitter import RawYieldFitter

    with open(cfg_path) as f:
        config = yaml.safe_load(f)
    method = config.get("method")
    if method != "MassBinning":
        print(f"[INFO] method='{method}' — skip.")
        return

    suffix = config["suffix"]
    outdir = PATH.Path(config["outdir"])
    extract_dir = outdir / f"CorrelExtract_{suffix}"
    inv_mass_path = outdir / "InvMass" / "InvMassVsPt.root"
    corr_results_path = (extract_dir / "CorrelationsResults"
                         / "CorrelationsResults.root")
    if not inv_mass_path.exists():
        print(f"[ERROR] {inv_mass_path} not found")
        sys.exit(1)
    if not corr_results_path.exists():
        print(f"[ERROR] {corr_results_path} not found")
        sys.exit(1)

    pt_bins_cand = [float(ptBinCand) for ptBinCand in config["ptBinsCand"]]
    pt_bins_had  = [float(ptBinHad) for ptBinHad in config["ptBinsHad"]]
    n_pt_cand, n_pt_had = len(pt_bins_cand)-1, len(pt_bins_had)-1
    fit_config = config.get("fitConfig", {})
    Dmeson = config.get("Dmeson", "Dzero")

    file_mass = ROOT.TFile.Open(str(inv_mass_path))
    htemp = file_mass.Get("hMassVsPt")

    inv_mass_bins_raw = config["invMassBins"]
    mass_edges_per_pt = []
    for i_pt in range(n_pt_cand):
        if i_pt < len(inv_mass_bins_raw):
            edges = [float(x) for x in inv_mass_bins_raw[i_pt]]
        else:
            print(f"[WARNING] invMassBins has only {len(inv_mass_bins_raw)}, but {n_pt_cand} pt cand bins — using last entry for remaining bins")
            edges = [float(x) for x in inv_mass_bins_raw[-1]]
        mass_edges_per_pt.append(edges)

    results = {}
    for i_pt_cand, (pt_cand_min, pt_cand_max) in enumerate(zip(pt_bins_cand[:-1], pt_bins_cand[1:])):
        y1 = htemp.GetYaxis().FindBin(pt_cand_min*1.0001)
        y2 = htemp.GetYaxis().FindBin(pt_cand_max*0.9999)
        h_mass_temp = htemp.ProjectionX(f"_ryproj_pc{i_pt_cand}", y1, y2)
        h_mass_temp.SetDirectory(0)
        total_count = h_mass_temp.Integral()

        mass_edges = mass_edges_per_pt[i_pt_cand]
        n_mass_bins = len(mass_edges) - 1

        mass_fit_range = (fit_config["MassFitRanges"][i_pt_cand] if isinstance(fit_config.get("MassFitRanges"), list)
               and len(fit_config["MassFitRanges"]) > i_pt_cand else [pt_bins_cand[0], pt_bins_cand[-1]])
        sgn_func = (fit_config["SgnFunc"][i_pt_cand] if isinstance(fit_config.get("SgnFunc"), list)
              else fit_config.get("SgnFunc", "kGaus"))
        bkg_func = (fit_config["BkgFunc"][i_pt_cand] if isinstance(fit_config.get("BkgFunc"), list)
              else fit_config.get("BkgFunc", "kExpo"))
        rebin = (fit_config["Rebin"][i_pt_cand] if isinstance(fit_config.get("Rebin"), list)
              else fit_config.get("Rebin", 4))

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
        fitter.plot_fit(logy=False, path=str(extract_dir / f"ry_trigger_fit_pc{i_pt_cand}.png"), show_extra_info=False)
        fit_info, _, _, _, _ = fitter.get_fit_info()
        # Compute per-mass-bin counts from the mass projection
        mass_bin_counts = []
        for i_ml in range(n_mass_bins):
            bin_low = h_mass_temp.GetXaxis().FindBin(mass_edges[i_ml] * 1.0001)
            bin_high = h_mass_temp.GetXaxis().FindBin(mass_edges[i_ml + 1] * 0.9999)
            cnt = int(h_mass_temp.Integral(bin_low, bin_high))
            mass_bin_counts.append(cnt)
        for i_ph in range(n_pt_had):
            results[(i_pt_cand, i_ph)] = (fit_info['sgn']['ry'], fit_info['sgn']['ry_unc'],
                                          total_count, mass_edges, mass_bin_counts)
        print(f"  ptCand [{pt_cand_min:.1f}, {pt_cand_max:.1f}]: ry_trigger = {fit_info['sgn']['ry']:.0f} ± {fit_info['sgn']['ry_unc']:.0f}, total_count = {total_count:.0f}")
        print(f"    mass bins: {n_mass_bins}, bin counts: {mass_bin_counts}")

    file_mass.Close()

    out_file = ROOT.TFile.Open(str(corr_results_path), "UPDATE")
    n_ry = 0
    for i_pt_cand, (pt_cand_min, pt_cand_max) in enumerate(zip(pt_bins_cand[:-1], pt_bins_cand[1:])):
        pt_cand_str = f"PtCandBin_{int(pt_cand_min*10):.0f}_{int(pt_cand_max*10):.0f}"
        for i_pt_had, (pt_had_min, pt_had_max) in enumerate(zip(pt_bins_had[:-1], pt_bins_had[1:])):
            pt_had_str = f"PtHadBin_{int(pt_had_min*10):.0f}_{int(pt_had_max*10):.0f}"
            dir_pt = f"{pt_cand_str}/{pt_had_str}"
            if not out_file.GetDirectory(dir_pt):
                out_file.mkdir(pt_cand_str); out_file.cd(pt_cand_str)
                if not ROOT.gDirectory.GetDirectory(pt_had_str):
                    ROOT.gDirectory.mkdir(pt_had_str)
            ry, ry_err, total_count, mass_edges, mass_bin_counts = results.get((i_pt_cand, i_pt_had),
                                                                               (0., 0., 0., [], []))
            # Save ry_trigger as multi-bin histogram (one bin per mass sub-bin)
            n_mass = len(mass_bin_counts)
            if n_mass > 0:
                hry = ROOT.TH1D("ry_trigger", "", n_mass,
                                np.array(mass_edges, dtype="d"))
                for i_ml in range(n_mass):
                    hry.SetBinContent(i_ml + 1, mass_bin_counts[i_ml])
            else:
                hry = ROOT.TH1D("ry_trigger", "", 1, 0., 1.)
                hry.SetBinContent(1, total_count)
            hry.SetDirectory(0)
            out_file.cd(dir_pt)
            hry.Write("ry_trigger", ROOT.TObject.kOverwrite)
            n_ry += 1
    out_file.Close()
    print(f"[INFO] Injected {n_ry} ry_trigger (per-mass-bin) into {corr_results_path}")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="MassBinning post-processing")
    sub_args = args.add_subparsers(dest="cmd", required=True)

    par1 = sub_args.add_parser("extract-ry-trigger", help="Inject ry_trigger (BEFORE FitCorrel)")
    par1.add_argument("config")

    par2 = sub_args.add_parser("build-mass-v2", help="Build v2-vs-mass histograms (AFTER FitCorrel)")
    par2.add_argument("config")

    pars = args.parse_args()
    if not os.path.exists(pars.config):
        logger(f"Config not found: {pars.config}", level="FATAL")
    if pars.cmd == "extract-ry-trigger":
        extract_ry_trigger(pars.config)
    else:
        logger(f"Running build_mass_v2 with config: {pars.config}")
        build_mass_v2(pars.config)
