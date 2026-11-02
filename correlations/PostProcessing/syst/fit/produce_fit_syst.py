#!/usr/bin/env python3
"""
Systematic uncertainty from the fit multitrial (sys) results.

For each pT bin, collect the prompt v2 and chi2/NDF of every trial, apply
the chi2/NDF cut (per pT bin), and compute the systematic uncertainty as
    syst_unc = sqrt(mean^2 + rms^2) of (v2_trial - v2_ref)
i.e. the same formula used by produce_fit_multitrial_syst_plots.py
(sqrt(mean^2 + rms^2) == RMS of the deviations about zero).

Usage:
    python3 produce_fit_syst.py <config.yaml> [--max_chi2 5] [--ref ref.root]
"""

import os
import sys
import glob
import argparse

import numpy as np
import yaml
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetCanvasPreferGL(True)
ROOT.gErrorIgnoreLevel = ROOT.kError
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

DMESON_NAME = {"Dzero": "D0", "D0": "D0", "Dplus": "Dplus", "Ds": "Ds"}


def read_v2(root_file, n_pt):
    f = ROOT.TFile.Open(root_file)
    if not f or f.IsZombie():
        raise RuntimeError(f"cannot open {root_file}")
    g = f.Get("gV2Prompt")
    if not g:
        # raise RuntimeError(f"gV2Prompt not found in {root_file}")
        print(f"[WARN] gV2Prompt not found in {root_file}, return zeros")
        return np.zeros(n_pt), np.zeros(n_pt)
    v2 = np.zeros(n_pt)
    unc = np.zeros(n_pt)
    for i in range(min(n_pt, g.GetN())):
        v2[i] = g.GetPointY(i)
        unc[i] = g.GetErrorY(i)
    f.Close()
    return v2, unc


def read_chi2ndf(trial_dir, n_pt, dmeson_name):
    root_dir = os.path.join(trial_dir, "CorrelationFitResults",
                            "Output_CorrelationFitting_Root")
    chi2 = [None] * n_pt
    for i_pt in range(n_pt):
        fname = (f"CorrPhi{dmeson_name}_PtBinCand{i_pt + 1}_"
                 f"PtBinAssoc1_InvMassBin1.root")
        hname = f"hChi2NDF_PtCand{i_pt + 1}_PtHad1_InvMassBin1"
        fpath = os.path.join(root_dir, fname)
        if not os.path.exists(fpath):
            continue
        f = ROOT.TFile.Open(fpath)
        if not f or f.IsZombie():
            continue
        h = f.Get(hname)
        if h:
            chi2[i_pt] = h.GetBinContent(1)
        f.Close()
    return chi2


def read_mass_chi2ndf(trial_dir, pt_bins, pt_had_bins):
    pair_file = os.path.join(trial_dir, "AssociatedPairsYields", "PairYieldsVsPhi.root")
    n_pt = len(pt_bins) - 1
    chi2 = [None] * n_pt
    if not os.path.exists(pair_file):
        return chi2
    f = ROOT.TFile.Open(pair_file)
    if not f or f.IsZombie():
        return chi2
    ph_dir = f"PtHadBin_{int(pt_had_bins[0] * 10)}_{int(pt_had_bins[1] * 10)}"
    for i_pt in range(n_pt):
        pc_dir = f"PtCandBin_{int(pt_bins[i_pt] * 10)}_{int(pt_bins[i_pt + 1] * 10)}"
        h = f.Get(f"{pc_dir}/{ph_dir}/chi2_over_ndf")
        if h:
            chi2[i_pt] = h.GetBinContent(1)
    f.Close()
    return chi2


def make_trial_graph(x_values, y_values, errors, name, ytitle):
    n = len(x_values)
    g = ROOT.TGraphErrors(n)
    g.SetName(name)
    g.SetTitle(f";trial index;{ytitle}")
    for j in range(n):
        g.SetPoint(j, float(x_values[j]), y_values[j])
        g.SetPointError(j, 0.0, errors[j])
    g.SetMarkerStyle(20)
    g.SetMarkerSize(1.2)
    g.SetMarkerColor(ROOT.kBlack)
    g.SetLineColor(ROOT.kBlack)
    g.SetLineWidth(1)
    g.GetXaxis().SetTitleSize(0.05)
    g.GetYaxis().SetTitleSize(0.05)
    g.GetXaxis().SetTitleOffset(1)
    g.GetYaxis().SetTitleOffset(1)
    return g


def main():
    ap = argparse.ArgumentParser(description="Fit multitrial systematic uncertainty")
    ap.add_argument("config", help="correlation config yaml")
    ap.add_argument("--max_chi2", "-mc2", type=float, default=5.0,
                    help="max correlation-fit chi2/NDF to accept a trial (per pT bin)")
    ap.add_argument("--max_mass_chi2", "-mmc2", type=float, default=10.0,
                    help="max mass-fit chi2/NDF to accept a trial (per pT bin)")
    ap.add_argument("--ref", "-r", default="",
                    help="reference v2_prompt root file (default: central value)")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sys_outdir = cfg["sys_outdir"]
    suffix = cfg["suffix"]
    pt_bins = [float(x) for x in cfg["ptBinsCand"]]
    pt_had_bins = [float(x) for x in cfg["ptBinsHad"]]
    n_pt = len(pt_bins) - 1
    dmeson_name = DMESON_NAME.get(cfg.get("Dmeson", "Dzero"), "D0")

    outdir = os.path.join(sys_outdir, "results", "fit")
    os.makedirs(outdir, exist_ok=True)
    print(f"[INFO] output dir: {outdir}")

    if args.ref:
        ref_file = args.ref
    else:
        ref_files = sorted(glob.glob(os.path.join(
            cfg["outdir"], f"CorrelExtract_{suffix}", "v2_prompt*.root")))
        if not ref_files:
            print(f"[ERROR] no v2_prompt file in "
                  f"{cfg['outdir']}/CorrelExtract_{suffix}")
            sys.exit(1)
        ref_file = ref_files[0]
    print(f"[INFO] ref file: {ref_file}")
    
    ref_v2, ref_unc = read_v2(ref_file, n_pt)

    sys_fit = os.path.join(sys_outdir, "sys_fit")
    trial_dirs = sorted(glob.glob(os.path.join(sys_fit, "CorrelExtract_trial_*")))
    if not trial_dirs:
        print(f"[ERROR] no trial dirs found under {sys_fit}")
        sys.exit(1)

    trials = []
    for td in trial_dirs:
        tid = os.path.basename(td).replace("CorrelExtract_trial_", "")
        v2_files = sorted(glob.glob(os.path.join(td, "v2_prompt*.root")))
        if not v2_files:
            continue
        v2, unc = read_v2(v2_files[0], n_pt)
        chi2 = read_chi2ndf(td, n_pt, dmeson_name)
        mass_chi2 = read_mass_chi2ndf(td, pt_bins, pt_had_bins)
        trials.append({"id": tid, "v2": v2, "unc": unc, "chi2": chi2, "mass_chi2": mass_chi2})
    trials.sort(key=lambda t: t["id"])
    # split into the 4 HM/LM binning combinations by the uniform naming
    # trial_<code>_H<hm>L<lm>; legacy variants (raw "xxxxx2", "_15"/"_32",
    # "HM_*", "LM_*") fall into no group.
    g_32_32 = [t for t in trials if t["id"].endswith("_H32L32")]   # 32H-32L
    g_32_16 = [t for t in trials if t["id"].endswith("_H32L16")]   # 32H-16L
    g_16_32 = [t for t in trials if t["id"].endswith("_H16L32")]   # 16H-32L
    g_16_16 = [t for t in trials if t["id"].endswith("_H16L16")]   # 16H-16L
    trials = g_32_32 + g_32_16 + g_16_32 + g_16_16
    print(f"[INFO] {len(trials)} trials found "
          f"(32H-32L={len(g_32_32)}, 32H-16L={len(g_32_16)}, "
          f"16H-32L={len(g_16_32)}, 16H-16L={len(g_16_16)})")

    h_syst_pt = ROOT.TH1F("hSystV2_vs_pT", ";#it{p}_{T} (GeV/#it{c});syst. unc. (fit procedure)",
                          n_pt, np.array(pt_bins, dtype=np.float64))
    h_syst_pt.GetYaxis().SetTitleOffset(2.0)
    
    h_rela_syst_pt = ROOT.TH1F("hRelaSystV2_vs_pT", ";#it{p}_{T} (GeV/#it{c});relative syst. unc. on #it{v}_{2}",
                               n_pt, np.array(pt_bins, dtype=np.float64))
    h_rela_syst_pt.GetYaxis().SetTitleOffset(2.0)

    for i_pt in range(n_pt):
        pt_min, pt_max = pt_bins[i_pt], pt_bins[i_pt + 1]
        pt_label = f"pt_{int(pt_min * 10)}_{int(pt_max * 10)}"

        v2s, uncs, chi2s, mass_chi2s = [], [], [], []
        for t in trials:
            v2 = t["v2"][i_pt]
            if v2 < -0.1 or v2 > 1.0:
                continue
            # if i_pt == 5 and (t["id"].endswith("31") or v2 > 0.096):
            if i_pt == (n_pt - 1) and (t["id"].endswith("15") and "HM" not in tid):
                continue
            if i_pt == 5 and v2 > 0.1:
                continue
            chi2_val = t["chi2"][i_pt]
            if chi2_val is not None and chi2_val <= args.max_chi2:
                mass_chi2_val = t["mass_chi2"][i_pt]
                if mass_chi2_val is not None and mass_chi2_val >= 0.0 and mass_chi2_val <= args.max_mass_chi2:
                    v2s.append(t["v2"][i_pt])
                    uncs.append(t["unc"][i_pt])
                    chi2s.append(chi2_val)
                    mass_chi2s.append(mass_chi2_val)
            # if i_pt == 5 and v2 > 0.096:
            #     print(f"[DEBUG] trial {t['id']} pt_bin {pt_label}: v2={v2:.4f}, chi2={chi2_val}, mass_chi2={mass_chi2_val}")
            #     print(f"[DEBUG] trial code: {t['id']}")
            # if (v2 < (ref_v2[i_pt] - ref_unc[i_pt]) or v2 > (ref_v2[i_pt] + ref_unc[i_pt])) and "HM" in t['id']:
            #     print(f"[DEBUG] trial {t['id']} pt_bin {pt_label}: v2={v2:.4f}, chi2={chi2_val}, mass_chi2={mass_chi2_val}")
            #     print(f"[DEBUG] trial code: {t['id']}")

        if not v2s:
            print(f"[WARN] no good trials for {pt_label}, skip")
            continue
            
        tids = np.arange(len(v2s))
        v2s = np.array(v2s)
        uncs = np.array(uncs)
        chi2s = np.array(chi2s)
        mass_chi2s = np.array(mass_chi2s)
        
        ref = ref_v2[i_pt]
        ref_stat_unc = ref_unc[i_pt]

        v2_dev = v2s - ref
        syst_unc = float(np.sqrt(np.mean(v2_dev ** 2)))
        h_syst_pt.SetBinContent(i_pt + 1, syst_unc)
        h_syst_pt.SetBinError(i_pt + 1, 0.0)

        rela_syst_unc = syst_unc / abs(ref) if ref != 0 else 0.0
        h_rela_syst_pt.SetBinContent(i_pt + 1, rela_syst_unc)
        h_rela_syst_pt.SetBinError(i_pt + 1, 0.0)

        plot_half = 1.8 * ref_stat_unc
        if plot_half == 0:
            plot_half = 1e-3
        hist_half = max(float(np.max(np.abs(v2_dev))), plot_half) * 1.1

        h_syst = ROOT.TH1F(f"h_syst_{pt_label}",
                           ";#it{v}_{2}(trial) - #it{v}_{2}(ref.);Counts", 50, -hist_half, hist_half)
        for d in v2_dev:
            h_syst.Fill(d)

        pt_dir = os.path.join(outdir, pt_label)
        os.makedirs(pt_dir, exist_ok=True)
        canvas = ROOT.TCanvas(f"c_syst_{pt_label}", "", 1920, 1536)
        canvas.Divide(2, 2)

        for ip in range(1, 5):
            pad = canvas.cd(ip)
            pad.SetTopMargin(0.05)
            pad.SetBottomMargin(0.12)
            pad.SetLeftMargin(0.12)
            pad.SetRightMargin(0.03)
            if ip != 2:
                pad.SetGrid()

        canvas.cd(1)
        g_v2 = make_trial_graph(tids, v2s, uncs, f"gV2_vs_trial_{pt_label}", "#it{v}_{2}")
        g_v2.Draw("AP")
        g_v2.GetXaxis().SetLimits(-0.5, len(v2s) - 0.5)
        ROOT.gPad.Update()

        xmin = g_v2.GetXaxis().GetXmin()
        xmax = g_v2.GetXaxis().GetXmax()

        box_ref = ROOT.TBox(xmin, ref - ref_stat_unc, xmax, ref + ref_stat_unc)
        box_ref.SetFillColorAlpha(ROOT.kAzure + 2, 0.3)
        box_ref.SetFillStyle(1001)
        box_ref.SetLineWidth(0)
        box_ref.Draw()
        
        line_ref = ROOT.TLine(xmin, ref, xmax, ref)
        line_ref.SetLineColor(ROOT.kBlack)
        line_ref.SetLineStyle(3)
        line_ref.SetLineWidth(3)
        line_ref.Draw()
        
        g_v2.Draw("P same")

        canvas.cd(2)
        h_syst.GetXaxis().SetRangeUser(-plot_half, plot_half)
        max_syst = h_syst.GetMaximum()
        h_syst.GetYaxis().SetRangeUser(0, max_syst * 1.8)
        h_syst.GetXaxis().SetTitleSize(0.05)
        h_syst.GetYaxis().SetTitleSize(0.05)
        h_syst.GetXaxis().SetTitleOffset(1.1)
        h_syst.GetYaxis().SetTitleOffset(1.1)
        h_syst.Draw("hist")

        box_syst = ROOT.TBox(-syst_unc, 0, syst_unc, max_syst * 1.8)
        box_syst.SetFillColorAlpha(ROOT.kOrange + 2, 0.3)
        box_syst.SetFillStyle(1001)
        box_syst.SetLineWidth(0)
        box_syst.Draw()

        box_ref_two = ROOT.TBox(-ref_stat_unc, 0, ref_stat_unc, max_syst * 1.8)
        box_ref_two.SetFillColorAlpha(ROOT.kAzure + 2, 0.3)
        box_ref_two.SetFillStyle(1001)
        box_ref_two.SetLineWidth(0)
        box_ref_two.Draw()

        h_syst.Draw("hist same")

        leg = ROOT.TLegend(0.18, 0.65, 0.92, 0.90)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetTextSize(0.045)
        leg.SetHeader(f'{pt_min} < #it{{p}}_{{T}} < {pt_max} GeV/#it{{c}}')
        leg.AddEntry(box_syst, f'#sqrt{{shift^{{2}} + rms^{{2}}}} = {syst_unc:.4f}', 'f')
        leg.AddEntry(box_ref_two, f'stat. unc. = {ref_stat_unc:.4f}', 'f')
        leg.Draw()

        canvas.cd(3)
        g_chi2 = make_trial_graph(tids, chi2s, np.zeros(len(chi2s)),
                                  f"gChi2NDF_vs_trial_{pt_label}", "correlation #chi^{2}/NDF")
        g_chi2.Draw("AP")
        g_chi2.GetXaxis().SetLimits(-0.5, len(v2s) - 0.5)
        g_chi2.GetYaxis().SetRangeUser(0, 5)
        ROOT.gPad.Update()

        canvas.cd(4)
        g_mass_chi2 = make_trial_graph(tids, mass_chi2s, np.zeros(len(mass_chi2s)),
                                       f"gMassChi2NDF_vs_trial_{pt_label}", "mass #chi^{2}/NDF")
        g_mass_chi2.Draw("AP")
        g_mass_chi2.GetXaxis().SetLimits(-0.5, len(v2s) - 0.5)
        g_mass_chi2.GetYaxis().SetRangeUser(0, 10)
        ROOT.gPad.Update()
        
        line_mass_chi2 = ROOT.TLine(g_mass_chi2.GetXaxis().GetXmin(), args.max_mass_chi2, 
                                    g_mass_chi2.GetXaxis().GetXmax(), args.max_mass_chi2)
        line_mass_chi2.SetLineColor(ROOT.kRed)
        line_mass_chi2.SetLineStyle(2)
        line_mass_chi2.SetLineWidth(2)
        line_mass_chi2.Draw()

        canvas.Update()

        canvas.SaveAs(os.path.join(pt_dir, f"syst_{pt_label}.png"))
        fout = ROOT.TFile(os.path.join(pt_dir, f"SystFit_{pt_label}.root"), "RECREATE")
        g_v2.Write()
        h_syst.Write()
        g_chi2.Write()
        g_mass_chi2.Write()
        canvas.Write()
        fout.Close()

        print(f"[OK] {pt_label}: n_trials={len(v2s)} "
              f"syst_unc={syst_unc:.4f} rela_syst={rela_syst_unc:.4f}")

    c_syst = ROOT.TCanvas("c_syst_vs_pT", "", 1920, 1536)
    c_syst.SetTopMargin(0.05)
    c_syst.SetRightMargin(0.03)
    c_syst.SetBottomMargin(0.12)
    c_syst.SetLeftMargin(0.12)
    # 左右上下都显示刻度
    c_syst.SetTickx()
    c_syst.SetTicky()
    
    h_syst_pt.GetXaxis().SetTitleSize(0.05)
    h_syst_pt.GetYaxis().SetTitleSize(0.05)
    h_syst_pt.GetXaxis().SetTitleOffset(1)
    h_syst_pt.GetYaxis().SetTitleOffset(1.1)
    h_syst_pt.SetFillColorAlpha(ROOT.kOrange + 1, 0.7)
    h_syst_pt.SetFillStyle(1001)
    h_syst_pt.SetLineColor(ROOT.kBlack)
    h_syst_pt.SetLineWidth(2)

    h_syst_pt.Draw("HIST")
    h_syst_pt.Draw("E SAME")
    
    c_syst.SaveAs(os.path.join(outdir, "SystV2_vs_pT.png"))
    c_syst.SaveAs(os.path.join(outdir, "SystV2_vs_pT.pdf"))

    c_rela_syst = ROOT.TCanvas("c_rela_syst_vs_pT", "", 1920, 1536)
    c_rela_syst.SetTopMargin(0.05)
    c_rela_syst.SetRightMargin(0.03)
    c_rela_syst.SetBottomMargin(0.12)
    c_rela_syst.SetLeftMargin(0.12)
    
    h_rela_syst_pt.GetXaxis().SetTitleSize(0.05)
    h_rela_syst_pt.GetYaxis().SetTitleSize(0.05)
    h_rela_syst_pt.GetXaxis().SetTitleOffset(1.1)
    h_rela_syst_pt.GetYaxis().SetTitleOffset(1.5)
    h_rela_syst_pt.SetFillColorAlpha(ROOT.kAzure + 1, 0.7)
    h_rela_syst_pt.SetFillStyle(1001)
    h_rela_syst_pt.SetLineColor(ROOT.kBlack)
    h_rela_syst_pt.SetLineWidth(2)


    h_rela_syst_pt.Draw("HIST")
    h_rela_syst_pt.Draw("E SAME")
    
    c_rela_syst.SaveAs(os.path.join(outdir, "RelaSystV2_vs_pT.png"))
    
    g_syst_pt = ROOT.TGraphErrors(h_syst_pt)
    g_syst_pt.SetName("gSystV2_vs_pT")
    g_syst_pt.SetTitle(";#it{p}_{T} (GeV/#it{c});syst. unc. from fit procedure")
    g_syst_pt.SetMarkerStyle(20)
    g_syst_pt.SetMarkerSize(1.2)
    g_syst_pt.SetMarkerColor(ROOT.kBlack)
    g_syst_pt.SetLineColor(ROOT.kBlack)
    g_syst_pt.SetLineWidth(2)
    for i in range(g_syst_pt.GetN()):
        g_syst_pt.SetPointError(i, 0.0, h_syst_pt.GetBinError(i + 1))

    fsum = ROOT.TFile(os.path.join(outdir, "TotalSystV2.root"), "RECREATE")
    h_syst_pt.Write()
    h_rela_syst_pt.Write()
    c_syst.Write()
    c_rela_syst.Write()
    g_syst_pt.Write()
    fsum.Close()

    print(f"[OK] summary -> {outdir}/TotalSystV2.root")


if __name__ == "__main__":
    main()