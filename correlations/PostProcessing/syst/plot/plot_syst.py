#!/usr/bin/env python3
"""Plot prompt v2 with statistical + systematic uncertainties.

Reads syst_config.yml (Dmeson -> centrality -> {pt_bins, fit_syst, lm_syst,
fd_syst_low, fd_syst_high}) and a central value file.

Draws (from widest to narrowest, then points):
  - total box:    outline box, +/-0.9 * bin-half-width
  - fd_syst  band: solid block, 70% transparent, +/-0.7 * bin-half-width, with gray outline
  - lm_syst  band: solid block, 70% transparent, +/-0.5 * bin-half-width, with gray outline
  - fit_syst band: solid block, 70% transparent, +/-0.3 * bin-half-width, with gray outline
  - central v2:    red star marker with stat errors
  - dashed line at y = 0
"""

import os
import math
import argparse
import yaml
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

HERE = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(HERE, "syst_config.yml")
CENTRAL_FILE = ("/home/wuct/MetaData/DATA/OO/apass2/corr/results/fifth/"
                "k020_gausPer/k60100_loose2to2d5_d20/sys/central/"
                "CorrelExtract_0d2_1d3/v2_prompt_ratio_0d2_1d3.root")
OUT_PNG = os.path.join(HERE, "v2_syst.png")
OUT_ROOT = os.path.join(HERE, "v2_syst.root")


def read_central(central_file):
    f = ROOT.TFile.Open(central_file)
    if not f or f.IsZombie():
        raise RuntimeError(f"cannot open {central_file}")
    g = f.Get("gV2Prompt")
    if not g:
        raise RuntimeError(f"gV2Prompt not found in {central_file}")
    n = g.GetN()
    pt_c = [g.GetPointX(i) for i in range(n)]
    central = [g.GetPointY(i) for i in range(n)]
    stat = [g.GetErrorY(i) for i in range(n)]
    f.Close()
    return pt_c, central, stat


def main():
    ap = argparse.ArgumentParser(description="Plot prompt v2 with syst uncertainties")
    ap.add_argument("--central-file", default=CENTRAL_FILE,
                    help="central v2_prompt root file (gV2Prompt)")
    ap.add_argument("--yaml", default=YAML_PATH, help="syst config yaml")
    args = ap.parse_args()

    with open(args.yaml) as f:
        cfg = yaml.safe_load(f)

    dmeson = next(iter(cfg))
    centrality = next(iter(cfg[dmeson]))
    d = cfg[dmeson][centrality]

    pt_bins = [float(x) for x in d["pt_bins"]]
    fit_syst = [float(x) for x in d["fit_syst"]]
    lm_syst = [float(x) for x in d["lm_syst"]]
    fd_low = [float(x) for x in d["fd_syst_low"]]
    fd_high = [float(x) for x in d["fd_syst_high"]]
    fp_syst = float(d["fp_syst"])  # universal scalar

    pt_c, central, stat = read_central(args.central_file)
    n = len(central)
    halfw = [0.5 * (pt_bins[i + 1] - pt_bins[i]) for i in range(n)]

    # total syst EXCLUDES FD (FD is drawn as its own band below)
    total = [math.sqrt(fit_syst[i] ** 2 + lm_syst[i] ** 2)
             for i in range(n)]

    # ---- central points with stat (vertical) + bin-width (horizontal) bars ----
    g_central = ROOT.TGraphErrors(n)
    g_central.SetName("gCentral")
    for i in range(n):
        g_central.SetPoint(i, pt_c[i], central[i])
        g_central.SetPointError(i, halfw[i], stat[i])
        
    g_central.SetMarkerStyle(29)
    g_central.SetMarkerSize(3.5)
    color_central = ROOT.TColor.GetColor("#D55E00") # Vermillion
    g_central.SetMarkerColor(color_central)
    g_central.SetLineColor(color_central)
    g_central.SetLineWidth(2)

    alpha = 0.3

    # Define custom hex colors for bands
    color_fit = ROOT.TColor.GetColor("#E69F00") # Soft orange
    color_lm  = ROOT.TColor.GetColor("#56B4E9") # Sky blue
    color_fd  = ROOT.TColor.GetColor("#009E73") # Bluish green
    color_outline = ROOT.kGray + 1              # Gray color for the outlines

    # ---- fit_syst band ----
    g_fit = ROOT.TGraphAsymmErrors(n)
    g_fit.SetName("gFitSyst")
    for i in range(n):
        g_fit.SetPoint(i, pt_c[i], central[i])
        g_fit.SetPointError(i, 0.3 * halfw[i], 0.3 * halfw[i], fit_syst[i], fit_syst[i])
    g_fit.SetFillStyle(1001)
    g_fit.SetFillColorAlpha(color_fit, alpha)
    g_fit.SetLineColor(color_fit)
    
    # Create outline clone for fit_syst
    g_fit_out = g_fit.Clone("gFitSyst_out")
    g_fit_out.SetFillStyle(0) # Hollow fill
    g_fit_out.SetFillColorAlpha(color_outline, alpha)
    g_fit_out.SetLineColor(color_outline)
    g_fit_out.SetLineWidth(1)

    # ---- lm_syst band ----
    g_lm = ROOT.TGraphAsymmErrors(n)
    g_lm.SetName("gLMSyst")
    for i in range(n):
        g_lm.SetPoint(i, pt_c[i], central[i])
        g_lm.SetPointError(i, 0.5 * halfw[i], 0.5 * halfw[i], lm_syst[i], lm_syst[i])
    g_lm.SetFillStyle(1001)
    g_lm.SetFillColorAlpha(color_lm, alpha)
    g_lm.SetLineColor(color_lm)
    
    # Create outline clone for lm_syst
    g_lm_out = g_lm.Clone("gLMSyst_out")
    g_lm_out.SetFillStyle(0)
    g_lm_out.SetFillColorAlpha(color_outline, alpha)
    g_lm_out.SetLineColor(color_outline)
    g_lm_out.SetLineWidth(1)

    # ---- fd_syst band (drawn: FD combined with FP = sqrt(fd^2 + fp^2)) ----
    g_fd = ROOT.TGraphAsymmErrors(n)
    g_fd.SetName("gFDSystTot")
    for i in range(n):
        fd_low_tot = math.sqrt(fd_low[i] ** 2 + fp_syst ** 2)
        fd_high_tot = math.sqrt(fd_high[i] ** 2 + fp_syst ** 2)
        g_fd.SetPoint(i, pt_c[i], central[i])
        g_fd.SetPointError(i, 0.7 * halfw[i], 0.7 * halfw[i], fd_low_tot, fd_high_tot)
    g_fd.SetFillStyle(1001)
    g_fd.SetFillColorAlpha(color_fd, alpha)
    g_fd.SetLineColor(color_fd)
    
    # Create outline clone for fd_syst
    g_fd_out = g_fd.Clone("gFDSystTot_out")
    g_fd_out.SetFillStyle(0)
    g_fd_out.SetFillColorAlpha(color_outline, alpha)
    g_fd_out.SetLineColor(color_outline)
    g_fd_out.SetLineWidth(1)

    # ---- fd-only graph (saved separately as gFDSyst) ----
    g_fd_raw = ROOT.TGraphAsymmErrors(n)
    g_fd_raw.SetName("gFDSyst")
    for i in range(n):
        g_fd_raw.SetPoint(i, pt_c[i], central[i])
        g_fd_raw.SetPointError(i, 0.7 * halfw[i], 0.7 * halfw[i], fd_low[i], fd_high[i])

    # ---- fp-only graph (saved separately as gFPSyst) ----
    g_fp = ROOT.TGraphAsymmErrors(n)
    g_fp.SetName("gFPSyst")
    for i in range(n):
        g_fp.SetPoint(i, pt_c[i], central[i])
        g_fp.SetPointError(i, 0.7 * halfw[i], 0.7 * halfw[i], fp_syst, fp_syst)

    # ---- total box: outline box (unchanged) ----
    g_tot = ROOT.TGraphAsymmErrors(n)
    g_tot.SetName("gTotal")
    for i in range(n):
        g_tot.SetPoint(i, pt_c[i], central[i])
        g_tot.SetPointError(i, 0.9 * halfw[i], 0.9 * halfw[i], total[i], total[i])
    g_tot.SetFillStyle(0) 
    g_tot.SetLineColor(ROOT.kBlack)
    g_tot.SetLineWidth(3)

    # ---- canvas ----
    c = ROOT.TCanvas("c_v2_syst", "", 1920, 1536)
    c.SetLeftMargin(0.12)
    c.SetRightMargin(0.03)
    c.SetBottomMargin(0.12)
    c.SetTopMargin(0.05)

    g_tot.GetXaxis().SetLimits(-0.2, 8.2)
    g_tot.SetMinimum(-0.02)
    g_tot.SetMaximum(0.16)
    g_tot.SetTitle(";#it{p}_{T}^{D} (GeV/#it{c});#it{v}_{2}^{prompt}")

    # Drawing order: Background boxes -> Widest to Narrowest -> Outlines -> Central points
    g_tot.Draw("A2")             # axes + total box (behind)
    
    g_fd.Draw("2 same")          # FD band fill
    g_fd_out.Draw("2 same")      # FD band outline
    
    g_lm.Draw("2 same")          # LM band fill
    g_lm_out.Draw("2 same")      # LM band outline
    
    g_fit.Draw("2 same")         # fit band fill
    g_fit_out.Draw("2 same")     # fit band outline
    
    g_central.Draw("P E1 same")  # central points + stat bars

    # dashed line at y = 0
    line0 = ROOT.TLine(-0.2, 0.0, 8.2, 0.0)
    line0.SetLineStyle(2)
    line0.SetLineWidth(2)
    line0.SetLineColor(ROOT.kGray + 2)
    line0.Draw()

    leg = ROOT.TLegend(0.15, 0.72, 0.5, 0.94)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(g_central, "Prompt #it{v}_{2} (stat. unc.)", "p")
    leg.AddEntry(g_fit, "Fit syst.", "f")
    leg.AddEntry(g_lm, "LM syst.", "f")
    leg.AddEntry(g_fd, "FD syst.", "f")
    leg.AddEntry(g_tot, "Total syst.", "f")
    leg.Draw()

    c.SaveAs(OUT_PNG)
    c.SaveAs(OUT_ROOT)

    fout = ROOT.TFile.Open(OUT_ROOT, "UPDATE")
    g_central.Write("gCentral", ROOT.TObject.kOverwrite)
    g_fit.Write("gFitSyst", ROOT.TObject.kOverwrite)
    g_lm.Write("gLMSyst", ROOT.TObject.kOverwrite)
    g_fd_raw.Write("gFDSyst", ROOT.TObject.kOverwrite)
    g_fp.Write("gFPSyst", ROOT.TObject.kOverwrite)
    g_tot.Write("gTotal", ROOT.TObject.kOverwrite)
    fout.Close()

    print(f"[OK] wrote {OUT_PNG}")
    print(f"[OK] wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()