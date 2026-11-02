#!/usr/bin/env python3
"""Compare prompt v2 from 2PC and SP methods and calculate their ratio.

Reads 2PC results from previously generated v2_syst.root.
Reads SP results from v2_prompt_wsyst_d0_020.root.
Plots comparison in top pad and 2PC/SP ratio in bottom pad.
"""

import os
import math
import argparse
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

def get_graph(filename, obj_name):
    """Safely retrieve a TGraph from a ROOT file."""
    f = ROOT.TFile.Open(filename)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open {filename}")
    g = f.Get(obj_name)
    if not g:
        raise RuntimeError(f"{obj_name} not found in {filename}")
    # Clone to detach from file
    g_clone = g.Clone()
    f.Close()
    return g_clone

def calculate_ratio(g_num, g_den, g_num_syst, g_den_syst):
    """Calculate ratio (num/den) with independent error propagation."""
    n = g_num.GetN()
    g_ratio_stat = ROOT.TGraphErrors(n)
    g_ratio_syst = ROOT.TGraphAsymmErrors(n)
    
    for i in range(n):
        x = g_num.GetPointX(i)
        y_num = g_num.GetPointY(i)
        y_den = g_den.GetPointY(i)
        
        # Avoid division by zero
        if y_den == 0:
            continue
            
        ratio = y_num / y_den
        
        # Stat errors
        stat_num = g_num.GetErrorY(i)
        stat_den = g_den.GetErrorY(i)
        # Assuming independent statistical errors (conservative)
        rel_stat_num = stat_num / y_num if y_num != 0 else 0
        rel_stat_den = stat_den / y_den
        stat_ratio = abs(ratio) * math.sqrt(rel_stat_num**2 + rel_stat_den**2)
        
        g_ratio_stat.SetPoint(i, x, ratio)
        g_ratio_stat.SetPointError(i, 0, stat_ratio)
        
        # Syst errors (handling asymmetric uncertainties)
        syst_num_high = g_num_syst.GetErrorYhigh(i)
        syst_num_low = g_num_syst.GetErrorYlow(i)
        syst_den_high = g_den_syst.GetErrorYhigh(i)
        syst_den_low = g_den_syst.GetErrorYlow(i)
        
        rel_syst_num_high = syst_num_high / y_num if y_num != 0 else 0
        rel_syst_num_low = syst_num_low / y_num if y_num != 0 else 0
        rel_syst_den_high = syst_den_high / y_den
        rel_syst_den_low = syst_den_low / y_den
        
        # Cross propagation for division
        syst_ratio_high = abs(ratio) * math.sqrt(rel_syst_num_high**2 + rel_syst_den_low**2)
        syst_ratio_low = abs(ratio) * math.sqrt(rel_syst_num_low**2 + rel_syst_den_high**2)
        
        half_width = g_num_syst.GetErrorXhigh(i)
        
        g_ratio_syst.SetPoint(i, x, ratio)
        g_ratio_syst.SetPointError(i, half_width, half_width, syst_ratio_low, syst_ratio_high)
        
    return g_ratio_stat, g_ratio_syst


def main():
    FILE_2PC = "v2_syst.root"  # Output from previous script
    FILE_SP = "v2_prompt_wsyst_d0_020.root"
    OUT_PNG = "v2_compare_SP_vs_2PC.png"
    
    # Load 2PC graphs
    g_2pc_stat = get_graph(FILE_2PC, "gCentral")
    g_2pc_fit  = get_graph(FILE_2PC, "gFitSyst")
    g_2pc_lm   = get_graph(FILE_2PC, "gLMSyst")
    g_2pc_fd   = get_graph(FILE_2PC, "gFDSyst")
    g_2pc_fp   = get_graph(FILE_2PC, "gFPSyst")
    g_2pc_tot  = get_graph(FILE_2PC, "gTotal")  # bin half-width reference only
    
    # Rebuild 2PC total syst WITHOUT FD (sqrt(fit^2 + lm^2))
    n2pc = g_2pc_stat.GetN()
    g_2pc_syst = ROOT.TGraphAsymmErrors(n2pc)
    for i in range(n2pc):
        x = g_2pc_stat.GetPointX(i)
        y = g_2pc_stat.GetPointY(i)
        half_width = g_2pc_tot.GetErrorX(i)      # 0.9 * bin half-width
        fit = g_2pc_fit.GetErrorY(i)
        lm = g_2pc_lm.GetErrorY(i)
        total_no_fd = math.sqrt(fit ** 2 + lm ** 2 + 0.0001**2)
        g_2pc_syst.SetPoint(i, x, y)
        g_2pc_syst.SetPointError(i, 0.8 * half_width, 0.8 * half_width,
                                 total_no_fd, total_no_fd)
    
    # FD band drawn separately (gray, transparent, same length as total box)
    # FD error = sqrt(fd^2 + fp^2)
    g_2pc_fd_band = ROOT.TGraphAsymmErrors(n2pc)
    for i in range(n2pc):
        x = g_2pc_fd.GetPointX(i)
        y = g_2pc_fd.GetPointY(i)
        half_width = g_2pc_tot.GetErrorX(i)
        fd_low = g_2pc_fd.GetErrorYlow(i)
        fd_high = g_2pc_fd.GetErrorYhigh(i)
        fp = g_2pc_fp.GetErrorYhigh(i)
        fd_low_tot = math.sqrt(fd_low ** 2 + fp ** 2)
        fd_high_tot = math.sqrt(fd_high ** 2 + fp ** 2)
        g_2pc_fd_band.SetPoint(i, x, y)
        g_2pc_fd_band.SetPointError(i, 0.8 * half_width, 0.8 * half_width,
                                    fd_low_tot, fd_high_tot)
    
    # Load SP graphs
    g_sp_stat = get_graph(FILE_SP, "gvn_prompt_stat")
    g_sp_syst = get_graph(FILE_SP, "tot_syst")
    
    # Calculate Ratios: 2PC / SP
    g_ratio_stat, g_ratio_syst = calculate_ratio(g_2pc_stat, g_sp_stat, g_2pc_syst, g_sp_syst)
    
    # Styling - Colors
    col_2pc = ROOT.TColor.GetColor("#D55E00") # Vermillion
    col_sp = ROOT.TColor.GetColor("#56B4E9")  # Sky Blue
    alpha = 0.3
    
    # Format 2PC
    g_2pc_stat.SetMarkerStyle(ROOT.kFullCircle)
    g_2pc_stat.SetMarkerSize(3.0)
    g_2pc_stat.SetMarkerColor(col_2pc)
    g_2pc_stat.SetLineColor(col_2pc)

    # 2PC systematic band (total WITHOUT FD) styling
    g_2pc_syst.SetFillColorAlpha(col_2pc, alpha)
    g_2pc_syst.SetFillStyle(1001)
    g_2pc_syst.SetLineColor(col_2pc)
    
    # FD band styling (gray, transparent)
    g_2pc_fd_band.SetFillColorAlpha(ROOT.kGray + 1, 0.7)
    g_2pc_fd_band.SetFillStyle(1001)
    g_2pc_fd_band.SetLineColor(ROOT.kGray + 1)
    
    # Format SP
    g_sp_stat.SetMarkerStyle(ROOT.kFullSquare) # Square
    g_sp_stat.SetMarkerSize(3.0)
    g_sp_stat.SetMarkerColor(col_sp)
    g_sp_stat.SetLineColor(col_sp)

    # reformat SP systematic band
    for i in range(g_sp_syst.GetN()):
        x = g_sp_syst.GetPointX(i)
        y = g_sp_syst.GetPointY(i)
        half_width = g_sp_stat.GetErrorX(i) # x error from stat graph, syst graph is wrong
        syst_low = g_sp_syst.GetErrorYlow(i)
        syst_high = g_sp_syst.GetErrorYhigh(i)
        g_sp_syst.SetPoint(i, x, y)
        g_sp_syst.SetPointError(i, 0.8*half_width, 0.8*half_width, syst_low, syst_high)
    g_sp_syst.SetFillColorAlpha(col_sp, alpha)
    g_sp_syst.SetFillStyle(1001)
    g_sp_syst.SetLineColor(col_sp)
    
    # Format Ratio
    g_ratio_stat.SetMarkerStyle(20) # Circle
    g_ratio_stat.SetMarkerSize(2.0)
    g_ratio_stat.SetMarkerColor(ROOT.kBlack)
    g_ratio_stat.SetLineColor(ROOT.kBlack)
    
    g_ratio_syst.SetFillColorAlpha(ROOT.kGray + 2, alpha)
    g_ratio_syst.SetFillStyle(1001)
    g_ratio_syst.SetLineColor(ROOT.kGray + 2)
    
    # Setup Canvas with 2 Pads (Standard physics layout)
    c = ROOT.TCanvas("c_comp", "", 1200, 1400)
    
    pad1 = ROOT.TPad("pad1", "pad1", 0, 0.35, 1, 1.0)
    pad1.SetBottomMargin(0.0) # Join with pad2
    pad1.SetLeftMargin(0.15)
    pad1.SetRightMargin(0.05)
    pad1.SetTopMargin(0.05)
    pad1.Draw()
    
    pad2 = ROOT.TPad("pad2", "pad2", 0, 0.0, 1, 0.35)
    pad2.SetTopMargin(0.0)
    pad2.SetBottomMargin(0.3)
    pad2.SetLeftMargin(0.15)
    pad2.SetRightMargin(0.05)
    pad2.Draw()
    
    # --- Draw Pad 1 (Comparison) ---
    pad1.cd()
    
    # Dummy histogram for axes
    h_dummy = ROOT.TH1F("h_dummy", "", 10, -0.2, 8.2)
    # Explicitly set top pad Y-axis limits
    h_dummy.SetMinimum(-0.015)
    h_dummy.SetMaximum(0.14)
    
    h_dummy.GetYaxis().SetTitle("Prompt #it{v}_{2}")
    h_dummy.GetYaxis().SetTitleSize(0.06)
    h_dummy.GetYaxis().SetLabelSize(0.05)
    h_dummy.GetYaxis().SetTitleOffset(1.1)
    h_dummy.Draw("AXIS")
    
    g_2pc_syst.Draw("2 same")
    g_2pc_fd_band.Draw("2 same")
    g_sp_syst.Draw("2 same")
    g_2pc_stat.Draw("P E same") # No horizontal error bars for central points by default
    g_sp_stat.Draw("P E same")
    
    line0 = ROOT.TLine(-0.2, 0.0, 8.2, 0.0)
    line0.SetLineStyle(2)
    line0.SetLineColor(ROOT.kGray + 1)
    line0.Draw()
    
    # Main legend
    leg = ROOT.TLegend(0.18, 0.72, 0.55, 0.92)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.05)
    leg.AddEntry(g_2pc_stat, "2PC", "p")
    leg.AddEntry("", "", "")  # Empty space for the second line
    leg.AddEntry("", "", "")  # Empty space for the second line
    leg.AddEntry(g_sp_stat, "SP", "p")
    leg.Draw()
    
    # Legend for feed-down band (indented)
    # Shift X coordinates by ~0.08 to align its marker with the text above
    leg_fd = ROOT.TLegend(0.27, 0.75, 0.64, 0.92) 
    leg_fd.SetBorderSize(0)
    leg_fd.SetFillStyle(0)
    leg_fd.SetMargin(0.12)
    leg_fd.SetTextSize(0.03)
    leg_fd.AddEntry("", "", "")  # Skip the first line
    leg_fd.AddEntry(g_2pc_fd_band, "Syst. from B feed-down", "f")
    leg_fd.AddEntry("", "", "")  # Skip the third line to match row heights exactly
    leg_fd.Draw()
    
    # --- Draw Pad 2 (Ratio) ---
    pad2.cd()
    h_dummy_ratio = ROOT.TH1F("h_dummy_ratio", "", 10, -0.2, 8.2)
    # Explicitly set bottom pad Y-axis limits
    h_dummy_ratio.SetMinimum(0.25)
    h_dummy_ratio.SetMaximum(1.75)
    
    h_dummy_ratio.GetXaxis().SetTitle("#it{p}_{T}^{D} (GeV/#it{c})")
    h_dummy_ratio.GetYaxis().SetTitle("2PC / SP") # Updated label
    h_dummy_ratio.GetXaxis().SetTitleSize(0.12)
    h_dummy_ratio.GetXaxis().SetLabelSize(0.1)
    h_dummy_ratio.GetYaxis().SetTitleSize(0.12)
    h_dummy_ratio.GetYaxis().SetLabelSize(0.1)
    h_dummy_ratio.GetYaxis().SetTitleOffset(0.55)
    h_dummy_ratio.GetXaxis().SetTitleOffset(1.0)
    h_dummy_ratio.GetYaxis().SetNdivisions(505)
    h_dummy_ratio.Draw("AXIS")
    
    g_ratio_syst.Draw("2 same")
    g_ratio_stat.Draw("P E same")
    
    line1 = ROOT.TLine(-0.2, 1.0, 8.2, 1.0)
    line1.SetLineStyle(2)
    line1.SetLineColor(ROOT.kBlack)
    line1.Draw()
    
    c.SaveAs(OUT_PNG)
    print(f"[OK] Wrote comparison plot to {OUT_PNG}")
    
    # plot without ratio
    c2 = ROOT.TCanvas("c_comp_no_ratio", "", 1200, 1000)
    c2.SetLeftMargin(0.15)
    c2.SetRightMargin(0.05)
    c2.SetBottomMargin(0.12)
    c2.SetTopMargin(0.05)
    h_dummy.SetXTitle("#it{p}_{T}^{D} (GeV/#it{c})")
    h_dummy.Draw("AXIS")
    g_2pc_syst.Draw("2 same")
    g_2pc_fd_band.Draw("2 same")
    g_sp_syst.Draw("2 same")
    g_2pc_stat.Draw("P E same")
    g_sp_stat.Draw("P E same")
    line0.Draw()
    leg.Draw()
    leg_fd.Draw()
    OUT_PNG_NO_RATIO = "v2_compare_SP_vs_2PC_no_ratio.png"
    c2.SaveAs(OUT_PNG_NO_RATIO)

if __name__ == "__main__":
    main()
