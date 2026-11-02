import sys
import argparse
import ROOT
from ROOT import TFile, TSpline3, TColor, TCanvas, TH1F, TLegend, TLine
import os
import numpy as np
from array import array
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import make_dir_root_file
from load_utils import load_ese_histos
from StyleFormatter import SetObjectStyle, SetGlobalStyle
SetGlobalStyle(padleftmargin=0.15, padbottommargin=0.15,
               padrightmargin=0.15, titleoffsety=1.1, maxdigits=3, titlesizex=0.03,
               labelsizey=0.04, setoptstat=0, setopttitle=0, palette=ROOT.kGreyScale)

ROOT.gROOT.SetBatch(True)


# Quantiles colors
QUANTILE_COLORS = [
    ROOT.kBlue + 1,
    ROOT.kOrange + 7,
    ROOT.kGreen + 2,
    ROOT.kRed + 1,
    ROOT.kViolet - 3,
    ROOT.kAzure + 2,
    ROOT.kCyan + 1,
    ROOT.kOrange + 7,
    ROOT.kGreen + 2,
    ROOT.kRed + 1,
    ROOT.kViolet - 3,
    ROOT.kAzure + 2,
    ROOT.kCyan + 1,
    ROOT.kMagenta + 2,
    ROOT.kYellow + 2,
    ROOT.kTeal + 1,
    ROOT.kPink + 1,
    ROOT.kSpring + 1,
    ROOT.kOrange - 3,
    ROOT.kGray + 1,
    ROOT.kBlack,
]

DETECTOR_COLORS = [
    ROOT.kBlue + 1,
    ROOT.kOrange + 7,
    ROOT.kGreen + 2,
    ROOT.kRed + 1,
    ROOT.kViolet - 3,
    ROOT.kAzure + 2,
    ROOT.kCyan + 1,
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("an_res_file", metavar="text",
                        default="an_res.root", help="input ROOT file with anres")
    parser.add_argument("--outputdir", "-o", metavar="text",
                        default=".", help="output directory")
    parser.add_argument("--suffix", "-s", metavar="text",
                        default="", help="suffix for output files")
    args = parser.parse_args()

    histos_red_q = load_ese_histos(args.an_res_file)

    # Get the centrality range
    cent_min = histos_red_q['FT0C'].GetXaxis().GetXmin()
    cent_max = histos_red_q['FT0C'].GetXaxis().GetXmax()


    out_file_summary = TFile(f'{args.outputdir}/ese_quantiles_summary.root', 'RECREATE')

    print("\n")
    total_cent_vals = len(range(int(cent_min), int(cent_max)))
    quantiles_thresholds = list(np.arange(0.05, 1.0, 0.05))  # Quantiles from 5% to 95% in steps of 5%
    summary_histos = {}
    for detector, red_q_vs_cent_histo in histos_red_q.items():
        print(f"Processing {detector} ... ")

        summary_histos[detector] = {
            "mean_vs_cent": TH1F(f"mean_vs_cent_{detector}", f"mean_vs_cent_{detector}", total_cent_vals, cent_min, cent_max),
            "rms_vs_cent": TH1F(f"rms_vs_cent_{detector}", f"rms_vs_cent_{detector}", total_cent_vals, cent_min, cent_max),
        }
        for thr in quantiles_thresholds:
            summary_histos[detector][f"quantile_{int(thr*100)}"] = TH1F(f"quantile_{int(thr*100)}_{detector}",
                                                                            f"quantile_{int(thr*100)}_{detector}",
                                                                            total_cent_vals, cent_min, cent_max)

        for cent_val in range(int(cent_min), int(cent_max)):
            suffix = f'{detector}_{int(cent_val)}_{int(cent_val + 1)}'
            make_dir_root_file(f"cent_bins/cent_{int(cent_val)}_{int(cent_val + 1)}", out_file_summary, verbose=False)
            out_file_summary.cd(f"cent_bins/cent_{int(cent_val)}_{int(cent_val + 1)}")
            bin_cent = red_q_vs_cent_histo.GetXaxis().FindBin(cent_val + 0.5)
            red_q_histo = red_q_vs_cent_histo.ProjectionY(f'proj_{suffix}', bin_cent, bin_cent)
            red_q_histo.SetDirectory(0)
            red_q_histo.Write(f"red_q_histo_{suffix}")

            # Obtain the cumulative distribution function (CDF) from the histogram
            cdf_histo = red_q_histo.GetCumulative()
            cdf_histo.Scale(1.0 / red_q_histo.Integral())  # Normalize to 1
            cdf_histo.SetDirectory(0)
            cdf_histo.Write(f'cdf_{suffix}')

            # Smooth with spline
            cdf_spline = TSpline3(cdf_histo)
            canvas = ROOT.TCanvas(f'canvas_{suffix}', f'canvas_{suffix}', 600, 600)
            cdf_histo.Draw()
            cdf_spline.Draw('same')
            canvas.Update()
            canvas.Write()

            # Fill summary histos
            cent_bin = summary_histos[detector]["mean_vs_cent"].GetXaxis().FindBin(cent_val + 0.5)
            summary_histos[detector]["mean_vs_cent"].SetBinContent(cent_bin, red_q_histo.GetMean())
            summary_histos[detector]["mean_vs_cent"].SetBinError(cent_bin, red_q_histo.GetMeanError())
            summary_histos[detector]["rms_vs_cent"].SetBinContent(cent_bin, red_q_histo.GetRMS())
            summary_histos[detector]["rms_vs_cent"].SetBinError(cent_bin, red_q_histo.GetRMSError())

            # Convert to TF1 to obtain quantiles
            xmin = red_q_histo.GetXaxis().GetXmin()
            xmax = red_q_histo.GetXaxis().GetXmax()
            spline_func = ROOT.TF1("spline_func", lambda x, p: cdf_spline.Eval(x[0]), xmin, xmax, 0)
            spline_func.SetNpx(1000)  # Set the number of points for evaluation
            spline_func.Write(f"spline_func_{suffix}")
            red_q_quantiles = [spline_func.GetX(quantile) for quantile in quantiles_thresholds]
            for thr, quantile in zip(red_q_quantiles, quantiles_thresholds):
                summary_histos[detector][f"quantile_{int(quantile*100)}"].SetBinContent(cent_bin, thr)
                summary_histos[detector][f"quantile_{int(quantile*100)}"].SetBinError(cent_bin, 0)

            # Canvas with threshold lines and red_q_histo
            canvas_quantiles = TCanvas(f"c_quantiles_{detector}_{cent_val}", f"c_quantiles_{detector}_{cent_val}", 600, 600)
            last_x_bin_not_empty = red_q_histo.FindLastBinAbove(1)
            red_q_histo.SetTitle("")
            red_q_histo.GetXaxis().SetTitle(f"Reduced Q vector {detector}")
            red_q_histo.GetYaxis().SetTitle("Counts")
            red_q_histo.Draw()
            lines = []
            legend_quantiles = TLegend(0.6, 0.7, 0.9, 0.9)
            for i_quant, (thr, quantile) in enumerate(zip(red_q_quantiles, quantiles_thresholds)):
                line = TLine(thr, 0, thr, red_q_histo.GetMaximum())
                line.SetLineColor(QUANTILE_COLORS[i_quant])
                line.SetLineStyle(2)
                line.SetLineWidth(2)
                line.Draw("pe same")
                legend_quantiles.AddEntry(line, f"Quantile {int(quantile*100)}%", "l") 
                lines.append(line)
            legend_quantiles.Draw()
            canvas_quantiles.Update()
            canvas_quantiles.Write()

        make_dir_root_file(detector, out_file_summary, verbose=False)
        out_file_summary.cd(detector)
        for histo_name, histo in summary_histos[detector].items():
            histo.SetDirectory(0)
            histo.Write()

    print("\n")

    # Summary plots
    out_file_summary.cd()
    first_detector = list(summary_histos.keys())[0]
    for histo_name in summary_histos[first_detector].keys():
        canvas = TCanvas(f'c_{histo_name}', f'c_{histo_name}', 600, 600)
        legend = TLegend(0.5, 0.7, 0.7, 0.9)

        # Calculate min/max for the current histo_name across all detectors
        current_histos = [histos_dict[histo_name] for histos_dict in summary_histos.values() if histo_name in histos_dict]
        max_y = max(h.GetMaximum() for h in current_histos)
        min_y = min(h.GetMinimum() for h in current_histos)

        # Create the dedicated axis frame
        y_max_margin = max_y + (max_y - min_y) * 0.1
        y_min_margin = min_y - (max_y - min_y) * 0.1
        frame = canvas.DrawFrame(float(cent_min), y_min_margin, float(cent_max), y_max_margin)
        frame.GetXaxis().SetTitle("Centrality (%)")
        frame.GetYaxis().SetTitle(f"Threshold highest {histo_name.split('_')[-1]}% |q_{{2}}|")

        for i_det, (detector, histos_dict) in enumerate(summary_histos.items()):
            histo = histos_dict[histo_name]
            histo.SetLineColor(DETECTOR_COLORS[i_det])
            histo.SetMarkerColor(DETECTOR_COLORS[i_det])
            histo.SetMarkerStyle(20)
            histo.SetMarkerSize(1)
            histo.SetLineWidth(2)
            histo.Draw("pe same")
            legend.AddEntry(histo, detector, "lp")

        legend.Draw()
        canvas.Update()
        canvas.Write()


    # All quantiles for each detector in one canvas
    for detector, histos_dict in summary_histos.items():
        canvas = TCanvas(f'c_quantiles_{detector}', f'c_quantiles_{detector}', 600, 600)
        legend = TLegend(0.5, 0.5, 0.7, 0.9)

        # Calculate min/max for the quantile histograms of the current detector
        quantile_histos = [h for name, h in histos_dict.items() if name.startswith("quantile_")]
        max_y = max(h.GetMaximum() for h in quantile_histos)
        min_y = min(h.GetMinimum() for h in quantile_histos)

        # Create the dedicated axis frame
        y_max_margin = max_y + (max_y - min_y) * 0.1
        y_min_margin = min_y - (max_y - min_y) * 0.1
        frame = canvas.DrawFrame(float(cent_min), y_min_margin, float(cent_max), y_max_margin)
        frame.GetXaxis().SetTitle("Centrality (%)")
        frame.GetYaxis().SetTitle(f"Reduced Q vector {detector}")

        for i_quant, (name, histo) in enumerate(histos_dict.items()):
            if name.startswith("quantile_"):
                histo.SetLineColor(QUANTILE_COLORS[i_quant])
                histo.SetMarkerColor(QUANTILE_COLORS[i_quant])
                histo.SetMarkerStyle(20)
                histo.SetMarkerSize(1)
                histo.SetLineWidth(2)
                histo.Draw("pe same")
                legend.AddEntry(histo, name.replace('_', ' ').replace(' thr', '').title(), "lp")

        legend.Draw()
        canvas.Update()
        canvas.Write()

    out_file_summary.Close()
    print(f"Output file with ESE percentiles summary: {out_file_summary.GetName()}")
