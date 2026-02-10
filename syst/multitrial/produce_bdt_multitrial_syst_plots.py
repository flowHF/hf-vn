import sys
import os
import numpy as np
import argparse
import re
import glob
import array
import pandas as pd
import yaml
import ast
import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kError  # Only show errors and above
from ROOT import TFile, TCanvas, TH1F, TGraph, TLine, TBox, TGraphAsymmErrors, TLegend, kOrange, kAzure, kBlue, kBlack, kRed, gStyle, gPad
script_dir = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(os.path.join(script_dir, '../..', 'utils'))
from utils import logger
from StyleFormatter import SetGlobalStyle, SetObjectStyle
from PyPDF2 import PdfMerger
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SetGlobalStyle(titleoffsety=1.1, maxdigits=3, topmargin=0.1, bottommargin=0.4, leftmargin=0.3, rightmargin=0.15,
               labelsizey=0.04, setoptstat=0, setopttitle=0, setdecimals=True,titleoffsetx=0.74)

# ------------------ ROOT Styling ------------------
def set_root_style(line_width=3, title_size=0.05, label_size=0.045, tick_length=0.03):
    gStyle.SetOptStat(0)
    gStyle.SetLineWidth(line_width)
    gStyle.SetPadTickX(1)
    gStyle.SetPadTickY(1)
    gStyle.SetTitleSize(title_size, "XYZ")
    gStyle.SetLabelSize(label_size, "XYZ")
    gStyle.SetTickLength(tick_length, "XYZ")


config = {
    'ry': {
        'filename_pattern': 'raw_yields_00.root',
        'histo_name': 'hRawYieldsSimFit',
        'y_label': 'Raw Yield',
    },
    'eff_prompt': {
        'filename_pattern': 'eff_00.root',
        'histo_name': 'hEffPrompt',
        'has_uncertainty': True,
        'y_label': 'Acc#times#font[152]{e} Prompt',
    },
    'eff_fd': {
        'filename_pattern': 'eff_00.root',
        'histo_name': 'hEffFD',
        'has_uncertainty': True,
        'y_label': 'Acc#times#font[152]{e} FD',
    },
    'vn': {
        'filename_pattern': 'raw_yields_00.root',
        'histo_name': 'hVnSimFit',
        'has_uncertainty': True,
        'y_label': 'Inclusive#kern[-0.5]{ }#it{v}_{n}',
    },
    'signif': {
        'filename_pattern': 'raw_yields_00.root',
        'histo_name': 'hRawYieldsSignificanceSimFit',
        'has_uncertainty': True,
        'y_label': 'Significance',
    },
    's_over_b': {
        'filename_pattern': 'raw_yields_00.root',
        'histo_name': 'hRawYieldsSoverBSimFit',
        'y_label': 'S/B',
    },
    'vn_bkg_coeff_0': {
        'filename_pattern': 'raw_yields_00.root',
        'histo_name': 'hVnBkgCoeff0',
        'has_uncertainty': True,
        'inside_pt_subdir': True,
        'y_label': '#it{v}_{n} bkg coeff 0',
    },
    'vn_bkg_coeff_1': {
        'filename_pattern': 'raw_yields_00.root',
        'histo_name': 'hVnBkgCoeff1',
        'has_uncertainty': True,
        'inside_pt_subdir': True,
        'y_label': '#it{v}_{n} bkg coeff 1',
    },
}

SUMMARY_FIGURE = [
    "vn",
    "vn_unc",              # derived
    "signif",
    "vn_bkg_coeff_0",
    # "s_over_b",
    "vn_bkg_coeff_1",
    "ry",
]

def produce_multitrial_syst_bdt_plots(default_cfg, results_dir):

    ry_files = [f for f in os.listdir(f"{results_dir}/raw_yields/") if re.match(r"raw_yields_\d+\.root$", f)]
    cutsets = sorted((re.search(r"raw_yields_(\d+)\.root", f).group(1) for f in ry_files), key=int)

    ry_cutsets_dict = {
        cutset: TFile.Open(os.path.join(f"{results_dir}/raw_yields/", f"raw_yields_{cutset}.root"), "READ")
        for cutset in cutsets
    }

    eff_files = [f for f in os.listdir(f"{results_dir}/effs/") if re.match(r"eff_\d+\.root$", f)]
    eff_cutsets_dict = {
        cutset: TFile.Open(os.path.join(f"{results_dir}/effs/", f"eff_{cutset}.root"), "READ")
        for cutset in sorted((re.search(r"eff_(\d+)\.root", f).group(1) for f in eff_files), key=int)
    }

    # List all pt-bins for which multitrial results are available
    multitrial_pt_dirs = [pt_dir for pt_dir in os.listdir(f"{results_dir}/syst/multitrial/bdt") \
                          if os.path.isdir(os.path.join(results_dir, 'syst/multitrial/bdt', pt_dir)) \
                          and pt_dir.startswith('pt_')]

    os.makedirs(f"{results_dir}/syst/multitrial/bdt/summary", exist_ok=True)
    out_file = TFile.Open(f"{results_dir}/syst/multitrial/bdt/summary/BkgScanSummary.root", "RECREATE")
    # Sort the pt dirs in increasing order
    multitrial_pt_dirs = sorted(multitrial_pt_dirs, key=lambda x: float(re.search(r"pt_([0-9]+)_([0-9]+)", x).group(1)))
    for mult_dir in multitrial_pt_dirs:
        out_file.mkdir(mult_dir)
        out_file.cd(mult_dir)
        _, pt_min_times_10, pt_max_times_10 = mult_dir.split('_')
        pt_min = float(pt_min_times_10) / 10.
        pt_max = float(pt_max_times_10) / 10.

        # Get pt bin index
        pt_bin_ref = None
        pt_center = (pt_min + pt_max) / 2.

        # Summary histogram for BDT variations

        h_syst_bdt = TH1F("h_syst_bdt", "h_syst_bdt;Cutset;#Delta#it{v}_{n} (BDT variations)", len(cutsets), 0.5, len(cutsets) + 0.5)
        h_shift_bdt = TH1F("h_shift_bdt", "h_shift_bdt;Cutset;#Delta#it{v}_{n} (BDT variations)", len(cutsets), 0.5, len(cutsets) + 0.5)
        vals = {}

        pt_trial_dir = f"{results_dir}/syst/multitrial/bdt/{mult_dir}"
        for i_cutset, ((cutset_suffix, ry_cutset_file), (_, eff_cutset_file)) in enumerate(zip(ry_cutsets_dict.items(), eff_cutsets_dict.items())):
            bkg_vals = []
            bkg_strs = []

            # Check if the directory exists
            if not os.path.isdir(f"{pt_trial_dir}/trials_cutset_{i_cutset}/"):
                logger(f"Directory {pt_trial_dir}/trials_cutset_{i_cutset}/ does not exist. Skipping cutset {cutset_suffix}.", "WARNING")
                continue

            for folder in os.listdir(f"{pt_trial_dir}/trials_cutset_{i_cutset}/"):
                m = re.search(r"bkg_([0-9]*\.?[0-9]+)", folder)
                if m:
                    bkg_vals.append(float(m.group(1)))
                    bkg_strs.append(m.group(1))
                    
            # Now sort
            bkg_vals, bkg_strs = zip(*sorted(zip(bkg_vals, bkg_strs)))
            bkg_vals = sorted([float(re.search(r"bkg_([0-9]*\.?[0-9]+)", os.path.basename(folder)).group(1)) for folder in \
                            os.listdir(f"{pt_trial_dir}/trials_cutset_{i_cutset}/") \
                            if os.path.isdir(os.path.join(f"{pt_trial_dir}/trials_cutset_{i_cutset}/", folder))])
            step = bkg_vals[1] - bkg_vals[0]
            bin_edges = array.array("d", [bkg_vals[0] - step] + bkg_vals)
            summary_histos = {}
            vals[cutset_suffix] = {}
            h_syst_bdt.GetXaxis().SetBinLabel(int(cutset_suffix)+1, cutset_suffix)
            h_shift_bdt.GetXaxis().SetBinLabel(int(cutset_suffix)+1, cutset_suffix)
            hist_vn = ry_cutset_file.Get('hVnSimFit')
            for i_bin in range(1, hist_vn.GetNbinsX()+1):
                pt_low_edge = hist_vn.GetBinLowEdge(i_bin)
                pt_up_edge = pt_low_edge + hist_vn.GetBinWidth(i_bin)
                if pt_low_edge <= pt_center < pt_up_edge:
                    pt_bin_ref = i_bin
                    break

            # Get reference bkg cut values
            bkg_score_ref = default_cfg['cut_variation']['uncorr_bdt_cut']['bkg_max'][pt_bin_ref-1][i_cutset]

            out_file.mkdir(f"{mult_dir}/cutset_{i_cutset}")
            out_file.cd(f"{mult_dir}/cutset_{i_cutset}")
            for variable, setting in config.items():
                summary_histos[variable] = TH1F(
                    f"hist_{variable}",
                    f"hist_{variable};Bkg Score Cut (<);{setting['y_label']}",
                    len(bin_edges) - 1,
                    bin_edges
                )
                if setting.get('has_uncertainty', False):
                    summary_histos[variable + "_unc"] = TH1F(
                        f"hist_{variable}_unc",
                        f"hist_{variable}_unc;Bkg Score Cut (<);{setting['y_label']} Uncertainty",
                        len(bin_edges) - 1,
                        bin_edges
                    )

                histo_name = f"pt_{pt_min_times_10}_{pt_max_times_10}/{setting['histo_name']}" if setting.get('inside_pt_subdir', False) else setting['histo_name']
                hist = ry_cutset_file.Get(histo_name) if 'raw_yields' in setting['filename_pattern'] else eff_cutset_file.Get(histo_name)
                print(f"ry_cutset_file.GetName(): {ry_cutset_file.GetName()}, trying to get histo {histo_name} for variable {variable}...")
                vals[cutset_suffix][variable] = {}
                ref_val = hist.GetBinContent(pt_bin_ref)
                ref_unc = hist.GetBinError(pt_bin_ref) if setting.get('has_uncertainty', False) else 0.0
                vals[cutset_suffix][variable]['ref_val'] = ref_val
                vals[cutset_suffix][variable]['ref_unc'] = ref_unc
                vals[cutset_suffix][variable]['trials'] = []
                vals[cutset_suffix][variable]['trials_unc'] = []
                file_type = 'raw_yields' if 'raw_yields' in setting['filename_pattern'] else 'effs'
                trials_files = glob.glob(f"{pt_trial_dir}/trials_cutset_{i_cutset}/bkg_*/{file_type}/{setting['filename_pattern']}")
                trials_files = sorted(trials_files, key=lambda x: float(re.search(r"bkg_([0-9]*\.?[0-9]+)", x).group(1)))

                for i_bkg, (bkg_val, bkg_str) in enumerate(zip(bkg_vals, bkg_strs)):
                    file_path = f"{pt_trial_dir}/trials_cutset_{i_cutset}/bkg_{bkg_str}/{file_type}/{setting['filename_pattern']}"
                    try:
                        file = TFile.Open(file_path, "READ")
                        hist = file.Get(histo_name)
                        val = hist.GetBinContent(1)
                        unc = hist.GetBinError(1) if setting.get('has_uncertainty', False) else 0.0
                        file.Close()
                    except Exception as e:
                        print(f"Error opening file {file_path}: {e}")
                        val = 0.0
                        unc = 0.0
                    vals[cutset_suffix][variable]['trials'].append(val)
                    vals[cutset_suffix][variable]['trials_unc'].append(unc)
                    summary_histos[variable].SetBinContent(i_bkg + 1, val)
                    summary_histos[variable].SetBinError(i_bkg + 1, unc)
                    if setting.get('has_uncertainty', False):
                        summary_histos[variable + "_unc"].SetBinContent(i_bkg + 1, unc)

                out_file.cd(f"{mult_dir}/cutset_{i_cutset}")
                summary_histos[variable].Write()
                if setting.get('has_uncertainty', False):
                    summary_histos[variable + "_unc"].Write()

                bkg_scan_figure(bkg_score_ref, bkg_vals, vals[cutset_suffix][variable]['trials'], vals[cutset_suffix][variable]['trials_unc'], \
                                vals[cutset_suffix][variable]['ref_val'], vals[cutset_suffix][variable]['ref_unc'], cutset_suffix, \
                                pt_min, pt_max, f"{pt_trial_dir}/{variable}_cutset_{cutset_suffix}.pdf",
                                f";Bkg score cut (<);{setting['y_label']}"
                               )

            panels = []

            for key in SUMMARY_FIGURE:

                if key == "vn_unc":
                    panels.append(dict(
                        trials_vals = vals[cutset_suffix]["vn"]["trials_unc"],
                        trials_uncs = [0.0] * len(bkg_vals),
                        val_ref     = vals[cutset_suffix]["vn"]["ref_unc"],
                        stat_unc_ref= 0.0,
                        labels      = ";Bkg score cut (<);v_{n} uncertainty"
                    ))
                    continue

                setting = config[key]
                panels.append(dict(
                    trials_vals  = vals[cutset_suffix][key]["trials"],
                    trials_uncs  = vals[cutset_suffix][key]["trials_unc"],
                    val_ref      = vals[cutset_suffix][key]["ref_val"],
                    stat_unc_ref = vals[cutset_suffix][key]["ref_unc"],
                    labels       = f";Bkg score cut (<);{setting['y_label']}"
                ))

            bkg_scan_summary(
                bkg_score_ref,
                bkg_vals,
                panels,
                cutset_suffix,
                pt_min,
                pt_max,
                f"{pt_trial_dir}/SummaryBdtBkgScan_cutset_{cutset_suffix}.pdf"
            )

        # Merge the Summary PDFs into a single file
        pdf_merger = PdfMerger()
        summary_pdfs = glob.glob(f"{pt_trial_dir}/SummaryBdtBkgScan_cutset_*.pdf")
        for pdf in sorted(summary_pdfs, key=lambda x: int(re.search(r"SummaryBdtBkgScan_cutset_(\d+)\.pdf", x).group(1))):
            pdf_merger.append(pdf)
        pdf_merger.write(f"{results_dir}/syst/multitrial/bdt/summary/BkgScanSummary_{mult_dir}.pdf")

    out_file.Close()


def bkg_scan_summary(bkg_score_ref, bkg_scores, panels, cutset_suffix, pt_min, pt_max, out_file_path):
    set_root_style()

    c = ROOT.TCanvas("c_summary", "Summary 3x2", 1200, 800)
    c.Divide(3, 2)

    drawn_objects = []   # <<< CRITICAL LINE

    step = bkg_scores[1] - bkg_scores[0]

    for i, panel in enumerate(panels, start=1):
        pad = c.cd(i)

        pad.SetTicks()
        pad.SetLeftMargin(0.18)
        pad.SetBottomMargin(0.18)
        pad.SetRightMargin(0.05)
        pad.SetTopMargin(0.05)

        trials_vals  = panel["trials_vals"]
        trials_uncs  = panel["trials_uncs"]
        val_ref      = panel["val_ref"]
        stat_unc_ref = panel["stat_unc_ref"]
        labels       = panel["labels"]

        min_val = min([v - u for v, u in zip(trials_vals, trials_uncs)] +
                      [val_ref - stat_unc_ref])
        max_val = max([v + u for v, u in zip(trials_vals, trials_uncs)] +
                      [val_ref + stat_unc_ref])
        y_margin = 0.25 * (max_val - min_val)

        frame = pad.DrawFrame(
            bkg_scores[0] - step,
            min_val - y_margin,
            bkg_scores[-1] + step,
            max_val + y_margin,
            labels
        )
        drawn_objects.append(frame)

        # Axes formatting
        frame.SetLineWidth(1)  # thickness of axes lines
        frame.GetXaxis().SetLabelSize(0.045)
        frame.GetYaxis().SetLabelSize(0.045)
        frame.GetXaxis().SetTitleOffset(1.1)  # space between title and tick labels
        frame.GetYaxis().SetTitleOffset(1.4)
        frame.GetXaxis().SetTickLength(0.03)
        frame.GetYaxis().SetTickLength(0.03)

        line = ROOT.TLine(
            bkg_scores[0] - step,
            val_ref,
            bkg_scores[-1] + step,
            val_ref
        )
        line.SetLineStyle(9)
        line.SetLineColor(ROOT.kRed)
        line.SetLineWidth(2)
        line.Draw("SAME")
        drawn_objects.append(line)

        # Reference value line
        ymin = pad.GetUymin()
        ymax = pad.GetUymax()

        if i == 2:
            y2 = ymin + 0.6 * (ymax - ymin)
        else:
            y2 = ymax

        bkg_ref_line = ROOT.TLine(
            bkg_score_ref,
            ymin,
            bkg_score_ref,
            y2
        )
        bkg_ref_line.SetLineStyle(9)
        bkg_ref_line.SetLineColor(ROOT.kRed)
        bkg_ref_line.SetLineWidth(2)
        bkg_ref_line.Draw("SAME")
        drawn_objects.append(bkg_ref_line)

        g = ROOT.TGraphAsymmErrors(len(bkg_scores))
        for j, (x, v, u) in enumerate(zip(bkg_scores, trials_vals, trials_uncs)):
            g.SetPoint(j, x, v)
            g.SetPointError(j, 0.5 * step, 0.5 * step, u, u)

        g.SetMarkerStyle(21)
        g.SetMarkerSize(1.1)
        g.SetMarkerColor(kBlue) # kBlack)
        g.SetLineColor(kBlue) # kBlack)
        g.SetLineWidth(2)
        g.Draw("P SAME")
        drawn_objects.append(g)


        if stat_unc_ref > 0:
            box = ROOT.TBox(
                bkg_scores[0] - step,
                val_ref - stat_unc_ref,
                bkg_scores[-1] + step,
                val_ref + stat_unc_ref
            )
            box.SetFillColorAlpha(ROOT.kRed, 0.35)
            box.SetLineColor(0)
            box.Draw("SAME")
            drawn_objects.append(box)

        if i == 2:
            leg = ROOT.TLegend(0.30, 0.70, 0.55, 0.90)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.SetTextSize(0.04)
            leg.SetHeader(
                f"Cutset {cutset_suffix}, "
                f"{pt_min} < p_{{T}} < {pt_max} GeV/#it{{c}}"
            )
            leg.AddEntry(g, "Trials", "lp")
            leg.AddEntry(line, "Reference", "l")
            leg.AddEntry(box, "Stat. Unc.", "f")
            leg.Draw()
            drawn_objects.append(leg)

    c.SaveAs(out_file_path)


# ------------------ Vn vs Bkg Score cut Figure ------------------
def bkg_scan_figure(bkg_score_ref, bkg_scores, trials_vals, trials_uncs, val_ref, stat_unc_ref, i_cutset, pt_min, pt_max, out_file_path, labels):
    set_root_style()
    step = bkg_scores[1] - bkg_scores[0]
    c = TCanvas(f"c_{labels}", f"c_{labels}", 800, 800)
    c.SetLeftMargin(0.20)
    c.SetBottomMargin(0.14)
    c.SetRightMargin(0.04)
    c.SetTopMargin(0.05)

    # Set y-range with margin
    min_val = min([val - unc for val, unc in zip(trials_vals, trials_uncs)] + [val_ref - stat_unc_ref])
    max_val = max([val + unc for val, unc in zip(trials_vals, trials_uncs)] + [val_ref + stat_unc_ref])
    y_margin = 1.5 * (max_val - min_val)

    frame = gPad.DrawFrame(
        bkg_scores[0] - step,
        min_val - y_margin,
        bkg_scores[-1] + step,
        max_val + y_margin,
        labels
    )

    # Graph markers
    g_markers = TGraphAsymmErrors(len(bkg_scores))
    for i, (bkg_score, val, unc) in enumerate(zip(bkg_scores, trials_vals, trials_uncs)):
        g_markers.SetPoint(i, bkg_score, val)
        # Set point errors
        g_markers.SetPointError(i, 0.5 * step, 0.5 * step, unc, unc)

    g_markers.SetMarkerStyle(21)
    g_markers.SetMarkerSize(1.2)
    g_markers.SetMarkerColor(kBlack)
    g_markers.SetLineColor(kBlack)
    g_markers.SetLineWidth(2)
    g_markers.Draw("P SAME")

    # Reference value line
    mean_line = TLine(bkg_scores[0] - step, val_ref, bkg_scores[-1] + step, val_ref)
    mean_line.SetLineStyle(9)
    mean_line.SetLineColor(kRed)
    mean_line.SetLineWidth(2)
    mean_line.Draw("SAME")

    # Reference value line
    ymin = gPad.GetUymin()
    ymax = gPad.GetUymax()
    bkg_ref_line = ROOT.TLine(bkg_score_ref, ymin, bkg_score_ref, ymin + 0.65 * (ymax - ymin))
    bkg_ref_line.SetLineStyle(9)
    bkg_ref_line.SetLineColor(ROOT.kRed)
    bkg_ref_line.SetLineWidth(2)
    bkg_ref_line.Draw("SAME")

    if stat_unc_ref > 0:
        # Syst. uncertainty band
        box = TBox(bkg_scores[0] - step, val_ref - stat_unc_ref, bkg_scores[-1] + step, val_ref + stat_unc_ref)
        box.SetFillColorAlpha(kRed, 0.35)  # 35% transparent red
        box.SetLineColor(0)                 # no border
        box.Draw("SAME")

    # ---- axis styling ----
    xaxis = frame.GetXaxis()
    yaxis = frame.GetYaxis()
    xaxis.SetTitleSize(0.05)
    yaxis.SetTitleSize(0.05)
    xaxis.SetLabelSize(0.045)
    yaxis.SetLabelSize(0.045)
    xaxis.SetTitleOffset(1.2)
    yaxis.SetTitleOffset(1.5)
    xaxis.CenterTitle()
    yaxis.CenterTitle()

    # Legend
    leg = TLegend(0.3, 0.69 if stat_unc_ref > 0 else 0.76, 0.5, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.04)
    leg.SetHeader(f"Cutset {i_cutset}, {pt_min}#kern[-0.35]{{ }}<#kern[-0.35]{{ }}#it{{p}}_{{T}}#kern[-0.35]{{ }}<#kern[-0.35]{{ }}{pt_max} GeV/#it{{c}}")
    leg.AddEntry(g_markers, "Trials", "lp")
    leg.AddEntry(mean_line, "Reference", "l")
    if stat_unc_ref > 0:
        leg.AddEntry(box, "Stat. Unc.", "f")
    leg.Draw()

    c.SaveAs(out_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('default_cfg', metavar='text', default='path to the reference yaml config')
    parser.add_argument('results_dir', metavar='text', default='path to the directory containing the .root files from multitrial')
    args = parser.parse_args()

    # Open reference config
    with open(args.default_cfg, 'r') as CfgFlow:
        cfg_ref = yaml.safe_load(CfgFlow)

    logger("Producing multitrial systematic plots for BDT variations...", "INFO")
    produce_multitrial_syst_bdt_plots(cfg_ref, args.results_dir)
