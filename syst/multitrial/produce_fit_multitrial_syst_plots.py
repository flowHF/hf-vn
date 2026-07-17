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
import traceback
import ROOT
ROOT.gErrorIgnoreLevel = ROOT.kError  # Only show errors and above
from ROOT import TFile, TCanvas, TH1F, TGraph, TLine, TBox, TGraphAsymmErrors, TLegend, kOrange, kAzure, kBlack, kRed, gStyle, gPad
script_dir = os.path.dirname(os.path.realpath(__file__))
os.sys.path.append(os.path.join(script_dir, '../..', 'utils'))
from utils import logger, make_dir_root_file
from StyleFormatter import SetGlobalStyle, SetObjectStyle
from PyPDF2 import PdfMerger
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SetGlobalStyle(titleoffsety=1.1, maxdigits=3, topmargin=0.1, bottommargin=0.4, leftmargin=0.3, rightmargin=0.15,
               labelsizey=0.04, setoptstat=0, setopttitle=0, setdecimals=True,titleoffsetx=0.74)

# Multitrial results
multitrial_results_cols = [
    "Significance",
    "SignificanceUnc",
    "Chi2",
    "Chi2Unc",
    "V2",
    "V2Unc",
]

# Multitrial configurations
multitrial_config_cols = [
    "BkgFunc",
    "BkgFuncVn",
    "SgnFunc",
    "Rebin",
    "VnVsMassBinWidths",
    "MassFitRanges",
    "TrialIdx"
]

# Global matplotlib style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13
})
colors = plt.cm.tab10.colors


def study_variations(sel_trials_df, sel_reference_df, v2_type, varied_var, pt_label, output_dir, pdf):
    """
    Produce 2-panel figures:
      - Left: individual trials with jitter
      - Right: binned distributions + means

    One PDF per cutset and per Prompt / NonPrompt.
    Each PDF contains one page per column in unique_cols.
    """
    pt_min, pt_max = get_pt_label_range(pt_label)

    jitter_val = 0.45

    # Extract unique category values
    unique_settings = sorted(sel_trials_df[varied_var].unique())
    labels = unique_settings
    # Setup labels for mass range
    if varied_var == "MassFitRanges":
        labels = [g.replace(", ", "- ").replace('[', '').replace(']', '') for g in unique_settings]

    x_pos = np.arange(len(unique_settings))

    fig = plt.figure(figsize=(11, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

    ax_scatter = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1])

    # ------------------ TITLE ------------------
    fig.suptitle(
        rf"{varied_var} variations, {v2_type} $v_2$, "
        rf"${pt_min} < p_T < {pt_max}$",
        y=0.96
    )

    # ================== SCATTER ==================
    for i, g in enumerate(unique_settings):
        df_setting = sel_trials_df[sel_trials_df[varied_var] == g]
        jitter = np.random.uniform(-jitter_val, jitter_val, len(df_setting))
        ax_scatter.errorbar(x_pos[i] + jitter,
            df_setting["V2"], yerr=df_setting["V2Unc"],
            fmt="o", color=colors[i % len(colors)],
            alpha=0.85, markersize=6, zorder=3, label=labels[i]
        )

    ax_scatter.set_xticks([])
    ax_scatter.set_xlim(-0.6, len(unique_settings) - 0.4)
    ax_scatter.set_xlabel(varied_var, labelpad=12)
    ax_scatter.set_ylabel(r"$v_2$")
    ax_scatter.grid(True, linestyle="--", alpha=0.25)
    ax_scatter.set_box_aspect(1)
    ax_scatter.legend(ncol=2, frameon=False)

    # ---------- SINGLE REFERENCE (ON TOP) ----------
    ref_val = sel_reference_df["V2"].iloc[0]
    ref_unc = sel_reference_df["V2Unc"].iloc[0]

    xmin, xmax = ax_scatter.get_xlim()
    ax_scatter.fill_between(
        [xmin, xmax], ref_val - ref_unc, ref_val + ref_unc,
        color="black", alpha=0.12, zorder=4)
    ax_scatter.axhline(
        ref_val, color="black",
        linestyle="--", linewidth=2.3, zorder=5
    )
    # ================== HISTOGRAM ==================
    vmin = sel_trials_df["V2"].min() - 0.02
    vmax = sel_trials_df["V2"].max() + 0.02
    bins = np.linspace(vmin, vmax, 120)

    for i, g in enumerate(unique_settings):
        df_setting = sel_trials_df[sel_trials_df[varied_var] == g]

        ax_hist.hist(
            df_setting["V2"], bins=bins, color=colors[i % len(colors)],
            alpha=0.55, edgecolor="black", linewidth=0.8
        )

        ax_hist.axvline(
            df_setting["V2"].mean(), color=colors[i % len(colors)],
            linestyle="--", linewidth=2, zorder=3
        )

        ax_hist.set_xlabel(r"$v_2$")
        ax_hist.set_ylabel("Counts")
        ax_hist.grid(True, linestyle="--", alpha=0.25)
        ax_hist.set_box_aspect(1)

        # ---------- SINGLE HISTO REFERENCE ----------
        if not sel_reference_df.empty:
            ymin, ymax = ax_hist.get_ylim()
            ax_hist.fill_betweenx(
                [ymin, ymax], ref_val - ref_unc, ref_val + ref_unc,
                color="black", alpha=0.12, zorder=4
            )
            ax_hist.axvline(
                ref_val, color="black", linestyle="--",
                linewidth=2.3, zorder=5
            )

        # ---------- LAYOUT CONTROL ----------
        fig.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.98, wspace=0.25)

        pdf.savefig(fig)
        plt.close(fig)


def study_syst_unc(trials_df, reference_df, v2_type, pt_label, output_dir):
    print(f"Entered study_syst_unc for {v2_type} in pt bin {pt_label}")
    pt_min, pt_max = get_pt_label_range(pt_label)
    reference_vn = reference_df["V2"].iloc[0]
    reference_vn_unc = reference_df["V2Unc"].iloc[0]

    n_trials = len(trials_df)-1
    max_trial_idx = int(trials_df["TrialIdx"].astype(int).max())
    min_trial_idx = int(trials_df["TrialIdx"].astype(int).min())

    h_vn_vs_trial_indexed = TH1F('h_vn_vs_trial_indexed', ';trial;#it{v}_{n}', max_trial_idx+1, -0.5, max_trial_idx+0.5)
    h_vn_vs_trial_indexed.GetXaxis().SetTitleSize(0.05)
    h_vn_vs_trial_indexed.GetXaxis().SetTitleOffset(0.85)
    h_vn_vs_trial_indexed.GetYaxis().SetTitleSize(0.05)
    h_vn_vs_trial_indexed.GetYaxis().SetTitleOffset(1.01)
    h_vn_vs_trial_indexed.SetStats(False)
    SetObjectStyle(h_vn_vs_trial_indexed, markerstyle=20, markercolor=kBlack, markersize=1., linecolor=kBlack)

    h_vn_vs_trial = TH1F('h_vn_vs_trial', ';trial;#it{v}_{n}', n_trials+1, -0.5, n_trials+0.5)
    h_vn_vs_trial.GetXaxis().SetTitleSize(0.05)
    h_vn_vs_trial.GetXaxis().SetTitleOffset(0.85)
    h_vn_vs_trial.GetYaxis().SetTitleSize(0.05)
    h_vn_vs_trial.GetYaxis().SetTitleOffset(1.01)
    h_vn_vs_trial.SetStats(False)
    x_low_lim = h_vn_vs_trial_indexed.GetXaxis().GetXmin()
    x_upp_lim = h_vn_vs_trial_indexed.GetXaxis().GetXmax()
    SetObjectStyle(h_vn_vs_trial, markerstyle=20, markercolor=kBlack, markersize=1., linecolor=kBlack)

    h_syst = TH1F('h_syst', ';#it{v}_{n}(trial) - #it{v}_{n}(ref.);Counts', 1000, -0.05, 0.05)
    h_syst.GetXaxis().SetTitleSize(0.05)
    h_syst.GetXaxis().SetTitleOffset(0.85)
    h_syst.GetYaxis().SetTitleSize(0.05)
    h_syst.GetYaxis().SetTitleOffset(1.01)
    h_syst.SetStats(False)
    SetObjectStyle(h_syst, markerstyle=20, markercolor=kBlack, markersize=1., linecolor=kBlack)

    if "Cutset" in v2_type:
        h_signif_vs_trial = TH1F('h_signif_vs_trial', ';trial;significance', n_trials+1, -0.5, n_trials+0.5)
        h_signif_vs_trial.GetXaxis().SetTitleSize(0.05)
        h_signif_vs_trial.GetXaxis().SetTitleOffset(0.85)
        h_signif_vs_trial.GetYaxis().SetTitleSize(0.05)
        h_signif_vs_trial.GetYaxis().SetTitleOffset(1.01)
        h_signif_vs_trial.SetStats(False)
        SetObjectStyle(h_signif_vs_trial, markerstyle=20, markercolor=kBlack, markersize=1., linecolor=kBlack)

        h_chi2_vs_trial = TH1F('h_chi2_vs_trial', ';trial;#chi^{2}', n_trials+1, -0.5, n_trials+0.5)
        h_chi2_vs_trial.GetXaxis().SetTitleSize(0.05)
        h_chi2_vs_trial.GetXaxis().SetTitleOffset(0.85)
        h_chi2_vs_trial.GetYaxis().SetTitleSize(0.05)
        h_chi2_vs_trial.GetYaxis().SetTitleOffset(1.01)
        h_chi2_vs_trial.SetStats(False)
        SetObjectStyle(h_chi2_vs_trial, markerstyle=20, markercolor=kBlack, markersize=1., linecolor=kBlack)

    for itrial, trial in enumerate(trials_df.itertuples(index=False)):
        h_vn_vs_trial_indexed.SetBinContent(int(trial.TrialIdx)+1, trial.V2)
        h_vn_vs_trial_indexed.SetBinError(int(trial.TrialIdx)+1, trial.V2Unc)
        h_vn_vs_trial.SetBinContent(itrial+1, trial.V2)
        h_vn_vs_trial.SetBinError(itrial+1, trial.V2Unc)
        h_syst.Fill(trial.V2 - reference_vn)
        if "Cutset" in v2_type:
            h_chi2_vs_trial.SetBinContent(itrial+1, trial.Chi2)
            h_chi2_vs_trial.SetBinError(itrial+1, trial.Chi2Unc)
            h_signif_vs_trial.SetBinContent(itrial+1, trial.Significance)
            h_signif_vs_trial.SetBinError(itrial+1, trial.SignificanceUnc)

    # Compute systematic uncertainty
    rms = h_syst.GetRMS()
    mean = h_syst.GetMean()
    syst_unc = np.sqrt(rms**2 + mean**2)
    max_syst = h_syst.GetMaximum()
    h_syst.GetYaxis().SetRangeUser(0, max_syst*1.8)
    h_syst.GetXaxis().SetRangeUser(h_syst.GetXaxis().GetXmin(), h_syst.GetXaxis().GetXmax())

    # Display systematic uncertainty
    g_syst = TGraphAsymmErrors()
    g_syst.SetPoint(0, 0, max_syst*0.5)
    g_syst.SetPointError(0, syst_unc, syst_unc, max_syst*0.5, max_syst*0.5)
    SetObjectStyle(g_syst, markerstyle=20, markercolor=kOrange+2,
                   markersize=1, linecolor=kOrange+2,
                   linewidth=2, fillcolor=kOrange+2, fillstyle=3153,
                   fillalpha=0.5, linestyle=9)
    # Display statistical uncertainty
    g_ref = TGraphAsymmErrors()
    g_ref.SetPoint(0, x_low_lim, reference_vn)
    g_ref.SetPointError(0, 0, 0, reference_vn_unc, reference_vn_unc)
    g_ref.SetPoint(1, x_upp_lim, reference_vn)
    g_ref.SetPointError(1, 0, 0, reference_vn_unc, reference_vn_unc)
    SetObjectStyle(g_ref, markerstyle=20, markercolor=kAzure+2,
                   markersize=0, linecolor=kAzure+2,
                   linewidth=2, fillcolor=kAzure+2, fillstyle=3135, fillalpha=0.5, linestyle=9)
    # Define reference vertical line at 1
    g_ref_two = g_ref.Clone()
    g_ref_two.SetPoint(0, 0, max_syst*0.5)
    g_ref_two.SetPointError(0, reference_vn_unc, reference_vn_unc, max_syst*0.5, max_syst*0.5)
    SetObjectStyle(g_ref_two, markerstyle=20, markercolor=kAzure+2,
                   markersize=1, linecolor=kAzure+2,
                   linewidth=2, fillcolor=kAzure+2, fillstyle=3135, fillalpha=0.5, linestyle=9)
    # Legend
    leg = TLegend(0.25, 0.65, 0.85, 0.85)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.04)
    leg.SetHeader(f'{v2_type.replace("_", " ")}, {pt_min} < #it{{p}}_{{T}} < {pt_max} GeV/#it{{c}}')
    leg.AddEntry(g_syst, f'#sqrt{{shift^{{2}} + rms^{{2}}}} = {syst_unc:.3f}', 'f')
    leg.AddEntry(g_ref, 'vn stat. unc.', 'f')

    canvas = TCanvas(f'c_syst_multitrial_{v2_type}', f'c_syst_multitrial_{v2_type}', 800 if "Cutset" in v2_type else 1600, 800)
    if "Cutset" in v2_type:
        canvas.cd().Divide(2, 2)
        canvas.cd(1).SetLeftMargin(0.12)
        canvas.cd(1).SetTopMargin(0.12)
        canvas.cd(2).SetRightMargin(0.12)
        canvas.cd(2).SetTopMargin(0.12)
    else:
        canvas.cd().Divide(2, 1)
        canvas.cd(1).SetLeftMargin(0.16)
        canvas.cd(1).SetRightMargin(0.12)
        canvas.cd(1).SetTopMargin(0.05)
        canvas.cd(1).SetBottomMargin(0.12)
        canvas.cd(2).SetLeftMargin(0.16)
        canvas.cd(2).SetRightMargin(0.12)
        canvas.cd(2).SetTopMargin(0.05)
        canvas.cd(2).SetBottomMargin(0.12)

    # Pad 1: vn vs trial
    canvas.cd(1).SetGrid()
    h_vn_vs_trial.Draw('same')
    g_ref.Draw('c3 same')
    canvas.cd(2)

    h_syst.Draw('same')
    g_syst.Draw('2')
    g_ref_two.Draw('2')
    g_syst.Draw('2')
    leg.Draw()

    if "Cutset" in v2_type:
        canvas.cd(3).SetLeftMargin(0.12)
        canvas.cd(3).SetBottomMargin(0.12)
        canvas.cd(4).SetRightMargin(0.12)
        canvas.cd(4).SetBottomMargin(0.12)
        # Pad 3: chi2 vs trial
        canvas.cd(3)
        h_chi2_vs_trial.Draw('same')
        canvas.Update()
        # Pad 4: significance vs trial
        canvas.cd(4)
        h_signif_vs_trial.Draw('same')

    canvas.Update()
    canvas.SaveAs(f'{output_dir}/{pt_label}/Syst{v2_type}.pdf')

    h_syst.Write()
    h_vn_vs_trial_indexed.Write()
    h_vn_vs_trial.Write()
    if "Cutset" in v2_type:
        h_chi2_vs_trial.Write()
        h_signif_vs_trial.Write()
    canvas.Write()

    return syst_unc


def get_cfg_entry(val):
    if isinstance(val, list):
        return val[0] if val else None
    return val


def get_pt_label_range(pt_label):
    """Convert 'pt_10_15' --> (1.0, 1.5)"""
    parts = pt_label.split('_')[1:]
    pt_min = float(parts[0]) / 10.0
    pt_max = float(parts[1]) / 10.0
    return pt_min, pt_max


def extract_cutset_results(cutset_file, pt_bin):
    """Return dict with V2, Chi2, Significance for one cutset at pt_bin"""
    result = {}
    try:
        result = dict(
            Significance=cutset_file.Get('hRawYieldsSignificanceSimFit').GetBinContent(pt_bin),
            SignificanceUnc=cutset_file.Get('hRawYieldsSignificanceSimFit').GetBinError(pt_bin),
            Chi2=cutset_file.Get('hRedChi2SimFit').GetBinContent(pt_bin),
            Chi2Unc=cutset_file.Get('hRedChi2SimFit').GetBinError(pt_bin),
            V2=cutset_file.Get('hVnSimFit').GetBinContent(pt_bin),
            V2Unc=cutset_file.Get('hVnSimFit').GetBinError(pt_bin),
            SoverB=cutset_file.Get('hRawYieldsSoverBSimFit').GetBinContent(pt_bin),
            Mean=cutset_file.Get('hMeanSimFit').GetBinContent(pt_bin),
            Sigma=cutset_file.Get('hSigmaSimFit').GetBinContent(pt_bin)
        )
    except Exception as e:
        result = dict(Significance=None, SignificanceUnc=None, Chi2=None, Chi2Unc=None,
                      V2=None, V2Unc=None, SoverB=None, Mean=None, Sigma=None)

    # Check if the result has second peak sigma
    try:
        sigma_sec_peak = cutset_file.Get('hSigmaSecPeakSimFit').GetBinContent(pt_bin)
    except Exception as e:
        sigma_sec_peak = None
    result['SigmaSecPeak'] = sigma_sec_peak

    return result


def get_trial_result(trial_dir, cutsets):
    """Extract a trial row dict for all cutsets"""
    idx = os.path.basename(trial_dir)
    pt_label = os.path.basename(os.path.dirname(os.path.dirname(trial_dir)))

    # Load trial config
    cfg_path = os.path.join(trial_dir, f'config_trial_{idx}.yml')
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Prepare config columns
    config_vals = dict(
        BkgFunc=get_cfg_entry(cfg['v2extraction']['BkgFunc']),
        BkgFuncVn=get_cfg_entry(cfg['v2extraction']['BkgFuncVn']),
        SgnFunc=get_cfg_entry(cfg['v2extraction']['SgnFunc']),
        Rebin=get_cfg_entry(cfg['v2extraction']['Rebin']),
        VnVsMassBinWidths=cfg['projections']['VnVsMassBins'][0][1] - \
                          cfg['projections']['VnVsMassBins'][0][0],
        MassFitRanges=cfg['v2extraction']['MassFitRanges'][0],
        TrialIdx=idx,
        PtLabel=pt_label
    )

    # Open V2 fraction hist
    try:
        v2f = TFile.Open(os.path.join(trial_dir, 'v2/v2VsFrac.root'), 'READ')
        prompt_row = dict(V2Type='Prompt', V2=v2f.Get('hV2VsPtPrompt').GetBinContent(1),
                          V2Unc=v2f.Get('hV2VsPtPrompt').GetBinError(1),**config_vals)
        nonprompt_row = dict(V2Type='NonPrompt',V2=v2f.Get('hV2VsPtFD').GetBinContent(1),
                             V2Unc=v2f.Get('hV2VsPtFD').GetBinError(1),**config_vals)
        v2f.Close()
    except Exception as e:
        raise e

    cutset_rows = []
    for cutset in cutsets:
        try:
            cutset_file = TFile.Open(os.path.join(trial_dir, f'raw_yields/raw_yields_0{cutset}.root'), 'READ')
            cutset_row = dict(V2Type=f"Cutset_{str(cutset)}", **extract_cutset_results(cutset_file, 1), **config_vals)
            cutset_file.Close()
        except Exception as e:
            logger(f"Failed to open cutset file in {os.path.basename(trial_dir)} for cutset {cutset}: {e}", "WARNING")
            cutset_row = dict(V2Type=f"Cutset_{str(cutset)}", **config_vals, Significance=None, 
                              SignificanceUnc=None, Chi2=None, Chi2Unc=None, V2=None, V2Unc=None, 
                              SoverB=None, Mean=None, Sigma=None, SigmaSecPeak=None)
        cutset_rows.append(cutset_row)

    return [prompt_row, nonprompt_row] + cutset_rows


def get_reference_result(results_dir, cutsets, pt_labels):
    """Extract reference results (Prompt, NonPrompt, and cutsets)"""
    ref_rows = []
    v2f = TFile.Open(os.path.join(results_dir, 'v2/v2VsFrac.root'), 'READ')

    for pt_label in pt_labels:
        pt_min, pt_max = get_pt_label_range(pt_label)
        pt_bin = v2f.Get('hV2VsPtPrompt').GetXaxis().FindBin((pt_min + pt_max)/2)

        ref_rows.append(dict(PtLabel=pt_label, V2=v2f.Get('hV2VsPtPrompt').GetBinContent(pt_bin),
                             V2Type='Prompt', V2Unc=v2f.Get('hV2VsPtPrompt').GetBinError(pt_bin)))
        ref_rows.append(dict(PtLabel=pt_label, V2=v2f.Get('hV2VsPtFD').GetBinContent(pt_bin),
                             V2Type='NonPrompt', V2Unc=v2f.Get('hV2VsPtFD').GetBinError(pt_bin)))

    v2f.Close()

    # Cutsets
    for cutset in cutsets:
        ry_file = TFile.Open(os.path.join(results_dir, f'raw_yields/raw_yields_0{cutset}.root'), 'READ')
        for pt_label in pt_labels:
            pt_min, pt_max = get_pt_label_range(pt_label)
            pt_bin = ry_file.Get('hRawYieldsSignificanceSimFit').GetXaxis().FindBin((pt_min + pt_max)/2)
            ref_rows.append(dict(
                PtLabel=pt_label,
                V2Type=f"Cutset_{str(cutset)}",
                **extract_cutset_results(ry_file, pt_bin)
            ))
        ry_file.Close()

    return ref_rows


def is_good_trial(trial, max_chi2, min_signif, max_signif):
    """Check if trial passes quality criteria"""

    # Sanity check that prompt v2 has absolute value < 1
    for row in trial:
        if row['V2Type'] == 'Prompt' and abs(row['V2']) > 1.0:
            logger(f"Rejecting trial {row['TrialIdx']} due to |V2| > 1.0", "WARNING")
            return False

    # Quality cuts on trials
    for row in trial:
        if "Cutset" in row['V2Type']:
            signif, chi2, SoverB, mean, sigma, sigma_sec_peak = row['Significance'], row['Chi2'], row['SoverB'], row['Mean'], row['Sigma'], row['SigmaSecPeak']
            if sigma is not None:
                if sigma > 0.07:
                    logger(f"Rejecting trial {row['TrialIdx']} due to sigma > 0.07 (sigma = {sigma})", "WARNING")
                    return False
            if mean is not None:
                if mean < 1.849 or mean > 1.879:
                    logger(f"Rejecting trial {row['TrialIdx']} due to mean outside range [1.849, 1.879] (mean = {mean})", "WARNING")
                    return False
            if SoverB is not None:
                if SoverB < 0:
                    logger(f"Rejecting trial {row['TrialIdx']} due to negative S/B = {SoverB}", "WARNING")
                    return False
            if chi2 is not None:
                if chi2 > max_chi2:
                    logger(f"Rejecting trial {row['TrialIdx']} due to chi2 > {max_chi2}", "WARNING")
                    return False
            if signif is not None:
                if signif < min_signif:
                    logger(f"Rejecting trial {row['TrialIdx']} due to significance < {min_signif}", "WARNING")
                    return False
                if signif > max_signif:
                    logger(f"Rejecting trial {row['TrialIdx']} due to significance > {max_signif}", "WARNING")
                    return False
            if sigma_sec_peak is not None:
                if (sigma_sec_peak / sigma) > 2:
                    logger(f"Rejecting trial {row['TrialIdx']} due to sigma of second peak > 2 * sigma (sigma_sec_peak = {sigma_sec_peak}, sigma = {sigma})", "WARNING")
                    return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('default_cfg', metavar='text', default='path to the reference yaml config')
    parser.add_argument('results_dir', metavar='text', default='path to the directory containing the .root files from multitrial')
    parser.add_argument('--multitrial_type', metavar='text', type=str, default='"fit" or "bdt" multitrial type')
    parser.add_argument('--max_chi2', metavar='number', type=float, default=20.0, help='Maximum reduced chi2 to accept a trial')
    parser.add_argument('--min_signif', metavar='number', type=float, default=0.0, help='Minimum significance to accept a trial')
    parser.add_argument('--max_signif', metavar='number', type=float, default=1000.0, help='Maximum significance to accept a trial')
    parser.add_argument('--force_prompt_enhanced', action='store_true', default=False, help='Force convergence of prompt-enhanced cutsets')
    args = parser.parse_args()

    # Print required trials quality cuts
    logger(f"Using max chi2 = {args.max_chi2}", "INFO")
    logger(f"Using min significance = {args.min_signif}", "INFO")
    logger(f"Using max significance = {args.max_signif}", "INFO")
    logger(f"Force prompt-enhanced cutset convergence = {args.force_prompt_enhanced}", "INFO")

    trials_path = f"{args.results_dir}/syst/multitrial/{args.multitrial_type}"
    # pt_dirs = [os.path.join(trials_path, "pt_30_35")]
    pt_dirs = sorted(glob.glob(os.path.join(trials_path, "pt_*")))
    pt_labels = [os.path.basename(d) for d in pt_dirs]

    # List cutsets
    cutsets = sorted(int(re.search(r"raw_yields_(\d+)\.root", f).group(1))
                    for f in os.listdir(os.path.join(args.results_dir, 'raw_yields'))
                    if re.match(r'raw_yields_\d+\.root', f))

    # Check if the trials dataframe has already been produced
    categories = ['BkgFunc', 'BkgFuncVn', 'SgnFunc', 'Rebin', 'VnVsMassBinWidths', 'MassFitRanges']
    if os.path.exists(f"{args.results_dir}/syst/multitrial/{args.multitrial_type}/summary/trials_results.parquet") and \
       os.path.exists(f"{args.results_dir}/syst/multitrial/{args.multitrial_type}/summary/reference_results.parquet"):
        logger("✔ Found existing parquet files, loading them...", "INFO")
        trials_df = pd.read_parquet(f"{args.results_dir}/syst/multitrial/{args.multitrial_type}/summary/trials_results.parquet")
        reference_df = pd.read_parquet(f"{args.results_dir}/syst/multitrial/{args.multitrial_type}/summary/reference_results.parquet")
    else:
        # Extract all trial rows
        trials_results = []
        for pt_dir in pt_dirs:
            trial_dirs = sorted(glob.glob(os.path.join(pt_dir, 'trials/*')))
            for i_trial, trial_dir in enumerate(trial_dirs):
                try:
                    trial_result = get_trial_result(trial_dir, cutsets)
                    if args.force_prompt_enhanced:
                        print(f"Checking prompt-enhanced cutset convergence for trial {os.path.basename(trial_dir)}...")
                        # Retrieve Cutset_0 row
                        cutset_0_row = next((row for row in trial_result if row['V2Type'] == 'Cutset_0'), None)
                        if cutset_0_row is None or cutset_0_row['Significance'] is None:
                            print(f"Rejecting trial {os.path.basename(trial_dir)} due to missing prompt-enhanced cutset")
                            continue
                    if is_good_trial(trial_result, args.max_chi2, args.min_signif, args.max_signif):
                        trials_results.extend(trial_result)
                except Exception as e:
                    logger(f"Failed trial {os.path.basename(trial_dir)}: {e}", "WARNING")
                    continue

        trials_df = pd.DataFrame(trials_results)
        logger(f"Extracted {len(trials_results)} trial results.", "INFO")

        # Convert categorical columns
        for col in categories:
            if col in trials_df:
                trials_df[col] = trials_df[col].astype(str).astype('category')
        trials_df['PtLabel'] = trials_df['PtLabel'].apply(lambda x: str(x)).astype('category')
        trials_df['V2Type'] = trials_df['V2Type'].apply(lambda x: str(x)).astype('category')

        # Extract reference
        reference_rows = get_reference_result(args.results_dir, cutsets, pt_labels)
        reference_df = pd.DataFrame(reference_rows)
        for col in ['PtLabel', 'V2Type']:
            reference_df[col] = reference_df[col].apply(lambda x: str(x)).astype('category')

        # Save trials_df and reference_df to .parquet files
        trials_df.to_parquet(f"{args.results_dir}/syst/multitrial/{args.multitrial_type}/summary/trials_results.parquet")
        reference_df.to_parquet(f"{args.results_dir}/syst/multitrial/{args.multitrial_type}/summary/reference_results.parquet")
        logger("✔ Saved trials and reference results to parquet files.", "INFO")

    # Summary files and figures
    output_dir = f"{args.results_dir}/syst/multitrial/{args.multitrial_type}/summary/"
    os.makedirs(output_dir, exist_ok=True)

    # Produce variations figures
    pt_bins = sorted(set(edge for pt in pt_labels for edge in get_pt_label_range(pt)))
    # # Only 3.0-3.5
    # pt_bins = [3.0, 3.5]

    syst_uncs_prompt, syst_uncs_non_prompt = {}, {}
    hist_prompt_syst = TH1F('h_prompt_syst', f';#it{{p}}_{{T}} (GeV/#it{{c}});{args.multitrial_type.capitalize()} syst. unc. on prompt #it{{v}}_{{2}}', \
                            len(pt_bins)-1, array.array('d', pt_bins))
    hist_non_prompt_syst = TH1F('h_non_prompt_syst', f';#it{{p}}_{{T}} (GeV/#it{{c}});{args.multitrial_type.capitalize()} syst. unc. on non-prompt #it{{v}}_{{2}}', \
                                len(pt_bins)-1, array.array('d', pt_bins))
    for pt_label in pt_labels:
        print(f"Processing pt bin {pt_label}...")
        os.makedirs(os.path.join(output_dir, pt_label), exist_ok=True)
        out_file_summary = TFile.Open(f"{output_dir}/SystSummary_{pt_label}.root", 'RECREATE')
        # Print all V2Type values
        for v2_type in ['Prompt', 'NonPrompt'] + [f"Cutset_{str(c)}" for c in cutsets]:
            # Apply category filtering
            sel_trials_df = trials_df[(trials_df["V2Type"] == v2_type) & (trials_df["PtLabel"] == pt_label)]
            sel_reference_df = reference_df[(reference_df["V2Type"] == v2_type) & (reference_df["PtLabel"] == pt_label)]

            # Cleanup entries with None in Cutsets (missing cutset, but v2 vs fFD could still be extracted)
            if "Cutset" in v2_type:
                sel_trials_df = sel_trials_df.dropna(subset=["V2", "V2Unc", "Chi2", "Chi2Unc", "Significance", "SignificanceUnc", "SoverB", "Mean", "Sigma"])

            # Reset row indices
            sel_trials_df = sel_trials_df.reset_index(drop=True)
            sel_reference_df = sel_reference_df.reset_index(drop=True)

            pdf_variations_path = os.path.join(output_dir, pt_label, f"V2_{v2_type}_variations.pdf")
            with PdfPages(pdf_variations_path) as pdf:
                for varied_var in categories:
                    output_dir = f"{args.results_dir}/syst/multitrial/{args.multitrial_type}/summary/"
                    study_variations(sel_trials_df, sel_reference_df, v2_type, varied_var, pt_label, output_dir, pdf)
            logger(f"✔ Saved merged PDF for {v2_type} in {pt_label} -> {pdf_variations_path}", "INFO")

            make_dir_root_file(v2_type, out_file_summary)
            out_file_summary.cd(v2_type)
            syst_unc = study_syst_unc(sel_trials_df, sel_reference_df, v2_type, pt_label, output_dir)
            if v2_type == "Prompt":
                hist_prompt_syst.SetBinContent(
                    hist_prompt_syst.GetXaxis().FindBin(
                        (get_pt_label_range(pt_label)[0] + get_pt_label_range(pt_label)[1]) / 2),
                    syst_unc)
            if v2_type == "NonPrompt":
                hist_non_prompt_syst.SetBinContent(
                    hist_non_prompt_syst.GetXaxis().FindBin(
                        (get_pt_label_range(pt_label)[0] + get_pt_label_range(pt_label)[1]) / 2),
                    syst_unc)
            logger(f"✔ Saved systematic uncertainty for {v2_type} in {pt_label}", "INFO")

        out_file_summary.Close()

    # Draw canvas for prompt
    canvas_prompt = TCanvas('c_prompt_syst', 'c_prompt_syst', 600, 600)
    canvas_prompt.SetTopMargin(0.12)
    canvas_prompt.SetLeftMargin(0.15)
    canvas_prompt.SetRightMargin(0.12)
    canvas_prompt.SetBottomMargin(0.15)
    SetObjectStyle(hist_prompt_syst, markerstyle=20, markercolor=kOrange+2,
                   markersize=1.,linecolor=kOrange+2,
                   linewidth=2, fillcolor=kOrange+2, fillstyle=3135, fillalpha=0.7)
    SetObjectStyle(hist_prompt_syst, markerstyle=20, markercolor=kBlack, markersize=1., linecolor=kBlack)
    hist_prompt_syst.Draw('hist')
    hist_prompt_syst.GetXaxis().SetTitleOffset(1.07)
    canvas_prompt.SaveAs(f'{args.results_dir}/syst/multitrial/{args.multitrial_type}/summary/SystV2Prompt_AllPtBins.pdf')

    # Draw canvas for non-prompt
    canvas_non_prompt = TCanvas('c_non_prompt_syst', 'c_non_prompt_syst', 600, 600)
    canvas_non_prompt.SetTopMargin(0.12)
    canvas_non_prompt.SetLeftMargin(0.15)
    canvas_non_prompt.SetRightMargin(0.12)
    canvas_non_prompt.SetBottomMargin(0.15)
    SetObjectStyle(hist_non_prompt_syst, markerstyle=20, markercolor=kOrange+2,
                   markersize=1.,linecolor=kOrange+2,
                   linewidth=2, fillcolor=kOrange+2, fillstyle=3135, fillalpha=0.7)
    SetObjectStyle(hist_non_prompt_syst, markerstyle=20, markercolor=kBlack, markersize=1., linecolor=kBlack)
    hist_non_prompt_syst.Draw('hist')
    hist_non_prompt_syst.GetXaxis().SetTitleOffset(1.07)
    canvas_non_prompt.SaveAs(f'{args.results_dir}/syst/multitrial/{args.multitrial_type}/summary/SystNonPrompt_AllPtBins.pdf')

    logger("Combining all cutsets pdfs into multi-page pdfs...", "INFO")
    for pt_dir in pt_dirs:
        pt_label = os.path.basename(pt_dir)
        cutset_files = sorted(glob.glob(f'{args.results_dir}/syst/multitrial/{args.multitrial_type}/summary/{pt_label}/SystCutset_*.pdf'))
        if cutset_files:
            merger = PdfMerger()
            for pdf in cutset_files:
                merger.append(pdf)
            out_cutsets = f"{args.results_dir}/syst/multitrial/{args.multitrial_type}/summary/CutsetsSyst_{pt_label}.pdf"
            merger.write(out_cutsets)
            merger.close()
            logger(f"✔ Saved merged PDF for cutsets in {pt_label} -> {out_cutsets}", "INFO")

    logger("Combining all SystPrompt and SystNonPrompt plots into multi-page pdfs...", "INFO")
    # Create a multi-page pdf with all SystPrompt and SystNonPrompt plots
    all_prompt_files = sorted(glob.glob(f'{args.results_dir}/syst/multitrial/{args.multitrial_type}/summary/**/SystPrompt.pdf', recursive=True))
    all_non_prompt_files = sorted(glob.glob(f'{args.results_dir}/syst/multitrial/{args.multitrial_type}/summary/**/SystNonPrompt.pdf', recursive=True))

    # --- Merge Prompt ---
    if all_prompt_files:
        merger = PdfMerger()
        for pdf in all_prompt_files:
            merger.append(pdf)
        out_prompt = f"{args.results_dir}/syst/multitrial/{args.multitrial_type}/summary/TrialsSystPrompt.pdf"
        merger.write(out_prompt)
        merger.close()

    # --- Merge Non-prompt ---
    if all_non_prompt_files:
        merger = PdfMerger()
        for pdf in all_non_prompt_files:
            merger.append(pdf)
        out_non_prompt = f"{args.results_dir}/syst/multitrial/{args.multitrial_type}/summary/TrialsSystNonPrompt.pdf"
        merger.write(out_non_prompt)
        merger.close()

    out_file_syst = TFile.Open(f"{args.results_dir}/syst/multitrial/{args.multitrial_type}/summary/TotalSystV2.root", 'RECREATE')
    out_file_syst.cd()
    hist_prompt_syst.Write()
    hist_non_prompt_syst.Write()
    out_file_syst.Close()
    logger(f"✔ Saved systematic uncertainties to {out_file_syst.GetName()}", "INFO")
