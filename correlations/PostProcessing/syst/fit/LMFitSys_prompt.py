#!/usr/bin/env python3
"""
LMFitSys_prompt.py — prompt v2 systematic uncertainty from LM template variation.

Builds on LMFitSys.py output (one v2_delta + error per trial per pT bin) and
propagates it through the prompt-v2 unfolding:

  1. Read the prompt fraction file (hPromptFracCorr / hFDFracCorr) exactly as
     compute_prompt_v2_unfold.py does (same yget / read_xy).
  2. For each pT bin, collect v2_delta + error from the central trial (trial_0000)
     and the sampled trials; scale each by 1/0.07.
  3. Compute prompt v2 and its statistical error per sample with the SAME
     v2prompt_ratio formula (incl. error propagation) used by
     compute_prompt_v2_unfold.py.
  4. Build the 1D prompt-v2 histogram per pT (like hProbV2_PtCand), gauss-fit it
     to get sigma = systematic uncertainty.
  5. Draw v2_vs_pT_LMSys (central prompt v2 + syst band) and v2Syst_vs_pT (hSyst),
     and save gSyst + hSyst.

The unfold is per-bin linear (v2p = v2obs / (f_prompt + r * f_FD)), so it can be
computed per pT bin without running the full subprocess per trial.
"""

import argparse
import math
import os

import numpy as np
import yaml
import ROOT
from array import array

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)

# ── Paths (mirror LMFitSys.py) ──────────────────────────────────────────
BASE = "/home/wuct/MetaData/DATA/OO/apass2/corr/results/fifth/k020_gausPer/k60100_loose2to2d5_d20/sys/central"
EXTRACT = f"{BASE}/CorrelExtract_0d2_1d3"
OUT_LM = f"{EXTRACT}/CorrelationFitResults/LMSysResults"
PROMPT_CONFIG = "/home/wuct/ALICE/reps/hf-vn-dev/dev/configs/v2_prompt_v2_method_check.yml"

PT_CAND = [0., 1., 1.5, 2., 2.5, 3., 3.5, 4., 5., 6., 7., 8.]
N_PT = len(PT_CAND) - 1
SCALE = 1.0 / 0.07

OUT_PROMPT = os.path.join(OUT_LM, "prompt")
OUT_TRIAL = os.path.join(OUT_PROMPT, "trial")


# ── Helpers (verbatim from compute_prompt_v2_unfold.py) ──────────────────
def yget(cfg, path, default=None, required=False):
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            if required:
                raise KeyError(f"Missing YAML key: {path}")
            return default
        cur = cur[k]
    return cur


def read_xy(obj):
    """Read x, y, ex, ey from a TH1."""
    xs, ys, exs, eys = [], [], [], []
    for i in range(1, obj.GetNbinsX() + 1):
        xs.append(float(obj.GetBinCenter(i)))
        exs.append(float(0.5 * obj.GetBinWidth(i)))
        ys.append(float(obj.GetBinContent(i)))
        eys.append(float(obj.GetBinError(i)))
    return xs, ys, exs, eys


def v2prompt_ratio(v2obs, ev2obs, fp, efp, ffd, effd, r, eps=1e-12):
    """v2p = v2obs / (fp + r * ffd) — identical to compute_prompt_v2_unfold.py."""
    v2p, ev2p = [], []
    for v, sv, a, sa, b, sb in zip(v2obs, ev2obs, fp, efp, ffd, effd):
        D = a + r * b
        if abs(D) < eps:
            v2p.append(float("nan"))
            ev2p.append(float("nan"))
            continue
        y = v / D
        sD2 = sa * sa + (r * sb) * (r * sb)
        sy2 = (sv / D) ** 2 + (v * v * sD2) / (D ** 4)
        v2p.append(y)
        ev2p.append(math.sqrt(max(0.0, sy2)))
    return v2p, ev2p


def _read_v2_from_txt(path):
    if not os.path.exists(path):
        return None, None
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v2_delta"):
                parts = line.strip().split()
                return float(parts[2]), float(parts[4])
    return None, None


def _read_prompt_config():
    with open(PROMPT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    frac_file = yget(cfg, "input.fractions.file", required=True)
    fp_name = yget(cfg, "input.fractions.objects_ratio.f_prompt", required=True)
    ffd_name = yget(cfg, "input.fractions.objects_ratio.f_FD", required=True)
    r = float(yget(cfg, "assumption.r", required=True))
    return frac_file, fp_name, ffd_name, r


def load_fractions():
    """Read fp / ffd (and errors) exactly like compute_prompt_v2_unfold.py."""
    frac_file, fp_name, ffd_name, r = _read_prompt_config()
    print(f"[INFO] fraction file: {frac_file}")
    print(f"[INFO] fractions: ({fp_name}, {ffd_name}), r={r}")
    ffrac = ROOT.TFile.Open(frac_file)
    if not ffrac or ffrac.IsZombie():
        raise RuntimeError(f"Cannot open {frac_file}")
    o_fp = ffrac.Get(fp_name)
    o_ffd = ffrac.Get(ffd_name)
    if not o_fp:
        raise RuntimeError(f"Missing {fp_name} in {frac_file}")
    if not o_ffd:
        raise RuntimeError(f"Missing {ffd_name} in {frac_file}")
    x_fp, fp, _, efp = read_xy(o_fp)
    x_ffd, ffd, _, effd = read_xy(o_ffd)
    ffrac.Close()
    return fp, efp, ffd, effd, r


def collect_v2_delta(pt_cand_idx, max_samples=None):
    """Return (central_v2, central_err, sample_v2s, sample_errs) for one pT bin."""
    pt_dir = os.path.join(OUT_LM, f"PtCand_{pt_cand_idx}")
    if not os.path.isdir(pt_dir):
        return None, None, [], []

    trials = sorted(d for d in os.listdir(pt_dir) if d.startswith("trial_"))
    central_v2 = central_err = None
    sample_v2s, sample_errs = [], []

    for t in trials:
        v2, err = _read_v2_from_txt(os.path.join(pt_dir, t, "v2_result.txt"))
        if v2 is None:
            continue
        idx = int(t.split("_")[1]) if "_" in t else -1
        if idx == 0:  # central trial
            central_v2, central_err = v2, err
        else:
            sample_v2s.append(v2)
            sample_errs.append(err)
            if max_samples is not None and len(sample_v2s) >= max_samples:
                break

    return central_v2, central_err, sample_v2s, sample_errs


def run_prompt_syst(n_samples=None):
    fp, efp, ffd, effd, r = load_fractions()
    n_frac = len(fp)
    n_bins = min(N_PT, n_frac)
    print(f"[INFO] pT bins: {N_PT} (LMFitSys) vs {n_frac} (fractions) -> using {n_bins}")

    pt_centers = [0.5 * (PT_CAND[i] + PT_CAND[i + 1]) for i in range(n_bins)]

    central_v2p = np.zeros(n_bins)
    central_ev2p = np.zeros(n_bins)
    all_prompt_samples = {}   # i_pt -> np.array of prompt v2
    sigmas = np.zeros(n_bins)
    mus = np.zeros(n_bins)

    for i_pt in range(n_bins):
        pt_cand_idx = i_pt + 1
        c_v2, c_err, sample_v2s, sample_errs = collect_v2_delta(pt_cand_idx, max_samples=n_samples)
        if c_v2 is None:
            print(f"[WARN] PtCand {pt_cand_idx}: no central trial, skip")
            continue

        # central prompt v2 (scaled observed v2 -> unfolded)
        c_v2obs = c_v2 * SCALE
        c_ev2obs = c_err * SCALE
        c_v2p, c_ev2p = v2prompt_ratio(
            [c_v2obs], [c_ev2obs], [fp[i_pt]], [efp[i_pt]], [ffd[i_pt]], [effd[i_pt]], r)
        central_v2p[i_pt] = c_v2p[0]
        central_ev2p[i_pt] = c_ev2p[0]

        if not sample_v2s:
            print(f"[WARN] PtCand {pt_cand_idx}: no samples, sigma=0")
            continue

        # per-sample prompt v2
        v2obs_arr = np.array(sample_v2s) * SCALE
        ev2obs_arr = np.array(sample_errs) * SCALE
        prompt_samples = []
        for v, sv in zip(v2obs_arr, ev2obs_arr):
            vp, _ = v2prompt_ratio(
                [v], [sv], [fp[i_pt]], [efp[i_pt]], [ffd[i_pt]], [effd[i_pt]], r)
            prompt_samples.append(vp[0])
        prompt_samples = np.array(prompt_samples)
        all_prompt_samples[i_pt] = prompt_samples

        # # 1D prompt-v2 histogram (like hProbV2) + gauss fit
        # v2_mean, v2_std = float(np.mean(prompt_samples)), float(np.std(prompt_samples))
        # half_width = max(float(np.max(np.abs(prompt_samples - c_v2p[0]))), 1e-6) * 1.05
        # h_pv2 = ROOT.TH1D(f"hProbV2_PtCand{pt_cand_idx}", "", 50,
        #                   c_v2p[0] - half_width, c_v2p[0] + half_width)
        # for v in prompt_samples:
        #     h_pv2.Fill(v)
        # h_pv2.Fit("gaus", "LSQ0+")
        # fit_func = h_pv2.GetFunction("gaus")
        # mu, sigma = v2_mean, v2_std
        # if fit_func:
        #     mu = fit_func.GetParameter(1)
        #     sigma = fit_func.GetParameter(2)
        # mus[i_pt] = mu
        # sigmas[i_pt] = sigma
        # # systematic uncertainty = sqrt(mean^2 + rms^2) of (v2_trial - v2_ref),
        # # same as produce_fit_syst.py (the fit step)
        # dev = prompt_samples - c_v2p[0]
        # syst_unc = float(np.sqrt(np.mean(dev ** 2)))
        # sigmas[i_pt] = syst_unc
        # calculate standard deviation as systematic uncertainty
        dev = prompt_samples - c_v2p[0]
        syst_unc = syst_unc = float(np.sqrt(np.mean(dev**2)))
        # take 95% confidence interval as systematic uncertainty
        syst_unc_95 = float(np.percentile(np.abs(dev), 95))
        print(f"[OK] PtCand {pt_cand_idx}: central={c_v2p[0]:.4f} +/- {c_ev2p[0]:.4f}, "
              f"n_samp={len(prompt_samples)}, syst=sqrt(mean^2+rms^2)={syst_unc:.4f}, 95%% CI={syst_unc_95:.4f}")

    # ── Build output objects ────────────────────────────────────────────
    os.makedirs(OUT_PROMPT, exist_ok=True)

    gSyst = ROOT.TGraphAsymmErrors(n_bins)
    gSyst.SetName("gSyst")
    gSyst.SetTitle(";#it{p}_{T}^{D} (GeV/#it{c});syst. unc.")
    hSyst = ROOT.TH1D("hSyst", ";#it{p}_{T}^{D} (GeV/#it{c});syst. unc.",
                      n_bins, array('d', PT_CAND[:n_bins + 1]))
    hSyst.SetStats(0)

    gr_cent = ROOT.TGraphErrors(n_bins)
    gr_cent.SetName("gV2Central")
    gr_cent.SetTitle(";#it{p}_{T}^{D} (GeV/#it{c});v_{2}^{prompt}")
    gr_cent.SetMarkerStyle(ROOT.kFullCircle)
    gr_cent.SetMarkerColor(ROOT.kBlack)
    gr_cent.SetLineColor(ROOT.kBlack)

    gr_band = ROOT.TGraphAsymmErrors(n_bins)
    gr_band.SetName("gV2SysBand")
    gr_band.SetTitle(";#it{p}_{T}^{D} (GeV/#it{c});#it{v}_{2}^{prompt}")
    gr_band.SetFillColorAlpha(ROOT.kBlue - 7, 0.35)

    for i_pt in range(n_bins):
        pt_c = pt_centers[i_pt]
        dx = 0.5 * (PT_CAND[i_pt + 1] - PT_CAND[i_pt])
        sigma = sigmas[i_pt]

        gr_cent.SetPoint(i_pt, pt_c, central_v2p[i_pt])
        gr_cent.SetPointError(i_pt, dx, central_ev2p[i_pt])

        gr_band.SetPoint(i_pt, pt_c, central_v2p[i_pt])
        gr_band.SetPointError(i_pt, 0.3 * dx, 0.3 * dx, sigma, sigma)

        gSyst.SetPoint(i_pt, pt_c, sigma)
        gSyst.SetPointError(i_pt, dx, dx, 0., 0.)

        hSyst.SetBinContent(i_pt + 1, sigma)
        hSyst.SetBinError(i_pt + 1, 0.)

    # ── Draw v2_vs_pT_LMSys ─────────────────────────────────────────────
    c1 = ROOT.TCanvas("c_v2_vs_pt", "", 1920, 1536)
    c1.SetLeftMargin(0.12)
    c1.SetRightMargin(0.03)
    c1.SetBottomMargin(0.12)
    c1.SetTopMargin(0.05)

    gr_band.SetMinimum(-0.05)
    gr_band.SetMaximum(0.15)
    gr_band.Draw("A2")
    gr_cent.Draw("P same")

    leg = ROOT.TLegend(0.65, 0.72, 0.95, 0.88)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.AddEntry(gr_cent, "central prompt v_{2} (stat.)", "lep")
    leg.AddEntry(gr_band, "LM syst. #sigma band", "f")
    leg.Draw()

    c1.SaveAs(os.path.join(OUT_PROMPT, "v2_vs_pT_LMSys.png"))
    c1.SaveAs(os.path.join(OUT_PROMPT, "v2_vs_pT_LMSys.root"))

    # ── Draw v2Syst_vs_pT ───────────────────────────────────────────────
    c_sys = ROOT.TCanvas("c_hSyst", "", 1920, 1536)
    c_sys.SetLeftMargin(0.12)
    c_sys.SetRightMargin(0.03)
    c_sys.SetBottomMargin(0.12)
    c_sys.SetTopMargin(0.05)

    hSyst.SetFillColorAlpha(ROOT.kRed - 7, 0.5)
    hSyst.SetFillStyle(3004)
    hSyst.SetLineColor(ROOT.kRed + 1)
    max_err = max(hSyst.GetBinContent(i + 1) for i in range(n_bins)) if n_bins else 0.1
    hSyst.GetYaxis().SetRangeUser(0., max_err * 1.5)
    hSyst.Draw("HIST")

    leg_sys = ROOT.TLegend(0.15, 0.75, 0.45, 0.88)
    leg_sys.SetBorderSize(0)
    leg_sys.SetFillStyle(0)
    leg_sys.AddEntry(hSyst, "prompt v_{2} LM syst. unc.", "f")
    leg_sys.Draw()

    c_sys.SaveAs(os.path.join(OUT_PROMPT, "v2Syst_vs_pT.png"))

    # ── Write gSyst / hSyst to ROOT ─────────────────────────────────────
    fout = ROOT.TFile.Open(os.path.join(OUT_PROMPT, "v2_vs_pT_LMSys.root"), "UPDATE")
    gr_cent.Write("gV2Central", ROOT.TObject.kOverwrite)
    gr_band.Write("gV2SysBand", ROOT.TObject.kOverwrite)
    gSyst.Write("gSyst", ROOT.TObject.kOverwrite)
    hSyst.Write("hSyst", ROOT.TObject.kOverwrite)
    fout.Close()

    # ── Central final_results.root (for reference / downstream) ─────────
    os.makedirs(OUT_TRIAL, exist_ok=True)
    h_final = ROOT.TH1D("hv2Delta_PtBinAssoc1_InvMassBin1", "", n_bins, array('d', PT_CAND[:n_bins + 1]))
    h_final.SetDirectory(0)
    for i_pt in range(n_bins):
        c_v2, c_err, _, _ = collect_v2_delta(i_pt + 1)
        if c_v2 is None:
            continue
        h_final.SetBinContent(i_pt + 1, c_v2 * SCALE)
        h_final.SetBinError(i_pt + 1, c_err * SCALE)
    ffinal = ROOT.TFile.Open(os.path.join(OUT_TRIAL, "final_results.root"), "RECREATE")
    h_final.Write()
    ffinal.Close()

    print(f"\n[DONE] outputs in {OUT_PROMPT}")
    print(f"  v2_vs_pT_LMSys.png/.root   — central prompt v2 + LM syst band (gSyst, hSyst)")
    print(f"  v2Syst_vs_pT.png           — prompt v2 syst vs pT (hSyst)")
    print(f"  trial/final_results.root   — central observed v2 (scaled 1/0.07)")

    # copy results to /home/wuct/MetaData/DATA/OO/apass2/corr/results/fifth/k020_gausPer/k60100_loose2to2d5_d20/sys/results
    os.makedirs("/home/wuct/MetaData/DATA/OO/apass2/corr/results/fifth/k020_gausPer/k60100_loose2to2d5_d20/sys/results", exist_ok=True)
    os.system(f"cp -r {OUT_PROMPT} /home/wuct/MetaData/DATA/OO/apass2/corr/results/fifth/k020_gausPer/k60100_loose2to2d5_d20/sys/results")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LM fit prompt-v2 systematic uncertainty")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Limit samples per pT (default: all)")
    args = parser.parse_args()
    run_prompt_syst(n_samples=args.n_samples)
