#!/usr/bin/env python3
"""
LM fit systematic uncertainty — vary LM template parameters via
multivariate Gaussian sampling, re-fit to get v2 spread.

Workflow per (PtCand, PtHad):
  1. Read data histogram from CorrPhiD0 canvas ROOT file
  2. Read LM template (fLMOutput) + covariance + parameters from hLMtemplate ROOT
  3. Sample LM parameters from multivariate Gaussian (N samples)
  4. Build varied GausPeriodic LM templates
  5. Fit with DhCorrelationFitter -> collect v2_delta per sample
  6. Plot v2 vs pT with systematic band, and probability vs v2 per pT
"""

import argparse
import gc
import math
import os
import sys
from array import array
from ctypes import c_double, c_int

import numpy as np
from scipy.stats import multivariate_normal
import ROOT
from alive_progress import alive_bar

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning
ROOT.gStyle.SetOptStat(0)  # No stat boxes
ROOT.gStyle.SetPalette(ROOT.kRainBow)  # kRainBow = 55
ROOT.gStyle.SetOptTitle(0)

# ── Global Caches to Prevent Memory & Performance Leaks ────────────────
_alpha_color_cache = {}
_lm_color_cache = {}

# ── Compile DhCorrelationFitter ────────────────────────────────────────
ROOT.gSystem.AddIncludePath("-I/home/wuct/Software/miniforge3/envs/alice/include")
_fitter_cxx = os.path.join(os.path.dirname(__file__), "..", "..", "DhCorrelationFitter.cxx")
ROOT.gSystem.CompileMacro(_fitter_cxx, "kO+")  # force recompile with ACLiC
from ROOT import DhCorrelationFitter

# ── Paths (from FitCorrel config) ──────────────────────────────────────
BASE = "/home/wuct/MetaData/DATA/OO/apass2/corr/results/fifth/k020_gausPer/k60100_loose2to2d5_d20/sys/central"
EXTRACT = f"{BASE}/CorrelExtract_0d2_1d3"
OUT_ROOT = f"{EXTRACT}/CorrelationFitResults/Output_CorrelationFitting_Root"
OUT_PNG  = f"{EXTRACT}/CorrelationFitResults/outputPathOutput_CorrelationFitting_png"
OUT_LM   = f"{EXTRACT}/CorrelationFitResults/LMSysResults"
os.makedirs(OUT_LM, exist_ok=True)

# Config values (mirror FitCorrel config)
DMESON = "D0"
PT_CAND = [0., 1., 1.5, 2., 2.5, 3., 3.5, 4., 5., 6., 7., 8.]
PT_HAD  = [0.2, 3.0]
MASS_RANGE = [[1.72, 2.02], [1.72, 2.02], # pt 2
            [1.72, 2.02], # pt 3
            [1.72, 2.02], # pt 4
            [1.72, 2.02], # pt 5
            [1.72, 2.02], # pt 6
            [1.72, 2.02], # pt 7
            [1.72, 2.04], # pt 8
            [1.72, 2.04], # pt 9
            [1.72, 2.04], # pt 10
            [1.72, 2.04] # pt 10
]
TEMP_FUNC = 4  # 4: GausPeriodic (default), 5: von Mises with peaks fixed at 0 / pi
FIX_LM_FACTOR = True
WITH_PED_LM = False
FIX_BASELINE = -4
FIT_FUNC_TYPE = 10  # kTemplateFit
F_MIN = -0.5 * math.pi
F_MAX = 1.5 * math.pi
N_SAMPLES = 100  # Default, override with --n-samples
RY_TRIGGER = None  # Will be read from file or set from config
LM_RY_TRIGGER = None
RY_ERR = 0.
LM_RY_ERR = 0.

_GAUS_PER_FORMULA = (
    "[0]"
    "+[1]/(TMath::Sqrt(2*TMath::Pi())*[2])"
    "*TMath::Exp(-x*x/(2*[2]*[2]))"
    "+[1]/(TMath::Sqrt(2*TMath::Pi())*[2])"
    "*TMath::Exp(-(x-2*TMath::Pi())*(x-2*TMath::Pi())/(2*[2]*[2]))"
    "+[1]/(TMath::Sqrt(2*TMath::Pi())*[2])"
    "*TMath::Exp(-(x+2*TMath::Pi())*(x+2*TMath::Pi())/(2*[2]*[2]))"
    "+[3]/(TMath::Sqrt(2*TMath::Pi())*[4])"
    "*TMath::Exp(-(x-TMath::Pi())*(x-TMath::Pi())/(2*[4]*[4]))"
    "+[3]/(TMath::Sqrt(2*TMath::Pi())*[4])"
    "*TMath::Exp(-(x-3*TMath::Pi())*(x-3*TMath::Pi())/(2*[4]*[4]))"
    "+[3]/(TMath::Sqrt(2*TMath::Pi())*[4])"
    "*TMath::Exp(-(x+TMath::Pi())*(x+TMath::Pi())/(2*[4]*[4]))"
)

# von Mises template, both peak positions fixed at 0 and pi (tempFunc = 5).
# par: 0 = baseline, 1 = NS yield (area), 2 = NS kappa, 3 = AS yield (area), 4 = AS kappa
# The amplitude is an AREA (the user's original formula). In the fitter it is bounded by the pairs
# observed in its own half-window of the template (ceil_NS over [-pi/2, pi/2), ceil_AS over
# [pi/2, 3pi/2], both summed from zero), so a broad component can no longer grow to an enormous
# unphysical yield (d20 PtBin 7 used to reach an area of 93000 with kappa_AS = 0.18).
# The kappa limits follow the GausPeriodic (tempFunc=4) ones through sigma_local = 1/sqrt(kappa):
# kappa in [1, 256/pi^2 = 25.94], kappa start 64/pi^2 = 6.485. The lower end stays at 1
# (= sigma <= 1 rad), NOT the 1/pi^2 that a sigma <= pi would give - kappa < 1 is a flat component
# degenerate with the baseline (that branch re-appears in the fit if the bound is loosened).
# Height form (kept for reference): par[1]/par[3] = peak heights, Y = h*2*pi*I0(kappa)/e^kappa.
_VONMISES_FORMULA = (
    "[0]"
    "+[1]/(2*TMath::Pi()*TMath::BesselI0([2]))"
    "*TMath::Exp([2]*TMath::Cos(x))"
    "+[3]/(2*TMath::Pi()*TMath::BesselI0([4]))"
    "*TMath::Exp([4]*TMath::Cos(x-TMath::Pi()))"
)

def _hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i %= 6
    if i == 0: return v, t, p
    if i == 1: return q, v, p
    if i == 2: return p, v, t
    if i == 3: return p, q, v
    if i == 4: return t, p, v
    return v, p, q

def get_custom_palette(name="rainbow", n_colors=100):
    colors = []
    for i in range(n_colors):
        frac = float(i) / max(n_colors - 1, 1)
        if name == "rainbow":
            h, s, v = frac, 0.8, 1.0
        elif name == "coolwarm":
            h, s, v = 0.66 * (1.0 - frac), abs(frac - 0.5) * 2.0, 0.95
        elif name == "viridis":
            h, s, v = 0.8 - frac * 0.65, 0.4 + 0.6 * frac, 0.3 + 0.7 * frac
        else:
            h, s, v = frac, 0.8, 1.0
        r, g, b = _hsv_to_rgb(h, s, v)
        c_idx = ROOT.TColor.GetColor(int(r * 255), int(g * 255), int(b * 255))
        colors.append(c_idx)
    return colors

def build_lm_tf1(par, name="fLMTemplate"):
    # TEMP_FUNC = 4 : GausPeriodic  (par: ped, A_NS, sigma_NS, A_AS, sigma_AS)
    # TEMP_FUNC = 5 : von Mises, peaks fixed at 0 and pi
    #                 (par: ped, Y_NS, kappa_NS, Y_AS, kappa_AS)
    formula = _VONMISES_FORMULA if TEMP_FUNC == 5 else _GAUS_PER_FORMULA
    func = ROOT.TF1(name, formula, F_MIN, F_MAX)
    for i in range(5):
        func.SetParameter(i, par[i])
    func.SetNpx(3 * 100)
    ROOT.SetOwnership(func, False)
    return func

def extract_lm_data_histo(lm_root_path):
    f = ROOT.TFile.Open(lm_root_path)
    if not f or f.IsZombie(): return None
    for k in f.GetListOfKeys():
        obj = f.Get(k.GetName())
        if obj and obj.InheritsFrom("TCanvas"):
            for ip in range(obj.GetListOfPrimitives().GetSize()):
                p = obj.GetListOfPrimitives().At(ip)
                if p and "LM" in p.GetName():
                    h = p.Clone("hLMData"); h.SetDirectory(0)
                    f.Close(); return h
            break
    f.Close(); return None

def extract_data_histo(canvas_root_path):
    f = ROOT.TFile.Open(canvas_root_path)
    if not f or f.IsZombie():
        return None, None
    keys = [k.GetName() for k in f.GetListOfKeys()]
    canvas = None
    for k in keys:
        obj = f.Get(k)
        if obj and obj.InheritsFrom("TCanvas"):
            canvas = obj; break
    if not canvas:
        f.Close(); return None, None

    h_data = None
    prims = canvas.GetListOfPrimitives()
    for ip in range(prims.GetSize()):
        obj = prims.At(ip)
        if obj and obj.InheritsFrom("TH1"):
            name = obj.GetName()
            if "h_corr" in name.lower() or "corr" in name.lower():
                h_data = obj.Clone("hData"); h_data.SetDirectory(0); break
    if not h_data:
        for ip in range(prims.GetSize()):
            obj = prims.At(ip)
            if obj and obj.InheritsFrom("TH1") and "hFit" in obj.GetName():
                h_data = obj.Clone("hData"); h_data.SetDirectory(0); break
    f.Close()
    return h_data, canvas

def extract_lm_covariance(lm_root_path, pt_cand_idx, pt_had_idx, mass_label):
    f = ROOT.TFile.Open(lm_root_path)
    if not f or f.IsZombie(): return None, None, None
    cov_name = f"hLMCovMatrix_PtCand{pt_cand_idx}_PtHad{pt_had_idx}_InvMassBin{mass_label}"
    h_cov = f.Get(cov_name)
    if not h_cov:
        f.Close(); return None, None, None
    npar = h_cov.GetNbinsX()
    mean_vec = np.zeros(npar)
    cov_mat = np.zeros((npar, npar))
    for i in range(npar):
        for j in range(npar):
            cov_mat[i, j] = h_cov.GetBinContent(i + 1, j + 1)
    par_name = f"hLMParValues_PtCand{pt_cand_idx}_PtHad{pt_had_idx}_InvMassBin{mass_label}"
    h_par = f.Get(par_name)
    if h_par:
        for i in range(npar): mean_vec[i] = h_par.GetBinContent(i + 1)
    f.Close()
    return mean_vec, cov_mat, npar

def read_lm_factor(canvas_root_path, pt_cand_idx, pt_had_idx, mass_label):
    f = ROOT.TFile.Open(canvas_root_path)
    if not f or f.IsZombie(): return None, None
    hname = f"hParValues_PtCand{pt_cand_idx}_PtHad{pt_had_idx}_InvMassBin{mass_label}"
    h = f.Get(hname)
    if not h:
        for k in f.GetListOfKeys():
            kn = k.GetName()
            if "hParValues" in kn and f"PtCand{pt_cand_idx}" in kn:
                h = f.Get(kn); break
    if not h:
        f.Close(); return None, None
    f_val = h.GetBinContent(1)
    f_err = h.GetBinError(1)
    f.Close()
    return f_val, f_err

def fit_with_lm_template(h_data, par, ry_val=1.0, ry_err=0.0, lm_ry_val=1.0, lm_ry_err=0.0, tag=""):
    h_fit = ROOT.TH1F(f"hFit_{tag}", "", h_data.GetNbinsX(),
                       h_data.GetXaxis().GetXmin(), h_data.GetXaxis().GetXmax())
    for ib in range(1, h_data.GetNbinsX() + 1):
        h_fit.SetBinContent(ib, h_data.GetBinContent(ib))
        h_fit.SetBinError(ib, h_data.GetBinError(ib))
    ROOT.SetOwnership(h_fit, False)

    c = ROOT.TCanvas(f"c_fit_{tag}", "", 1920, 1536)
    c.SetLeftMargin(0.12); c.SetRightMargin(0.03); c.SetBottomMargin(0.12); c.SetTopMargin(0.05)
    c.cd()

    fitter = DhCorrelationFitter(h_fit, F_MIN, F_MAX)
    ROOT.SetOwnership(fitter, True)
    fitter.SetHistoIsReflected(False)
    fitter.SetWithPedLM(WITH_PED_LM)
    fitter.SetFixLMFactor(FIX_LM_FACTOR)
    fitter.SetTempFunc(TEMP_FUNC)
    fitter.SetFixBaseline(FIX_BASELINE)
    fitter.SetBaselineUpOrDown(False, False)
    fitter.SetReflectedCorrHisto(True)
    fitter.SetFixMean(0)
    fitter.SetRyTrigger(ry_val, ry_err)
    fitter.SetLMRyTrigger(lm_ry_val, lm_ry_err)
    
    h_lm = ROOT.TH1D(f"hLM_{tag}", "", h_data.GetNbinsX(),
                      h_data.GetXaxis().GetXmin(), h_data.GetXaxis().GetXmax())
    f_tmp = build_lm_tf1(par, f"_tmp_{tag}")
    for ib in range(1, h_data.GetNbinsX() + 1):
        h_lm.SetBinContent(ib, f_tmp.Eval(h_lm.GetBinCenter(ib)))
    h_lm.SetDirectory(0)
    fitter.SetLMTemplate(h_lm)
    c_arr = (c_double * len(par))(*par)
    fitter.SetLMTemplateParams(len(par), c_arr)
    fitter.SetFuncType(DhCorrelationFitter.FunctionType(FIT_FUNC_TYPE))

    try: fitter.Fitting(True, False)
    except Exception as e:
        c.Close(); del fitter, h_fit; return None, None, None, None

    h_fit.GetYaxis().SetRangeUser(h_fit.GetMinimum() * 0.96, h_fit.GetMaximum() * 1.05)
    c.Update()
    h_data.SetStats(0); h_data.SetMinimum(0)
    h_data.SetMarkerStyle(ROOT.kFullCircle)
    h_data.SetMarkerColor(ROOT.kRed + 1)
    h_data.SetLineColor(ROOT.kRed + 1)
    h_data.Draw("same")

    # return prompt v2 instead of v2Delta

    return fitter.Getv2Delta(), fitter.Getv2DeltaError(), c, fitter

def save_trial(trial_dir, canvas, fitter, f_fit, f_lm,
               pt_cand_idx, pt_had_idx, mass_label, trial_id,
               pt_cand_min, pt_cand_max, pt_had_min, pt_had_max,
               mass_min, mass_max):
    os.makedirs(trial_dir, exist_ok=True)
    with open(os.path.join(trial_dir, "v2_result.txt"), "w") as fp:
        fp.write(f"v2_delta = {fitter.Getv2Delta():.8f} +/- {fitter.Getv2DeltaError():.8f}\n")
    canvas.SaveAs(os.path.join(trial_dir, "CorrFit.png"))
    canvas.SaveAs(os.path.join(trial_dir, "CorrFit.root"))

    if f_fit:
        with open(os.path.join(trial_dir, "parFit.txt"), "w") as fp:
            fp.write(f"# Trial {trial_id}  PtCand [{pt_cand_min:.1f},{pt_cand_max:.1f}]\n")
            for ip in range(f_fit.GetNpar()):
                fp.write(f"{ip:<6} {f_fit.GetParName(ip):<20} {f_fit.GetParameter(ip):<18.8f}\n")
    if f_lm:
        with open(os.path.join(trial_dir, "parLM.txt"), "w") as fp:
            fp.write(f"# Trial {trial_id}  LM tempFunc={TEMP_FUNC}\n")
            for ip in range(f_lm.GetNpar()):
                fp.write(f"{ip:<6} {f_lm.GetParName(ip):<20} {f_lm.GetParameter(ip):<18.8f}\n")

def _read_v2_from_txt(path):
    if not os.path.exists(path): return None, None
    with open(path, "r") as f:
        for line in f:
            if line.startswith("v2_delta"):
                parts = line.strip().split()
                return float(parts[2]), float(parts[4])
    return None, None

def _read_parlm_from_txt(path):
    if not os.path.exists(path): return None
    pars = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            # 完整过滤表头和分隔符
            if not line or line.startswith("#") or line.startswith("-") or line.startswith("Param"): 
                continue
            parts = line.rsplit(maxsplit=2)
            if len(parts) >= 3: 
                pars.append(float(parts[-2]))
    return pars if len(pars) == 5 else None

def draw_colored_scatter(x_vals, p_vals, title, xtitle, ytitle, is_par_plot=False):
    mg = ROOT.TMultiGraph()
    mg.SetTitle(f";{xtitle};{ytitle}")
    p_min, p_max = np.min(p_vals), np.max(p_vals)
    n_colors = ROOT.TColor.GetNumberOfColors()
    graphs = {}
    sorted_indices = np.argsort(p_vals)
    
    for idx in sorted_indices:
        x, p = x_vals[idx], p_vals[idx]
        frac = (p - p_min) / (p_max - p_min) if p_max > p_min else 0.5
        c_idx = int(frac * (n_colors - 1))
        
        if c_idx not in graphs:
            graphs[c_idx] = ROOT.TGraph()
            graphs[c_idx].SetMarkerStyle(ROOT.kFullCircle)
            graphs[c_idx].SetMarkerSize(1.2 if is_par_plot else 1.5) 
            
            # --- COLOR CACHE OPTIMIZATION ---
            color = ROOT.TColor.GetColorPalette(c_idx)
            if color not in _alpha_color_cache:
                _alpha_color_cache[color] = ROOT.TColor.GetColorTransparent(color, 0.6)
            graphs[c_idx].SetMarkerColor(_alpha_color_cache[color])
            
        n_pts = graphs[c_idx].GetN()
        graphs[c_idx].SetPoint(n_pts, x, p)
        
    for c_idx in sorted(graphs.keys()): 
        mg.Add(graphs[c_idx], "P")
        
    return mg, graphs 

def run_systematics(n_samples=100, load_from_exist=False, pt_only=0):
    N_DIGITS = int(math.ceil(math.log10(n_samples + 1)))
    pt_cand_bins = PT_CAND
    n_pt_cand = len(pt_cand_bins) - 1
    pt_had_idx = 1
    mass_label = 1

    all_v2_central, all_v2_err = np.zeros(n_pt_cand), np.zeros(n_pt_cand)
    all_prob_central = np.zeros(n_pt_cand)
    all_v2_samples, all_probs, all_fit_funcs = {}, {}, {}
    all_h_data, all_par_samples, all_lm_data, all_central_pars = {}, {}, {}, {}

    for i_pt in range(n_pt_cand):
        pt_cand_idx = i_pt + 1
        if pt_only > 0 and pt_cand_idx != pt_only: continue
        pt_min = pt_cand_bins[i_pt]
        pt_max = pt_cand_bins[i_pt + 1]
        print(f"\n{'='*60}")
        print(f"[PtCand {pt_cand_idx}/{n_pt_cand}]  [{pt_min}, {pt_max}] GeV/#it{{c}}")

        canvas_path = os.path.join(OUT_ROOT, f"CorrPhi{DMESON}_PtBinCand{pt_cand_idx}_PtBinAssoc{pt_had_idx}_InvMassBin{mass_label}.root")
        h_data, _ = extract_data_histo(canvas_path)
        if not h_data: continue
        all_h_data[i_pt] = h_data

        if load_from_exist:
            pt_dir = os.path.join(OUT_LM, f"PtCand_{pt_cand_idx}")
            if not os.path.isdir(pt_dir): continue

            lm_path = os.path.join(OUT_ROOT, f"hLMtemplate_PtBinCand{pt_cand_idx}_PtBinAssoc{pt_had_idx}.root")
            mean_vec, cov_mat, npar = extract_lm_covariance(lm_path, pt_cand_idx, pt_had_idx, mass_label)
            if mean_vec is None: continue

            central_path = os.path.join(pt_dir, f"trial_{0:0{N_DIGITS}d}", "v2_result.txt")
            v2_cent, v2_cent_err = _read_v2_from_txt(central_path)
            central_par = _read_parlm_from_txt(os.path.join(pt_dir, f"trial_{0:0{N_DIGITS}d}", "parLM.txt"))
            if central_par is None: central_par = mean_vec.copy()
            else: central_par = np.array(central_par)
            
            lm_func_central = build_lm_tf1(central_par, "lm_central")
            all_v2_central[i_pt] = v2_cent if v2_cent is not None else 0.
            all_v2_err[i_pt] = v2_cent_err if v2_cent_err is not None else 0.

            dist = multivariate_normal(mean=mean_vec, cov=cov_mat, allow_singular=True)
            v2_list, prob_list, lm_funcs, par_list, pdf_list = [], [], [], [], []
            with alive_bar(n_samples + 1, title=f"Loading PtCand {pt_cand_idx} trials") as bar:
                for itrial in range(0, n_samples + 1):
                    trial_path = os.path.join(pt_dir, f"trial_{itrial:0{N_DIGITS}d}")
                    if not os.path.isdir(trial_path): break
                    v2_val, _ = _read_v2_from_txt(os.path.join(trial_path, "v2_result.txt"))
                    par_lm = _read_parlm_from_txt(os.path.join(trial_path, "parLM.txt"))
                    if v2_val is None or par_lm is None: continue
                    v2_list.append(v2_val)
                    pdf_list.append(dist.pdf(par_lm))
                    par_list.append(par_lm)
                    lm_funcs.append(build_lm_tf1(par_lm, f"lm_{itrial}"))
                    bar()
            
            prob_list = pdf_list
            all_prob_central[i_pt] = prob_list[0] if prob_list else 1.0
            all_v2_samples[i_pt] = np.array(v2_list)
            all_probs[i_pt] = np.array(prob_list)
            all_fit_funcs[i_pt] = (lm_func_central, lm_funcs)
            all_par_samples[i_pt] = par_list
            all_lm_data[i_pt] = extract_lm_data_histo(lm_path)
            all_central_pars[i_pt] = central_par
            continue

        f_lm, f_lm_err = read_lm_factor(canvas_path, pt_cand_idx, pt_had_idx, mass_label)
        ry_val, lm_ry_val, ry_err, lm_ry_err = (f_lm, 1.0, 0.0, 0.0) if f_lm is not None else (1.0, 1.0, 0.0, 0.0)

        lm_path = os.path.join(OUT_ROOT, f"hLMtemplate_PtBinCand{pt_cand_idx}_PtBinAssoc{pt_had_idx}.root")
        mean_vec, cov_mat, npar = extract_lm_covariance(lm_path, pt_cand_idx, pt_had_idx, mass_label)
        if mean_vec is None: continue

        dist = multivariate_normal(mean=mean_vec, cov=cov_mat, allow_singular=True)
        samples = dist.rvs(size=n_samples)
        pdf_vals = dist.pdf(samples)
        probs = pdf_vals
        central_par = mean_vec.copy()
        all_prob_central[i_pt] = dist.pdf(central_par)

        v2_central, v2_central_err, c_cent, fitter_cent = fit_with_lm_template(
                h_data, central_par, ry_val=ry_val, ry_err=ry_err,
                lm_ry_val=lm_ry_val, lm_ry_err=lm_ry_err, tag="central"
        )
        all_v2_central[i_pt] = v2_central if v2_central is not None else 0.
        all_v2_err[i_pt] = v2_central_err if v2_central_err is not None else 0.

        pt_dir = os.path.join(OUT_LM, f"PtCand_{pt_cand_idx}")
        trial_central = os.path.join(pt_dir, f"trial_{0:0{N_DIGITS}d}")
        lm_func_central = build_lm_tf1(central_par, "lm_central")
        if c_cent and fitter_cent:
                save_trial(trial_central, c_cent, fitter_cent, fitter_cent.GetFitFunction(), fitter_cent.GetLMTemplateFunc(),
                           pt_cand_idx, pt_had_idx, mass_label, "central",
                           pt_min, pt_max, PT_HAD[0], PT_HAD[1], MASS_RANGE[i_pt][0], MASS_RANGE[i_pt][1])
                c_cent.Close(); del c_cent, fitter_cent
        gc.collect()

        v2_list, prob_list, fit_funcs, par_list_fit = [], [], [], []
        n_converged = 0
        with alive_bar(n_samples, title=f"Fitting PtCand {pt_cand_idx} trials") as bar:
            for isamp in range(n_samples):
                par = samples[isamp]
                # check the sampled template parameters:
                # tempFunc=4 -> par[1]/par[3] amplitudes, par[2]/par[4] Gaussian widths
                # tempFunc=5 -> par[1]/par[3] yields (areas), par[2]/par[4] von Mises kappas
                if par[1] <= 0 or par[3] <= 0: continue
                if TEMP_FUNC == 5:
                    # kappa is a concentration, not a width: below ~0.5 the component becomes an
                    # almost flat pedestal that is degenerate with the baseline (the fit is bounded
                    # at kappa >= 1, but the sampled errors can still push it down there)
                    if par[2] < 0.5 or par[4] < 0.5: continue
                elif par[2] <= 0.05 or par[4] <= 0.05: continue
                v2, _, c_var, fitter_var = fit_with_lm_template(
                    h_data, par, ry_val=ry_val, ry_err=ry_err, lm_ry_val=lm_ry_val, lm_ry_err=lm_ry_err, tag=f"s{isamp}"
                )
                if v2 is not None:
                    v2_list.append(v2); prob_list.append(probs[isamp]); n_converged += 1
                    fit_funcs.append(build_lm_tf1(par, f"lm_{isamp}")); par_list_fit.append(par.copy())
                    trial_dir = os.path.join(pt_dir, f"trial_{n_converged:0{N_DIGITS}d}")
                    save_trial(trial_dir, c_var, fitter_var, fitter_var.GetFitFunction(), fitter_var.GetLMTemplateFunc(),
                                pt_cand_idx, pt_had_idx, mass_label, f"{n_converged}",
                                pt_min, pt_max, PT_HAD[0], PT_HAD[1], MASS_RANGE[i_pt][0], MASS_RANGE[i_pt][1])
                    c_var.Close(); del c_var, fitter_var
                if isamp % 20 == 19:
                    ROOT.gROOT.GetListOfCanvases().Clear(); ROOT.gROOT.GetListOfFunctions().Clear()
                    ROOT.gDirectory = ROOT.nullptr; gc.collect()
                bar()

        all_v2_samples[i_pt] = np.array(v2_list); all_probs[i_pt] = np.array(prob_list)
        all_fit_funcs[i_pt] = (lm_func_central, fit_funcs); all_par_samples[i_pt] = par_list_fit
        all_lm_data[i_pt] = extract_lm_data_histo(lm_path); all_central_pars[i_pt] = central_par

    # ═══════════════════════════════════════════════════════════════════
    # ── PLOT ───────────────────────────────────────────────────────────
    # ═══════════════════════════════════════════════════════════════════
    f_out = ROOT.TFile.Open(os.path.join(OUT_LM, "probV2_PtCand.root"), "RECREATE")
    n_colors = ROOT.TColor.GetNumberOfColors()
    sigmas, mus = [], []

    with alive_bar(n_pt_cand, title="Plotting Probability vs v2") as bar:
        for i_pt in range(n_pt_cand):
            if i_pt not in all_v2_samples or len(all_v2_samples[i_pt]) < 5: 
                mus.append(all_v2_central[i_pt])
                sigmas.append(0.0)
                continue
            
            v2_arr = all_v2_samples[i_pt]
            prob_arr = all_probs[i_pt]
            p_min, p_max = np.min(prob_arr), np.max(prob_arr)
            d = f_out.GetDirectory(f"PtCand_{i_pt + 1}") or f_out.mkdir(f"PtCand_{i_pt + 1}")
            d.cd()

            v2_mean, v2_std = np.mean(v2_arr), np.std(v2_arr)
            v2_cent = all_v2_central[i_pt]
            prob_cent = all_prob_central[i_pt] if i_pt < len(all_prob_central) else p_max
            
            lat_pt = ROOT.TLatex(); lat_pt.SetNDC(); lat_pt.SetTextFont(42); lat_pt.SetTextSize(0.04)

            # ── Scatter Plot ──
            c2 = ROOT.TCanvas(f"c_pV2_PtCand{i_pt + 1}", "", 1920, 1536)
            c2.SetLeftMargin(0.12); c2.SetRightMargin(0.03); c2.SetBottomMargin(0.12); c2.SetTopMargin(0.05)
            mg_pv2, _h_refs_1 = draw_colored_scatter(v2_arr, prob_arr, "", "v_{2}^{#Delta}", "Probability density", is_par_plot=False)
            mg_pv2.Draw("A")
            ROOT.gPad.Update()
            
            y_min = p_min
            y_max = np.max([p_max, prob_cent]) * 1.4 
            y_bottom = y_min * 0.9 if y_min > 0 else 0
            mg_pv2.GetYaxis().SetRangeUser(y_bottom, y_max)
            
            line_mean = ROOT.TLine(v2_mean, y_bottom, v2_mean, y_max * 0.75)
            line_mean.SetLineColor(ROOT.kGreen + 1); line_mean.SetLineStyle(2); line_mean.Draw(); line_mean.SetLineWidth(3)
            line_std_low = ROOT.TLine(v2_mean - v2_std, y_bottom, v2_mean - v2_std, y_max * 0.75)
            line_std_low.SetLineColor(ROOT.kGreen + 1); line_std_low.SetLineStyle(3); line_std_low.Draw(); line_std_low.SetLineWidth(3)
            line_std_high = ROOT.TLine(v2_mean + v2_std, y_bottom, v2_mean + v2_std, y_max * 0.75)
            line_std_high.SetLineColor(ROOT.kGreen + 1); line_std_high.SetLineStyle(3); line_std_high.Draw(); line_std_high.SetLineWidth(3)

            line = ROOT.TLine(v2_cent, y_bottom, v2_cent, prob_cent)
            line.SetLineColor(ROOT.kBlack); line.SetLineStyle(5); line.Draw(); line.SetLineWidth(3)
            
            gr_cent_v2 = ROOT.TGraph(1)
            gr_cent_v2.SetPoint(0, v2_cent, prob_cent)
            gr_cent_v2.SetMarkerStyle(ROOT.kFullStar); gr_cent_v2.SetMarkerSize(3.0); gr_cent_v2.SetMarkerColor(ROOT.kRed + 1)
            gr_cent_v2.Draw("P SAME")

            leg_c2 = ROOT.TLegend(0.15, 0.65, 0.45, 0.82)
            leg_c2.SetBorderSize(0); leg_c2.SetFillStyle(0)
            leg_c2.AddEntry(gr_cent_v2, "Central Fit", "p")
            leg_c2.AddEntry(line_mean, f"Mean: {v2_mean:.5f}", "l")
            leg_c2.AddEntry(line_std_high, f"Sample #pm1#sigma ({v2_std:.5f})", "l")
            leg_c2.Draw()

            lat_pt.DrawLatex(0.15, 0.85, f"{PT_CAND[i_pt]} <#it{{p}}_{{T}}^{{D}} < {PT_CAND[i_pt+1]} GeV/#it{{c}}")
            c2.SaveAs(os.path.join(OUT_LM, f"probV2_PtCand{i_pt + 1}.png")); c2.Write(); c2.Close()

            # ── 1D Probability vs v2 Histogram ──
            c_h_pv2 = ROOT.TCanvas(f"c_hProbV2_PtCand{i_pt + 1}", "", 1920, 1536)
            c_h_pv2.SetLeftMargin(0.12); c_h_pv2.SetRightMargin(0.03); c_h_pv2.SetBottomMargin(0.12); c_h_pv2.SetTopMargin(0.05)
            
            diffs = np.abs(v2_arr - v2_cent)
            half_width = np.max(diffs) * 1.05
            v2_min_sym = v2_cent - half_width; v2_max_sym = v2_cent + half_width
            
            h_pv2 = ROOT.TH1D(f"hProbV2_PtCand{i_pt + 1}", f";v_{{2}}^{{#Delta}};Counts", 50, v2_min_sym, v2_max_sym)
            for v in v2_arr: h_pv2.Fill(v)
            h_pv2.SetFillColorAlpha(ROOT.kBlue - 7, 0.5)
            h_pv2.Draw("HIST")
            
            l_mean_hist = ROOT.TLine(v2_mean, 0, v2_mean, 1)
            l_mean_hist.SetLineColor(ROOT.kRed + 1); l_mean_hist.SetLineStyle(2); l_mean_hist.SetLineWidth(3)
            l_std_low_hist = ROOT.TLine(v2_mean - v2_std, 0, v2_mean - v2_std, 1)
            l_std_low_hist.SetLineColor(ROOT.kRed + 1); l_std_low_hist.SetLineStyle(5); l_std_low_hist.SetLineWidth(3)
            l_std_high_hist = ROOT.TLine(v2_mean + v2_std, 0, v2_mean + v2_std, 1)
            l_std_high_hist.SetLineColor(ROOT.kRed + 1); l_std_high_hist.SetLineStyle(5); l_std_high_hist.SetLineWidth(3)
            line_cent_hist = ROOT.TLine(v2_cent, 0, v2_cent, 1)
            line_cent_hist.SetLineColor(ROOT.kGreen + 1); line_cent_hist.SetLineWidth(3); line_cent_hist.SetLineStyle(2)
            
            fit_res = h_pv2.Fit("gaus", "LSQ0+")
            fit_func = h_pv2.GetFunction("gaus")
            cloned_fit_func = None

            mu, sigma = v2_mean, v2_std
            if fit_func:
                mu, sigma = fit_func.GetParameter(1), fit_func.GetParameter(2)
                cloned_fit_func = fit_func.Clone(f"fit_gaus_PtCand{i_pt + 1}")
                cloned_fit_func.SetLineColor(ROOT.kMagenta + 1); cloned_fit_func.SetLineWidth(2)
                cloned_fit_func.Draw("SAME")
            
            mus.append(mu)
            sigmas.append(sigma)

            max_y_val = h_pv2.GetMaximum()
            if fit_func and fit_func.GetMaximum() > max_y_val: max_y_val = fit_func.GetMaximum()
            h_pv2.SetMaximum(max_y_val * 1.5)
            
            # update l_mean_hist position to match Gauss fit mean for proper display
            l_mean_hist.SetX1(mu); l_mean_hist.SetX2(mu)

            l_mean_hist.SetY2(max_y_val * 1.1); l_std_low_hist.SetY2(max_y_val * 1.1); l_std_high_hist.SetY2(max_y_val * 1.1); line_cent_hist.SetY2(max_y_val * 1.1)
            l_mean_hist.Draw("SAME"); l_std_low_hist.Draw("SAME"); l_std_high_hist.Draw("SAME"); line_cent_hist.Draw("SAME")

            leg_hist = ROOT.TLegend(0.15, 0.60, 0.50, 0.82)
            leg_hist.SetBorderSize(0); leg_hist.SetFillStyle(0)
            
            leg_hist.AddEntry(h_pv2, "Sampled v_{2}^{#Delta}", "f")
            leg_hist.AddEntry(line_cent_hist, f"Central v_{{2}}^{{#Delta}} = {v2_cent:.5f}", "l")
            leg_hist.AddEntry(l_std_low_hist, f"Dev #sigma={v2_std:.5f}", "l")
            if cloned_fit_func: 
                leg_hist.AddEntry(l_mean_hist, f"Gauss Mean = {mu:.5f}", "l")
                leg_hist.AddEntry(cloned_fit_func, f"Gauss Fit #sigma = {sigma:.5f}", "l")
            
            leg_hist.Draw()
            lat_pt.DrawLatex(0.15, 0.85, f"{PT_CAND[i_pt]} < #it{{p}}_{{T}}^{{D}} < {PT_CAND[i_pt+1]} GeV/#it{{c}}")
            c_h_pv2.Update(); h_pv2.Write(); c_h_pv2.SaveAs(os.path.join(OUT_LM, f"hProbV2_PtCand{i_pt + 1}.png")); c_h_pv2.Write(f"c_hProbV2_PtCand{i_pt + 1}"); c_h_pv2.Close()

            # ── LM Fit Functions Overlay ──
            lm_central, lm_funcs = all_fit_funcs.get(i_pt, (None, []))
            h_lm_data = all_lm_data.get(i_pt)
            if h_lm_data and lm_central and lm_funcs:
                c_lm = ROOT.TCanvas(f"c_LMFits_PtCand{i_pt + 1}", "", 1920, 1536)
                c_lm.SetLeftMargin(0.12); c_lm.SetRightMargin(0.03); c_lm.SetBottomMargin(0.12); c_lm.SetTopMargin(0.05)
                h_lm_data.SetTitle(";#Delta#varphi;Counts"); h_lm_data.SetStats(0); h_lm_data.SetMarkerStyle(ROOT.kFullCircle)
                h_lm_data.SetMarkerSize(1.2); h_lm_data.SetMarkerColor(ROOT.kBlack); h_lm_data.SetLineColor(ROOT.kBlack)

                # Set dynamic Y range based on data points
                y_max = h_lm_data.GetBinContent(h_lm_data.GetMaximumBin())
                y_min = h_lm_data.GetBinContent(h_lm_data.GetMinimumBin())
                r_max = y_max * 1.12 if y_max > 0 else y_max * 0.8
                r_min = y_min * 0.95 if y_min > 0 else y_min * 1.1
                h_lm_data.GetYaxis().SetRangeUser(r_min, r_max)

                h_lm_data.Draw("E")
                
                sorted_indices = np.argsort(prob_arr)
                for idx in sorted_indices:
                    f, p = lm_funcs[idx], prob_arr[idx]
                    frac = (p - p_min) / (p_max - p_min) if p_max > p_min else 0.5
                    
                    # --- OPTIMIZE LM FIT COLORS TO AVOID MEMORY LEAK ---
                    cache_key = int(frac * 100) # Bucket colors to 100 bins
                    if cache_key not in _lm_color_cache:
                        c_idx = ROOT.TColor.GetColor(0.0, float(frac), float(1.0 - frac))
                        _lm_color_cache[cache_key] = ROOT.TColor.GetColorTransparent(c_idx, 0.15)
                    
                    f.SetLineColor(_lm_color_cache[cache_key])
                    f.SetLineWidth(1); f.Draw("same")
                
                lm_central.SetLineColor(ROOT.kBlack); lm_central.SetLineWidth(3); lm_central.Draw("same")  
                h_lm_data.Draw("same E") 
                
                leg_lm = ROOT.TLegend(0.40, 0.72, 0.92, 0.88)
                leg_lm.SetNColumns(2); leg_lm.SetBorderSize(0); leg_lm.SetFillStyle(0)
                leg_lm.AddEntry(h_lm_data, "Data", "pe"); leg_lm.AddEntry(lm_central, "Central LM Fit", "l")
                
                d_high = ROOT.TLine(); d_high.SetLineColor(ROOT.kGreen + 1); d_high.SetLineWidth(2)
                d_low  = ROOT.TLine(); d_low.SetLineColor(ROOT.kBlue - 1); d_low.SetLineWidth(2)
                leg_lm.AddEntry(d_high, "High Prob", "l")
                leg_lm.AddEntry(d_low, "Low Prob", "l")
                leg_lm.Draw()

                lat_pt.DrawLatex(0.15, 0.88, f"{PT_CAND[i_pt]} < #it{{p}}_{{T}}^{{D}} < {PT_CAND[i_pt+1]} GeV/#it{{c}}")
                c_lm.Update()
                out_lm_png = os.path.join(OUT_LM, "LMSys", f"LMFits_PtCand{i_pt + 1}.png")
                os.makedirs(os.path.dirname(out_lm_png), exist_ok=True)
                lm_central.Write(f"lm_central_PtCand{i_pt + 1}")
                c_lm.SaveAs(out_lm_png); c_lm.Write(f"c_LMFits_PtCand{i_pt + 1}"); c_lm.Close()
            bar()  
        f_out.cd()
    f_out.Close()

    # ═══════════════════════════════════════════════════════════════════
    # ── Final Global Systematic Summaries & gSyst/hSyst ──────────────
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n[PLOT] Summaries and Systematics...")
    pt_centers = np.array([0.5 * (PT_CAND[i] + PT_CAND[i + 1]) for i in range(n_pt_cand)])

    c1 = ROOT.TCanvas("c_v2_vs_pt", "", 1920, 1536)
    c1.SetLeftMargin(0.12); c1.SetRightMargin(0.03); c1.SetBottomMargin(0.12); c1.SetTopMargin(0.05)
    
    gr_cent = ROOT.TGraphErrors(n_pt_cand)
    gr_cent.SetName("gV2Central"); gr_cent.SetTitle(";#it{{p}}_{{T}}^{{D}} (GeV/#it{{c}});v_{2}^{#Delta}")
    gr_cent.SetMarkerStyle(ROOT.kFullCircle); gr_cent.SetMarkerColor(ROOT.kBlack); gr_cent.SetLineColor(ROOT.kBlack)

    gr_band = ROOT.TGraphAsymmErrors(n_pt_cand)
    gr_band.SetName("gV2SysBand"); gr_band.SetFillColorAlpha(ROOT.kBlue - 7, 0.35)
    
    gr_band_fit = ROOT.TGraphAsymmErrors(n_pt_cand)
    gr_band_fit.SetName("gV2SysBandFit"); gr_band_fit.SetFillColorAlpha(ROOT.kRed - 7, 0.35)

    gSyst = ROOT.TGraphAsymmErrors(n_pt_cand)
    gSyst.SetName("gSyst"); gSyst.SetTitle(f";#it{{p}}_{{T}}^{{D}} (GeV/#it{{c}});LM Syst. Unc. #sigma")
    
    hSyst = ROOT.TH1D("hSyst", f";#it{{p}}_{{T}}^{{D}} (GeV/#it{{c}});LM Syst. Unc. #sigma", n_pt_cand, array('d', PT_CAND))
    hSyst.SetStats(0)

    for i_pt in range(n_pt_cand):
        pt_c = pt_centers[i_pt]
        dx = 0.5 * (PT_CAND[i_pt + 1] - PT_CAND[i_pt])
        gr_cent.SetPoint(i_pt, pt_c, all_v2_central[i_pt])
        gr_cent.SetPointError(i_pt, dx, all_v2_err[i_pt])

        if i_pt in all_v2_samples and len(all_v2_samples[i_pt]) > 0:
            v2_arr = all_v2_samples[i_pt]
            v2_mean, v2_std = np.mean(v2_arr), np.std(v2_arr)
            gr_band.SetPoint(i_pt, pt_c, v2_mean)
            gr_band.SetPointError(i_pt, 0.3 * (PT_CAND[i_pt + 1] - PT_CAND[i_pt]), 0.3 * (PT_CAND[i_pt + 1] - PT_CAND[i_pt]), v2_std, v2_std)
        else:
            gr_band.SetPoint(i_pt, pt_c, all_v2_central[i_pt])
            gr_band.SetPointError(i_pt, 0., 0., 0., 0.)
            v2_std = 0.
            
        mu = mus[i_pt] if i_pt < len(mus) else all_v2_central[i_pt]
        sigma = sigmas[i_pt] if i_pt < len(sigmas) else v2_std
        
        gr_band_fit.SetPoint(i_pt, pt_c, mu)
        gr_band_fit.SetPointError(i_pt, 0.3 * (PT_CAND[i_pt + 1] - PT_CAND[i_pt]), 0.3 * (PT_CAND[i_pt + 1] - PT_CAND[i_pt]), sigma, sigma)

        gSyst.SetPoint(i_pt, pt_c, sigma)
        gSyst.SetPointError(i_pt, dx, dx, 0., 0.)
        
        hSyst.SetBinContent(i_pt + 1, sigma)
        hSyst.SetBinError(i_pt + 1, 0.)

    gr_band.SetTitle(";#it{p}_{T}^{D} (GeV/#it{c});V_{2#Delta}")
    gr_band.SetMinimum(-0.005); gr_band.SetMaximum(0.01)
    gr_band.Draw("A2")
    gr_band_fit.Draw("2 SAME")
    gr_cent.Draw("P SAME")

    leg = ROOT.TLegend(0.65, 0.72, 0.95, 0.88)
    leg.SetFillStyle(0); leg.SetBorderSize(0)
    leg.AddEntry(gr_cent, "Central fit", "lep")
    leg.AddEntry(gr_band, "Mean #pm 1#sigma band (Std)", "f")
    leg.AddEntry(gr_band_fit, "Gaussian #pm 1#sigma band (Fit)", "f")
    leg.Draw()

    c1.SaveAs(os.path.join(OUT_LM, "v2_vs_pT_LMSys.png"))

    # ── Plot Syst Error distribution vs pT (gSyst and hSyst) ──
    c_sys = ROOT.TCanvas("c_hSyst", "", 1920, 1536)
    c_sys.SetLeftMargin(0.12); c_sys.SetRightMargin(0.03); c_sys.SetBottomMargin(0.12); c_sys.SetTopMargin(0.05)
    
    hSyst.SetFillColorAlpha(ROOT.kRed - 7, 0.5)
    hSyst.SetFillStyle(3004) 
    hSyst.SetLineColor(ROOT.kRed + 1)
    
    max_err_overall = max([hSyst.GetBinContent(i+1) for i in range(n_pt_cand)]) if n_pt_cand > 0 else 0.1
    hSyst.GetYaxis().SetRangeUser(0., max_err_overall * 1.5)
    
    hSyst.Draw("HIST") 
    
    leg_sys = ROOT.TLegend(0.15, 0.75, 0.45, 0.88)
    leg_sys.SetBorderSize(0); leg_sys.SetFillStyle(0)
    leg_sys.AddEntry(hSyst, "LM Systematic Unc.", "f")
    leg_sys.Draw()
    
    c_sys.SaveAs(os.path.join(OUT_LM, "v2Syst_vs_pT.png"))

    # Write root file
    out_file = ROOT.TFile.Open(os.path.join(OUT_LM, "v2_vs_pT_LMSys.root"), "RECREATE")
    gr_cent.Write()
    gr_band.Write()
    gr_band_fit.Write()
    c1.Write()
    gSyst.Write()
    hSyst.Write()
    c_sys.Write()
    out_file.Close()

    # # ═══════════════════════════════════════════════════════════════════
    # # ── Parameter-Probability Scatter Plots ──
    # # ═══════════════════════════════════════════════════════════════════
    # print(f"\n[INFO] Parameter-probability scatter plots...")
    # par_names = ["Baseline", "Amplitude NS", "Sigma NS", "Amplitude AS", "Sigma AS"]
    # for i_pt in range(n_pt_cand):
    #     if i_pt not in all_par_samples or not all_par_samples[i_pt]: continue
        
    #     pars, probs_arr = all_par_samples[i_pt], all_probs.get(i_pt, [])
    #     p_max = np.max(probs_arr) if len(probs_arr) > 0 else 1.0
    #     prob_cent = all_prob_central[i_pt] if i_pt < len(all_prob_central) else p_max
    #     curr_central_par = all_central_pars[i_pt] if i_pt in all_central_pars else None

    #     for ipar in range(5):
    #         print(f"  [INFO] PtCand {i_pt + 1}, Parameter {par_names[ipar]} vs Probability...")
    #         c_pp = ROOT.TCanvas(f"c_ParProb_PtCand{i_pt+1}_par{ipar}", "", 1920, 1536)
    #         c_pp.SetLeftMargin(0.12); c_pp.SetRightMargin(0.03); c_pp.SetBottomMargin(0.12); c_pp.SetTopMargin(0.05)
            
    #         x_vals = np.array([p[ipar] for p in pars])
    #         x_min, x_max = np.min(x_vals), np.max(x_vals)
    #         if x_min == x_max: pad = max(abs(x_min) * 0.1, 1e-6); x_min, x_max = x_min - pad, x_max + pad
                
    #         mg_pp, _h_refs_2 = draw_colored_scatter(x_vals, probs_arr, "", par_names[ipar], "Probability density", is_par_plot=True)
    #         mg_pp.Draw("A")
            
    #         margin = (x_max - x_min) * 0.1
    #         mg_pp.GetXaxis().SetLimits(x_min - margin, x_max + margin)
    #         mg_pp.GetYaxis().SetRangeUser(0, np.max([p_max, prob_cent]) * 1.3)
    #         ROOT.gPad.Update()
            
    #         cent_val = curr_central_par[ipar] if curr_central_par is not None else np.mean(x_vals)
    #         line = ROOT.TLine(cent_val, 0, cent_val, prob_cent); line.SetLineColor(ROOT.kBlack); line.SetLineStyle(5); line.Draw(); line.SetLineWidth(3)
            
    #         gr_cent_p = ROOT.TGraph(1); gr_cent_p.SetPoint(0, cent_val, prob_cent)
    #         gr_cent_p.SetMarkerStyle(ROOT.kFullStar); gr_cent_p.SetMarkerSize(3.0); gr_cent_p.SetMarkerColor(ROOT.kRed + 1)
    #         gr_cent_p.Draw("P SAME")

    #         lat_pp = ROOT.TLatex(); lat_pp.SetNDC(); lat_pp.SetTextFont(42); lat_pp.SetTextSize(0.04)
    #         lat_pp.DrawLatex(0.15, 0.88, f"{PT_CAND[i_pt]} < #it{{p}}_{{T}}^{{D}} < {PT_CAND[i_pt+1]} GeV/#it{{c}}")
    #         c_pp.Update()
            
    #         out_png = os.path.join(OUT_LM, "LMSys", f"ParProb_{par_names[ipar].replace(' ','_')}_PtCand{i_pt+1}.png")
    #         os.makedirs(os.path.dirname(out_png), exist_ok=True)
    #         c_pp.SaveAs(out_png)
            
    #         # --- FORCE CLEANUP TO PREVENT MEMORY LEAK ---
    #         c_pp.Close()
    #         del mg_pp, _h_refs_2, line, gr_cent_p, c_pp, lat_pp

    print(f"\n[DONE] Outputs in: {OUT_LM}")
    print(f"  v2_vs_pT_LMSys.png         — v2 vs pT with systematics band")
    print(f"  v2Syst_vs_pT.png           — Systematics error distribution vs pT")
    print(f"  probV2_PtCand*.png         — probability vs v2 per pT bin (Colored & Sorted)")
    print(f"  hProbV2_PtCand*.png        — 1D counts distribution of v2")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LM fit systematic uncertainty via parameter sampling")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES, help="Number of samples")
    parser.add_argument("--pt-only", type=int, default=0, help="Process only one PtCand bin")
    parser.add_argument("--load-from-exist", '-l', action="store_true", help="Load v2 from exist")
    args = parser.parse_args()
    run_systematics(n_samples=args.n_samples, load_from_exist=args.load_from_exist, pt_only=args.pt_only)