#!/usr/bin/env python3
"""
Compute prompt-hadron v2 with systematic uncertainty from ratio variation.

Scans the ratio r = v2_FD / v2_prompt from 0 to 1, computes v2_prompt for each r,
and derives:
  - sys_unc: half the spread (max-min)/2 across all r values per pT bin
  - central value at r=0.5 with statistical error
  - total uncertainty: tot_unc = sqrt(stat_unc^2 + sys_unc^2)

Output: a ROOT file containing the central result, sys_unc, and tot_unc.
Also saves a summary PNG for quick inspection.
"""

import argparse
import math
import ctypes
import ROOT
import yaml

ROOT.gROOT.SetBatch(True)


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
    """Read x, y, ex, ey from a TH1 or TGraph* object."""
    # TH1
    if obj.InheritsFrom("TH1"):
        xs, ys, exs, eys = [], [], [], []
        for i in range(1, obj.GetNbinsX() + 1):
            xs.append(float(obj.GetBinCenter(i)))
            exs.append(float(0.5 * obj.GetBinWidth(i)))
            ys.append(float(obj.GetBinContent(i)))
            eys.append(float(obj.GetBinError(i)))
        return xs, ys, exs, eys

    # TGraph*
    if obj.InheritsFrom("TGraph"):
        n = obj.GetN()
        xs, ys, exs, eys = [], [], [], []
        for i in range(n):
            x = ctypes.c_double(0.0)
            y = ctypes.c_double(0.0)
            obj.GetPoint(i, x, y)
            xs.append(float(x.value))
            ys.append(float(y.value))

            exl = exh = eyl = eyh = 0.0
            if obj.InheritsFrom("TGraphAsymmErrors"):
                exl, exh = obj.GetErrorXlow(i), obj.GetErrorXhigh(i)
                eyl, eyh = obj.GetErrorYlow(i), obj.GetErrorYhigh(i)
            elif obj.InheritsFrom("TGraphErrors"):
                exl = exh = obj.GetErrorX(i)
                eyl = eyh = obj.GetErrorY(i)

            exs.append(0.5 * (exl + exh))
            eys.append(0.5 * (eyl + eyh))
        return xs, ys, exs, eys

    raise TypeError(f"Unsupported object type: {obj.ClassName()}")


def check_fraction_sum(fp, ffd, tol):
    bad = [(i, fp[i] + ffd[i]) for i in range(len(fp)) if abs(fp[i] + ffd[i] - 1.0) > tol]
    if bad:
        msg = ", ".join([f"(i={i}, sum={s:.3f})" for i, s in bad[:10]])
        raise RuntimeError(f"Fraction sum check failed (tol={tol}). Examples: {msg}")


def v2prompt_ratio(v2obs, ev2obs, fp, efp, ffd, effd, r, eps=1e-12):
    """v2p = v2obs / (fp + r * ffd)"""
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


def write_th1(out_file, name, title, xs, exs, ys, eys, xtitle="p_{T} (GeV/c)", ytitle="v_{2}^{prompt}"):
    """Write a TH1F from bin centers, half-widths, values and errors."""
    n = len(xs)
    binning = ROOT.TArrayD(n + 1)
    for i in range(n):
        binning[i] = xs[i] - exs[i]
    binning[n] = xs[-1] + exs[-1]

    h = ROOT.TH1F(name, title, n, binning.GetArray())
    h.SetDirectory(0)
    h.GetXaxis().SetTitle(xtitle)
    h.GetYaxis().SetTitle(ytitle)
    for i, (y, ey) in enumerate(zip(ys, eys)):
        h.SetBinContent(i + 1, y)
        h.SetBinError(i + 1, ey)

    # Write to file (APPEND mode — file handled externally)
    out_file.cd()
    h.Write()
    return h


def write_graph(out_file, name, title, xs, ys, exs, eys):
    """Write a TGraphErrors."""
    g = ROOT.TGraphErrors(len(xs))
    g.SetName(name)
    g.SetTitle(title)
    for i, (x, y, ex, ey) in enumerate(zip(xs, ys, exs, eys)):
        g.SetPoint(i, x, y)
        g.SetPointError(i, ex, ey)
    out_file.cd()
    g.Write()
    return g


def write_graph_asymm(out_file, name, title, xs, ys, exs, eylows, eyhighs):
    """Write a TGraphAsymmErrors with asymmetric y-errors."""
    g = ROOT.TGraphAsymmErrors(len(xs))
    g.SetName(name)
    g.SetTitle(title)
    for i, (x, y, ex, eylo, eyhi) in enumerate(zip(xs, ys, exs, eylows, eyhighs)):
        g.SetPoint(i, x, y)
        g.SetPointError(i, ex, ex, eylo, eyhi)
    out_file.cd()
    g.Write()
    return g


# ── Main ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Compute prompt v2 with systematic uncertainty from ratio scan"
    )
    ap.add_argument("config", help="YAML config file")
    ap.add_argument(
        "--r-step", type=float, default=0.05,
        help="Step size for ratio scan (default: 0.05)"
    )
    ap.add_argument(
        "--r-start", type=float, default=0.0,
        help="Starting ratio (default: 0.0)"
    )
    ap.add_argument(
        "--r-end", type=float, default=1.0,
        help="Ending ratio (default: 1.0)"
    )
    ap.add_argument(
        "--central-r", type=float, default=0.5,
        help="Central ratio for statistical error and central value (default: 0.5)"
    )
    ap.add_argument(
        "--sys-method", type=str, default="asymmetric",
        choices=["half-range", "rms", "asymmetric", "geom-mean"],
        help="Systematic uncertainty method: half-range=(max-min)/2, "
             "rms=sqrt(mean((v-c)^2)), asymmetric=max|dev| per side, "
             "geom-mean=sqrt(max_dev*min_dev) (default: asymmetric)"
    )
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ── Read inputs ──────────────────────────────────────────────────
    frac_file = yget(cfg, "input.fractions.file", required=True)
    v2_file = yget(cfg, "input.v2obs.file", required=True)
    v2_name = yget(cfg, "input.v2obs.object", required=True)

    out_file = yget(cfg, "output.file", "v2_prompt_sys.root")
    out_name = yget(cfg, "output.object", "gV2Prompt")
    out_title = yget(cfg, "output.title", "v2^{prompt}")

    check_n = bool(yget(cfg, "checks.check_npoints", True))
    check_sum = bool(yget(cfg, "checks.check_fraction_sum", True))
    tol_sum = float(yget(cfg, "checks.fraction_sum_tolerance", 0.05))

    # fractions (ratio method)
    fp_name = yget(cfg, "input.fractions.objects_ratio.f_prompt", required=True)
    ffd_name = yget(cfg, "input.fractions.objects_ratio.f_FD", required=True)

    ffrac = ROOT.TFile.Open(frac_file)
    fv2 = ROOT.TFile.Open(v2_file)
    if not ffrac or ffrac.IsZombie():
        raise RuntimeError(f"Cannot open {frac_file}")
    if not fv2 or fv2.IsZombie():
        raise RuntimeError(f"Cannot open {v2_file}")

    # v2obs
    o_v2 = fv2.Get(v2_name)
    if not o_v2:
        raise RuntimeError(f"Missing {v2_name} in {v2_file}")
    x_v2, v2obs, ex_v2, ev2obs = read_xy(o_v2)

    # fractions
    o_fp = ffrac.Get(fp_name)
    o_ffd = ffrac.Get(ffd_name)
    if not o_fp:
        raise RuntimeError(f"Missing {fp_name} in {frac_file}")
    if not o_ffd:
        raise RuntimeError(f"Missing {ffd_name} in {frac_file}")

    x_fp, fp, _, efp = read_xy(o_fp)
    x_ffd, ffd, _, effd = read_xy(o_ffd)

    # align lengths
    n_pts = min(len(fp), len(ffd), len(v2obs))
    if check_n and not (len(fp) == len(ffd) == len(v2obs)):
        print(f"[WARN] Npoints mismatch: fp={len(fp)} ffd={len(ffd)} v2={len(v2obs)} → using first {n_pts}")
        v2obs = v2obs[:n_pts]
        ev2obs = ev2obs[:n_pts]
        fp = fp[:n_pts]
        efp = efp[:n_pts]
        ffd = ffd[:n_pts]
        effd = effd[:n_pts]
        x_v2 = x_v2[:n_pts]
        ex_v2 = ex_v2[:n_pts]

    if check_sum:
        check_fraction_sum(fp, ffd, tol_sum)

    # ── Ratio scan ────────────────────────────────────────────────────
    r_vals = []
    import numpy as np
    r_vals = np.arange(args.r_start+args.r_step, args.r_end, args.r_step)
    print(r_vals)

    print(f"Scanning r = {r_vals[0]:.3f} → {r_vals[-1]:.3f} (step={args.r_step}, N={len(r_vals)})")
    print(f"Central r = {args.central_r}")

    # Store all v2_prompt values: all_v2p[i_bin][i_r]
    all_v2p = [[] for _ in range(n_pts)]
    all_ev2p = [[] for _ in range(n_pts)]

    for ir, r_val in enumerate(r_vals):
        v2p, ev2p = v2prompt_ratio(v2obs, ev2obs, fp, efp, ffd, effd, r_val)
        for ib in range(n_pts):
            all_v2p[ib].append(v2p[ib])
            all_ev2p[ib].append(ev2p[ib])

    # ── Compute central values (r = central_r) ─────────────────────
    idx_central = None
    for ir, r_val in enumerate(r_vals):
        # if abs(r_val - args.central_r) < 1e-9:
        if r_val == args.central_r:
            idx_central = ir
            break
    if idx_central is None:
        raise RuntimeError(f"central_r={args.central_r} not found in scan. "
                           f"Ensure it is a multiple of r_step={args.r_step}.")

    central_v2p = [all_v2p[ib][idx_central] for ib in range(n_pts)]
    central_ev2p = [all_ev2p[ib][idx_central] for ib in range(n_pts)]

    # ── Ratio scan envelope (always full range, for band visualization) ──
    scan_lows = []
    scan_highs = []
    for ib in range(n_pts):
        vals = [v for v in all_v2p[ib] if not math.isnan(v)]
        if len(vals) < 2:
            scan_lows.append(0.0)
            scan_highs.append(0.0)
        else:
            scan_lows.append(max(central_v2p[ib] - min(vals), 0.0))
            scan_highs.append(max(max(vals) - central_v2p[ib], 0.0))

    # ── Compute systematic uncertainty ───────────────────────────────
    sys_lows = []
    sys_highs = []
    print(f"\nSys method: {args.sys_method}")
    for ib in range(n_pts):
        vals = [v for v in all_v2p[ib] if not math.isnan(v)]
        if len(vals) < 2:
            sys_lows.append(0.0)
            sys_highs.append(0.0)
        elif args.sys_method == "half-range":
            hr = 0.5 * (max(vals) - min(vals))
            sys_lows.append(hr)
            sys_highs.append(hr)
        elif args.sys_method == "rms":
            rms = math.sqrt(sum((v - central_v2p[ib]) ** 2 for v in vals) / len(vals))
            sys_lows.append(rms)
            sys_highs.append(rms)
        elif args.sys_method == "geom-mean":
            max_dev = max(vals) - central_v2p[ib]
            min_dev = central_v2p[ib] - min(vals)
            gm = math.sqrt(max(max_dev, 0.0) * max(min_dev, 0.0))
            sys_lows.append(gm)
            sys_highs.append(gm)
        else:  # asymmetric
            max_dev = 0.68*max(max(vals) - central_v2p[ib], 0.0)
            min_dev = 0.68*max(central_v2p[ib] - min(vals), 0.0)
            sys_lows.append(min_dev)
            sys_highs.append(max_dev)

    sys_v2p = [max(lo, hi) for lo, hi in zip(sys_lows, sys_highs)]  # conservative symmetric for tot

    # ── Compute total uncertainty (asymmetric) ──────────────────────────
    tot_lows = []
    tot_highs = []
    for ib in range(n_pts):
        tot_lows.append(math.sqrt(central_ev2p[ib] ** 2 + sys_lows[ib] ** 2))
        tot_highs.append(math.sqrt(central_ev2p[ib] ** 2 + sys_highs[ib] ** 2))
    tot_unc_conservative = [max(lo, hi) for lo, hi in zip(tot_lows, tot_highs)]

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'pT':>8s}  {'central':>10s}  {'stat':>8s}  {'sys_low':>8s}  {'sys_high':>8s}  {'tot_low':>8s}  {'tot_high':>8s}")
    print("-" * 71)
    for ib in range(n_pts):
        print(f"{x_v2[ib]:8.2f}  {central_v2p[ib]:10.5f}  {central_ev2p[ib]:8.5f}  {sys_lows[ib]:8.5f}  {sys_highs[ib]:8.5f}  {tot_lows[ib]:8.5f}  {tot_highs[ib]:8.5f}")

    # ── Write output ──────────────────────────────────────────────────
    fout = ROOT.TFile.Open(out_file, "RECREATE")

    # Central result (TH1F with stat errors)
    h_central = write_th1(
        fout, "hV2Prompt",
        out_title + f"; r={args.central_r}",
        x_v2, ex_v2, central_v2p, central_ev2p,
        xtitle="p_{T} (GeV/c)", ytitle="v_{2}^{prompt}"
    )

    # Also as TGraphErrors
    write_graph(
        fout, "gV2Prompt",
        out_title + f"; r={args.central_r}",
        x_v2, central_v2p, ex_v2, central_ev2p
    )

    # Systematic uncertainty — always asymmetric (low/high); equal for RMS/half-range
    g_sys_asym = write_graph_asymm(
        fout, "gSysUncAsym",
        f"Sys. unc. from ratio scan ({args.sys_method})",
        x_v2, [0.0] * n_pts, ex_v2, sys_lows, sys_highs
    )

    # Systematic uncertainty — graph form "gSys" (0 +/- syst band)
    g_sys = g_sys_asym.Clone("gSys")
    g_sys.Write()

    # Systematic uncertainty — histogram form "hSys" (conservative symmetric, error=0)
    h_sys = write_th1(
        fout, "hSys",
        f"Sys. unc. from ratio scan ({args.sys_method})",
        x_v2, ex_v2, sys_v2p, [0.0] * n_pts,
        xtitle="p_{T} (GeV/c)", ytitle="syst. unc."
    )

    # Total uncertainty as TGraphAsymmErrors
    g_tot_asym = write_graph_asymm(
        fout, "gTotUnc",
        f"Total uncertainty ({args.sys_method})",
        x_v2, [0.0] * n_pts, ex_v2, tot_lows, tot_highs
    )

    # Also write all individual ratio curves (optional, for debugging)
    dir_scan = fout.mkdir("ratio_scan")
    dir_scan.cd()
    for ir, r_val in enumerate(r_vals):
        g = ROOT.TGraphErrors(n_pts)
        g.SetName(f"gV2Prompt_r{r_val:.2f}".replace(".", "p"))
        g.SetTitle(f"v2_prompt; r={r_val:.3f}")
        for ib in range(n_pts):
            g.SetPoint(ib, x_v2[ib], all_v2p[ib][ir])
            g.SetPointError(ib, ex_v2[ib], all_ev2p[ib][ir])
        g.Write()

    # ── Draw summary canvases ──────────────────────────────────────────────────
    fout.cd()  # back to root file (ratio_scan loop left gDirectory inside the subdir)

    # (1) systematic uncertainty vs pT (hSys filled bars)
    c_sys = ROOT.TCanvas("c_sys", "Systematic uncertainty", 1200, 800)
    c_sys.SetTopMargin(0.06)
    c_sys.SetRightMargin(0.06)
    c_sys.SetBottomMargin(0.14)
    c_sys.SetLeftMargin(0.14)
    g_sys.SetFillColorAlpha(ROOT.kOrange + 2, 0.6)
    g_sys.SetFillStyle(1001)
    g_sys.SetLineColor(ROOT.kOrange + 2)
    g_sys.SetMarkerStyle(20)
    g_sys.SetMarkerColor(ROOT.kOrange + 2)
    max_sys = max(sys_v2p) if sys_v2p else 0.01
    g_sys.Draw("A2")
    g_sys.GetXaxis().SetTitle("p_{T} (GeV/c)")
    g_sys.GetYaxis().SetTitle("Systematic uncertainty")
    ROOT.gPad.SetGrid()
    c_sys.Write("c_syst_vs_pT")
    c_sys.SaveAs(out_file.replace(".root", "_syst_vs_pT.png"))

    # (2) v2 with statistical + systematic uncertainties
    c_v2 = ROOT.TCanvas("c_v2", "v2 with stat + syst", 1200, 800)
    c_v2.SetTopMargin(0.06)
    c_v2.SetRightMargin(0.06)
    c_v2.SetBottomMargin(0.14)
    c_v2.SetLeftMargin(0.14)

    band = ROOT.TGraphAsymmErrors(n_pts)
    band.SetName("band_syst")
    for ib in range(n_pts):
        band.SetPoint(ib, x_v2[ib], central_v2p[ib])
        band.SetPointError(ib, ex_v2[ib], ex_v2[ib], sys_lows[ib]/0.68, sys_highs[ib]/0.68)
    band.SetFillColorAlpha(ROOT.kOrange + 2, 0.35)
    band.SetFillStyle(1001)
    band.SetLineWidth(0)
    band.SetTitle(";#it{p}_{T} (GeV/#it{c});#it{v}_{2}^{prompt}")
    band.SetMinimum(-0.02)
    band.SetMaximum(0.18)
    band.Draw("A2")

    g_cent = ROOT.TGraphErrors(n_pts)
    g_cent.SetName("gV2Prompt_plot")
    for ib in range(n_pts):
        g_cent.SetPoint(ib, x_v2[ib], central_v2p[ib])
        g_cent.SetPointError(ib, ex_v2[ib], central_ev2p[ib])
    g_cent.SetMarkerStyle(20)
    g_cent.SetMarkerSize(1.2)
    g_cent.SetMarkerColor(ROOT.kBlack)
    g_cent.SetLineColor(ROOT.kBlack)
    g_cent.SetLineWidth(2)
    g_cent.Draw("P E1 same")

    leg = ROOT.TLegend(0.2, 0.72, 0.5, 0.9)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.AddEntry(g_cent, "central (stat. unc.)", "lep")
    leg.AddEntry(band, f"Possible value range", "f")
    leg.Draw()
    ROOT.gPad.SetGrid()
    c_v2.Write("c_v2_stat_syst")
    c_v2.SaveAs(out_file.replace(".root", "_v2_stat_syst.png"))

    print(f"\n[OK] Output written to {out_file}")
    print(f"[OK] Summary PNG: {out_file.replace('.root', '_syst_vs_pT.png')}")
    print(f"[OK] Summary PNG: {out_file.replace('.root', '_v2_stat_syst.png')}")

    fv2.Close()
    ffrac.Close()
    fout.Close()


if __name__ == "__main__":
    main()
