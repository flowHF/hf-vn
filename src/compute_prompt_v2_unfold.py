"""
Compute prompt-hadron v2 from the observed v2 using fraction-based unfolding.

The script supports multiple unfolding assumptions:
  - ratio method: v2_FD = r * v2_prompt
  - fixed-FD method: v2_FD = const

All inputs and assumptions are configured via a YAML file.
"""

#!/usr/bin/env python3
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

def write_graph(out_file, name, title, xs, ys, exs, eys):
    g = ROOT.TGraphErrors()
    g.SetName(name)
    g.SetTitle(title)
    for i, (x, y, ex, ey) in enumerate(zip(xs, ys, exs, eys)):
        g.SetPoint(i, x, y)
        g.SetPointError(i, ex, ey)
    print(xs)
    print(exs)
    binning = ROOT.TArrayD(len(xs) + 1)
    for i in range(len(xs)):
        binning[i] = xs[i] - exs[i]
    binning[len(xs)] = xs[-1] + exs[-1]
    h = ROOT.TH1F("h", "h", len(xs), binning.GetArray())
    h.SetDirectory(0)
    h.GetXaxis().SetTitle("p_{T} (GeV/c)")
    h.GetYaxis().SetTitle("v_{2}^{prompt}")
    for i, (x, y, ex, ey) in enumerate(zip(xs, ys, exs, eys)):
        h.SetBinContent(i + 1, y)
        h.SetBinError(i + 1, ey)
    h.SetName(name.replace("g", "h"))
    

    fout = ROOT.TFile.Open(out_file, "RECREATE")
    g.Write()
    h.Write()
    fout.Close()

def v2prompt_ratio(v2obs, ev2obs, fp, efp, ffd, effd, r, eps=1e-12):
    # v2p = v2obs / (fp + r ffd)
    v2p, ev2p = [], []
    for v, sv, a, sa, b, sb in zip(v2obs, ev2obs, fp, efp, ffd, effd):
        D = a + r * b
        if abs(D) < eps:
            v2p.append(float("nan")); ev2p.append(float("nan")); continue
        y = v / D
        sD2 = sa*sa + (r*sb)*(r*sb)
        sy2 = (sv / D)**2 + (v*v * sD2) / (D**4)
        v2p.append(y)
        ev2p.append(math.sqrt(max(0.0, sy2)))
    return v2p, ev2p

def v2prompt_fixedFD(v2obs, ev2obs, fp, efp, ffd, effd, v2fd, sv2fd=0.0, min_fp=1e-6):
    # v2p = (v2obs - ffd*v2fd) / fp
    v2p, ev2p = [], []
    for v, sv, a, sa, b, sb in zip(v2obs, ev2obs, fp, efp, ffd, effd):
        if a < min_fp:
            v2p.append(float("nan")); ev2p.append(float("nan")); continue

        num = v - b * v2fd
        y = num / a

        dv = 1.0 / a
        da = -num / (a * a)
        db = -v2fd / a
        dc = -b / a

        sy2 = (dv*sv)**2 + (da*sa)**2 + (db*sb)**2 + (dc*sv2fd)**2
        v2p.append(y)
        ev2p.append(math.sqrt(max(0.0, sy2)))
    return v2p, ev2p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="YAML config file")
    ap.add_argument("final_results_file", help="Optional final results file from DeltaPhiBinning method", nargs="?")
    ap.add_argument("--outpath", help="Optional output path for the results", default=None)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    method = str(yget(cfg, "mode.method", "ratio")).strip()

    frac_file = yget(cfg, "input.fractions.file", required=True)
    v2_file   = yget(cfg, "input.v2obs.file", required=True)
    final_results_file = args.final_results_file
    v2_file  = final_results_file if final_results_file else v2_file
    v2_name   = yget(cfg, "input.v2obs.object", required=True)

    out_file  = yget(cfg, "output.file", "v2_prompt.root")
    if args.outpath:
        import os
        out_file = os.path.join(args.outpath, os.path.basename(out_file))
    out_name  = yget(cfg, "output.object", "gV2Prompt")
    out_title = yget(cfg, "output.title", "v2^{prompt}")

    check_n   = bool(yget(cfg, "checks.check_npoints", True))
    check_sum = bool(yget(cfg, "checks.check_fraction_sum", True))
    tol_sum   = float(yget(cfg, "checks.fraction_sum_tolerance", 0.05))
    min_fp    = float(yget(cfg, "checks.min_f_prompt", 1e-6))

    ffrac = ROOT.TFile.Open(frac_file)
    fv2   = ROOT.TFile.Open(v2_file)
    if not ffrac or ffrac.IsZombie(): raise RuntimeError(f"Cannot open {frac_file}")
    if not fv2 or fv2.IsZombie():     raise RuntimeError(f"Cannot open {v2_file}")

    # v2obs
    o_v2 = fv2.Get(v2_name)
    if not o_v2:
        raise RuntimeError(f"Missing {v2_name} in {v2_file}")
    x_v2, v2obs, ex_v2, ev2obs = read_xy(o_v2)

    # fractions: choose by method
    if method == "ratio":
        fp_name  = yget(cfg, "input.fractions.objects_ratio.f_prompt", required=True)
        ffd_name = yget(cfg, "input.fractions.objects_ratio.f_FD", required=True)
        r = float(yget(cfg, "assumption.r", required=True))
        info = f"method=ratio, fractions=({fp_name},{ffd_name}), r={r}"

    elif method == "fixedFD":
        fp_name  = yget(cfg, "input.fractions.objects_fixedFD.f_prompt", required=True)
        ffd_name = yget(cfg, "input.fractions.objects_fixedFD.f_FD", required=True)
        v2fd   = float(yget(cfg, "assumption.v2_FD_fixed", required=True))
        sv2fd  = float(yget(cfg, "assumption.v2_FD_fixed_err", 0.0))
        info = f"method=fixedFD, fractions=({fp_name},{ffd_name}), v2FD={v2fd}"

    else:
        raise ValueError("mode.method must be 'ratio' or 'fixedFD'")

    o_fp  = ffrac.Get(fp_name)
    o_ffd = ffrac.Get(ffd_name)
    if not o_fp:  raise RuntimeError(f"Missing {fp_name} in {frac_file}")
    if not o_ffd: raise RuntimeError(f"Missing {ffd_name} in {frac_file}")

    x_fp,  fp,  _, efp   = read_xy(o_fp)
    x_ffd, ffd, _, effd  = read_xy(o_ffd)

    if check_n and not (len(fp) == len(ffd) == len(v2obs)):
        v2obs = v2obs[:min(len(fp), len(ffd))]
        ev2obs = ev2obs[:min(len(fp), len(ffd))]
        # raise RuntimeError(f"Npoints mismatch: fp={len(fp)} ffd={len(ffd)} v2={len(v2obs)}")

    if check_sum:
        check_fraction_sum(fp, ffd, tol_sum)

    if method == "ratio":
        v2p, ev2p = v2prompt_ratio(v2obs, ev2obs, fp, efp, ffd, effd, r)
        title = out_title + f";({info})"
    else:
        v2fd   = float(yget(cfg, "assumption.v2_FD_fixed", required=True))
        sv2fd  = float(yget(cfg, "assumption.v2_FD_fixed_err", 0.0))
        v2p, ev2p = v2prompt_fixedFD(v2obs, ev2obs, fp, efp, ffd, effd, v2fd, sv2fd=sv2fd, min_fp=min_fp)
        title = out_title + f";({info})"

    write_graph(out_file, out_name, title, x_v2, v2p, ex_v2, ev2p)
    print(f"[OK] {info}")
    print(f"[OK] wrote {out_file}::{out_name}")

if __name__ == "__main__":
    main()

