import pandas as pd
import numpy as np
import argparse

import yaml
import awkward as ak
import uproot
import pyarrow.parquet as pq
import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import logger, get_centrality_bins
import ROOT  # imported here so the rest of the script works without ROOT
from ROOT import gROOT, gErrorIgnoreLevel, TStyle, TCanvas, TPad, TH1D, TH2D, TLine, TLegend, TColor, kWarning, kBlack, kGray, kFullSquare, gPad, TLatex


def _read_columns_aligned(parquet_file, cols):
    """Read several numeric columns while preserving event-by-event row alignment."""
    df = pd.read_parquet(parquet_file, columns=list(dict.fromkeys(cols)))
    out = {}
    for col in cols:
        out[col] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    return out


def _make_th2(name, xvalues, yvalues, xrange, yrange, xbins, ybins):
    """Build a ROOT TH2D from two arrays, restricted to the configured ranges."""
    mask = (
        np.isfinite(xvalues) & np.isfinite(yvalues) &
        (xvalues >= xrange[0]) & (xvalues <= xrange[1]) &
        (yvalues >= yrange[0]) & (yvalues <= yrange[1])
    )
    x = xvalues[mask]
    y = yvalues[mask]
    h = TH2D(
        name, "", int(xbins), float(xrange[0]), float(xrange[1]),
        int(ybins), float(yrange[0]), float(yrange[1])
    )
    h.SetDirectory(0)
    if len(x):
        counts, _, _ = np.histogram2d(
            x, y,
            bins=[int(xbins), int(ybins)],
            range=[xrange, yrange],
        )
        for ix in range(int(xbins)):
            for iy in range(int(ybins)):
                h.SetBinContent(ix + 1, iy + 1, float(counts[ix, iy]))
    return h


def plot_correlations(data_file, mc_prompt_file, mc_fd_file, output_pdf,
                      corr_pairs, columns_cfg,
                      labels=('Combinatorial bkg', 'MC prompt', 'MC non-prompt'),
                      header=None, max_entries=None, pt_range=None,
                      pt_column="fPt"):
    """Plot one 3-panel page for each explicit (x, y) correlation pair."""
    gROOT.SetBatch(True)

    files = [data_file, mc_prompt_file, mc_fd_file]
    schemas = [pq.read_schema(f) for f in files]
    common = set(schemas[0].names)
    for schema in schemas[1:]:
        common &= set(schema.names)

    pairs = []
    for pair in corr_pairs:
        if len(pair) != 2:
            logger(f"Invalid correlation pair, skipped: {pair}", level='WARNING')
            continue
        missing = [v for v in pair if v not in common or v not in columns_cfg]
        if missing:
            logger(f"Correlation pair {pair} skipped; missing: {missing}",
                   level='WARNING')
            continue
        pairs.append(tuple(pair))

    if not pairs:
        logger("No valid 2D correlation pairs to plot", level='WARNING')
        return

    rng = np.random.default_rng(42)
    colors = [
        TColor.GetColor("#355070"),
        TColor.GetColor("#e56b6f"),
        TColor.GetColor("#2a9d8f"),
    ]

    corr_cols = list(dict.fromkeys(v for pair in pairs for v in pair))
    cache = {col: [] for col in corr_cols}

    for f in files:
        cols = corr_cols + ([pt_column] if pt_range is not None else [])
        try:
            vals = _read_columns_aligned(f, list(dict.fromkeys(cols)))
        except Exception as e:
            logger(f"Cannot read correlation columns from {f}: {e}",
                   level='WARNING')
            return

        if pt_range is not None:
            pt = vals[pt_column]
            event_mask = (
                np.isfinite(pt)
                & (pt >= pt_range[0])
                & (pt < pt_range[1])
            )
        else:
            event_mask = np.ones(
                len(next(iter(vals.values()))), dtype=bool
            )

        for col in corr_cols:
            cache[col].append(vals[col][event_mask])

    for ipage, (xcol, ycol) in enumerate(pairs):
        xcfg, ycfg = columns_cfg[xcol], columns_cfg[ycol]

        # Configured ranges are used as hard boundaries.
        xcfg_range = tuple(map(float, xcfg["range"]))
        ycfg_range = tuple(map(float, ycfg["range"]))

        # Combine the three samples when determining the displayed range.
        xall = np.concatenate(cache[xcol])
        yall = np.concatenate(cache[ycol])

        xall = xall[np.isfinite(xall)]
        yall = yall[np.isfinite(yall)]

        if len(xall) == 0 or len(yall) == 0:
            logger(f"No finite values for correlation {xcol} vs {ycol}",
                   level='WARNING')
            continue

        # 95th percentile of the absolute distributions.
        x95 = np.percentile(xall, 95)
        y95 = np.percentile(yall, 95)

        # Keep the lower side of the configured range and extend only
        # far enough to contain 95% of the observed distribution.
        xmin = xcfg_range[0]
        ymin = ycfg_range[0]

        xmax = min(xcfg_range[1], x95)
        ymax = min(ycfg_range[1], y95)

        # Add 5% breathing room.
        xspan = xmax - xmin
        yspan = ymax - ymin

        xmax += 0.05 * xspan
        ymax += 0.05 * yspan

        xmax = min(xmax, xcfg_range[1])
        ymax = min(ymax, ycfg_range[1])

        xr = (xmin, xmax)
        yr = (ymin, ymax)

        xb = int(xcfg["bins"])
        yb = int(ycfg["bins"])

        canvas = TCanvas(
            f"corr_{ipage}",
            f"{xcol} vs {ycol}",
            3000,
            900
        )

        pads = []
        hists = []
        latex_objects = []

        for isample, (sample_label, color) in enumerate(zip(labels, colors)):
            x0 = isample / 3.0
            x1 = (isample + 1) / 3.0

            canvas.cd()

            pad = TPad(
                f"corr_pad_{ipage}_{isample}",
                sample_label,
                x0, 0.0, x1, 1.0
            )

            # More room for axis labels and the colour bar.
            pad.SetLeftMargin(0.16)
            pad.SetRightMargin(0.18)
            pad.SetBottomMargin(0.16)
            pad.SetTopMargin(0.11)
            pad.SetLogz()
            pad.Draw()
            pads.append(pad)

            pad.cd()

            xvals = cache[xcol][isample]
            yvals = cache[ycol][isample]
            mask = np.isfinite(xvals) & np.isfinite(yvals)

            # Only plot points inside the displayed range.
            mask &= (xvals >= xr[0]) & (xvals <= xr[1])
            mask &= (yvals >= yr[0]) & (yvals <= yr[1])

            if max_entries is not None and np.count_nonzero(mask) > max_entries:
                good = np.flatnonzero(mask)
                keep = rng.choice(good, max_entries, replace=False)
                mask = np.zeros(len(mask), dtype=bool)
                mask[keep] = True

            h = _make_th2(
                f"h2_{ipage}_{isample}",
                xvals[mask],
                yvals[mask],
                xr,
                yr,
                xb,
                yb
            )

            h.SetDirectory(0)
            h.SetStats(0)
            h.SetContour(100)
            h.SetMinimum(0.5)
            h.SetTitle(f";{xcol};{ycol}")

            # Axis styling.
            h.GetXaxis().SetTitleSize(0.050)
            h.GetXaxis().SetLabelSize(0.040)
            h.GetXaxis().SetTitleOffset(1.15)

            h.GetYaxis().SetTitleSize(0.050)
            h.GetYaxis().SetLabelSize(0.040)
            h.GetYaxis().SetTitleOffset(1.45)

            h.GetZaxis().SetTitle("Entries")
            h.GetZaxis().SetTitleSize(0.045)
            h.GetZaxis().SetLabelSize(0.035)
            h.GetZaxis().SetTitleOffset(1.15)

            h.Draw("COLZ")
            hists.append(h)

            # Class title.
            title = TLatex()
            title.SetNDC(True)
            title.SetTextFont(43)
            title.SetTextSize(42)
            title.DrawLatex(0.16, 0.94, sample_label.replace(",", ""))
            latex_objects.append(title)

            # if header:
            #     head = TLatex()
            #     head.SetNDC(True)
            #     head.SetTextFont(43)
            #     head.SetTextSize(20)
            #     head.DrawLatex(0.16, 0.985, header)
            #     latex_objects.append(head)

        canvas.cd()
        canvas.Update()

        suffix = (
            "" if len(pairs) == 1
            else "(" if ipage == 0
            else ")" if ipage == len(pairs) - 1
            else ""
        )

        canvas.Print(
            f"{output_pdf}{suffix}",
            f"Title:{xcol} vs {ycol}"
        )
        canvas.Close()

    logger(
        f"Saved 2D correlation plots ({len(pairs)} pages) to {output_pdf}",
        level='INFO'
    )


def _parse_pt_bins(pt_cfg):
    """Normalize pT-bin config to a list of (low, high, label)."""
    bins = []
    for item in pt_cfg or []:
        if isinstance(item, dict):
            lo, hi = item["range"]
            label = item.get("label", f"{lo:g} < #it{{p}}_{{T}} < {hi:g}")
        else:
            lo, hi = item
            label = f"{lo:g} < #it{{p}}_{{T}} < {hi:g}"
        lo, hi = float(lo), float(hi)
        if hi <= lo:
            raise ValueError(f"Invalid pT range [{lo}, {hi}]")
        bins.append((lo, hi, label))
    return bins


def main(config):
    """Run inclusive and pT-differential feature comparisons and correlations."""
    inputs = config["inputs"]
    output = config.get("output", {})
    options = config.get("options", {})
    columns_cfg = config["columns"]
    for key in ("bkg", "prompt", "non_prompt"):
        if key not in inputs:
            raise KeyError(f"Missing config.inputs.{key}")
        if not os.path.isfile(inputs[key]):
            raise FileNotFoundError(f"Input file not found: {inputs[key]}")

    labels = tuple(options.get("labels", ["Combinatorial bkg", "MC prompt", "MC non-prompt"]))
    pt_bins = _parse_pt_bins(config.get("pt_differential", []))
    pt_column = config.get("pt_column", "fPt")
    corr_vars = config.get("correlations", [])

    # Optional inclusive outputs.
    if output.get("distributions"):
        plot_comparison_configured(
            inputs["bkg"], inputs["prompt"], inputs["non_prompt"], output["distributions"],
            columns_cfg, labels=labels, logy=options.get("logy", True),
            header=options.get("header"), max_entries=options.get("max_entries"))
    if corr_vars and output.get("correlations"):
        plot_correlations(
            inputs["bkg"], inputs["prompt"], inputs["non_prompt"], output["correlations"],
            corr_vars, columns_cfg, labels=labels, header=options.get("header"),
            max_entries=options.get("max_entries"))

    # Every pT bin gets BOTH its own 1D feature-comparison PDF and its own
    # 2D-correlation PDF.
    if pt_bins:
        pt_outputs = output.get("pt_differential", {})
        dist_base = pt_outputs.get("distributions", output.get("distributions_pt"))
        corr_base = pt_outputs.get("correlations", output.get("correlations_pt"))
        if not dist_base:
            raise KeyError("pt_differential is configured but no output for feature comparisons was provided")
        if corr_vars and not corr_base:
            raise KeyError("pt_differential is configured with correlations but no correlation output was provided")

        def pt_pdf(base, prefix, lo, hi):
            if base.endswith(".pdf"):
                return f"{base[:-4]}_pt_{lo:g}_{hi:g}.pdf"
            os.makedirs(base, exist_ok=True)
            return os.path.join(base, f"{prefix}_pt_{lo:g}_{hi:g}.pdf")

        for lo, hi, pt_label in pt_bins:
            d_pdf = pt_pdf(dist_base, "feature_comparisons", lo, hi)
            plot_comparison_configured(
                inputs["bkg"], inputs["prompt"], inputs["non_prompt"], d_pdf,
                columns_cfg, labels=labels, logy=options.get("logy", True),
                header=pt_label, max_entries=options.get("max_entries"),
                pt_range=(lo, hi), pt_column=pt_column)

            if corr_vars:
                c_pdf = pt_pdf(corr_base, "feature_correlations", lo, hi)
                plot_correlations(
                    inputs["bkg"], inputs["prompt"], inputs["non_prompt"], c_pdf,
                    corr_vars, columns_cfg, labels=labels, header=pt_label,
                    max_entries=options.get("max_entries"), pt_range=(lo, hi), pt_column=pt_column)


def plot_comparison_configured(data_file, mc_prompt_file, mc_fd_file, output_pdf,
                               columns_cfg,
                               labels=('Combinatorial bkg', 'MC prompt', 'MC non-prompt'),
                               logy=True, header=None, max_entries=None,
                               pt_range=None, pt_column="fPt"):
    """Configured-range/bin 1D comparison plots without ratio pad."""
    gROOT.SetBatch(True)

    files = [data_file, mc_prompt_file, mc_fd_file]
    cols = list(columns_cfg.keys())

    schemas = [pq.read_schema(f) for f in files]
    common = [n for n in schemas[0].names if all(n in s.names for s in schemas[1:])]
    cols = [c for c in cols if c in common]

    style = TStyle("pubStyleConfigured", "Publication style")
    style.SetOptStat(0)
    style.SetOptTitle(0)
    style.SetCanvasColor(0)
    style.SetPadColor(0)
    style.SetFrameLineWidth(3)
    style.SetPadTickX(1)
    style.SetPadTickY(1)
    style.SetTickLength(0.035, "X")
    style.SetTickLength(0.035, "Y")
    style.SetLegendBorderSize(0)
    style.SetEndErrorSize(0)

    for axis in ("X", "Y", "Z"):
        style.SetLabelFont(43, axis)
        style.SetTitleFont(43, axis)
        style.SetLabelSize(24, axis)
        style.SetTitleSize(28, axis)

    style.SetNdivisions(510, "X")
    style.SetNdivisions(506, "Y")
    gROOT.SetStyle("pubStyleConfigured")
    gROOT.ForceStyle()

    c_comb = TColor.GetColor("#355070")
    c_prompt = TColor.GetColor("#e56b6f")
    c_fd = TColor.GetColor("#2a9d8f")

    colors = [c_comb, c_prompt, c_fd]
    markers = [kFullSquare, kFullSquare, kFullSquare]
    rng = np.random.default_rng(42)

    pages = []

    for col in cols:
        try:
            arrays = []

            for f in files:
                if pt_range is None:
                    vals = _read_columns_aligned(f, [col])[col]
                    vals = vals[np.isfinite(vals)]
                else:
                    vals = _read_columns_aligned(f, [col, pt_column])
                    pt = vals[pt_column]
                    mask = (
                        np.isfinite(pt)
                        & (pt >= pt_range[0])
                        & (pt < pt_range[1])
                    )
                    vals = vals[col][mask]
                    vals = vals[np.isfinite(vals)]

                arrays.append(vals)

        except Exception as e:
            logger(
                f"Cannot read column {col} as numeric ({e}), skipped",
                level='WARNING'
            )
            continue

        if max_entries is not None:
            arrays = [
                a if len(a) <= max_entries
                else rng.choice(a, max_entries, replace=False)
                for a in arrays
            ]

        cfg = columns_cfg[col]
        xmin, xmax = map(float, cfg["range"])
        nbins = int(cfg["bins"])

        if xmax <= xmin or nbins <= 0:
            logger(
                f"Invalid range/binning for {col}, skipped",
                level='WARNING'
            )
            continue

        edges = np.linspace(xmin, xmax, nbins + 1)
        pages.append((col, arrays, edges))

    if not pages:
        logger("Nothing to plot", level='WARNING')
        return

    def make_hist(name, values, edges, color, marker):
        counts, _ = np.histogram(values, bins=edges)

        h = TH1D(
            name,
            "",
            len(edges) - 1,
            np.asarray(edges, dtype=np.float64)
        )
        h.SetDirectory(0)

        for i, c in enumerate(counts, start=1):
            h.SetBinContent(i, float(c))
            h.SetBinError(i, float(np.sqrt(c)))

        integral = h.Integral()
        if integral > 0:
            h.Scale(1.0 / integral, "width")

        h.SetLineColor(color)
        h.SetLineWidth(2)
        h.SetFillColorAlpha(color, 0.30)
        h.SetFillStyle(1001)
        h.SetMarkerColor(color)
        h.SetMarkerStyle(marker)
        h.SetMarkerSize(1.1)

        return h

    npages = len(pages)

    for ipage, (col, arrays, edges) in enumerate(pages):
        hists = [
            make_hist(
                f"h_{k}_{ipage}_cfg",
                arrays[k],
                edges,
                colors[k],
                markers[k]
            )
            for k in range(3)
        ]

        h_comb, h_prompt, h_fd = hists

        cfg = columns_cfg[col]
        x_label = cfg.get("label", col)

        canvas = TCanvas(
            f"c_{ipage}_cfg",
            col,
            800,
            700
        )

        pad = TPad(
            f"p_{ipage}_cfg",
            "",
            0., 0., 1., 1.
        )

        pad.SetLeftMargin(0.15)
        pad.SetRightMargin(0.05)
        pad.SetTopMargin(0.05)
        pad.SetBottomMargin(0.14)
        pad.SetFillStyle(4000)
        pad.Draw()
        pad.cd()

        ymax = max(h.GetMaximum() for h in hists)

        positive = [
            h.GetBinContent(i)
            for h in hists
            for i in range(1, h.GetNbinsX() + 1)
            if h.GetBinContent(i) > 0
        ]

        if logy and positive:
            pad.SetLogy()

            ymin = 0.3 * min(positive)
            span = np.log10(ymax / ymin)
            ymax_draw = ymin * 10 ** (span / 0.65)
        else:
            ymin = 0.
            ymax_draw = ymax * 1.55

        frame = h_prompt.Clone(f"frame_{ipage}_cfg")
        frame.Reset()
        frame.SetDirectory(0)

        frame.GetYaxis().SetRangeUser(ymin, ymax_draw)
        frame.GetYaxis().SetTitle("Normalized Entries")
        frame.GetYaxis().SetTitleOffset(1.9)

        frame.GetXaxis().SetTitle(x_label)
        frame.GetXaxis().SetTitleOffset(1.3)

        frame.Draw("AXIS")

        for h in hists:
            h.Draw("HIST SAME")

        for h in hists:
            h.Draw("PE SAME")

        frame.Draw("AXIS SAME")

        nlines = 3 + (1 if header else 0)

        leg = TLegend(
            0.48,
            0.93 - 0.075 * nlines,
            0.93,
            0.93
        )

        leg.SetFillStyle(0)
        leg.SetTextFont(43)
        leg.SetTextSize(26)

        if header:
            leg.SetHeader(header)

        for h, lab in zip(hists, labels):
            leg.AddEntry(h, lab, "fp")

        leg.Draw()

        canvas.cd()

        suffix = (
            ""
            if npages == 1
            else "(" if ipage == 0
            else ")" if ipage == npages - 1
            else ""
        )

        canvas.Print(
            f"{output_pdf}{suffix}",
            f"Title:{col}"
        )

        canvas.Close()

    logger(
        f"Saved configured comparison plots ({npages} pages) to {output_pdf}",
        level='INFO'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare bkg, prompt MC and non-prompt MC distributions.")
    parser.add_argument("config", help="YAML configuration file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    main(config)
