#!/usr/bin/env python3
"""Shared helpers for rendering correlation and Fourier outputs."""
import hashlib
import math
import os
import re
from array import array
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import ROOT
PDF_SUBDIR = "pdf"
MASS_UNIT = "GeV/#it{c}^{2}"
PT_UNIT = "GeV/#it{c}"
MAX_COLUMNS = 4
CELL_WIDTH = 560       # pixels per grid cell
CELL_HEIGHT = 480
PAD_GAP_PX = 44        # empty space between neighbouring pads
TITLE_LINE_PX = 56     # height of one title line in the top strip
DEFAULT_PALETTE_ALPHA = 0.5
PALETTE_COLORS = 50
PALETTE_STOPS = (0.0, 0.55, 1.0)
PALETTE_ANCHORS = (ROOT.kBlue + 2, ROOT.kAzure+4, ROOT.kOrange + 7)  # cold, mid, warm
HARMONIC_COLORS = (
    ROOT.kOrange + 1,
    ROOT.kAzure + 2,
    ROOT.kCyan + 2,
    ROOT.kCyan + 2,
    ROOT.kMagenta + 2,
)
# --------------------------------------------------------------------------
# File structure and labels
# --------------------------------------------------------------------------


def subdirectories(directory, prefix):
    """Sub-directories named <prefix><int>, sorted numerically."""
    found = {}
    for key in directory.GetListOfKeys():
        name = key.GetName()

        if not name.startswith(prefix) or not directory.GetDirectory(name):
            continue
        try:
            found[name] = int(name[len(prefix):])
        except ValueError:
            continue  # ignore e.g. "mass_summary"
    return sorted(found, key=found.get)
PT_RANGE_OBJECTS = ("pt_range", "pt_assoc_range")
MASS_RANGE_OBJECTS = ("mass_range",)
_NUMBER = re.compile(r"[-+]?(?:\d+\\.?\d*|\\.\d+)(?:[eE][-+]?\d+)?")
_warned = set()


def _warn_once(message):
    if message not in _warned:
        _warned.add(message)
        print(f"[plots] WARNING: {message}")


def read_pt_assoc_bins(cfg):
    """Optional pt_assoc bin edges (correlation / preprocess section or top level)."""
    edges = None
    for section in (cfg.get("correlation", {}), cfg.get("preprocess", {}), cfg):
        if isinstance(section, dict) and section.get("pt_assoc_bins") is not None:
            edges = section["pt_assoc_bins"]
            break

    if edges is None:
        return None
    edges = tuple(float(x) for x in edges)

    if len(edges) < 2 or any(hi <= lo for lo, hi in zip(edges, edges[1:])):
        raise RuntimeError(
            "pt_assoc_bins must be strictly increasing with at least two edges"
        )
    return edges


def read_range(directory, object_names):
    """(low, high) from a TNamed / TObjString holding exactly two numbers, else None."""
    for object_name in object_names:
        obj = directory.Get(object_name)

        if not obj:
            continue
        for getter in ("GetTitle", "GetString", "GetName"):
            if not hasattr(obj, getter):
                continue
            numbers = [float(x) for x in _NUMBER.findall(str(getattr(obj, getter)()))]

            if len(numbers) == 2 and numbers[1] > numbers[0]:
                return (numbers[0], numbers[1])
    return None


def _config_ranges(edges, count, what):
    if edges is None:
        return None

    if len(edges) != count + 1:
        _warn_once(
            f"{what}: config defines {len(edges) - 1} bin(s) but the file has {count}; "
            "ignoring the config edges"
        )
        return None
    return list(zip(edges, edges[1:]))


def with_name(label, name):
    """Display a label and directory name when they differ."""
    return label if label == name else f"{label}   [{name}]"


def pt_labels(fin, names, edges, context):
    """Labels from ROOT metadata, configured edges, or directory names."""
    from_config = _config_ranges(edges, len(names), f"{context}: pt_assoc_bins")
    labels = []
    missing = False
    for i, name in enumerate(names):
        limits = read_range(fin.GetDirectory(name), PT_RANGE_OBJECTS)

        if limits is None and from_config is not None:
            limits = from_config[i]

        if limits is None:
            missing = True
            labels.append(name)
            continue
        lo, hi = limits
        labels.append(f"{lo:g} < #it{{p}}_{{T}}^{{assoc}} < {hi:g} {PT_UNIT}")

    if missing:
        _warn_once(
            f"{context}: associated-pT ranges unavailable in ROOT and config; "
            "using directory names for those bins"
        )
    return labels


def mass_labels(fin, pt_name, mass_names, edges, context):
    """One label per mass directory of a pt_assoc bin (mass_range, then config)."""
    from_config = _config_ranges(edges, len(mass_names), f"{context}: correlation.mass_bins")
    labels, missing = [], False
    for j, name in enumerate(mass_names):
        directory = fin.GetDirectory(f"{pt_name}/{name}")
        limits = read_range(directory, MASS_RANGE_OBJECTS) if directory else None

        if limits is None and from_config:
            limits = from_config[j]

        if limits is None:
            missing = True
            labels.append(name)
        else:
            lo, hi = limits
            labels.append(f"{lo:g} < M < {hi:g} {MASS_UNIT}")

    if missing:
        _warn_once(f"{context}: mass limits not found; labels show the directory names")
    return labels
# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _make_palette(alpha):
    """Cold-to-warm gradient used for the correlation maps; returns colour indices."""
    red, green, blue = [], [], []
    for color_index in PALETTE_ANCHORS:
        color = ROOT.gROOT.GetColor(color_index)

        if not color:
            raise RuntimeError(f"ROOT colour {color_index} is not defined")
        red.append(color.GetRed())
        green.append(color.GetGreen())
        blue.append(color.GetBlue())
    first = ROOT.TColor.CreateGradientColorTable(
        len(PALETTE_STOPS),
        array("d", PALETTE_STOPS),
        array("d", red),
        array("d", green),
        array("d", blue),
        PALETTE_COLORS,
        alpha,
    )
    return array("i", [first + i for i in range(PALETTE_COLORS)])
@lru_cache(maxsize=None)


def _make_style(palette_alpha):
    """Private TStyle (cached per alpha, so the palette is created only once)."""
    style = ROOT.TStyle(
        f"pipelinePdfStyle_{palette_alpha:g}", "2PC pipeline PDF style"
    )
    palette = _make_palette(palette_alpha)
    style.SetPalette(PALETTE_COLORS, palette)
    style.SetNumberContours(PALETTE_COLORS)
    style.SetOptStat(0)
    style.SetOptTitle(0)
    style.SetCanvasColor(0)
    style.SetPadColor(0)
    style.SetCanvasBorderMode(0)
    style.SetPadBorderMode(0)
    style.SetFrameBorderMode(0)
    style.SetFrameLineWidth(2)
    style.SetLineScalePS(1.2)
    style.SetPadTickX(1)
    style.SetPadTickY(1)
    style.SetTextFont(42)
    for axis in "XYZ":
        style.SetLabelFont(42, axis)
        style.SetTitleFont(42, axis)
        style.SetLabelSize(0.045, axis)
        style.SetTitleSize(0.05, axis)
    return style
@contextmanager


def pdf_rendering(palette_alpha=DEFAULT_PALETTE_ALPHA):
    """Batch mode + private style, restoring the global state afterwards."""
    previous_style = ROOT.gStyle
    was_batch = ROOT.gROOT.IsBatch()
    ROOT.gROOT.SetBatch(True)
    _make_style(float(palette_alpha)).cd()
    try:
        yield
    finally:
        previous_style.cd()
        ROOT.gROOT.SetBatch(was_batch)


def draw_pad_label(pad, text):
    """Small label above the frame, aligned with the left axis."""
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.05)
    latex.SetTextAlign(11)
    latex.DrawLatexNDC(pad.GetLeftMargin(), 1.0 - pad.GetTopMargin() + 0.012, text)


def write_pdf_pages(
    pdf_path, pages, max_columns=MAX_COLUMNS, palette_alpha=DEFAULT_PALETTE_ALPHA
):
    """Write a multi-page PDF.

    pages: list of (title_lines, cells)

        title_lines  list of strings drawn in the top strip of the page

        cells        list of callables draw(pad) -> iterable of objects to keep alive

    palette_alpha: opacity of the colour palette used for 2D maps

    """

    if not pages:
        return
    os.makedirs(os.path.dirname(os.path.abspath(pdf_path)), exist_ok=True)
    max_cells = max(len(cells) for _, cells in pages)
    max_lines = max(len(lines) for lines, _ in pages)
    columns = max(1, min(max_columns, math.ceil(math.sqrt(max_cells))))
    rows = max(1, math.ceil(max_cells / columns))
    strip_px = TITLE_LINE_PX * max_lines + 20
    canvas_w = columns * CELL_WIDTH
    canvas_h = rows * CELL_HEIGHT + strip_px
    plot_height = 1.0 - strip_px / canvas_h
    cell_w = 1.0 / columns
    cell_h = plot_height / rows
    gap_x = PAD_GAP_PX / canvas_w
    gap_y = PAD_GAP_PX / canvas_h
    with pdf_rendering(palette_alpha):
        canvas = ROOT.TCanvas("c2pcPdf", "2PC pdf", canvas_w, canvas_h)
        try:
            npages = len(pages)
            for ipage, (title_lines, cells) in enumerate(pages):
                canvas.Clear()
                canvas.cd()
                keep = []
                header = ROOT.TLatex()
                header.SetNDC()
                header.SetTextFont(42)
                header.SetTextAlign(12)
                for iline, line in enumerate(title_lines):
                    header.SetTextSize((30 if iline == 0 else 24) / canvas_h)
                    y = 1.0 - (10 + TITLE_LINE_PX * (iline + 0.5)) / canvas_h
                    header.DrawLatex(0.01, y, line)
                for icell, draw_cell in enumerate(cells):
                    row, col = divmod(icell, columns)
                    x1 = col * cell_w
                    y2 = plot_height - row * cell_h
                    pad = ROOT.TPad(
                        f"pad_{ipage}_{icell}",
                        "",
                        x1 + gap_x / 2,
                        y2 - cell_h + gap_y / 2,
                        x1 + cell_w - gap_x / 2,
                        y2 - gap_y / 2,
                    )
                    pad.Draw()
                    pad.cd()
                    keep.append(pad)
                    keep.extend(draw_cell(pad) or [])
                    canvas.cd()

                if npages > 1 and ipage == 0:
                    flag = "("
                elif npages > 1 and ipage == npages - 1:
                    flag = ")"
                else:
                    flag = ""
                canvas.Print(f"{pdf_path}{flag}")
                del keep
        finally:
            canvas.Close()


def save_mass_fit_qa(fitter, output_path):
    """Delegate the invariant-mass fit QA plot to RawYieldFitter."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fitter.plot_fit(logy=False, path=str(output_path), show_extra_info=False)


def _draw_summary_hist(pad, hist, y_title):
    """Draw a single DeltaPhi summary observable (no changes to styling)."""
    pad.SetMargin(0.15, 0.04, 0.14, 0.08)
    hist.SetStats(False)
    hist.SetTitle("")
    hist.SetMarkerStyle(20)
    hist.SetMarkerSize(0.8)
    hist.GetXaxis().SetTitle("#Delta#varphi (rad)")
    hist.GetYaxis().SetTitle(y_title)
    hist.GetYaxis().SetTitleOffset(1.4)
    hist.Draw("PE")
    return hist


def save_summary_pdf(root_file, pdf_path):
    """Write the existing three-pad, multipage summary PDF."""
    fin = ROOT.TFile.Open(str(root_file), "READ")

    if not fin or fin.IsZombie():
        raise RuntimeError(f"Cannot open {root_file}")
    pdf_path = str(pdf_path)
    try:
        pt_names = subdirectories(fin, "pt_assoc_")
        opener = ROOT.TCanvas("cDeltaPhiSummaryOpen", "", 10, 10)
        try:
            opener.Print(pdf_path + "[")
        finally:
            opener.Close()
        for ipt, pt_name in enumerate(pt_names):
            pt_dir = fin.GetDirectory(pt_name)
            canvas = ROOT.TCanvas(f"cDeltaPhiSummary_{ipt}", pt_name, 900, 1000)
            canvas.Divide(1, 3)
            keep = [
                _draw_summary_hist(
                    canvas.cd(1),
                    pt_dir.Get("hRawYieldVsDeltaPhi"),
                    "Signal pairs / rad",
                ),
                _draw_summary_hist(canvas.cd(2), pt_dir.Get("hMeanVsDeltaPhi"), "mean"),
                _draw_summary_hist(canvas.cd(3), pt_dir.Get("hSigmaVsDeltaPhi"), "sigma"),
            ]
            canvas._keep = keep  # retain ROOT references until canvas is printed
            canvas.Print(pdf_path)
        closer = ROOT.TCanvas("cDeltaPhiSummaryClose", "", 10, 10)
        closer.Print(pdf_path + "]")
    finally:
        fin.Close()


def _mass_directory_structure(fin, filename):
    """Discover and validate pt_assoc/mass directory structure once."""
    pt_names = subdirectories(fin, "pt_assoc_")

    if not pt_names:
        raise RuntimeError(f"No pt_assoc_* directories found in {filename}")
    mass_names = {
        pt: subdirectories(fin.GetDirectory(pt), "mass_")
        for pt in pt_names
    }
    counts = [len(names) for names in mass_names.values()]

    if not counts[0] or len(set(counts)) != 1:
        raise RuntimeError(
            f"Inconsistent or missing mass_* directories across pt bins in {filename}"
        )
    return pt_names, mass_names


def _candidate_pt_label(filename, pt_cand_bins):
    if pt_cand_bins is None or len(pt_cand_bins) < 2:
        raise RuntimeError("Provide bins.pt_cand for PDF candidate-pT labels")
    match = re.fullmatch(r"mult_\d+_pt_(\d+)\.root", filename)

    if not match:
        raise RuntimeError(f"Cannot determine candidate-pT bin from {filename}")
    i_pt = int(match.group(1))

    if i_pt + 1 >= len(pt_cand_bins):
        raise RuntimeError(f"Candidate-pT index {i_pt} outside bins.pt_cand in {filename}")
    return (
        f"{pt_cand_bins[i_pt]:g} < #it{{p}}_{{T}}^{{D}} < "
        f"{pt_cand_bins[i_pt + 1]:g} {PT_UNIT}"
    )


def _map_pages(fin, filename, mass_bins, option, eta_min, eta_max,
               histogram_name, pt_label, pt_names, mass_names,
               check_duplicates=False):
    """Build PDF pages from an already-open ROOT file."""
    seen = {}
    pages = []
    for pt in pt_names:
        labels = mass_labels(fin, pt, mass_names[pt], mass_bins, filename)
        cells = []
        for mass_name, mass_label in zip(mass_names[pt], labels):
            path = f"{pt}/{mass_name}/{histogram_name}"
            source = fin.Get(path)

            if not source:
                raise RuntimeError(f"Missing {path} in {filename}")
            hist = source.Clone(f"{histogram_name}_{pt}_{mass_name}")
            hist.SetDirectory(0)
            hist.GetXaxis().SetRangeUser(eta_min, eta_max)
            hist.SetLineColor(ROOT.kBlack)

            if check_duplicates:
                signature, nonzero = _signature(hist)

                if not nonzero:
                    _warn_once(f"{filename}: {path} is empty")
                elif signature in seen:
                    _warn_once(f"{filename}: {path} is identical to {seen[signature]}")
                else:
                    seen[signature] = path
            cells.append(_map_cell(hist, mass_label, option))
        pages.append(([pt_label], cells))
    return pages


def save_maps_pdfs(corr_file, pdf_paths, mass_bins, option, palette_alpha,
                   eta_min, eta_max, pt_cand_bins, check_duplicates=False):
    """Render hCorr/hSE/hME PDFs, opening the input ROOT file only once.

    pdf_paths maps ROOT histogram names to output PDF paths, e.g.
    {"hCorr": "file.pdf", "hSE": "file_SE.pdf", "hME": "file_ME.pdf"}.
    """
    fin = ROOT.TFile.Open(str(corr_file), "READ")

    if not fin or fin.IsZombie():
        raise RuntimeError(f"Cannot open {corr_file}")
    filename = os.path.basename(str(corr_file))
    try:
        pt_names, mass_names = _mass_directory_structure(fin, filename)
        pt_label = _candidate_pt_label(filename, pt_cand_bins)
        for hist_name, pdf_path in pdf_paths.items():
            pages = _map_pages(
                fin, filename, mass_bins, option, eta_min, eta_max,
                hist_name, pt_label, pt_names, mass_names,
                check_duplicates=check_duplicates,
            )
            write_pdf_pages(str(pdf_path), pages, palette_alpha=palette_alpha)
    finally:
        fin.Close()


def _save_maps_pdf(corr_file, pdf_path, mass_bins, option, palette_alpha,
                   eta_min, eta_max, histogram_name="hCorr", pt_cand_bins=None,
                   check_duplicates=False):
    """Backwards-compatible single-map PDF output, candidate-pT header only."""
    save_maps_pdfs(
        corr_file, {histogram_name: pdf_path}, mass_bins, option, palette_alpha,
        eta_min, eta_max, pt_cand_bins, check_duplicates=check_duplicates,
    )


def _draw_map(pad, hist, label, option):
    """Draw one correlation map into a pad."""
    is_3d = option.upper().startswith(("SURF", "LEGO"))

    if is_3d:
        pad.SetMargin(0.15, 0.10, 0.09, 0.09)  # room for the z labels / #Delta#eta title
        offsets = (1.8, 2.0, 1.7)
    else:
        pad.SetMargin(0.17, 0.20, 0.15, 0.09)
        offsets = (1.2, 1.6, 1.7)
    hist.SetStats(False)
    hist.SetTitle("")
    for axis, title, offset in zip(
        (hist.GetXaxis(), hist.GetYaxis(), hist.GetZaxis()),
        ("#Delta#eta", "#Delta#varphi (rad)", "C(#Delta#eta, #Delta#varphi)"),
        offsets,
    ):
        axis.SetTitle(title)
        axis.SetTitleOffset(offset)
        axis.CenterTitle(False)

        if title == "C(#Delta#eta, #Delta#varphi)":
            axis.SetMaxDigits(3)
    hist.Draw(option)
    draw_pad_label(pad, label)
    return [hist]


def _map_cell(hist, label, option):
    return lambda pad: _draw_map(pad, hist, label, option)


def _signature(hist):
    """(hash of all bin contents, True if any bin is non-zero)."""
    nbins = (hist.GetNbinsX() + 2) * (hist.GetNbinsY() + 2)
    values = array("d", (hist.GetBinContent(b) for b in range(nbins)))
    return hashlib.md5(values.tobytes()).hexdigest(), any(values)


def _read_v_values(hpars):
    """[(n, V_nΔ, error), ...] from the labelled hFitParameters bins."""
    values = []
    for ibin in range(1, hpars.GetNbinsX() + 1):
        label = hpars.GetXaxis().GetBinLabel(ibin)

        if label.startswith("V") and label.endswith("Delta"):
            values.append(
                (label[1:-5], hpars.GetBinContent(ibin), hpars.GetBinError(ibin))
            )
    return values


def _read_fit_status(in_mass):
    info = in_mass.Get("fit_info")

    if not info:
        return None
    match = re.search(r"status=(-?\d+)", info.GetTitle())
    return int(match.group(1)) if match else None


def _draw_fit(pad, in_mass, config, mass_label, context):
    """Draw data, total fit, baseline and harmonics for one mass bin."""
    hist = in_mass.Get("hDeltaPhi")
    func = in_mass.Get("fFourier")
    baseline = in_mass.Get("fBaseline")
    flow = in_mass.Get("fFlow")
    lm_template = in_mass.Get("fLMTemplate")
    hpars = in_mass.Get("hFitParameters")

    if not (hist and func and baseline and hpars):
        raise RuntimeError(f"Missing fit objects in {context}")
    pad.SetMargin(0.18, 0.04, 0.14, 0.09)
    # y-range with head-room for the text block
    points = [
        (hist.GetBinContent(i), hist.GetBinError(i))
        for i in range(1, hist.GetNbinsX() + 1)
    ]
    ymin = min(c - e for c, e in points)
    ymax = max(c + e for c, e in points)
    span = (ymax - ymin) or abs(ymax) or 1.0
    hist.SetStats(False)
    hist.SetTitle("")
    hist.SetMinimum(ymin - 0.15 * span)
    hist.SetMaximum(ymax + 0.50 * span)
    hist.GetXaxis().SetTitle("#Delta#varphi (rad)")
    hist.GetYaxis().SetTitle("C(#Delta#varphi)")
    hist.GetYaxis().SetTitleOffset(1.8)
    hist.GetXaxis().SetTitleOffset(1.1)
    hist.SetMarkerStyle(20)
    hist.SetMarkerSize(0.8)
    hist.SetMarkerColor(ROOT.kBlack)
    hist.SetLineColor(ROOT.kBlack)
    hist.Draw("PE")
    keep = [hist, func, baseline]

    if bool(flow) != bool(lm_template):
        raise RuntimeError(f"Incomplete LM decomposition in {context}")
    has_lm = bool(flow and lm_template)
    xmin, xmax = func.GetXmin(), func.GetXmax()
    a0 = func.GetParameter(0)
    # In HM, draw the separated contributions, not the raw harmonics.

    if not has_lm:
        for i, harmonic in enumerate(config.harmonics):
            component = _make_tf1(
                f"fPlotHarmonic{harmonic}", f"[0]+[1]*cos({harmonic}*x)", xmin, xmax
            )
            component.SetParameters(a0, func.GetParameter(i + 1))
            component.SetNpx(300)
            component.SetLineColor(HARMONIC_COLORS[i % len(HARMONIC_COLORS)])
            component.SetLineStyle(ROOT.kDashed)
            component.SetLineWidth(2)
            component.Draw("SAME")
            keep.append(component)

    if has_lm:
        for curve, color, style in ((flow, ROOT.kBlue + 1, ROOT.kDashed), (lm_template, ROOT.kGreen + 2, ROOT.kDashDotted)):
            curve.SetNpx(400)
            curve.SetLineColor(color)
            curve.SetLineStyle(style)
            curve.SetLineWidth(3)
            curve.Draw("SAME")
            keep.append(curve)
    baseline.SetLineColor(ROOT.kGray + 2)
    baseline.SetLineStyle(ROOT.kDotted)
    baseline.SetLineWidth(2)
    baseline.Draw("SAME")
    func.SetNpx(300)
    func.SetLineColor(ROOT.kRed + 1)
    func.SetLineWidth(3)
    func.Draw("SAME")
    hist.Draw("PE SAME")  # data points on top of the curves
    # V_nΔ and χ²/ndf
    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextFont(42)
    text.SetTextSize(0.042)
    text.SetTextAlign(13)
    x = pad.GetLeftMargin() + 0.04
    y = 1.0 - pad.GetTopMargin() - 0.03
    displayed = in_mass.Get("hLMCorrectionParameters") if has_lm else hpars
    flow_values = []

    if has_lm:
        for ibin in range(1, displayed.GetNbinsX() + 1):
            name = displayed.GetXaxis().GetBinLabel(ibin)

            if name.startswith("V") and name.endswith("DeltaFlow") and name != "V1DeltaFlow":
                flow_values.append((name[1:-9], displayed.GetBinContent(ibin), displayed.GetBinError(ibin)))
    else:
        flow_values = list(_read_v_values(hpars))
    for harmonic, value, error in flow_values:
        text.DrawLatexNDC(
            x, y, f"#it{{V}}_{{{harmonic}#Delta}} = {value:.4f} #pm {error:.4f}"
        )
        y -= 0.055
    text.DrawLatexNDC(
        x, y, f"#chi^{{2}}/ndf = {func.GetChisquare():.1f}/{func.GetNDF()}"
    )
    y -= 0.055
    status = _read_fit_status(in_mass)

    if status not in (None, 0):
        text.SetTextColor(ROOT.kRed + 1)
        text.DrawLatexNDC(x + 0.02, y, f"fit status {status}")
    # Mass interval: top-right corner of each pad.
    mass_text = ROOT.TLatex()
    mass_text.SetNDC()
    mass_text.SetTextFont(42)
    mass_text.SetTextSize(0.040)
    mass_text.SetTextAlign(33)  # right/top aligned
    mass_text.DrawLatexNDC(
        1.0 - pad.GetRightMargin() - 0.02,
        1.0 - pad.GetTopMargin() - 0.025,
        mass_label,
    )
    keep.extend([mass_text, text])
    return keep


def _fit_cell(in_mass, config, mass_label, context):
    return lambda pad: _draw_fit(pad, in_mass, config, mass_label, context)


def _legend_cell(config, has_lm=False):
    """Dedicated pad explaining the curves, the model and the Δη selection."""
    def dummy(line_color=None, line_style=1, line_width=2, marker=False):
        graph = ROOT.TGraph()

        if marker:
            graph.SetMarkerStyle(20)
            graph.SetMarkerColor(ROOT.kBlack)
            graph.SetLineColor(ROOT.kBlack)
        else:
            graph.SetLineColor(line_color)
            graph.SetLineStyle(line_style)
            graph.SetLineWidth(line_width)
        return graph
    def draw(pad):
        pad.SetMargin(0.02, 0.02, 0.02, 0.02)
        entries = [
            (dummy(marker=True), "Data", "pez"),
            (dummy(ROOT.kRed + 1, 1, 3), "Total Fit", "l"),
            (dummy(ROOT.kGray + 2, ROOT.kDotted), "Baseline  a_{0}", "l"),
        ]

        if has_lm:
            entries.extend([
                (dummy(ROOT.kBlue + 1, ROOT.kDashed, 3), "Flow baseline + flow harmonics", "l"),
                (dummy(ROOT.kGreen + 2, ROOT.kDashDotted, 3), "Scaled LM template (unit baseline)", "l"),
            ])
        else:
            for i, harmonic in enumerate(config.harmonics):
                color = HARMONIC_COLORS[i % len(HARMONIC_COLORS)]
                entries.append((dummy(color, ROOT.kDashed), f"Harmonic {harmonic}: a_{{0}} + a_{{{harmonic}}}cos({harmonic}#Delta#varphi)", "l"))
        legend = ROOT.TLegend(0.04, 0.36, 0.98, 0.97)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextFont(42)
        legend.SetTextSize(min(0.065, 0.35 / len(entries)))
        for graph, label, option in entries:
            legend.AddEntry(graph, label, option)
        legend.Draw()
        eta_text = " #cup ".join(f"[{lo:g}, {hi:g})" for lo, hi in config.eta_ranges)
        lines = [
            f"#Delta#eta ranges: {eta_text}",
        ]
        text = ROOT.TLatex()
        text.SetNDC()
        text.SetTextFont(42)
        text.SetTextSize(0.05)
        text.SetTextAlign(12)
        y = 0.28
        for line in lines:
            text.DrawLatexNDC(0.05, y, line)
            y -= 0.09
        return [legend] + [g for g, _, _ in entries]
    return draw


def _save_fit_pdf(fourier_file, pdf_path, config, pt_cand_bins=None):
    """One page per pt_assoc bin, one pad per invariant-mass bin (fits)."""
    fin = ROOT.TFile.Open(fourier_file, "READ")

    if not fin or fin.IsZombie():
        raise RuntimeError(f"Cannot open {fourier_file}")
    name = os.path.basename(fourier_file)
    stem = os.path.splitext(name)[0]
    try:
        pt_names, mass_names = _mass_directory_structure(fin, name)
        pt_label_list = pt_labels(fin, pt_names, config.pt_assoc_bins, name)
        pages = []
        for pt, pt_label in zip(pt_names, pt_label_list):
            mass_label_list = mass_labels(
                fin, pt, mass_names[pt], config.mass_bins, name
            )
            cells = []
            for mass_name, mass_label in zip(mass_names[pt], mass_label_list):
                path = f"{pt}/{mass_name}"
                in_mass = fin.GetDirectory(path)

                if not in_mass:
                    raise RuntimeError(f"Missing directory {path} in {name}")
                cells.append(_fit_cell(in_mass, config, mass_label, f"{name}:{path}"))
            has_lm = any(bool(fin.GetDirectory(f"{pt}/{mass_name}").Get("fLMTemplate")) for mass_name in mass_names[pt])
            cells.append(_legend_cell(config, has_lm=has_lm))
            heading = _candidate_pt_label(name, pt_cand_bins) if pt_cand_bins is not None else stem
            pages.append(([f"{heading}   |   {with_name(pt_label, pt)}"], cells))
        write_pdf_pages(pdf_path, pages)
    finally:
        fin.Close()


def _draw_v2_mass(pad, hist, pt_label):
    """Draw V2Delta(mass) with the same data style as the Fourier projections."""
    pad.SetMargin(0.18, 0.04, 0.14, 0.09)
    hist.SetStats(False)
    hist.SetTitle("")
    hist.SetMarkerStyle(20)
    hist.SetMarkerSize(0.8)
    hist.SetMarkerColor(ROOT.kBlack)
    hist.SetLineColor(ROOT.kBlack)
    hist.GetXaxis().SetTitle(f"Invariant mass ({MASS_UNIT})")
    hist.GetYaxis().SetTitle("#it{V}_{2#Delta}")
    hist.GetXaxis().SetTitleOffset(1.1)
    hist.GetYaxis().SetTitleOffset(1.8)
    values = [
        (hist.GetBinContent(i), hist.GetBinError(i))
        for i in range(1, hist.GetNbinsX() + 1)

        if math.isfinite(hist.GetBinContent(i)) and math.isfinite(hist.GetBinError(i))
    ]

    if values:
        low = min(v - e for v, e in values)
        high = max(v + e for v, e in values)
        span = high - low or abs(high) or 1.0
        hist.SetMinimum(low - 0.15 * span)
        hist.SetMaximum(high + 0.30 * span)

    hist.Draw("PE")
    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextFont(42)
    text.SetTextSize(0.040)
    text.SetTextAlign(33)
    text.DrawLatexNDC(1.0 - pad.GetRightMargin() - 0.02,
                      1.0 - pad.GetTopMargin() - 0.025, pt_label)
    return [hist, text]


def save_v2_mass_summary_pdf(fourier_files, pdf_path, pt_cand_bins):
    """One multipage summary: each associated-pT bin has all candidate-pT panels.

    Input files must belong to the same multiplicity interval. Panel labels are
    taken from bins.pt_cand via the pt index in each ROOT filename.
    """

    if not fourier_files:
        raise RuntimeError("No Fourier files supplied for V2Delta(mass) summary")

    files_by_pt = {}
    multiplicities = set()
    for path in fourier_files:
        name = os.path.basename(str(path))
        match = re.fullmatch(r"mult_(\d+)_pt_(\d+)\.root", name)

        if not match:
            raise RuntimeError(f"Unrecognized Fourier filename: {name}")
        mult_index, pt_index = map(int, match.groups())
        multiplicities.add(mult_index)

        if pt_index in files_by_pt:
            raise RuntimeError(f"Duplicate candidate-pT index {pt_index}")
        files_by_pt[pt_index] = str(path)

    if len(multiplicities) != 1:
        raise RuntimeError("V2Delta(mass) summary requires one multiplicity bin")

    # Clone and detach histograms before closing input files; retain one page per
    # associated-pT interval, with candidate-pT bins ordered numerically.
    by_assoc = {}
    expected_assoc = None
    for pt_index, path in sorted(files_by_pt.items()):
        filename = os.path.basename(path)
        label = _candidate_pt_label(filename, pt_cand_bins)
        fin = ROOT.TFile.Open(path, "READ")

        if not fin or fin.IsZombie():
            raise RuntimeError(f"Cannot open {path}")
        try:
            assoc_dirs = subdirectories(fin, "pt_assoc_")

            if not assoc_dirs:
                raise RuntimeError(f"No pt_assoc_* directories in {path}")

            if expected_assoc is None:
                expected_assoc = assoc_dirs
            elif assoc_dirs != expected_assoc:
                raise RuntimeError(f"Inconsistent associated-pT directories in {path}")
            for assoc_name in assoc_dirs:
                source = fin.Get(f"{assoc_name}/hV2DeltaVsMass")

                if not source:
                    raise RuntimeError(f"Missing {assoc_name}/hV2DeltaVsMass in {path}")
                hist = source.Clone(f"hV2DeltaVsMass_mult{next(iter(multiplicities))}_pt{pt_index}_{assoc_name}")
                hist.SetDirectory(0)
                by_assoc.setdefault(assoc_name, []).append((pt_index, label, hist))
        finally:
            fin.Close()

    pages = []
    for assoc_name in expected_assoc:
        cells = [
            (lambda pad, h=hist, label=pt_label: _draw_v2_mass(pad, h, label))
            for _, pt_label, hist in sorted(by_assoc[assoc_name])
        ]
        pages.append(([], cells))

    write_pdf_pages(str(pdf_path), pages)


def _make_tf1(name, formula, xmin, xmax):
    return ROOT.TF1(name, formula, xmin, xmax)
