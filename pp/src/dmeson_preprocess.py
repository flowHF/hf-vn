"""Prefilter the dedicated D-candidate THnSparse, separately from D-hadron pairs.

One output file per multiplicity/candidate-pT bin. The output contains both a
reduced (mass, BDT background, BDT feed-down) THnSparse and its mass projection. Files are
merged only with other D-candidate files, never with SE/ME pair sparses.
"""
import math
from array import array
from pathlib import Path


def _cut_name(cfg, raw_axis):
    if isinstance(raw_axis, str):
        return raw_axis
    for name, index in cfg.get("axes", {}).items():
        if index == raw_axis:
            return name
    raise ValueError(f"Cannot map pair cut axis {raw_axis} to the dedicated D sparse")


def validate_mass_config(cfg):
    se = cfg["inputs"]["SE"]

    if not se.get("hsparsemass"):
        return False
    axes = cfg.setdefault("mass_axes", {"mass": 0, "pt_cand": 1, "bdt_background": 2, "bdt_prompt": 3})

    if not isinstance(axes, dict) or not all(k in axes for k in ("mass", "pt_cand", "bdt_background", "bdt_prompt")):
        raise ValueError("inputs.SE.hsparsemass requires mass_axes.mass, pt_cand, bdt_background and bdt_prompt")

    if cfg["bins"].get("multiplicity") is not None and axes.get("multiplicity") is None:
        raise ValueError("mass_axes.multiplicity is required for multiplicity splitting")

    if len(set(v for v in axes.values() if v is not None)) != len([v for v in axes.values() if v is not None]):
        raise ValueError("mass_axes indices must be distinct")

    if any(type(v) is not int or v < 0 for v in axes.values() if v is not None):
        raise ValueError("mass_axes indices must be nonnegative integers")
    # Inherit named cuts only, with the dedicated sparse's own axis numbering.
    # A cut without its matching axis must not be silently dropped.
    for cut in cfg.get("cuts", []):
        name = _cut_name(cfg, cut["axis"])

        if name not in axes:
            raise ValueError(f"D sparse: cut axis '{name}' missing from mass_axes; add the corresponding axis to mass_axes")

        if name in ("mass", "pt_cand", "multiplicity"):
            raise ValueError(f"D sparse: '{name}' is a splitting/retained axis, not a mass_cuts axis")
    return True


def _edges(axis):
    return [float(axis.GetBinLowEdge(i)) for i in range(1, axis.GetNbins() + 2)]


def _mass_hist(ROOT, axis, factor):
    native = _edges(axis)
    n = len(native) - 1

    if n % factor:
        raise ValueError(f"D mass axis has {n} bins, not divisible by rebin factor {factor}")
    new_edges = native[::factor]

    if new_edges[-1] != native[-1]:
        new_edges.append(native[-1])
    h = ROOT.TH1D("hInvariantMass", "D candidates;invariant mass;counts", len(new_edges) - 1, array("d", new_edges))
    h.Sumw2()
    h.SetDirectory(0)
    return h, new_edges


def _interval_index(value, edges):
    # Half-open [low,high) selection, with the final edge included.
    for i, (low, high) in enumerate(zip(edges, edges[1:])):
        if low <= value < high or (i == len(edges) - 2 and value == high):
            return i
    return None


def _cut_ranges(cfg, npt):
    cuts = []
    for cut in cfg.get("cuts", []):
        ranges = cut["ranges"]

        if len(ranges) == 2 and all(isinstance(x, (int, float)) for x in ranges):
            ranges = [ranges] * npt

        if len(ranges) != npt or any(len(r) != 2 for r in ranges):
            raise ValueError(f"Invalid D-sparse cut ranges for {cut['axis']}")
        cuts.append((_cut_name(cfg, cut["axis"]), ranges))
    return cuts


def split_dmeson_file(input_file, histogram_path, output_dir, cfg):
    """Scan occupied source bins once, preserving contents and sum-of-weight variances."""
    import ROOT
    axes = cfg["mass_axes"]
    bins = cfg["bins"]
    pt_edges = bins["pt_cand"]
    mult_edges = bins.get("multiplicity")
    npt = len(pt_edges) - 1
    nmult = len(mult_edges) - 1 if mult_edges else 1
    factor = cfg.get("mass_rebin", cfg.get("rebin", {}).get("mass", 1))

    if type(factor) is not int or factor < 1:
        raise ValueError("mass_rebin must be a positive integer")
    cuts = _cut_ranges(cfg, npt)
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    fin = ROOT.TFile.Open(str(input_file), "READ")

    if not fin or fin.IsZombie():
        raise RuntimeError(f"Cannot open {input_file}")
    try:
        src = fin.Get(histogram_path)

        if not src or not src.InheritsFrom("THnSparse"):
            raise RuntimeError(f"Missing dedicated THnSparse '{histogram_path}' in {input_file}")
        ndim = src.GetNdimensions()

        if any(i >= ndim for i in axes.values() if i is not None):
            raise ValueError(f"mass_axes index out of range (D sparse has {ndim} axes)")
        histos = {}
        sumerr2 = {}
        sparse_bins = {}
        mass_axis = src.GetAxis(axes["mass"])
        for im in range(nmult):
            for ip in range(npt):
                histos[(im, ip)], edges = _mass_hist(ROOT, mass_axis, factor)
                sumerr2[(im, ip)] = [0.0] * histos[(im, ip)].GetNbinsX()
                sparse_bins[(im, ip)] = {}
        has_errors = bool(src.GetCalculateErrors())

        if not has_errors and not cfg.get("allow_poisson_errors", False):
            raise RuntimeError("D THnSparse lacks stored errors; enable allow_poisson_errors only if Poisson errors are valid")
        coords = array("i", [0] * ndim)
        n_selected = 0
        used_axes = {axes[name] for name in ("mass", "pt_cand", "bdt_background", "bdt_prompt")} | {axes[name] for name, _ in cuts}

        if mult_edges:
            used_axes.add(axes["multiplicity"])
        for ibin in range(src.GetNbins()):
            value = float(src.GetBinContent(ibin, coords))

            if not value:
                continue

            if any(coords[i] < 1 or coords[i] > src.GetAxis(i).GetNbins() for i in used_axes):
                continue
            pt = src.GetAxis(axes["pt_cand"]).GetBinCenter(coords[axes["pt_cand"]])
            ip = _interval_index(pt, pt_edges)

            if ip is None:
                continue
            im = 0

            if mult_edges:
                mult = src.GetAxis(axes["multiplicity"]).GetBinCenter(coords[axes["multiplicity"]])
                im = _interval_index(mult, mult_edges)

                if im is None:
                    continue

            if any(not (bounds[ip][0] <= src.GetAxis(axes[name]).GetBinCenter(coords[axes[name]]) < bounds[ip][1]) for name, bounds in cuts):
                continue
            mass = mass_axis.GetBinCenter(coords[axes["mass"]])
            target = histos[(im, ip)]
            j = target.GetXaxis().FindFixBin(mass)

            if not (1 <= j <= target.GetNbinsX()):
                continue
            target.SetBinContent(j, target.GetBinContent(j) + value)
            err2 = float(src.GetBinError2(ibin)) if has_errors else max(0.0, value)
            sumerr2[(im, ip)][j - 1] += err2
            key = (j, coords[axes["bdt_background"]], coords[axes["bdt_prompt"]])
            old_sum, old_var = sparse_bins[(im, ip)].get(key, (0.0, 0.0))
            sparse_bins[(im, ip)][key] = (old_sum + value, old_var + err2)
            n_selected += 1
        generated = []
        for (im, ip), hist in histos.items():
            for j, err2 in enumerate(sumerr2[(im, ip)], 1):
                hist.SetBinError(j, math.sqrt(max(0.0, err2)))
            # Preserve both BDT axes at their original binning; only mass is rebinned.
            mass_edges = _edges(hist.GetXaxis())
            bkg_edges = _edges(src.GetAxis(axes["bdt_background"]))
            fd_edges = _edges(src.GetAxis(axes["bdt_prompt"]))
            axis_edges = (mass_edges, bkg_edges, fd_edges)
            nbins = array("i", [len(edges) - 1 for edges in axis_edges])
            minima = array("d", [edges[0] for edges in axis_edges])
            maxima = array("d", [edges[-1] for edges in axis_edges])
            sp = ROOT.THnSparseD("hSparseD", "Filtered D candidates;mass;BDT background;BDT feed-down",
                                 3, nbins, minima, maxima)
            for iaxis, edges in enumerate(axis_edges):
                sp.SetBinEdges(iaxis, array("d", edges))
            sp.Sumw2()
            point = array("d", [0.0, 0.0, 0.0])
            for (j, jbkg, jfd), (content, variance) in sparse_bins[(im, ip)].items():
                if content == 0.0 and variance == 0.0:
                    continue
                point[0] = hist.GetXaxis().GetBinCenter(j)
                point[1] = src.GetAxis(axes["bdt_background"]).GetBinCenter(jbkg)
                point[2] = src.GetAxis(axes["bdt_prompt"]).GetBinCenter(jfd)
                sp.Fill(point, content)
            # Restore the source sumw2 values (Fill would otherwise use content**2).
            coord = array("i", [0, 0, 0])
            for index in range(sp.GetNbins()):
                sp.GetBinContent(index, coord)
                key = tuple(coord)
                sp.SetBinError2(index, sparse_bins[(im, ip)][key][1])
            dst = path / f"mult_{im}_pt_{ip}.root"
            fout = ROOT.TFile.Open(str(dst), "RECREATE")

            if not fout or fout.IsZombie():
                raise RuntimeError(f"Cannot create {dst}")
            try:
                fout.WriteTObject(hist, "hInvariantMass")
                fout.WriteTObject(sp, "hSparseD")
                fout.WriteTObject(ROOT.TNamed("source_histogram", histogram_path), "source_histogram")
            finally:
                fout.Close()
            generated.append(str(dst))
        return {"file": str(input_file), "selected_source_bins": n_selected}, generated
    finally:
        fin.Close()
