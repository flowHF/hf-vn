#!/usr/bin/env python3
"""
    Shared helpers for the handling of THnSparse correlation outputs.
"""
import ROOT


def _get_axis_indices(cfg, REQUIRED_AXES):
    try:
        keep_axes = cfg["preprocess"]["keep_axes"]
    except KeyError as exc:
        raise RuntimeError(f"Missing config key: preprocess.keep_axes ({exc})") from exc

    missing = [a for a in REQUIRED_AXES if a not in keep_axes]

    if missing:
        raise RuntimeError(
            f"Correlation step requires retained axes {list(REQUIRED_AXES)}; "
            f"missing {missing}"
        )

    return {name: keep_axes.index(name) for name in REQUIRED_AXES}


def _get_mass_bins(corr_cfg):
    mass_bins = corr_cfg.get("mass_bins")

    if mass_bins is None or len(mass_bins) < 2:
        raise RuntimeError("correlation.mass_bins must contain at least two edges")

    edges = [float(x) for x in mass_bins]

    if any(hi <= lo for lo, hi in zip(edges, edges[1:])):
        raise RuntimeError(
            f"correlation.mass_bins must be strictly increasing, got {edges}"
        )
    return edges


def _edge_index(axis, value):
    """Return ROOT edge index (1..nbins+1), requiring an existing edge."""
    for ibin in range(1, axis.GetNbins() + 2):
        edge = axis.GetBinLowEdge(ibin)

        if abs(edge - value) <= 1.e-7 * max(1.0, abs(value)):
            return ibin

    raise RuntimeError(
        f"Δη boundary {value} is not an existing edge of axis "
        f"'{axis.GetTitle()}'"
    )


def _project_delta_phi(h2, eta_ranges, name="hDeltaPhi"):
    """Project hCorr(Δη, Δφ) onto Δφ, summing one or more Δη ranges [low, high)."""

    if not h2:
        raise RuntimeError("Null TH2 passed to _project_delta_phi")

    axis = h2.GetXaxis()
    projection = None
    for irange, (low, high) in enumerate(eta_ranges):
        first_bin = _edge_index(axis, low)
        last_bin = _edge_index(axis, high) - 1

        if last_bin < first_bin:
            raise RuntimeError(f"Empty Δη interval [{low}, {high})")

        part = h2.ProjectionY(f"{name}_range_{irange}", first_bin, last_bin, "e")

        if not part:
            raise RuntimeError(f"Failed to project Δη bins {first_bin}..{last_bin}")
        part.SetDirectory(0)

        if projection is None:
            projection = part
        else:
            projection.Add(part)

    projection.SetName(name)
    projection.SetTitle(name)
    return projection


def _mkdir(parent, name):
    directory = parent.GetDirectory(name)

    if directory:
        return directory

    directory = parent.mkdir(name)

    if not directory:
        raise RuntimeError(f"Could not create ROOT directory '{name}'")
    return directory


def _write(directory, obj, name=None):
    directory.WriteTObject(obj, name or obj.GetName(), "overwrite")


def _make_tf1(name, formula, xmin, xmax):
    """Create a TF1 that is NOT registered in ROOT's global function list.

    Re-using a name in the global list makes ROOT delete the earlier function,
    which is dangerous when Python still holds a reference to it.
    """
    func = ROOT.TF1(name, formula, xmin, xmax)
    ROOT.TF1.AddToGlobalList(func, False)
    return func


def _copy_if_present(source_dir, output_dir, name):
    obj = source_dir.Get(name)

    if obj:
        _write(output_dir, obj, name)
