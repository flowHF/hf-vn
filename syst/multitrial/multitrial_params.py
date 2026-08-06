#!/usr/bin/env python3
"""
To run the script you need to have previously run "run_multitrial_fit.sh" to generate the trials_results.parquet file with the trial results.

For Rebin, Chi2, Significance, MassMin, MassMax, Sigma and Mean the script produces a plot of variable vs trial
In addition by using "--plot A:B A:C B:C" you can produce a plot of whichever variable you like 
    as long as it has the name of one of the .parquet columns  
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from collections import defaultdict

try:
    import ROOT
except ImportError:
    ROOT = None

# ----------------------------------------------------------------------
def parse_mass_fit_ranges(ranges_series):
    """Parse mass fit ranges from string or list format."""
    mass_min, mass_max = [], []
    for val in ranges_series:
        if isinstance(val, str):
            try:
                s = val.strip("[]")
                parts = [float(x.strip()) for x in s.split(",")]
                if len(parts) == 2:
                    mass_min.append(parts[0])
                    mass_max.append(parts[1])
                else:
                    mass_min.append(np.nan)
                    mass_max.append(np.nan)
            except:
                mass_min.append(np.nan)
                mass_max.append(np.nan)
        elif isinstance(val, (list, tuple)) and len(val) == 2:
            mass_min.append(float(val[0]))
            mass_max.append(float(val[1]))
        else:
            mass_min.append(np.nan)
            mass_max.append(np.nan)
    return np.array(mass_min, dtype=float), np.array(mass_max, dtype=float)

# ----------------------------------------------------------------------
def compute_ylimits(var_name, y_vals):
    """Calculate y_min and y_max based on the variable name (without suffixes)."""
    valid = y_vals[~np.isnan(y_vals)]
    if len(valid) == 0:
        return 0, 1

    if var_name in ["Rebin"]:
        y_min = np.nanmin(y_vals) - 1
        y_max = np.nanmax(y_vals) + 1
    elif var_name == "Chi2":
        y_min = 0.0
        y_max = np.ceil(np.nanmax(y_vals))
        if y_max == 0:
            y_max = 1
    elif var_name == "Significance":
        y_min = np.nanmin(y_vals) - 5
        y_max = np.nanmax(y_vals) + 5
    elif var_name in ["MassMin", "MassMax"]:
        y_min = np.nanmin(y_vals) - 0.01
        y_max = np.nanmax(y_vals) + 0.01
    elif var_name == "Sigma":
        y_min = np.nanmin(y_vals) - 0.00005
        y_max = np.nanmax(y_vals) + 0.00005
    else:  # Mean and anything else
        y_min = np.nanmin(y_vals) * 0.9995
        y_max = np.nanmax(y_vals) * 1.0005

    if y_min == y_max:
        y_min -= 0.001
        y_max += 0.001
    return y_min, y_max

# ----------------------------------------------------------------------
def clean_name(name):
    """Remove _BkgFunc or _BkgFuncVn suffixes from the task name."""
    for suffix in ["_BkgFuncVn", "_BkgFunc"]:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name  # fallback (should not happen)

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Variable vs Trial plots with optional custom plots"
    )
    parser.add_argument("--centrality", required=True,
                        choices=["0-20", "20-50", "50-100"],
                        help="Centrality class")
    parser.add_argument("--output", required=True,
                        help="Output directory")
    parser.add_argument("--plot", nargs='+', type=str, default=None,
                        help='Custom scatter plot in the form "Y:X" (e.g. "Mean:MassMin"). '
                             'Multiple pairs can be added separated by spaces.')
    args = parser.parse_args()

    # Process custom pairs
    custom_plots = []  # list of tuples (y_col, x_col)
    if args.plot:
        for pair in args.plot:
            parts = pair.split(':')
            if len(parts) != 2:
                parser.error(f"Invalid format '{pair}'. Use 'Y:X'.")
            y_col, x_col = parts[0], parts[1]
            custom_plots.append((y_col, x_col))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = Path(
        f"/home/rdrosu/alice/analisi/misure_v2_inclusive/crystal_ball/"
        f"{args.centrality}/misure/multitrial/syst/multitrial/fit/summary/trials_results.parquet"
    )
    if not parquet_path.exists():
        print(f"ERROR: file not found: {parquet_path}")
        return

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"ERROR reading Parquet file: {e}")
        return

    df_filtered = df[df["V2Type"] == "Cutset_0"].copy()
    if df_filtered.empty:
        print("No rows with V2Type == 'Cutset_0'.")
        return

    required = ["TrialIdx", "PtLabel", "BkgFunc", "BkgFuncVn"]
    for c in required:
        if c not in df_filtered.columns:
            print(f"ERROR: column '{c}' missing.")
            return

    if "MassFitRanges" in df_filtered.columns:
        mass_min, mass_max = parse_mass_fit_ranges(df_filtered["MassFitRanges"])
        df_filtered["MassMin"] = mass_min
        df_filtered["MassMax"] = mass_max
    else:
        df_filtered["MassMin"] = np.nan
        df_filtered["MassMax"] = np.nan

    for y_col, x_col in custom_plots:
        for col in [y_col, x_col]:
            if col not in df_filtered.columns:
                print(f"ERROR: column '{col}' required for custom plot not found.")
                return

    color_map_mpl = {"kLin": "red", "kPol2": "green", "kExpo": "blue"}
    default_color_mpl = "black"
    if ROOT is not None:
        color_map_root = {"kLin": ROOT.kRed, "kPol2": ROOT.kGreen + 2, "kExpo": ROOT.kBlue}
        default_color_root = ROOT.kBlack

    for pt_label, group in df_filtered.groupby("PtLabel", observed=True):
        group = group.sort_values("TrialIdx")
        x_trial = group["TrialIdx"].astype(int).values
        bkgfunc = group["BkgFunc"].values
        bkgfuncvn = group["BkgFuncVn"].values

        standard_vars = []
        for var in ["Mean", 
                    "Sigma", 
                    "Significance", 
                    "Chi2", 
                    #"Rebin", 
                    #"MassMin", 
                    #"MassMax", 
                    ]:
            if var in group.columns:
                y_vals = pd.to_numeric(group[var], errors='coerce').astype(float).values
                if not np.all(np.isnan(y_vals)):
                    standard_vars.append((var, y_vals))

        plot_tasks = []
        for var_name, y_vals in standard_vars:
            plot_tasks.append((f"{var_name}_BkgFunc", x_trial, y_vals, bkgfunc))
            plot_tasks.append((f"{var_name}_BkgFuncVn", x_trial, y_vals, bkgfuncvn))

        for y_col, x_col in custom_plots:
            y_c = pd.to_numeric(group[y_col], errors='coerce').astype(float).values
            x_c = pd.to_numeric(group[x_col], errors='coerce').astype(float).values
            valid = ~np.isnan(y_c) & ~np.isnan(x_c)
            if np.any(valid):
                x_cut = x_c[valid]
                y_cut = y_c[valid]
                bkgfunc_cut = bkgfunc[valid]
                bkgfuncvn_cut = bkgfuncvn[valid]
                plot_tasks.append((f"{y_col}_vs_{x_col}_BkgFunc", x_cut, y_cut, bkgfunc_cut))
                plot_tasks.append((f"{y_col}_vs_{x_col}_BkgFuncVn", x_cut, y_cut, bkgfuncvn_cut))
            else:
                print(f"Warning: no valid data for {y_col} vs {x_col} in {pt_label}")

        if not plot_tasks:
            print(f"No plot tasks for {pt_label}")
            continue

        # ----- Calculate shared Y limits for each variable -----
        groups = defaultdict(list)
        for name, x_data, y_data, color_var in plot_tasks:
            base_name = clean_name(name)
            groups[base_name].append((name, x_data, y_data, color_var))

        group_ylims = {}
        for base, tasks in groups.items():
            all_y = np.concatenate([t[2] for t in tasks])
            y_min, y_max = compute_ylimits(base, all_y)
            group_ylims[base] = (y_min, y_max)

        safe_label = str(pt_label).replace("/", "_").replace(" ", "_")


        pdf_path = output_dir / f"summary_{safe_label}_{args.centrality}.pdf"
        with PdfPages(pdf_path) as pdf:
            n_tasks = len(plot_tasks)
            for start in range(0, n_tasks, 4):
                subset = plot_tasks[start:start+4]
                n_sub = len(subset)
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                axes = axes.flatten()

                for idx, (name, x_data, y_data, color_var) in enumerate(subset):
                    ax = axes[idx]
                    colors = [color_map_mpl.get(c, default_color_mpl) for c in color_var]
                    ax.scatter(x_data, y_data, c=colors, s=8, edgecolors='none', alpha=0.7)

                    # X limits
                    x_min, x_max = np.nanmin(x_data), np.nanmax(x_data)
                    if x_min == x_max:
                        x_min -= 0.5
                        x_max += 0.5
                    ax.set_xlim(x_min - 0.05*(x_max-x_min), x_max + 0.05*(x_max-x_min))

                    # Y limits from group (shared)
                    base_name = clean_name(name)
                    y_min, y_max = group_ylims[base_name]
                    ax.set_ylim(y_min, y_max)

                    # Ticks
                    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
                    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

                    # Legend
                    unique_vals = sorted(set(color_var))
                    legend_elements = [
                        plt.Line2D([0], [0], marker='o', color='w', label=val,
                                   markerfacecolor=color_map_mpl.get(val, default_color_mpl),
                                   markersize=8)
                        for val in unique_vals
                    ]
                    ax.legend(handles=legend_elements, loc='best', fontsize='x-small')

                    # Axis labels
                    if "_vs_" in base_name:
                        parts = base_name.split("_vs_")
                        y_label = parts[0]
                        x_label = parts[1] if len(parts) > 1 else "X"
                    else:
                        x_label = "TrialIdx"
                        y_label = base_name  
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.set_title(f"{name}  ({args.centrality}%)")
                    ax.grid(True, alpha=0.3)

                for j in range(n_sub, 4):
                    axes[j].set_visible(False)

                fig.suptitle(f"{pt_label}  (Centrality {args.centrality}%)", fontsize=14, fontweight='bold')
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                pdf.savefig(fig)
                plt.close(fig)
        print(f"PDF saved: {pdf_path}")

        # ----- ROOT -----
        if ROOT is not None:
            root_path = output_dir / f"summary_{safe_label}_{args.centrality}.root"
            f_root = ROOT.TFile(str(root_path), "RECREATE")

            for name, x_data, y_data, color_var in plot_tasks:
                mg = ROOT.TMultiGraph()
                base_name = clean_name(name)
                if "_vs_" in base_name:
                    parts = base_name.split("_vs_")
                    y_ax_label = parts[0]
                    x_ax_label = parts[1] if len(parts) > 1 else "X"
                else:
                    x_ax_label = "TrialIdx"
                    y_ax_label = base_name
                mg.SetTitle(f"{name} - {pt_label} ({args.centrality}%);{x_ax_label};{y_ax_label}")

                for cat in sorted(set(color_var)):
                    mask = color_var == cat
                    x_b = x_data[mask]
                    y_b = y_data[mask]
                    n = len(x_b)
                    if n == 0:
                        continue

                    gr = ROOT.TGraph(n, x_b.astype(np.float64), y_b.astype(np.float64))
                    gr.SetName(f"gr_{cat.replace('k', '')}")
                    gr.SetTitle(cat)
                    gr.SetLineWidth(0)
                    gr.SetMarkerStyle(20)
                    gr.SetMarkerSize(0.5)
                    col = color_map_root.get(cat, default_color_root)
                    gr.SetMarkerColor(col)
                    gr.SetLineColor(col)
                    mg.Add(gr)

                mg.Write(name)

            f_root.Close()
            print(f"ROOT saved: {root_path}")

if __name__ == "__main__":
    main()