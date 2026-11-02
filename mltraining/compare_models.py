import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import argparse
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import check_dir, logger

parser = argparse.ArgumentParser(description="Compare ML models over pt bins.")
parser.add_argument("firstdir", type=str, help="Path to the first folder.")
parser.add_argument("firstlabel", type=str, help="Name/label for the first folder.")
parser.add_argument("secdir", type=str, help="Path to the second folder.")
parser.add_argument("seclabel", type=str, help="Name/label for the second folder.")
parser.add_argument("--ptbins", nargs="+", type=float, default=None, help="List of pt bin edges, e.g. --ptbins 0 3 6 12")

args = parser.parse_args()
dfs_folders = [args.firstdir, args.secdir]
dfs_labels = [args.firstlabel, args.seclabel]
out_folder = f'{args.firstdir}/compare_vs_{args.seclabel}'
pt_bins = []
if args.ptbins is not None:
    for pt_min, pt_max in zip(args.ptbins[:-1], args.ptbins[1:]):
        print(f"{pt_min}, {pt_max}")
        pt_bins.append([int(pt_min), int(pt_max)])
else:
    for directory in os.listdir(args.firstdir):
        if directory.startswith('pt_'):
            pt_bins.append([directory.split('_')[1], directory.split('_')[2]])

classes = ["Data", "Prompt", "FD"]
label_classes = ["Bkg", "Prompt", "FD"]
label_colors = ['#1f77b4', '#ff7f0e', '#9467bd']

nbins = 100
bins = np.linspace(0, 1, nbins + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
bin_widths = bins[1:] - bins[:-1]

for ipt in pt_bins:
    for iclass in classes:
        logger(f"Processing class {iclass} for pT bin {ipt[0]}-{ipt[1]} GeV/c", "INFO")
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        ratio_maxs = []
        ratio_mins = []
        for label_class, label_color in zip(label_classes, label_colors):
            histos_counts = []
            for ifolder, (ifolder_name, idf_label) in enumerate(zip(dfs_folders, dfs_labels)):
                for pt_dir in os.listdir(ifolder_name):
                    if pt_dir.startswith('pt_'):
                        pt_min, pt_max = pt_dir.split('_')[1], pt_dir.split('_')[2]
                        if ipt[0] >= int(pt_min) and ipt[1] <= int(pt_max):
                            df = pd.read_parquet(f"{ifolder_name}/pt_{pt_min}_{pt_max}/application/{iclass}_pT_{pt_min}_{pt_max}_ModelApplied.parquet.gzip")
                            logger(f"Loaded model {ifolder_name}/pt_{pt_min}_{pt_max}/application/{iclass}_pT_{pt_min}_{pt_max}_ModelApplied.parquet.gzip "
                                   f"for pt bin {ipt[0]}-{ipt[1]}", "INFO")

                df_pt_sel = df[(df['fPt'] >= int(ipt[0])) & (df['fPt'] < int(ipt[1]))]
                counts, edges = np.histogram(df_pt_sel[f"ML_output_{label_class}"], bins=bins)
                histos_counts.append(counts)
                integral = np.sum(counts)

                # Plot the markers on top of the histogram
                if ifolder == 0:
                    ax1.plot(bin_centers, counts, 'o', color=label_color, label=f"{idf_label}_{label_class}")
                else:
                    # Fill the area under the histogram
                    ax1.bar(bin_centers, counts, width=bin_widths, color=label_color, alpha=0.5,
                            label=f"{idf_label}_{label_class}", log=True)

            with np.errstate(divide='ignore', invalid='ignore'):
                ratios = np.array(histos_counts[1]) / np.array(histos_counts[0])
                ratios_unc = np.sqrt( (np.array(np.sqrt(histos_counts[1])) / np.array(histos_counts[0]))**2 + 
                                    ( (np.array(histos_counts[1])*np.array(np.sqrt(histos_counts[0]))) / np.array(histos_counts[0])**2)**2 )
            ax2.errorbar(bin_centers, ratios, yerr=ratios_unc, xerr=bin_widths, label=f'{label_class}', fmt='o', color=label_color)
            
            max_ratio_idx = np.argmax([ratio + ratio_unc for ratio, ratio_unc in zip(ratios, ratios_unc)])
            ratio_maxs.append(ratios[max_ratio_idx] + ratios_unc[max_ratio_idx])
            min_ratio_idx = np.argmin([ratio - ratio_unc for ratio, ratio_unc in zip(ratios, ratios_unc)])
            ratio_mins.append(ratios[min_ratio_idx] - ratios_unc[max_ratio_idx])

        ax1.set_title(f"Distributions of ML Outputs for {iclass}, pt_{ipt[0]}_{ipt[1]}", fontsize=18)
        ax1.set_xlabel("Score")
        ax1.set_ylabel("Frequency (log scale)")
        # ax1.set_yscale('log')
        ax1.legend()

        ax2.set_xlabel("Score")
        ax2.set_ylabel(f"{dfs_labels[1]}/{dfs_labels[0]}")
        # ax2.set_ylim([min(ratio_mins)-min(ratio_mins)*0.001,max(ratio_maxs)+max(ratio_maxs)*0.1])
        ax2.grid()
        # ax2.set_yscale('log')
        ax2.legend()

        dir_path = f"{out_folder}/pt_{ipt[0]}_{ipt[1]}/"
        os.makedirs(dir_path, exist_ok=True)
        logger(f"Saving plot to {dir_path}/{iclass}Distrs.pdf", "INFO")
        plt.savefig(f"{dir_path}/{iclass}Distrs.pdf", format="pdf", bbox_inches="tight")
