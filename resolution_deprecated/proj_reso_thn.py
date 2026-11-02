import os
import sys
import argparse
import ROOT
from ROOT import TFile
import yaml
import glob
import shutil
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append('/home/mdicosta/alice/DmesonAnalysis/run3/flow')
from flow_analysis_utils import get_resolution, getListOfHisots
sys.path.append('/home/mdicosta/alice/DmesonAnalysis')
from utils.StyleFormatter import SetObjectStyle, SetGlobalStyle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
SetGlobalStyle(padleftmargin=0.15, padbottommargin=0.15,
               padrightmargin=0.15, titleoffsety=1.1, maxdigits=3, titlesizex=0.03,
               labelsizey=0.04, setoptstat=0, setopttitle=0, palette=ROOT.kGreyScale)

ax_reso = {
    'cent': 0,
    'FT0cFV0a': 1,
    'FT0cTPCtot': 2,
    'FV0aTPCtot': 3,
    'occ': 4,
}

def get_reso_sparses(files_paths):
    thn_resos = []
    
    with ThreadPoolExecutor(22) as executor:
        reso_files = []
        for path in files_paths:
            file = executor.submit(TFile.Open, path, 'read')  # Submit job
            reso_files.append((file, path))  # Store as tuple
            print(f'Loading {path}')
        
        for iFile, (file, path) in enumerate(reso_files):
            reso = file.result()  # Get TFile object
            if not reso or reso.IsZombie():
                print(f"ERROR: Could not open file {path}")
                continue

            thn_reso = reso.Get('hf-task-flow-charm-hadrons/spReso/hSparseReso')
            if not thn_reso:
                print(f"WARNING: hSparseReso not found in {path}")
            else:
                thn_resos.append(thn_reso)
                print(f"[{iFile}, {path}] {thn_reso}")

            reso.Close()

    print(f"thn_resos: {thn_resos}")
    return thn_resos

def process_reso_sparse(isparse, sparse, iOcc, occ, ax_evsels, ax_reso):
    sparse.GetAxis(ax_reso['occ']).SetRangeUser(occ[0], occ[1])
    for ax in ax_evsels:
        sparse.GetAxis(ax).SetRange(1,1)    # Binary flags

    hFT0cFV0a = sparse.Projection(ax_reso['FT0cFV0a'], ax_reso['cent'])
    hFT0cFV0a.SetName(f'hFT0cFV0a_{iOcc}_{isparse}')
    hFT0cFV0a.SetDirectory(0)

    hFT0cTPCtot = sparse.Projection(ax_reso['FT0cTPCtot'], ax_reso['cent'])
    hFT0cTPCtot.SetName(f'hFT0cTPCtot_{iOcc}_{isparse}')
    hFT0cTPCtot.SetDirectory(0)
    
    hFV0aTPCtot = sparse.Projection(ax_reso['FV0aTPCtot'], ax_reso['cent'])
    hFV0aTPCtot.SetName(f'hFV0aTPCtot_{iOcc}_{isparse}')
    hFV0aTPCtot.SetDirectory(0)

    return hFT0cFV0a, hFT0cTPCtot, hFV0aTPCtot

def proj_reso(config_flow, ax_evsels):
    
    reso_sparses = get_reso_sparses(config['reso_files'])
    proj_dir = f'{config["out_dir"]}/proj/'
    if os.path.exists(proj_dir):
        shutil.rmtree(proj_dir)
    os.makedirs(proj_dir)
    
    for iOcc, occ in enumerate(config["occ_classes"]):
        for name, sel_axes in ax_evsels.items():
            with ProcessPoolExecutor(24) as executor:
                reso_histos = [
                    executor.submit(
                        process_reso_sparse,
                        i,
                        reso_sparse,
                        iOcc,
                        occ,
                        sel_axes,
                        ax_reso
                    )
                    for i, reso_sparse in enumerate(reso_sparses)
                ]
            results = []
            for hist in reso_histos:
                results.append(hist.result())
            
            for i, (hFT0cFV0a_i, hFT0cTPCtot_i, hFV0aTPCtot_i) in enumerate(results):
                if i == 0:
                    hFT0cFV0a = hFT0cFV0a_i.Clone('hFT0cFV0a')
                    hFT0cFV0a.SetDirectory(0)
                    hFT0cFV0a.Reset()
                    
                    hFT0cTPCtot = hFT0cTPCtot_i.Clone('hFT0cTPCtot')
                    hFT0cTPCtot.SetDirectory(0)
                    hFT0cTPCtot.Reset()
                    
                    hFV0aTPCtot = hFV0aTPCtot_i.Clone('hFV0aTPCtot')
                    hFV0aTPCtot.SetDirectory(0)
                    hFV0aTPCtot.Reset()

                hFT0cFV0a.Add(hFT0cFV0a_i)
                hFT0cTPCtot.Add(hFT0cTPCtot_i)
                hFV0aTPCtot.Add(hFV0aTPCtot_i)


            outfile = TFile.Open(f'{proj_dir}/proj_reso_occ_{occ[0]}_{occ[1]}_{name}_{config["suffix"]}.root', 'RECREATE')
            outfile.mkdir('hf-task-flow-charm-hadrons/spReso')
            outfile.cd('hf-task-flow-charm-hadrons/spReso')
            hFT0cFV0a.Write(f'hSpResoFT0cFV0a')
            hFT0cTPCtot.Write(f'hSpResoFT0cTPCtot')
            hFV0aTPCtot.Write(f'hSpResoFV0aTPCtot')
            outfile.Close()

def create_resolution_dataframe(config):

    rows = []
    for occ_class in config['occ_classes']:
        occ_label = f"{occ_class[0]} < Occ < {occ_class[1]}"
        for cent_class in config['cent_classes']:
            cent_label_df = f"{cent_class[0]} < Cent < {cent_class[1]}"
            cent_label = f"k{cent_class[0]}{cent_class[1]}"
            outdir = f"{config['out_dir']}/cent/{cent_label}/"

            for file in os.listdir(outdir):
                if file.startswith(f"resosp_occ_{occ_class[0]}_{occ_class[1]}") and file.endswith(".root"):
                    label = file.replace(f"resosp_occ_{occ_class[0]}_{occ_class[1]}_", "") \
                                .replace(".root", "") \
                                .replace(f"_{config['suffix']}", "")
                    f = TFile(os.path.join(outdir, file))
                    reso = f.Get("FT0c_FV0a_TPCtot/histo_reso_delta_cent").GetBinContent(1)
                    f.Close()
                    rows.append({
                        "Centrality": cent_label_df,
                        "Occupancy": occ_label,
                        "Evsel": label,
                        "Resolution": reso
                    })

    # Create the DataFrame after collecting all rows
    reso_dataframe = pd.DataFrame(rows)
    os.makedirs(f"{config['out_dir']}/df/", exist_ok=True)
    reso_dataframe.to_parquet(f"{config['out_dir']}/df/reso_dataframe.parquet")
    
    
    # reso_dataframe = pd.DataFrame({"Centrality", "Occupancy", "Evsel", "Resolution"})
    # for occ_class in config['occ_classes']:
    #     occ_label = f"{occ_class[0]} < Occ < {occ_class[1]}"
    #     for cent_class in config['cent_classes']:
    #         cent_label_df = f"{cent_class[0]} < Cent < {cent_class[1]}"
    #         cent_label = f"k{cent_class[0]}{cent_class[1]}"
    #         outdir = f"{config['out_dir']}/cent/{cent_label}/"
    #         for file in os.listdir(outdir):
    #             if file.startswith(f"resosp_occ_{occ_class[0]}_{occ_class[1]}") and file.endswith(".root"):
    #                 label = file.replace(f"resosp_occ_{occ_class[0]}_{occ_class[1]}_", "").replace(".root", "").replace(f"_{config['suffix']}", "")
    #                 f = TFile(os.path.join(outdir, file))
    #                 reso = f.Get("FT0c_FV0a_TPCtot/histo_reso_delta_cent").GetBinContent(1)
    #                 reso_dataframe = reso_dataframe.append({"Centrality": cent_label_df, "Occupancy": occ_label, "Evsel": label, "Resolution": reso}, ignore_index=True)
    
    # os.makedirs(f"{config['out_dir']}/df/", exist_ok=True)
    # reso_dataframe.to_parquet(f"{config['out_dir']}/df/reso_dataframe.parquet")

# def compute_reso_all_cases(config):
    
#     proj_inputdir = f"{config['out_dir']}/proj/"
#     files, suffixes = [], []
#     for file in glob.glob(os.path.join(proj_inputdir, "proj_reso*.root")):
#         files.append(file)
#         suffixes.append(os.path.basename(file).replace("proj_reso", "").replace(".root", ""))

#     resos, labels = {}, {}
#     cent_classes = config['cent_classes']
#     for cent in cent_classes:
#         cent_string = f"k{cent[0]}{cent[1]}"
#         print(f'Processing centrality class: {cent_string}')
#         outdir = f"{config['out_dir']}/cent/{cent_string}/"
#         for file, suffix in zip(files, suffixes):
#             print(f"Processing file: {file}")
#             os.system(f"python3 /home/mdicosta/alice/DmesonAnalysis/run3/flow/compute_reso.py \
#                         {file} -c {cent_string} -vn {'sp'} -o {outdir} -s {suffix}")
#         resos, labels = produce_plots(config, outdir, cent_string, suffixes)

# def produce_plots(config, outdir, cent_string, suffixes, all_key='all'):
#     for occ_class in config['occ_classes']:
#         resos, labels = [], []
        
#         # Assuming "all_key" is available in `resos` for comparison
#         for file in os.listdir(outdir):
#             if file.startswith(f"resosp_occ{occ_class[0]}_{occ_class[1]}") and file.endswith(".root"):
#                 f = TFile(os.path.join(outdir, file))
#                 resos.append(f.Get("FT0c_FV0a_TPCtot/histo_reso_delta_cent").GetBinContent(1))
#                 labels.append(file.replace(f"resosp_occ{occ_class[0]}_{occ_class[1]}", "").replace(".root", ""))
#                 f.Close()
        
#         # Assuming `resos` and `labels` are ready
#         all_reso = resos[labels.index(all_key)]

#         # Create the figure and axes for subplots
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))  # Two subplots (1 row, 2 columns)

#         # --- Left panel: Plot the resolution values ---
#         # Create the x-values corresponding to the bins
#         x = np.arange(len(resos))
#         x_edges = np.repeat(x, 2)[1:-1]  # Create edges for the steps
#         resos_edges = np.repeat(resos, 2)[1:-1]  # Adjust resos for the edges

#         # Plot resolution (step plot with filled area under curve)
#         ax1.step(x_edges, resos_edges, where='post', linestyle='-', color='blue', label='Resolution')
#         ax1.fill_between(x_edges, resos_edges, step='post', alpha=0.3, color='blue')
#         ax1.set_xticks(x)
#         ax1.set_xticklabels(labels, rotation=90, ha='left', fontsize=12)
#         ax1.set_xlabel("Configuration Label", fontsize=14)
#         ax1.set_ylabel("Resolution", fontsize=14)
#         ax1.set_title(f"Resolution for {occ_class[0]} < Occ < {occ_class[1]}", fontsize=16)
#         ax1.grid(True)

#         # --- Right panel: Plot the ratio with respect to 'all' ---
#         ratios = [reso / all_reso for reso in resos]  # Compute the ratio with respect to 'all'

#         ax2.step(x, ratios, where='post', linestyle='-', color='green', label='Resolution Ratio')
#         ax2.fill_between(x, ratios, step='post', alpha=0.3, color='green')
#         ax2.set_xticks(x)
#         ax2.set_xticklabels(labels, rotation=90, ha='left', fontsize=12)
#         ax2.set_xlabel("Configuration Label", fontsize=14)
#         ax2.set_ylabel("Resolution Ratio", fontsize=14)
#         ax2.set_title(f"Resolution Ratio for {occ_class[0]} < Occ < {occ_class[1]}", fontsize=16)
#         ax2.grid(True)

#         # Adjust the layout to prevent overlap
#         plt.tight_layout()

#         # Save the plot
#         output_path = f"{outdir}/reso_occ_{occ_class[0]}_{occ_class[1]}.png"
#         print(f"Saving plot for occupancy class {output_path}")
#         plt.savefig(output_path, dpi=300)
#         plt.close()

def produce_plots(config, outdir, cent_string, suffixes):
    
    resos = {}
    labels = {}
    for occ_class in config['occ_classes']:
        occ_label = f"{occ_class[0]} < Occ < {occ_class[1]}"
        resos[occ_label] = []
        labels[occ_label] = []
        for file in os.listdir(outdir):
            if file.startswith(f"resosp_occ_{occ_class[0]}_{occ_class[1]}") and file.endswith(".root"):
                f = TFile(os.path.join(outdir, file))
                hist = f.Get("FT0c_FV0a_TPCtot/histo_reso_delta_cent")
                if hist:
                    resos[occ_label].append(hist.GetBinContent(1))
                    label = file.replace(f"resosp_occ_{occ_class[0]}_{occ_class[1]}_", "").replace(".root", "").replace(f"_{config['suffix']}", "")
                    labels[occ_label].append(label)
                f.Close()

        # Plot
        plt.figure(figsize=(10, 6))
        plt.step(list(range(len(resos[occ_label]))), resos[occ_label], where='mid', linestyle='-', color='blue', label='Resolution')
        plt.fill_between(list(range(len(resos[occ_label]))), resos[occ_label], step='mid', alpha=0.3, color='blue')
        plt.xticks(list(range(len(resos[occ_label]))), labels[occ_label], rotation=90, ha='left', fontsize=18)
        plt.xlabel("Configuration Label", fontsize=14)
        plt.ylabel("Resolution", fontsize=14)
        plt.title(f"Resolution Values for {occ_class[0]} < Occ < {occ_class[1]}")
        plt.ylim(min(resos[occ_label]) - 0.001, max(resos[occ_label]) + 0.001)
        plt.grid(True)
        plt.tight_layout()
        output_path = f"{outdir}/reso_occ_{occ_class[0]}_{occ_class[1]}.png"
        print(f"Saving plot for occupancy class {output_path}")
        plt.savefig(output_path, dpi=300)
        plt.close()

    return resos, labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument("config", metavar="text",
                        default="config.yaml", help="flow configuration file")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if config["project"]:
        print("Projection enabled")
        proj_reso(config, config["ax_evsels"])
    else:
        print("No projection, computing resolution directly")
    
    create_resolution_dataframe(config)
    print("Resolution computation finished")
