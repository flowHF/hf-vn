import os
import sys
import yaml
from ROOT import TFile
import argparse

CORRECTIONS = {
    0: "recenter_x",
    1: "recenter_y",
    2: "twist_x",
    3: "twist_y",
    4: "rescale_x",
    5: "rescale_y",
}

DETECTORS = [
    "FT0C",
    "FT0A",
    "FT0M",
    "FV0A",
    "TPCPos",
    "TPCNeg",
    "TPCAll",
]

parser = argparse.ArgumentParser(description="Check CCDB corrections")
parser.add_argument("config", type=str, default="cfg.yml", help="Directory containing CCDB correction files")

with open(parser.parse_args().config, "r") as f:
    cfg = yaml.safe_load(f)

out_file = TFile.Open(cfg['outfile'], "RECREATE")

print(f"\n\n ---> Processing CCDB Junlee corrections")
for cfg_file in cfg['files']:

    os.system(f"alien.py cp {cfg_file['alien_path']} file:{cfg_file['path']}")

    out_file.mkdir(cfg_file['name'])
    out_file.cd(cfg_file['name'])

    ccdb_file = TFile.Open(cfg_file['path'], "READ")
    h_corrs = ccdb_file.Get("ccdb_object")
    h_corrs.GetXaxis().SetRangeUser(0, 100)
    for i_detector, det_name in enumerate(DETECTORS):
        out_file.mkdir(f"{cfg_file['name']}/{det_name}")
        out_file.cd(f"{cfg_file['name']}/{det_name}")
        h_corrs.GetZaxis().SetRangeUser(i_detector+1, i_detector+1)
        for i_corr_type in CORRECTIONS:
            corr_type = CORRECTIONS[i_corr_type]
            h_corrs.GetYaxis().SetRange(i_corr_type+1, i_corr_type+1)
            h_corrs.Project3D("x").Write(f"h_corrs_{corr_type}")

print(f"\n\n Finished! \n")
out_file.Close()
