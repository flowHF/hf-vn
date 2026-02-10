import pandas as pd
import numpy as np
import argparse
import awkward as ak
import uproot
import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import logger, get_centrality_bins

def converter(file, table, cent_min, cent_max, queries, labels):

    # Pick data tree, convert to pandas dataframe, and save as parquet
    with uproot.open(file) as f:
        keys = f.keys()
        print(keys)

        matched_keys = [key for key in keys if table in key]
        dfs = []
        for key in matched_keys:
            logger(f"Reading {key}", level='INFO')
            array = f[key].arrays(library="ak")  # get awkward array
            df = ak.to_dataframe(array)          # convert to pandas DataFrame
            if (len(df) == 0):
                logger(f"Empty DataFrame for key: {key}", level='WARNING')
            dfs.append(df)

        full_df = pd.concat(dfs, ignore_index=True)

    # Create parquet file name
    cent_query = f"fCentrality >= {cent_min} and fCentrality < {cent_max} and "
    for query, label in zip(queries, labels):
        if query != "":
            queried_df = full_df.query(cent_query + query)
        else:
            queried_df = full_df.query(cent_query)

        parquet_file = file.replace('.root', f'_{table}_{label}_cent_{cent_min}_{cent_max}.parquet')
        queried_df.to_parquet(parquet_file)
        logger(f"Saved {label} parquet file: {parquet_file} with entries {len(queried_df)}", level='INFO')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for ML training.")
    parser.add_argument('--data', '-d', default='AO2D_data.root', help='Path to the input AO2D data file.')
    parser.add_argument('--mc', '-m', default='AO2D_mc.root', help='Path to the input AO2D MC file.')
    parser.add_argument('--cent_class', '-c', default='AO2D_mc.root', help='Path to the input AO2D MC file.')
    parser.add_argument('--table', '-t', default='o2hfcandX', help='Name of the tree to convert.')
    args = parser.parse_args()

    _, (cent_min, cent_max) = get_centrality_bins(args.cent_class)
    if args.data != "AO2D_data.root":
        converter(args.data, args.table, cent_min, cent_max, queries=["fM > 1.6 and fM < 2.5"], labels=["data"])
    if args.mc != "AO2D_mc.root":
        print("Converting MC file...")
        converter(args.mc, args.table, cent_min, cent_max, queries=["fM > 1.6 and fM < 2.5 and fOriginMcRec == 1", "fM > 1.6 and fM < 2.5 and fOriginMcRec == 2"], labels=["prompt", "fd"])