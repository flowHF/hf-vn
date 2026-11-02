import pandas as pd
import numpy as np
import argparse

import yaml
import awkward as ak
import uproot
import os
import sys
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import logger, get_centrality_bins

def converter(file, table, query, cols_to_keep, cols_to_add, label):

    # Pick data tree, convert to pandas dataframe, and save as parquet
    with upopen(file) as f:
        keys = f.keys()

        matched_keys = [key for key in keys if table in key and len(key.split('/')) == 2]
        dfs = []
        for key in matched_keys:
            logger(f"Reading {key}", level='INFO')
            array = f[key].arrays(library="ak")  # get awkward array
            df = ak.to_dataframe(array)          # convert to pandas DataFrame
            if (len(df) == 0):
                logger(f"Empty DataFrame for key: {key}", level='WARNING')
            dfs.append(df)

        # Print all columns before filtering
        if cols_to_keep is not None:
            dfs = [df[cols_to_keep] for df in dfs]
        full_df = pd.concat(dfs, ignore_index=True)

        # Add new columns
        if cols_to_add is not None:
            for col in cols_to_add:
                print(f"Adding new column: {col['name']} with formula: {col['formula']}")
                full_df[col['name']] = full_df.eval(col['formula'])

    # Create parquet file name
    parquet_file = file.replace('.root', f'_{table}_{label}.parquet')
    if query != "":
        print(f"Applying query: {query}")
        queried_df = full_df.query(query)
        queried_df.to_parquet(parquet_file)
        logger(f"Saved {label} parquet file: {parquet_file} with entries {len(queried_df)}", level='INFO')
    else:
        full_df.to_parquet(parquet_file)
        logger(f"Saved {label} parquet file: {parquet_file} with entries {len(full_df)}", level='INFO')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for ML training.")
    parser = argparse.ArgumentParser(description='Arguments to pass')
    parser.add_argument('cfgFileName', metavar='text', default='cfg_prepare_inputs.yml', help='config file name for ml')
    args = parser.parse_args()

    print('Loading analysis configuration: ...', end='\r')
    with open(args.cfgFileName, 'r') as ymlCfgFile:
        cfg = yaml.load(ymlCfgFile, yaml.FullLoader)
    print('Loading analysis configuration: Done!')

    data_file_path = cfg['data']
    mc_prompt_file_path = cfg['mc_prompt']
    mc_fd_file_path = cfg['mc_fd']
    cent_class = cfg.get('cent_class', None)
    table = cfg['table']
    cols = cfg.get('cols_to_keep', None)
    common_query = cfg.get('common_query', '').strip()

    # Append centrality
    if cent_class is not None:
        _, (cent_min, cent_max) = get_centrality_bins(cent_class)
        cent_str = f"fCentrality >= {cent_min} and fCentrality < {cent_max}"
        common_query = f"{common_query} and {cent_str}" if common_query else cent_str

    # prompt
    print(f"\n\n----> Preparing prompt mc parquet file from {mc_prompt_file_path} for table {table} with query: {common_query}")
    prompt_part = cfg.get('prompt_query', '').strip()
    if common_query and prompt_part:
        prompt_query = f"{common_query} and {prompt_part}"
    else:
        prompt_query = common_query or prompt_part
    converter(mc_prompt_file_path, table, query=prompt_query, cols_to_keep=cols, cols_to_add=cfg.get('extend_cols', []), label='mc_prompt')

    # fd
    print(f"\n\n----> Preparing fd mc parquet file from {mc_fd_file_path} for table {table} with query: {common_query}")
    fd_part = cfg.get('fd_query', '').strip()
    if common_query and fd_part:
        fd_query = f"{common_query} and {fd_part}"
    else:
        fd_query = common_query or fd_part
    converter(mc_fd_file_path, table, query=fd_query, cols_to_keep=cols, cols_to_add=cfg.get('extend_cols', []), label='mc_fd')

    print(f"\n\n----> Preparing data parquet file from {data_file_path} for table {table} with query: {common_query}")
    converter(data_file_path, table, query=common_query, cols_to_keep=cols, cols_to_add=cfg.get('extend_cols', []), label='data')

    # Now prepare plots of all columns in the parquet files, comparing data and mc, and save them in a multi-page pdf file
