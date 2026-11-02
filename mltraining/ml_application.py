'''
python script to apply trained models using the hipe4ml package
run: python MLApplication.py cfgFileNameML.yml [--full_dataset, --test_set]
'''

import os
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ROOT
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, '..', 'utils'))
from utils import logger

from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler

LABEL_BKG = 0
LABEL_PROMPT = 1
LABEL_NONPROMPT = 2


def get_output_labels(cfg):
    labels = [cfg['output']['out_labels']['Bkg'], cfg['output']['out_labels']['Prompt']]
    if cfg['output']['out_labels']['FD'] is not None:
        labels.append(cfg['output']['out_labels']['FD'])
    return labels


def load_models(cfg, n_bins):
    model_list = cfg['application']['saved_models']
    model_handls = []
    for iBin in range(n_bins):
        model_path = model_list[iBin]
        if not isinstance(model_path, str):
            logger('path to model not correctly defined!', 'ERROR')
            sys.exit()
        model_path = os.path.expanduser(model_path)
        logger(f'Loaded saved model: {model_path}')
        model_handl = ModelHandler()
        model_handl.load_model_handler(model_path)
        model_handls.append(model_handl)
    return model_handls


def get_columns_to_save(cfg):
    '''
    returns a fresh copy of the list of columns to save (never modifies the config list),
    dropping the MC-only columns not present in df
    '''
    cols = cfg['application']['column_to_save_list']
    if not isinstance(cols, list):
        logger('column_to_save_list must be defined!', 'ERROR')
        sys.exit()
    cols = list(cols)
    if 'fM' not in cols:
        logger('fM is not going to be saved in the output dataframe!', 'WARNING')
    if 'fPt' not in cols:
        logger('fPt is not going to be saved in the output dataframe!', 'WARNING')
    return cols


def apply_and_save(cfg, model_handl, df, output_labels, out_path):
    '''
    apply the model to df, keep only the requested columns, add the ML scores and save to parquet
    returns the dataframe that has been saved
    '''
    y_pred = model_handl.predict(df, cfg['ml']['raw_output'])
    df_out = df.loc[:, get_columns_to_save(cfg)].copy()
    if model_handl.get_n_classes() < 3:
        df_out['ML_output'] = y_pred
    else:
        for iPred, lab in enumerate(output_labels):
            df_out[f'ML_output_{lab}'] = y_pred[:, iPred]
    df_out.to_parquet(out_path)
    return df_out


def save_score_distrs(df_out, out_dir, name):
    plt.figure(figsize=(10, 6))
    for col in df_out.columns:
        if 'ML_output' in col:
            plt.hist(df_out[col], bins=100, alpha=0.5, label=col, log=True)
    plt.title(f'Distributions of ML Outputs for {name}')
    plt.xlabel('Score')
    plt.ylabel('Frequency (log scale)')
    plt.legend()
    plt.savefig(f'{out_dir}/{name}Distrs.pdf', format='pdf', bbox_inches='tight')
    plt.close('all')


def score_thresholds(y_true, y_score, efficiencies):
    """Per-class score cuts at the requested signal efficiencies."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out = {}
    for eff in efficiencies:
        row = {}
        for iclass, name in [(LABEL_BKG, 'Bkg'), (LABEL_PROMPT, 'Prompt'), (LABEL_NONPROMPT, 'NonPrompt')]:
            score = y_score[:, iclass]
            mask = y_true == iclass
            if mask.sum() == 0:
                row[name] = (0., np.nan)
                continue
            thr = float(np.quantile(score[mask], 1. - eff))
            actual_eff = float((score[mask] >= thr).mean())
            row[name] = (thr, actual_eff)
        out[eff] = row
    return out


def make_score_hist(name, title, scores, cfg):
    root_cfg = cfg.get('root_score_hist', {})
    nbins = int(root_cfg.get('nbins', 500))
    xmin = float(root_cfg.get('xmin', 0.))
    xmax = float(root_cfg.get('xmax', 1.))
    hist = ROOT.TH1D(name, title, nbins, xmin, xmax)
    for score in np.asarray(scores):
        if np.isfinite(score):
            hist.Fill(float(score))
    return hist


def save_test_root(cfg, out_path, y_test, y_pred, histograms):
    root_file = ROOT.TFile(out_path, 'RECREATE')
    class_names = ['Bkg', 'Prompt', 'NonPrompt']

    for ipop, pop_name in enumerate(class_names):
        mask = y_test == ipop
        for iscore, score_name in enumerate(class_names):
            hist = make_score_hist(f'hScore_True{pop_name}_{score_name}',
                                   f'True {pop_name};{score_name} score;Candidates',
                                   y_pred[mask, iscore], cfg)
            hist.Write()

    for hist in histograms.values():
        hist.Write()

    root_file.Close()



def save_application_root(cfg, out_path, y_pred):
    root_file = ROOT.TFile(out_path, 'RECREATE')
    class_names = ['Bkg', 'Prompt', 'NonPrompt']
    for iclass, name in enumerate(class_names):
        hist = make_score_hist(f'hScore_{name}', f'{name} score;Score;Candidates',
                               y_pred[:, iclass], cfg)
        hist.Write()
    root_file.Close()


def report_working_points(cfg, model_handl, test_set, y_test, out_dir_pt, pt_bin):
    '''
    compute, print and save the fraction of candidates surviving a score cut
    for each score and each true population
    '''
    score_tol = 0.005
    scores = np.arange(0.0 - score_tol, 1.0 + score_tol, score_tol)
    y_pred_test = model_handl.predict(test_set, cfg['ml']['raw_output'])

    class_names = ['Bkg', 'Prompt', 'NonPrompt']
    histograms = {}

    for ipop, pop_name in enumerate(class_names):
        mask = y_test == ipop
        n_candidates = mask.sum()

        for iscore, score_name in enumerate(class_names):
            h = ROOT.TH1D(f'h{pop_name}Score_{score_name}',
                          f'{pop_name} candidates;{score_name} score cut;Fraction kept',
                          len(scores), 0.0 - score_tol, 1.0 + score_tol)

            for ibin, score_cut in enumerate(scores, 1):
                # First score is bkg score, cut is <=
                if iscore == 0:
                    fraction = ((y_pred_test[mask, iscore] <= score_cut).sum() / n_candidates
                                if n_candidates else 0.)
                # Second score is prompt score, cut is >=
                if iscore == 1:
                    fraction = ((y_pred_test[mask, iscore] >= score_cut).sum() / n_candidates
                                if n_candidates else 0.)
                # Third score is bkg score, cut is <=
                if iscore == 2:
                    fraction = ((y_pred_test[mask, iscore] <= score_cut).sum() / n_candidates
                                if n_candidates else 0.)
                h.SetBinContent(ibin, fraction)

            histograms[(ipop, iscore)] = h

    return y_pred_test, histograms


# full_dataset application: any file, any sample
def run_full_dataset(cfg, pt_bins, model_handls, output_labels):
    sa = cfg['application']
    out_base = os.path.expanduser(sa['output_dir'])
    treename = sa['treename']

    for inputFile, outName in zip(sa['inputs'], sa['output_names']):
        logger(f'Loading and preparing data file {inputFile}: ...')
        data_handler = TreeHandler(inputFile) if treename is None else TreeHandler(inputFile, treename)
        data_handler.slice_data_frame('fPt', pt_bins, True)
        logger(f'Loading and preparing data file {inputFile}: Done!')

        logger('Applying ML model to dataframes: ...')
        for iBin, ptBin in enumerate(pt_bins):
            out_dir_pt = os.path.join(out_base, f'pt{ptBin[0]}_{ptBin[1]}', 'apply_all_data')
            if os.path.isdir(out_dir_pt):
                logger(f"Output directory '{out_dir_pt}' already exists, overwrites possibly ongoing!", 'WARNING')
            else:
                os.makedirs(out_dir_pt)

            df = data_handler.get_slice(iBin)
            df_out = apply_and_save(cfg, model_handls[iBin], df, output_labels,
                                    f'{out_dir_pt}/{outName}_pT_{ptBin[0]}_{ptBin[1]}_ModelApplied.parquet.gzip')

            y_pred = model_handls[iBin].predict(df, cfg['ml']['raw_output'])

            if cfg.get('savedistrs'):
                save_score_distrs(df_out, out_dir_pt, outName)

            save_application_root(
                cfg,
                f'{out_dir_pt}/{outName}_pT_{ptBin[0]}_{ptBin[1]}_ModelApplied.root',
                y_pred
            )

            del df, df_out, y_pred

        logger('Applying ML model to dataframes: Done!')


# test_set-based application: reproduces the test set of ml_classification.py
def build_test_sets(cfg, iBin, prompt_df, fd_df, bkg_df):
    '''
    Same logic as data_prep in ml_classification.py (without plots), so that with the same seed_split,
    input files and settings the same test set is obtained.
    Returns the complete test set and its labels.
    '''
    prep = cfg['data_prep']
    dataset_opt = prep['dataset_opt']
    seed_split = prep['seed_split']
    test_f = prep['test_fraction']
    n_prompt, n_fd, n_bkg = len(prompt_df), len(fd_df), len(bkg_df)

    if dataset_opt == 'equal':
        n_keep = min([n_prompt, n_bkg]) if fd_df.empty else min([n_prompt, n_fd, n_bkg])
        bkg_keep, prompt_keep, fd_keep = bkg_df.iloc[:n_keep], prompt_df.iloc[:n_keep], fd_df.iloc[:n_keep]
    elif dataset_opt == 'max_signal':
        n_bkg_keep = round(prep['bkg_mult'][iBin] * (n_prompt + n_fd))
        if n_bkg_keep >= n_bkg:
            n_bkg_keep = n_bkg
            logger('Using all bkg available, not good!', 'WARNING')
        bkg_keep, prompt_keep, fd_keep = bkg_df.iloc[:n_bkg_keep], prompt_df, fd_df
    else:
        logger(f'{dataset_opt} is not a valid option!', 'ERROR')
        sys.exit()

    tot_df = pd.concat([bkg_keep, prompt_keep, fd_keep], sort=True)
    labels = np.array([LABEL_BKG] * len(bkg_keep) + [LABEL_PROMPT] * len(prompt_keep) +
                      [LABEL_NONPROMPT] * len(fd_keep))

    if test_f < 1:
        _, test_set, _, y_test = train_test_split(tot_df, labels, test_size=test_f, random_state=seed_split)
    else:
        test_set, y_test = tot_df, labels

    return test_set, y_test


def run_test_set(cfg, pt_bins, model_handls, output_labels):
    if cfg['data_prep']['test_fraction'] < 1.:
        logger('Using only a fraction of the MC for the application! Are you sure?', 'WARNING')

    logger('Loading and preparing data files: ...')
    tree = cfg['input']['treename']
    prompt_handler = TreeHandler(cfg['input']['prompt'], tree)
    fd_handler = None if cfg['input']['FD'] is None else TreeHandler(cfg['input']['FD'], tree)
    data_handler = TreeHandler(cfg['input']['data'], tree)

    if cfg['data_prep']['filt_bkg_mass']:
        bkg_handler = data_handler.get_subset(cfg['data_prep']['filt_bkg_mass'], frac=1.,
                                              rndm_state=cfg['data_prep']['seed_split'])
    else:
        bkg_handler = data_handler

    prompt_handler.slice_data_frame('fPt', pt_bins, True)
    if fd_handler is not None:
        fd_handler.slice_data_frame('fPt', pt_bins, True)
    data_handler.slice_data_frame('fPt', pt_bins, True)
    bkg_handler.slice_data_frame('fPt', pt_bins, True)
    logger('Loading and preparing data files: Done!\n')

    out_base = os.path.expanduser(cfg['output']['dir'])

    for iBin, ptBin in enumerate(pt_bins):
        logger(f'Starting ML application --- {ptBin[0]} < pT < {ptBin[1]} GeV/c')
        out_dir_pt = os.path.join(out_base, f'pt_{ptBin[0]}_{ptBin[1]}', 'apply_test_set')
        if os.path.isdir(out_dir_pt):
            logger(f"Output directory '{out_dir_pt}' already exists, overwrites possibly ongoing!", 'WARNING')
        else:
            os.makedirs(out_dir_pt)

        fd_df = pd.DataFrame() if fd_handler is None else fd_handler.get_slice(iBin)
        test_set, y_test = build_test_sets(cfg, iBin, prompt_handler.get_slice(iBin), fd_df,
                                           bkg_handler.get_slice(iBin))
        model = model_handls[iBin]
        suffix = f'pT_{ptBin[0]}_{ptBin[1]}_ModelApplied.parquet.gzip'

        # Apply to the complete test set
        y_pred_test = model.predict(test_set, cfg['ml']['raw_output'])

        # Save separate parquet files for the three true classes
        bkg_test = test_set[y_test == LABEL_BKG]
        prompt_test = test_set[y_test == LABEL_PROMPT]
        fd_test = test_set[y_test == LABEL_NONPROMPT]

        apply_and_save(cfg, model, bkg_test, output_labels, f'{out_dir_pt}/Bkg_{suffix}')
        apply_and_save(cfg, model, prompt_test, output_labels, f'{out_dir_pt}/Prompt_{suffix}')
        apply_and_save(cfg, model, fd_test, output_labels, f'{out_dir_pt}/NonPrompt_{suffix}')

        # Working points and efficiency TH1Ds
        _, eff_hists = report_working_points(cfg, model, test_set, y_test, out_dir_pt, ptBin)

        # Test-set ROOT file: 9 score histograms + 3 efficiency histograms
        save_test_root(
            cfg,
            f'{out_dir_pt}/TestSetScores_pT_{ptBin[0]}_{ptBin[1]}.root',
            y_test,
            y_pred_test,
            eff_hists
        )

        # Full data sample
        logger('Applying ML model to data dataframe: ...')
        data_df = data_handler.get_slice(iBin)
        df_out = apply_and_save(cfg, model, data_df, output_labels,
                                f'{out_dir_pt}/Data_{suffix}')
        y_pred_data = model.predict(data_df, cfg['ml']['raw_output'])

        save_application_root(
            cfg,
            f'{out_dir_pt}/DataScores_pT_{ptBin[0]}_{ptBin[1]}.root',
            y_pred_data
        )

        logger('Applying ML model to data dataframe: Done!')
        del bkg_test, prompt_test, fd_test, test_set, y_test, y_pred_test, data_df, df_out, y_pred_data


def main():
    parser = argparse.ArgumentParser(description='Arguments to pass')
    parser.add_argument('cfgFileName', metavar='text', default='cfgFileNameML.yml', help='config file name for ml')
    parser.add_argument('--full_dataset', help='apply the models to the files in full_dataset_appl.inputs',
                        action='store_true')
    parser.add_argument('--test_set', help='apply the models to the test-set MC and to the data of the training config',
                        action='store_true')
    args = parser.parse_args()

    logger('Loading analysis configuration: ...')
    with open(args.cfgFileName, 'r') as ymlCfgFile:
        cfg = yaml.load(ymlCfgFile, yaml.FullLoader)
    logger('Loading analysis configuration: Done!')

    apply_full_dataset, apply_test_set = args.full_dataset, args.test_set
    if not apply_full_dataset and not apply_test_set:
        logger('No flag provided, returning ... ')
        sys.exit(1)

    pt_bins = [[a, b] for a, b in zip(cfg['pt_ranges']['min'], cfg['pt_ranges']['max'])]
    output_labels = get_output_labels(cfg)
    model_handls = load_models(cfg, len(pt_bins))

    if apply_test_set:
        run_test_set(cfg, pt_bins, model_handls, output_labels)
    if apply_full_dataset:
        run_full_dataset(cfg, pt_bins, model_handls, output_labels)


if __name__ == '__main__':
    main()
