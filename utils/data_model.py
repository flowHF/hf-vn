import sys
import os
from ROOT import TFile # pyright: ignore # type: ignore
sys.path.append("./")
from utils import logger, get_centrality_bins

def get_pt_preprocessed_sparses(config, pt_label):
    """ Load preprocessed sparses for a given pt bin label from the preprocess output files.
    Args:
        config (dict): Configuration dictionary.
        pt_label (str): Pt bin label to load sparses for.
    Returns:
        sparses_data (dict): Dictionary of data sparses.
        sparses_reco (dict): Dictionary of reconstructed MC sparses.
        sparses_gen (dict): Dictionary of generated MC sparses.
        axes (dict): Dictionary of axes for each sparse.
    """

    sparses_data, sparses_reco, sparses_gen, axes = {}, {}, {}, {}
    pre_cfg = config['preprocess']

    # Find preprocess config of sparse with name "FlowSP" (this is the one to be projected)
    for input_cfg in pre_cfg['inputs']:
        # print(f"input_cfg: {input_cfg}")
        if not 'sparses' in input_cfg:
            continue
        for sparse_cfg in input_cfg['sparses']:
            if sparse_cfg['name'] == 'FlowSP':
                sparse_proj_data_cfg = sparse_cfg
                break

    prep_dir = config.get("outdirPrep", config["outdir"])
    if config["operations"].get("proj_data") or config["operations"].get("perform_fits_syst_reso"):
        infile_prep_data = TFile.Open(f"{prep_dir}/preprocess/{pt_label}/FlowSP/AnalysisResults_{pt_label}.root", "read")
        logger(f"Loading preprocessed sparse from: {prep_dir}/preprocess/{pt_label}/FlowSP/AnalysisResults_{pt_label}.root", level='INFO')
        axes['FlowSP'] = {ax: iax for iax, ax in enumerate(sparse_proj_data_cfg['axes']['names'])}
        sparses_data["FlowSP"] = infile_prep_data.Get("FlowSP/hSparseFlowSP")
        infile_prep_data.Close()

    if config["operations"].get("proj_mc"):
        infile_prep_mc = TFile.Open(f"{prep_dir}/preprocess/{pt_label}/MC/AnalysisResults_{pt_label}.root", "read")
        logger(f"Loading preprocessed sparse from: {prep_dir}/preprocess/{pt_label}/MC/AnalysisResults_{pt_label}.root", level='INFO')
        for key in infile_prep_mc.GetListOfKeys():
            key_name = key.GetName()
            if "Reco" in key_name:
                sparses_reco[key_name] = infile_prep_mc.Get(f"{key_name}/hSparse{key_name}")
            elif "Gen" in key_name:
                sparses_gen[key_name] = infile_prep_mc.Get(f"{key_name}/hSparse{key_name}")
            else:
                logger(f"Unknown sparse type in MC folder: {key_name}", level='ERROR')
            # Retrieve axes
            sparse_proj_mc_cfg = None
            for input_cfg in pre_cfg['inputs']:
                if not 'sparses' in input_cfg:
                    continue
                for sparse_cfg in input_cfg['sparses']:
                    if sparse_cfg['name'] == key_name:
                        sparse_proj_mc_cfg = sparse_cfg
                        break
            axes[key_name] = {ax: iax for iax, ax in enumerate(sparse_proj_mc_cfg['axes']['names'])}
        infile_prep_mc.Close()

    return sparses_data, sparses_reco, sparses_gen, axes

def get_tree_dict(tree_name):

    if tree_name == "O2hfcandflowinfo":
        return {
            'Mass': 'fM',
            'Pt': 'fPt',
            'ScoreBkg': 'fMlScore0',
            'ScoreFD': 'fMlScore1',
            'Sp': 'fScalarProd',
            'Cent': 'fCent',
        }
    if tree_name == "O2hfcandmptinfo":
        return {
            'Mass': 'fM',
            'Pt': 'fPt',
            'ScoreBkg': 'fScoreBkg',
            'ScoreFD': 'fScoreFD',
        }
    if tree_name == "O2hfcandd0lite":
        return {
            'Mass': 'fM',
            'Pt': 'fPt',
            'FlagMcMatchRec': 'fFlagMc',
            'FlagOriginMcRec': 'fOriginMcRec',
        }
    if tree_name == "O2hfcandd0ml":
        return {
            'ScoreBkg': 'fBdtOutputBkg',
            'ScoreFD': 'fBdtOutputNonPrompt',
        }
    if tree_name == "O2hfcanddplite":
        return {
            'Mass': 'fM',
            'Pt': 'fPt',
            'Cent': 'fCentrality',
            'FlagMcMatchRec': 'fFlagMcMatchRec',
            'FlagOriginMcRec': 'fOriginMcRec',
        }
    if tree_name == "O2hfcanddpml":
        return {
            'ScoreBkg': 'fMlScore0',
            'ScoreFD': 'fMlScore1',
        }

def get_sparse_dict(sparse_name, dmeson):
    """ Get dictionary mapping variable names to their respective axis indices in the sparse.
    Args:
        sparse_name (str): Name of the sparse.
        dmeson (str): Type of D meson ('Dzero', 'Dplus', 'Ds').
    Returns:
        sparse_dict (dict): Dictionary mapping variable names to axis indices.
    """

    if sparse_name == "CorrelMaps":
        return {
                'PoolBin': 0,
                'PtTrig': 1,
                'PtAssoc': 2,
                'DeltaEta': 3,
                'DeltaPhi': 4,
                'Mass': 5,
                'ScoreBkg': 6,
                'ScoreFD': 7
                }
    elif sparse_name == "CorrelTrig":
        return {
                'Mass': 0,
                'PtTrig': 1,
                'ScoreBkg': 2,
                'ScoreFD': 3
                }
    elif sparse_name == "FlowSP":
        return {
                'Mass': 0,
                'Pt': 1,
                'Cent': 2,
                'Sp': 3,
                'ScoreBkg': 4,
                'ScoreFD': 5,
                'Qvec': 6
                }
    else:
        if dmeson == 'Dzero':
            if sparse_name == "RecoPrompt" or sparse_name == "RecoFD" or sparse_name == "RecoRefl" or sparse_name == "RecoReflPrompt" or sparse_name == "RecoReflFD":
                return {
                    'ScoreBkg': 0,
                    'ScorePrompt': 1,
                    'ScoreFD': 2,
                    'Mass': 3,
                    'Pt': 4,
                    'Y': 5,
                    'CandType': 6,
                    'PtBMoth': 7,
                    'Origin': 8,
                    'NPvContr': 9,
                    'Cent': 10,
                    'Occ': 11,
                }
            elif sparse_name == "GenPrompt" or sparse_name == "GenFD":
                return {
                    'Pt': 0,
                    'PtBMoth': 1,
                    'Y': 2,
                    'Origin': 3,
                    'NPvContr': 4,
                    'Cent': 5,
                    'Occ': 6
                }
            else:
                logger(f"Unknown sparse type for Ds {sparse_name}", level='ERROR')
        elif dmeson == 'Dplus':
            if sparse_name == "RecoPrompt":
                return {
                    'Mass': 0,
                    'Pt': 1,
                    'ScoreBkg': 2,
                    'ScorePrompt': 3,
                    'ScoreFD': 4,
                    'Cent': 5,
                    'Occ': 6,
                }
            elif sparse_name == "RecoFD":
                return {
                    'Mass': 0,
                    'Pt': 1,
                    'ScoreBkg': 2,
                    'ScorePrompt': 3,
                    'ScoreFD': 4,
                    'Cent': 5,
                    'PtBMoth': 6,
                    'FlagBHad': 7,
                }
            elif sparse_name == "GenPrompt":
                return {
                    'Pt': 0,
                    'Y': 1,
                    'Cent': 2,
                    'Occ': 3
                }
            elif sparse_name == "GenFD":
                return {
                'Pt': 0,
                'Y': 1,
                'Cent': 2,
                'PtBMoth': 3,
                'FlagBHad': 4,
            }
            else:
                logger(f"Unknown sparse type for Ds {sparse_name}", level='ERROR')
        elif dmeson == 'Ds':
            if sparse_name == "RecoPrompt":
                return {
                    'Mass': 0,
                    'Pt': 1,
                    'Cent': 2,
                    'ScoreBkg': 3,
                    'ScorePrompt': 4,
                    'ScoreFD': 5,
                    'NPvContr': 6
                }
            elif sparse_name == "RecoFD":
                return {
                    'Mass': 0,
                    'Pt': 1,
                    'Cent': 2,
                    'ScoreBkg': 3,
                    'ScorePrompt': 4,
                    'ScoreFD': 5,
                    'NPvContr': 6,
                    'PtBMoth': 7,
                    'FlagBHad': 8
                }
            elif sparse_name == "GenPrompt":
                return {
                    'Pt': 0,
                    'Y': 1,
                    'NPvContr': 2,
                    'Cent': 3
                }
            elif sparse_name == "GenFD":
                return {
                    'Pt': 0,
                    'Y': 1,
                    'NPvContr': 2,
                    'Cent': 3,
                    'PtBMoth': 4,
                    'FlagBHad': 5
                }
            else:
                logger(f"Unknown sparse type for Ds {sparse_name}", level='ERROR')
        else:
            logger(f"Sparse dictionary {sparse_name} not defined for Dmeson type {dmeson}", level='ERROR')

CORR_BKG_COLUMNS = {
    'Cent': 'fCentrality',
    'ScoreBkg': 'fMlScore0',
    'ScoreFD': 'fMlScore1',
    'FlagMcMatchRec': 'fFlagMcMatchRec',
}

def get_corr_bkg_rename(tree_name):
    """
    Map TTree columns onto the names used in correlated_bkgs.py
    """

    tree_dict = get_tree_dict(tree_name)
    if tree_dict is None:
        return {}
    return {column: CORR_BKG_COLUMNS[variable]
            for variable, column in tree_dict.items()
            if variable in CORR_BKG_COLUMNS and column != CORR_BKG_COLUMNS[variable]}
