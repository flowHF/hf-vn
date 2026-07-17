'''
Script for the computation of pT shape weights
run: 
python ./compute_pt_weights.py \
        <cfg>
'''
import os
import sys
import argparse
import yaml
from ROOT import TFile, TCanvas, TLegend  # pylint: disable=import-error,no-name-in-module
from ROOT import kBlack, kRed, kAzure # pylint: disable=import-error,no-name-in-module
work_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('../')
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../utils")
from utils import logger
from ReadModel import ReadFONLL, ReadTAMU  #pylint: disable=wrong-import-position,import-error
from StyleFormatter import SetObjectStyle     #pylint: disable=wrong-import-position,import-error
sys.path.append(f"{os.path.dirname(os.path.abspath(__file__))}/../src")
from pre_process import get_inputs_sparse, get_input_paths  #pylint: disable=wrong-import-position,import-error

def fill_pt_spectrum(hist, pp_cross_sect, pp_cross_sect_pt_min=None, pp_cross_sect_pt_max=None,
                     RAA=None, RAA_pt_min=None, RAA_pt_max=None):
    """
    Fill histogram h using cross_sect(pt)
    optionally multiplied by RAA(pt) (with boundary saturation).
    """

    for i in range(1, hist.GetNbinsX() + 1):
        pt = hist.GetBinCenter(i)
        if pt < pp_cross_sect_pt_min or pt > pp_cross_sect_pt_max:
            continue  # skip out-of-range bins
        
        pt_low = hist.GetBinLowEdge(i)
        pt_up = hist.GetBinLowEdge(i+1)
        val = pp_cross_sect.integral(pt_low, pt_up)
        # val = pp_cross_sect(pt)
        if RAA is not None:
            if RAA_pt_min <= pt <= RAA_pt_max:
                val *= RAA(pt)
            elif pt > RAA_pt_max:
                val *= RAA(RAA_pt_max)
            else:
                val *= RAA(RAA_pt_min)
        hist.SetBinContent(i, val)

    hist.Sumw2()
    hist.Scale(1.0 / hist.Integral())
    return hist

def compute_pt_weights(cfg):

    # load input cfguration
    with open(cfg, 'r') as ymlCfgFile:
        cfg = yaml.load(ymlCfgFile, yaml.FullLoader)

    cfg_pt_weights = cfg['ptWeights']
    charmSpecie = cfg['Dmeson']

    # 'Ds', 'Dplus', 'Dzero', 'Lc'
    if charmSpecie not in ['Ds', 'Dplus', 'Dzero', 'Lc']:
        print(f'ERROR: D specie {charmSpecie} not supported! Only Ds, Dplus, Dzero, Lc is supported! Exit')
        sys.exit()

    cent = cfg['centrality']
    beautySpecie = cfg_pt_weights.get('BeautySpecie', None)
    rebin = cfg_pt_weights.get('Rebin', 1) # default rebin
    smooth = cfg_pt_weights.get('Smooth', 100) # default smooth
    suffix = cfg_pt_weights.get('Suffix', '')
    if suffix != '':
        suffix = '_' + suffix

    # Retrieve sparse inputs
    #___________________________________________________________________________________________________________________________
    for sparses_cfg in cfg['preprocess']['inputs']:
        if not 'sparses' in sparses_cfg:
            continue
        for sparse_cfg in sparses_cfg['sparses']:
            if sparse_cfg['name'] == 'GenPrompt':
                sparse_gen_prompt = sparse_cfg
                sparse_file_path = get_input_paths(sparses_cfg['files'], "AnalysisResults")[0]
            if sparse_cfg['name'] == 'GenFD':
                sparse_gen_FD = sparse_cfg
                sparse_file_path = get_input_paths(sparses_cfg['files'], "AnalysisResults")[0]

    sparse_file = TFile.Open(sparse_file_path, "read")
    sparseGenPrompt, axesPrompt = get_inputs_sparse(sparse_file, cfg, sparse_gen_prompt)
    sparseGenNonPrompt, axesNonPrompt = get_inputs_sparse(sparse_file, cfg, sparse_gen_FD)
    sparse_file.Close()

    sparseGenPrompt.SetName('sparseGenPrompt')
    sparseGenNonPrompt.SetName('sparseGenNonPrompt')

    # Pt shape of generated prompt cands from MC
    hPtGenPrompt = sparseGenPrompt.Projection(axesPrompt['Pt'])
    hPtGenPrompt.SetDirectory(0)
    hPtGenPrompt.SetName('hPtGenPrompt')
    hPtGenPrompt.Sumw2()
    hPtGenPrompt.Rebin(rebin)
    hPtGenPrompt.Scale(1./hPtGenPrompt.Integral())

    if beautySpecie:
        if charmSpecie == 'Ds' and beautySpecie == 'BsBmix':
            # Add LambdaB
            sparseGenDFromLambdaB = sparseGenNonPrompt.Clone('sparseGenDFromLambdaB')
            sparseGenDFromLambdaB.GetAxis(axesNonPrompt['FlagBHad']).SetRange(4, 4)
            hPtGenB = sparseGenDFromLambdaB.Projection(axesNonPrompt['PtBMoth'])

            # Add B+ and B0
            sparseGenDFromBPlusBZero = sparseGenNonPrompt.Clone('sparseGenDFromBPlusBZero')
            sparseGenDFromBPlusBZero.GetAxis(axesNonPrompt['FlagBHad']).SetRange(1, 2)
            hPtGenB.Add(sparseGenDFromBPlusBZero.Projection(axesNonPrompt['PtBMoth']))

            # Add Bs
            sparseGenDFromBs = sparseGenNonPrompt.Clone('sparseGenDFromBs')
            sparseGenDFromBs.GetAxis(axesNonPrompt['FlagBHad']).SetRange(3, 3)
            hPtGenBs = sparseGenDFromBs.Projection(axesNonPrompt['PtBMoth'])
            hPtGenBs.SetDirectory(0)
            hPtGenBs.Scale(1./2 * hPtGenBs.Integral())
            hPtGenB.Scale(1./2 * hPtGenB.Integral())
            hPtGenB.Add(hPtGenBs) # assuming 50% Bs and 50% B, reasonable for non-prompt Ds
        else:
            #TODO: modifications for other B mesons
            hPtGenB = sparseGenNonPrompt.Projection(axesNonPrompt['PtBMoth'])

        # Pt shape of generated non-prompt cands from MC
        hPtGenB.SetName('hPtGenB')
        hPtGenB.SetDirectory(0)
        hPtGenB.Sumw2()
        hPtGenB.Rebin(rebin)
        hPtGenB.Scale(1./hPtGenB.Integral())

    logger('MC input loaded', level='INFO')
    # load models predictions
    #___________________________________________________________________________________________________________________________
    sCharmFONLL, _, ptMinCharmFONLL, ptMaxCharmFONLL = ReadFONLL(cfg_pt_weights['CharmPtShapeFONLL'], True, charmSpecie)
    sBeautyFONLL, _, ptMinBeautyFONLL, ptMaxBeautyFONLL = ReadFONLL(cfg_pt_weights['BeautyPtShapeFONLL'], True, 'B')
    sCharmTAMU, _, PtMinCharmTAMU, ptMaxCharmTAMU = ReadTAMU(cfg_pt_weights['CharmRaaTAMU'])
    sBeautyTAMU, _, ptMinBeautyTAMU, ptMaxBeautyTAMU = ReadTAMU(cfg_pt_weights['BeautyRaaTAMU'])
    if charmSpecie == 'Ds' and beautySpecie == 'BsBmix' and cfg_pt_weights.get('BsRaaTAMU'):
        sBsTAMU, _, ptMinBsTAMU, ptMaxBsTAMU = ReadTAMU(cfg_pt_weights['BsRaaTAMU'])

    histoCharmNames = ['hPtCharmFONLLcent', 'hPtCharmFONLLmin', 'hPtCharmFONLLmax']
    histoBeautyNames = ['hPtBeautyFONLLBcent', 'hPtBeautyFONLLBmin', 'hPtBeautyFONLLBmax']
    modelPred = ['yCent', 'yMin', 'yMax']

    hPtCharmFONLL, hPtBeautyFONLL, hPtCharmFONLLtimesTAMU, hPtBeautyFONLLtimesTAMU = [], [], [], []
    hPtWeightsCharmFONLL, hPtWeightsBeautyFONLL, hPtWeightsCharmFONLLtimesTAMU, hPtWeightsBeautyFONLLtimesTAMU = [], [], [], []
    logger("Start computing pT weights", level='INFO')

    # D meson weights
    #___________________________________________________________________________________________________________________________
    for histoName, pred in zip(histoCharmNames, modelPred):
        hPtFONLL = hPtGenPrompt.Clone(histoName)
        hPtFONLL.SetTitle("Charm FONLL #it{p_{T}} shape;#it{p_{T}} (GeV/c);Norm. Counts")
        hPtFONLLtimesTAMU = hPtGenPrompt.Clone(histoName.replace("FONLL", "FONLLtimesTAMU"))
        hPtFONLLtimesTAMU.SetTitle("Charm FONLL #times TAMU #it{p_{T}} shape;#it{p_{T}} (GeV/c);Norm. Counts")

        fill_pt_spectrum(hPtFONLL, pp_cross_sect = sCharmFONLL[pred],
                         pp_cross_sect_pt_min = ptMinCharmFONLL, pp_cross_sect_pt_max = ptMaxCharmFONLL)
        hPtCharmFONLL.append(hPtFONLL)
        fill_pt_spectrum(hPtFONLLtimesTAMU,
                         pp_cross_sect = sCharmFONLL[pred], pp_cross_sect_pt_min = ptMinCharmFONLL, pp_cross_sect_pt_max = ptMaxCharmFONLL,
                         RAA = sCharmTAMU["yCent"], RAA_pt_min = PtMinCharmTAMU, RAA_pt_max = ptMaxCharmTAMU)
        hPtCharmFONLLtimesTAMU.append(hPtFONLLtimesTAMU)

        hPtWeightsFONLL = hPtFONLL.Clone(histoName.replace("Pt", "PtWeights"))
        hPtWeightsFONLL.Divide(hPtFONLL, hPtGenPrompt)
        hPtWeightsFONLL.Smooth(smooth)
        hPtWeightsFONLL.SetTitle("Charm #it{p_{T}} weights FONLL;#it{p_{T}} (GeV/c);Weights")
        hPtWeightsCharmFONLL.append(hPtWeightsFONLL)

        hPtWeightsFONLLtimesTAMU = hPtFONLLtimesTAMU.Clone(hPtFONLLtimesTAMU.GetName().replace("Pt", "PtWeights"))
        hPtWeightsFONLLtimesTAMU.Divide(hPtFONLLtimesTAMU, hPtGenPrompt)
        hPtWeightsFONLLtimesTAMU.Smooth(smooth)
        hPtWeightsFONLLtimesTAMU.SetTitle("Charm #it{p_{T}} weights FONLL #times TAMU;#it{p_{T}} (GeV/c);Weights")
        hPtWeightsCharmFONLLtimesTAMU.append(hPtWeightsFONLLtimesTAMU)

    # B meson weights
    #___________________________________________________________________________________________________________________________
    if beautySpecie:
        for histoName, pred in zip(histoBeautyNames, modelPred):
            hPtFONLL = hPtGenB.Clone(histoName)
            hPtFONLL.SetTitle("Beauty FONLL #it{p_{T}} shape;#it{p_{T}} (GeV/c);Norm. Counts")
            hPtFONLLtimesTAMU = hPtGenB.Clone(histoName.replace("FONLL", "FONLLtimesTAMU"))
            hPtFONLLtimesTAMU.SetTitle("Beauty FONLL #times TAMU #it{p_{T}} shape;#it{p_{T}} (GeV/c);Norm. Counts")

            fill_pt_spectrum(hPtFONLL, pp_cross_sect = sBeautyFONLL[pred],
                             pp_cross_sect_pt_min = ptMinCharmFONLL, pp_cross_sect_pt_max = ptMaxCharmFONLL)
            hPtBeautyFONLL.append(hPtFONLL)
            fill_pt_spectrum(hPtFONLLtimesTAMU,
                             pp_cross_sect = sBeautyFONLL[pred], pp_cross_sect_pt_min = ptMinCharmFONLL, pp_cross_sect_pt_max = ptMaxCharmFONLL,
                             RAA = sBeautyTAMU["yCent"] if beautySpecie != 'BsBmix' else (lambda pt: (sBeautyTAMU['yCent'](pt) + sBsTAMU['yCent'](pt)) / 2),
                             RAA_pt_min = ptMinBeautyTAMU if beautySpecie != 'BsBmix' else min([ptMinBeautyTAMU, ptMinBsTAMU]),
                             RAA_pt_max = ptMaxBeautyTAMU if beautySpecie != 'BsBmix' else max([ptMaxBeautyTAMU, ptMaxBsTAMU])
            )
            hPtBeautyFONLLtimesTAMU.append(hPtFONLLtimesTAMU)

            hPtWeightsFONLL = hPtFONLL.Clone(histoName.replace("Pt", "PtWeights"))
            hPtWeightsFONLL.Divide(hPtFONLL, hPtGenB)
            hPtWeightsFONLL.Smooth(smooth)
            hPtWeightsFONLL.SetTitle("Beauty #it{p_{T}} weights FONLL;#it{p_{T}} (GeV/c);Weights")
            hPtWeightsBeautyFONLL.append(hPtWeightsFONLL)

            hPtWeightsFONLLtimesTAMU = hPtFONLLtimesTAMU.Clone(hPtFONLLtimesTAMU.GetName().replace("Pt", "PtWeights"))
            hPtWeightsFONLLtimesTAMU.Divide(hPtFONLLtimesTAMU, hPtGenB)
            hPtWeightsFONLLtimesTAMU.Smooth(smooth)
            hPtWeightsFONLLtimesTAMU.SetTitle("Beauty #it{p_{T}} weights FONLL #times TAMU;#it{p_{T}} (GeV/c);Weights")
            hPtWeightsBeautyFONLLtimesTAMU.append(hPtWeightsFONLLtimesTAMU)

    logger("B pT weights calculated", level='INFO')

    # save output
    #___________________________________________________________________________________________________________________________
    outputDir = f'{cfg.get('outdir', work_dir)}/ptweights/{charmSpecie}/{cent}/'
    os.makedirs(outputDir, exist_ok=True)
    outfile = TFile(f'{outputDir}/pTweight_{charmSpecie}_{cent}{suffix}.root', 'recreate')
    outfile.mkdir('Generated')
    outfile.cd('Generated')
    hPtGenPrompt.Write()
    if beautySpecie:
        hPtGenB.Write()
    outfile.mkdir('FONLL')
    outfile.mkdir('FONLL/charm')
    outfile.mkdir('FONLL/beauty')
    outfile.mkdir('FONLLtimesTAMU')
    outfile.mkdir('FONLLtimesTAMU/charm')
    outfile.mkdir('FONLLtimesTAMU/beauty')
    for iHisto, _ in enumerate(hPtCharmFONLL):
        outfile.cd('FONLL/charm')
        hPtCharmFONLL[iHisto].Write()
        hPtWeightsCharmFONLL[iHisto].Write()
        outfile.cd('FONLLtimesTAMU/charm')
        hPtCharmFONLLtimesTAMU[iHisto].Write()
        hPtWeightsCharmFONLLtimesTAMU[iHisto].Write()
        if beautySpecie:
            outfile.cd('FONLL/beauty')
            hPtBeautyFONLL[iHisto].Write()
            hPtWeightsBeautyFONLL[iHisto].Write()
            outfile.cd('FONLLtimesTAMU/beauty')
            hPtBeautyFONLLtimesTAMU[iHisto].Write()
            hPtWeightsBeautyFONLLtimesTAMU[iHisto].Write()

    outfile.cd()
    # pT shape D
    #___________________________________________________________________________________________________________________________
    canvPtshape = TCanvas('pTshape', 'pTshape', 2000, 900)
    canvPtshape.Divide(2, 1)
    ptD = [0, 36]
    canvPtshape.cd(1).DrawFrame(ptD[0], 0.0000001, ptD[1], 1,
                        f';#it{{p_{{T}}}} (GeV/c);Prompt {cfg["Dmeson"]}')
    canvPtshape.cd(1)
    canvPtshape.cd(1).SetLogy()

    leg = TLegend(0.5, 0.63, 0.7, 0.83)
    leg.SetFillStyle(0)
    leg.SetBorderSize(0)
    leg.SetTextSize(0.04)

    SetObjectStyle(hPtGenPrompt, color=kRed, markersize=0.5)
    leg.AddEntry(hPtGenPrompt, 'Gen Prompt', 'lp')
    SetObjectStyle(hPtCharmFONLLtimesTAMU[0], color=kBlack, markersize=0.5)
    leg.AddEntry(hPtCharmFONLLtimesTAMU[0], 'FONLL #times TAMU (R_{AA})', 'lp')
    SetObjectStyle(hPtCharmFONLL[0], color=kAzure, markersize=0.5)

    hPtCharmFONLLtimesTAMU[0].Draw('same')
    hPtGenPrompt.Draw('same')
    leg.Draw()

    canvPtshape.cd(2).DrawFrame(0, 0., ptD[1], 5, ';#it{p_{T}} (GeV/c);Ratio')
    canvPtshape.cd(2)

    legR = TLegend(0.5, 0.63, 0.7, 0.83)
    legR.SetFillStyle(0)
    legR.SetBorderSize(0)
    legR.SetTextSize(0.04)

    SetObjectStyle(hPtWeightsCharmFONLLtimesTAMU[0], color=kBlack, markersize=1)
    legR.AddEntry(hPtWeightsCharmFONLLtimesTAMU[0], 'FONLL #times TAMU (R_{AA})', 'lp')
    SetObjectStyle(hPtWeightsCharmFONLL[0], color=kAzure, markersize=1)

    hPtWeightsCharmFONLLtimesTAMU[0].Draw('same')
    legR.Draw()

    canvPtshape.Write()
    canvPtshape.SaveAs(f'{outputDir}/pTweight_{charmSpecie}_{cent}{suffix}.png')
    canvPtshape.SaveAs(f'{outputDir}/pTweight_{charmSpecie}_{cent}{suffix}.pdf')

    # pT shape B
    #___________________________________________________________________________________________________________________________
    canvPtshapeB = TCanvas('pTshapeB', 'pTshapeB', 2000, 900)
    canvPtshapeB.Divide(2, 1)
    ptB = [0, 72]
    canvPtshapeB.cd(1).DrawFrame(0, 0.0000001, ptB[1], 1, ';#it{p_{T}^{B}} (GeV/c); B hadron')
    canvPtshapeB.cd(1)
    canvPtshapeB.cd(1).SetLogy()

    legB = TLegend(0.5, 0.63, 0.7, 0.83)
    legB.SetFillStyle(0)
    legB.SetBorderSize(0)
    legB.SetTextSize(0.04)

    SetObjectStyle(hPtGenB, color=kRed, markersize=0.5)
    legB.AddEntry(hPtGenB, 'Gen B', 'lp')
    SetObjectStyle(hPtBeautyFONLLtimesTAMU[0], color=kBlack, markersize=0.5)
    legB.AddEntry(hPtBeautyFONLLtimesTAMU[0], 'FONLL #times TAMU (R_{AA})', 'lp')
    SetObjectStyle(hPtBeautyFONLL[0], color=kAzure, markersize=0.5)

    hPtBeautyFONLLtimesTAMU[0].Draw('same')
    hPtGenB.Draw('same')
    legB.Draw()

    canvPtshapeB.cd(2).DrawFrame(0, 0., ptB[1], 10, ';#it{p_{T}}^{B} (GeV/c);Ratio')
    canvPtshapeB.cd(2)

    legBR = TLegend(0.5, 0.63, 0.7, 0.83)
    legBR.SetFillStyle(0)
    legBR.SetBorderSize(0)
    legBR.SetTextSize(0.04)

    SetObjectStyle(hPtWeightsBeautyFONLLtimesTAMU[0], color=kBlack, markersize=1)
    legBR.AddEntry(hPtWeightsBeautyFONLLtimesTAMU[0], 'FONLL #times TAMU (R_{AA})', 'lp')
    SetObjectStyle(hPtWeightsBeautyFONLL[0], color=kAzure, markersize=1)

    hPtWeightsBeautyFONLLtimesTAMU[0].Draw('same')
    legBR.Draw()

    canvPtshapeB.Write()
    canvPtshapeB.SaveAs(f'{outputDir}/pTweightB_{charmSpecie}_{cent}{suffix}.png')
    canvPtshapeB.SaveAs(f'{outputDir}/pTweightB_{charmSpecie}_{cent}{suffix}.pdf')

    outfile.Close()
    print(f'pT weights saved in {outputDir}/pTweight_{charmSpecie}_{cent}{suffix}.root')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument("cfg", type=str, help="flow cfg file")
    args = parser.parse_args()

    compute_pt_weights(args.cfg)
