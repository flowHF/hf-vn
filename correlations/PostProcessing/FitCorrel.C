// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DhCorrelationExtraction.cxx
/// \brief class for D-h correlation extraction
/// \author Samuele Cattaruzzi <samuele.cattaruzzi@cern.ch>
/// \author Swapnesh Santosh Khade <swapnesh.santosh.khade@cern.ch>

#include "DhCorrelationFitter.h"
#include "Riostream.h"

#include <TF1.h>
#include <TH1D.h>
#include <TMath.h>
#include <TPaveText.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TFile.h>

#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <yaml-cpp/yaml.h>

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace rapidjson;

bool removeNSPeakLowPt = false;
bool doCorrelation = false;

template <typename ValueType>
void readArray(const Value& jsonArray, vector<ValueType>& output)
{
  for (auto it = jsonArray.Begin(); it != jsonArray.End(); it++) {
    auto value = it->template Get<ValueType>();
    output.emplace_back(value);
  }
}

void SetTH1HistoStyle(TH1D*& histo, TString hTitle, TString hXaxisTitle, TString hYaxisTitle,
                      Style_t markerStyle, Color_t markerColor, Double_t markerSize,
                      Color_t lineColor, Int_t lineWidth, Float_t hTitleXaxisOffset = 1.3, Float_t hTitleYaxisOffset = 1.3,
                      Float_t hTitleXaxisSize = 0.045, Float_t hTitleYaxisSize = 0.045, Float_t hLabelXaxisSize = 0.045, Float_t hLabelYaxisSize = 0.045,
                      Bool_t centerXaxisTitle = false, Bool_t centerYaxisTitle = false);
void SetTH1HistoStyle(TH1F*& histo, TString hTitle, TString hXaxisTitle, TString hYaxisTitle,
                      Style_t markerStyle, Color_t markerColor, Double_t markerSize,
                      Color_t lineColor, Int_t lineWidth, Float_t hTitleXaxisOffset = 1.3, Float_t hTitleYaxisOffset = 1.3,
                      Float_t hTitleXaxisSize = 0.045, Float_t hTitleYaxisSize = 0.045, Float_t hLabelXaxisSize = 0.045, Float_t hLabelYaxisSize = 0.045,
                      Bool_t centerXaxisTitle = false, Bool_t centerYaxisTitle = false);

void FitCorrel(const TString cfgFileName = "config_CorrAnalysis.json")
{
  gStyle->SetOptStat(0);
  gStyle->SetPadLeftMargin(0.2);
  gStyle->SetPadRightMargin(0.005);
  gStyle->SetPadBottomMargin(0.2);
  gStyle->SetFrameLineWidth(2);
  gStyle->SetLineWidth(2);
  gStyle->SetCanvasDefH(1126);
  gStyle->SetCanvasDefW(1840);

  // Load config
  FILE* configFile = fopen(cfgFileName.Data(), "r");
  Document config;
  char readBuffer[65536];
  FileReadStream is(configFile, readBuffer, sizeof(readBuffer));
  config.ParseStream(is);
  fclose(configFile);

  string CodeNameAnalysis = config["CodeName"].GetString();
  gSystem->Exec(Form("rm -rf Output_CorrelationFitting_%s_Root/ Output_CorrelationFitting_%s_png/", CodeNameAnalysis.data(), CodeNameAnalysis.data()));
  gSystem->Exec(Form("mkdir Output_CorrelationFitting_%s_Root/ Output_CorrelationFitting_%s_png/", CodeNameAnalysis.data(), CodeNameAnalysis.data()));

  string inputFileNameFit = config["InputFileNameFitCorr"].GetString();
  // const TString inFileName = Form("Output_CorrelationExtraction_%s_Root/%s", CodeNameAnalysis.data(), inputFileNameFit.data());
  const TString inFileName = TString(inputFileNameFit);
  const int DmesonSpecies = config["DmesonSpecies"].GetInt();
  const TString DmesonName = (DmesonSpecies == 0) ? "D0" : (DmesonSpecies == 1) ? "Dplus" : (DmesonSpecies == 2) ? "Ds" : "DStar";
  const TString fDmesonLabel = (DmesonSpecies == 0) ? "D^{0}" : (DmesonSpecies == 1) ? "D^{+}" : (DmesonSpecies == 2) ? "D_{s}^{+}" : "D^{*+}";

  bool isReflected = config["IsRiflected"].GetBool();
  bool drawSystematicErrors = config["DrawSystematics"].GetBool();
  bool sameSystematics = config["SameSystematics"].GetBool();
  bool shiftBaseUp = config["ShiftBaseUp"].GetBool();
  bool shiftBaseDown = config["ShiftBaseDown"].GetBool();

  std::vector<double> binsInvMassIntervals;
  std::vector<double> binsPtCandIntervalsVec;
  std::vector<double> binsPtHadIntervals;
  std::vector<int> fitFunc;

  const Value& InvMassValue = config["binsInvMassIntervals"];
  readArray(InvMassValue, binsInvMassIntervals);
  binsInvMassIntervals.clear();
  binsInvMassIntervals.push_back(1.72);
  binsInvMassIntervals.push_back(2.02);

  const Value& PtCandValue = config["binsPtCandIntervals"];
  readArray(PtCandValue, binsPtCandIntervalsVec);

  const Value& PtHadValue = config["binsPtHadIntervals"];
  readArray(PtHadValue, binsPtHadIntervals);

  std::vector<double> parVals;
  std::vector<double> parLowBounds;
  std::vector<double> parUpperBounds;

  const Value& ParVals = config["parVals"];
  readArray(ParVals, parVals);

  const Value& ParLowBounds = config["parLowBounds"];
  readArray(ParLowBounds, parLowBounds);

  const Value& ParUpperBounds = config["parUpperBounds"];
  readArray(ParUpperBounds, parUpperBounds);

  // const int nBinsInvMass = binsInvMassIntervals.size() - 1;
  const int nBinsInvMass = 1;
  const int nBinsPtCand = binsPtCandIntervalsVec.size() - 1;
  const int nBinsPtHad = binsPtHadIntervals.size() - 1;
  const int npars = parVals.size();

  double binsPtCandIntervals[nBinsPtCand + 1];
  for (int i = 0; i < nBinsPtCand + 1; i++) {
    binsPtCandIntervals[i] = binsPtCandIntervalsVec[i];
  }

  const Value& FitFuncValue = config["FitFunction"];
  readArray(FitFuncValue, fitFunc);

  int fixBase = config["FixBaseline"].GetInt();
  int fixMean = config["FixMean"].GetInt();

  int nBaselinePoints = config["nBaselinePoints"].GetInt();
  vector<int> pointsForBaselineVec;
  const Value& pointsForBaselineValue = config["binsForBaseline"];
  readArray(pointsForBaselineValue, pointsForBaselineVec);
  if (pointsForBaselineVec.size() != nBaselinePoints) {
    cout << "ERROR: size of the vector pointsForBaseline is different from the number of nBaselinePoints" << endl;
    return;
  }
  int pointsForBaseline[nBaselinePoints];
  for (int i = 0; i < nBaselinePoints; i++) {
    pointsForBaseline[i] = pointsForBaselineVec[i];
  }

  // TODO: reflections
  bool refl = false;
  bool lmTemplate = true;

  // YAML config
  string yamlConfigFile = "/home/wuct/MetaData/DATA/OO/apass2/corr/results/fthlook_Biao/large/k020/CorrelExtract_0d8_1d3_1limits/config_CorrAnalysis_v2_010_negDeta_0d8_1d3_1limits.yaml";
  YAML::Node yamlConfig = YAML::LoadFile(yamlConfigFile);
  string outdir = yamlConfig["outdir"].as<std::string>();
  string suffix = yamlConfig["suffix"].as<std::string>();
  string outdirLM;
  string inputLMFile;
  if (yamlConfig["task_LM"] && yamlConfig["task_LM"]["do"].as<bool>()) {
    outdirLM = yamlConfig["task_LM"]["outdir"].as<std::string>();
    inputLMFile = outdirLM + "/CorrelExtract_" + suffix + "/AssociatedPairsYields/PairYieldsVsPhi.root";

  }
  string inputFile = outdir + "/CorrelExtract_" + suffix + "/AssociatedPairsYields/PairYieldsVsPhi.root";
  string outputPath = outdir + "/CorrelExtract_" + suffix + "/CorrelationFitResults/";

  if (inputLMFile.empty()) {
    cout << "No input LM template file provided, will not perform template fit" << endl;
    for (int iFunc=0; iFunc < nBinsPtCand; iFunc++) {
      fitFunc[iFunc] = 8;
    }  
  } else {
    cout << "Input LM template file: " << inputLMFile << endl;
  }

  std::cout << "=========================== " << std::endl;
  std::cout << "Input variables from config" << std::endl;
  for (int iBinPtCand = 0; iBinPtCand < nBinsPtCand; iBinPtCand++) {
    std::cout << "iPt = " << iBinPtCand + 1 << "  FitFunction    = " << fitFunc[iBinPtCand] << std::endl;
  }
  std::cout << "FixBaseline    = " << fixBase << std::endl;
  std::cout << "FixMean    = " << fixMean << std::endl;
  std::cout << "=========================== " << std::endl;
  std::cout << " " << std::endl;

  // create output directory if it doesn't exist
  gSystem->Exec(Form("mkdir -p %s/%s", outputPath.c_str(), "outputPathOutput_CorrelationFitting_png"));
  gSystem->Exec(Form("mkdir -p %s/%s", outputPath.c_str(), "Output_CorrelationFitting_Root"));

  // Input file
  // TFile* inFile = new TFile(inFileName.Data());
  TFile* inFile = new TFile(inputFile.c_str());
  // TFile* inFileSystematicErrors = new TFile("OutputSystematicUncertainties/SystematicUncertaintesAngCorrMerged.root");
  // TFile* inFileFitSystematicErrors = new TFile("OutputSystematicUncertainties/SystematicUncertaintesFitPhysObsMerged.root");
  TFile* inFileLMTemplate = nullptr;
  if (!inputLMFile.empty()) {
    inFileLMTemplate = new TFile(inputLMFile.c_str());
  }

  // Canvas
  TCanvas* CanvasCorrPhi[nBinsPtCand][nBinsPtHad][nBinsInvMass];

  // Histograms
  TH1D* hCorrPhi[nBinsPtCand][nBinsPtHad][nBinsInvMass];
  TH1D* hCorrPhiLMTemplate[nBinsPtCand][nBinsPtHad][nBinsInvMass];
  // TH1F* hSystematicErrors[nBinsPtCand][nBinsPtHad][nBinsInvMass];
  // TH1D* hSystematicErrorsPlot[nBinsPtCand][nBinsPtHad][nBinsInvMass];

  // DhCorrelationFitter
  const double fMin{-0.5 * TMath::Pi()}, fMax{1.5 * TMath::Pi()}; // limits for the fitting function
  DhCorrelationFitter* corrFitter[nBinsPtHad][nBinsPtCand][nBinsInvMass];

  // Output histograms
  TH1D* hBaselin[nBinsPtHad][nBinsInvMass];
  TH1D* hNSYield[nBinsPtHad][nBinsInvMass];
  TH1D* hNSSigma[nBinsPtHad][nBinsInvMass];
  TH1D* hASYield[nBinsPtHad][nBinsInvMass];
  TH1D* hASSigma[nBinsPtHad][nBinsInvMass];
  TH1D* hBeta[nBinsPtHad][nBinsInvMass];
  TH1D* hNSYieldBinCount[nBinsPtHad][nBinsInvMass];
  TH1D* hASYieldBinCount[nBinsPtHad][nBinsInvMass];
  TH1D* hv2Delta[nBinsPtHad][nBinsInvMass];

  // extract TH1D and prepare fit
  for (int iBinInvMass = 0; iBinInvMass < nBinsInvMass; iBinInvMass++) {
    for (int iBinPtCand = 0; iBinPtCand < nBinsPtCand; iBinPtCand++) {
      for (int iBinPtHad = 0; iBinPtHad < nBinsPtHad; iBinPtHad++) {
        if (isReflected) {
          hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass] = reinterpret_cast<TH1D*>(inFile->Get(Form("PtCandBin_%0.f_%0.f/PtHadBin_4_50/InvMassBin_%0.f_%0.f/hCorrectedCorrel_Reflected_%d", binsPtCandIntervals[iBinPtCand], binsPtCandIntervals[iBinPtCand + 1], binsInvMassIntervals[iBinInvMass]*1000, binsInvMassIntervals[iBinInvMass + 1]*1000, iBinInvMass+1)));
        } else {
          hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass] = reinterpret_cast<TH1D*>(inFile->Get(Form("PtCandBin_%0.f_%0.f/PtHadBin_4_30/hPairsYields_vs_DeltaPhi", binsPtCandIntervals[iBinPtCand]*10, binsPtCandIntervals[iBinPtCand + 1]*10)));
          if (!inputLMFile.empty()) {
            hCorrPhiLMTemplate[iBinPtCand][iBinPtHad][iBinInvMass] = reinterpret_cast<TH1D*>(inFileLMTemplate->Get(Form("PtCandBin_%0.f_%0.f/PtHadBin_4_30/hPairsYields_vs_DeltaPhi", binsPtCandIntervals[iBinPtCand]*10, binsPtCandIntervals[iBinPtCand + 1]*10)));
          }
          if (!hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass]) {
            printf("Not found Fitting histogram: PtCandBin_%0.f_%0.f/PtHadBin_4_50/InvMassBin_%0.f_%0.f/hCorrectedCorr\n", binsPtCandIntervals[iBinPtCand], binsPtCandIntervals[iBinPtCand + 1], binsInvMassIntervals[iBinInvMass]*1000, binsInvMassIntervals[iBinInvMass + 1]*1000);
            continue;
          }
        }
        std::string histoName = hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass]->GetName();
        printf("Fitting histogram: %s\n", histoName.c_str());
        double minHisto = hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass]->GetXaxis()->GetXmin();
        double maxHisto = hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass]->GetXaxis()->GetXmax();
        printf("Histo range: [%.2f, %.2f]\n", minHisto, maxHisto);
        double integralHisto = hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass]->Integral();
        printf("Histo integral: %.5f\n", integralHisto);
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass] = new DhCorrelationFitter(reinterpret_cast<TH1F*>(hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass]), fMin, fMax);
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetHistoIsReflected(refl);
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetFixBaseline(fixBase);
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetBaselineUpOrDown(shiftBaseUp, shiftBaseDown);
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetPointsForBaseline(nBaselinePoints, pointsForBaseline);
        //corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->Setv2(v2AssocPart[iBinPtCand], v2Dmeson[iBinPtCand]);
        if (!inputLMFile.empty()) {
          corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetLMTemplate(hCorrPhiLMTemplate[iBinPtCand][iBinPtHad][iBinInvMass]);
        }
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetReflectedCorrHisto(isReflected);

        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetFixMean(fixMean);
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetPtRanges(binsPtCandIntervals[iBinPtCand], binsPtCandIntervals[iBinPtCand + 1], binsPtHadIntervals[iBinPtHad], binsPtHadIntervals[iBinPtHad + 1]);
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetExternalValsAndBounds(npars, parVals.data(), parLowBounds.data(), parUpperBounds.data()); // these are starting points and limits...
      }
    }
  }

  // Plots and fit
  for (int iBinInvMass = 0; iBinInvMass < nBinsInvMass; iBinInvMass++) {
    std::cout << "[INFO] InvMass: " << binsInvMassIntervals[iBinInvMass] << " - "<< binsInvMassIntervals[iBinInvMass+1] << std::endl;
    for (int iBinPtHad = 0; iBinPtHad < nBinsPtHad; iBinPtHad++) {
      std::cout << "[INFO] PtHad: " << binsPtHadIntervals[iBinPtHad] << " - "<< binsPtHadIntervals[iBinPtHad+1] << std::endl;

      // histograms with fir parameters
      hBaselin[iBinPtHad][iBinInvMass] = new TH1D(Form("hBaselin_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), "", nBinsPtCand, binsPtCandIntervals);
      hNSYield[iBinPtHad][iBinInvMass] = new TH1D(Form("hNSYield_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), "", nBinsPtCand, binsPtCandIntervals);
      hNSSigma[iBinPtHad][iBinInvMass] = new TH1D(Form("hNSSigma_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), "", nBinsPtCand, binsPtCandIntervals);
      hASYield[iBinPtHad][iBinInvMass] = new TH1D(Form("hASYield_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), "", nBinsPtCand, binsPtCandIntervals);
      hASSigma[iBinPtHad][iBinInvMass] = new TH1D(Form("hASSigma_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), "", nBinsPtCand, binsPtCandIntervals);
      hBeta[iBinPtHad][iBinInvMass] = new TH1D(Form("hBeta_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), "", nBinsPtCand, binsPtCandIntervals);
      hNSYieldBinCount[iBinPtHad][iBinInvMass] = new TH1D(Form("hNSYieldBinCount_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), "", nBinsPtCand, binsPtCandIntervals);
      hASYieldBinCount[iBinPtHad][iBinInvMass] = new TH1D(Form("hASYieldBinCount_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), "", nBinsPtCand, binsPtCandIntervals);
      hv2Delta[iBinPtHad][iBinInvMass] = new TH1D(Form("hv2Delta_PtBinAssoc%d_InvMassBin%d", iBinPtHad + 1, iBinInvMass+1), "", nBinsPtCand, binsPtCandIntervals);

      for (int iBinPtCand = 0; iBinPtCand < nBinsPtCand; iBinPtCand++) {
        std::cout << "[INFO] PtCand: " << binsPtCandIntervals[iBinPtCand] << " - "<< binsPtCandIntervals[iBinPtCand+1] << std::endl;
        CanvasCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass] = new TCanvas(Form("CanvasCorrPhi_PtBinCand%d_PtBinAssoc%d_InvMassBin%d", iBinPtCand + 1, iBinPtHad + 1, iBinInvMass+1), Form("CorrPhi%s_PtBinCand%d_PtBinAssoc%d_InvMassBin%d", DmesonName.Data(), iBinPtCand + 1, iBinPtHad + 1, iBinInvMass+1));

        SetTH1HistoStyle(hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass], "", "#Delta#phi [rad]", "#frac{dN^{assoc}}{d#Delta#phi} [rad^{-1}]", kFullCircle, kRed + 1, 1.4, kRed + 1, 3);

        CanvasCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass]->cd(iBinPtCand + 1);
        CanvasCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass]->SetTickx();
        CanvasCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass]->SetTicky();
        hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass]->SetStats(0);
        hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass]->SetMinimum(0);

        // Fit
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->SetFuncType(static_cast<DhCorrelationFitter::FunctionType>(fitFunc[iBinPtCand]));
        corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->Fitting(kTRUE, kTRUE); // the first term is for drawing the fit functions, the second argument is useExternalParams

        TF1* fFit = corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetFitFunction();

        // Title of the histogram
        TPaveText* pttext = new TPaveText(0.15, 0.9, 0.85, 0.95, "NDC");
        pttext->SetFillStyle(0);
        pttext->SetBorderSize(0);

        TText* tpT = pttext->AddText(0., 0.8, Form("%.0f < p_{T}^{%s} < %.0f GeV/c, p_{T}^{assoc} > %.1f GeV/c", binsPtCandIntervals[iBinPtCand], fDmesonLabel.Data(), binsPtCandIntervals[iBinPtCand + 1], binsPtHadIntervals[iBinPtHad]));

        // Fill the histograms with the fit parameters
        // if (doCorrelation) {
        //   hBaselin[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetPedestal());
        //   hBaselin[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetPedestalError());
        //   // if (iBinPtCand == 0 && removeNSPeakLowPt) {
        //   //   hNSYield[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, -1);
        //   //   hNSYield[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, 0);

        //   //   hNSSigma[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, -1);
        //   //   hNSSigma[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, 0);

        //   //   hBeta[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, -1);
        //   //   hBeta[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, 0);
        //   // } else {
        //   //   hNSYield[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetNSYield());
        //   //   hNSYield[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetNSYieldError());

        //   //   if (fitFunc[iBinPtCand] != 5 && fitFunc[iBinPtCand] != 6) {
        //   //     hNSSigma[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetNSSigma());
        //   //     hNSSigma[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetNSSigmaError());
        //   //   } else {
        //   //     hNSSigma[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, TMath::Sqrt(1. / corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetNSSigma()));
        //   //     Double_t errrel = corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetNSSigmaError() / corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetNSSigma() / 2.;
        //   //     hNSSigma[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, errrel * TMath::Sqrt(1. / corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetNSSigma()));
        //   //   }
        //   // }
        //   // hNSYieldBinCount[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetBinCountingNSYield());
        //   // hNSYieldBinCount[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetBinCountingNSYieldErr());

        //   // hASYield[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetASYield());
        //   // hASYield[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetASYieldError());

        //   // hASYieldBinCount[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetBinCountingASYield());
        //   // hASYieldBinCount[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetBinCountingASYieldErr());
        //   if (fitFunc[iBinPtCand] != 5 && fitFunc[iBinPtCand] != 6) {
        //     hASSigma[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetASSigma());
        //     hASSigma[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetASSigmaError());
        //   } else {
        //     hASSigma[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, TMath::Sqrt(1. / corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetASSigma()));
        //     Double_t errrel = corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetASSigmaError() / corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetASSigma() / 2.;
        //     hASSigma[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, errrel * TMath::Sqrt(1. / corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetASSigma()));
        //   }
        //   if (fitFunc[iBinPtCand] == 4) { // param beta for gen. gauss
        //     hBeta[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetBeta());
        //     hBeta[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->GetBetaError());
        //   }
        // } else {
        //   hv2Delta[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->Getv2Delta());
        //   hv2Delta[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->Getv2DeltaError()); 
        // }
        hv2Delta[iBinPtHad][iBinInvMass]->SetBinContent(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->Getv2Delta());
        hv2Delta[iBinPtHad][iBinInvMass]->SetBinError(iBinPtCand + 1, corrFitter[iBinPtHad][iBinPtCand][iBinInvMass]->Getv2DeltaError());

        // Draw
        hCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass]->Draw("same");
        pttext->Draw("same");
        CanvasCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass]->SaveAs(Form("%soutputPathOutput_CorrelationFitting_png/CorrPhi%s_PtBinCand%d_PtBinAssoc%d_InvMassBin%d.png", outputPath.c_str(), DmesonName.Data(), iBinPtCand + 1, iBinPtHad + 1, iBinInvMass+1));
        CanvasCorrPhi[iBinPtCand][iBinPtHad][iBinInvMass]->SaveAs(Form("%sOutput_CorrelationFitting_Root/CorrPhi%s_PtBinCand%d_PtBinAssoc%d_InvMassBin%d.root", outputPath.c_str(), DmesonName.Data(), iBinPtCand + 1, iBinPtHad + 1, iBinInvMass+1));
      }
    }
  }

  // histogram with fit parameter and errors
  TFile* outFile = new TFile(Form("%sOutput_CorrelationFitting_Root/CorrPhi%s_FinalPlots.root", outputPath.c_str(), DmesonName.Data()), "RECREATE");
  outFile->cd();
  for (int iBinInvMass = 0; iBinInvMass < nBinsInvMass; iBinInvMass++) {
    for (int iBinPtHad = 0; iBinPtHad < nBinsPtHad; iBinPtHad++) {
      if (doCorrelation) {
        hBaselin[iBinPtHad][iBinInvMass]->Write();
        hNSYield[iBinPtHad][iBinInvMass]->Write();
        hNSSigma[iBinPtHad][iBinInvMass]->Write();
        hASYield[iBinPtHad][iBinInvMass]->Write();
        hASSigma[iBinPtHad][iBinInvMass]->Write();
        hBeta[iBinPtHad][iBinInvMass]->Write();
        hNSYieldBinCount[iBinPtHad][iBinInvMass]->Write();
        hASYieldBinCount[iBinPtHad][iBinInvMass]->Write();
      } else {
        hv2Delta[iBinPtHad][iBinInvMass]->Write();
      }
    }
  }
  outFile->Close();

  return;
}

void SetTH1HistoStyle(TH1D*& histo, TString hTitle, TString hXaxisTitle, TString hYaxisTitle,
                      Style_t markerStyle, Color_t markerColor, Double_t markerSize,
                      Color_t lineColor, Int_t lineWidth, Float_t hTitleXaxisOffset = 1.3, Float_t hTitleYaxisOffset = 1.3,
                      Float_t hTitleXaxisSize = 0.045, Float_t hTitleYaxisSize = 0.045, Float_t hLabelXaxisSize = 0.045, Float_t hLabelYaxisSize = 0.045,
                      Bool_t centerXaxisTitle = false, Bool_t centerYaxisTitle = false)
{

  histo->SetTitle(hTitle.Data());
  histo->GetXaxis()->SetTitle(hXaxisTitle.Data());
  histo->GetYaxis()->SetTitle(hYaxisTitle.Data());
  histo->SetMarkerStyle(markerStyle);
  histo->SetMarkerColor(markerColor);
  histo->SetMarkerSize(markerSize);
  histo->SetLineColor(lineColor);
  histo->SetLineWidth(lineWidth);
  histo->GetXaxis()->SetTitleOffset(hTitleXaxisOffset);
  histo->GetYaxis()->SetTitleOffset(hTitleYaxisOffset);
  histo->GetXaxis()->SetTitleSize(hTitleXaxisSize);
  histo->GetYaxis()->SetTitleSize(hTitleYaxisSize);
  histo->GetXaxis()->SetLabelSize(hLabelXaxisSize);
  histo->GetYaxis()->SetLabelSize(hLabelYaxisSize);
  histo->GetXaxis()->CenterTitle(centerXaxisTitle);
  histo->GetYaxis()->CenterTitle(centerYaxisTitle);

  return;
}

void SetTH1HistoStyle(TH1F*& histo, TString hTitle, TString hXaxisTitle, TString hYaxisTitle,
                      Style_t markerStyle, Color_t markerColor, Double_t markerSize,
                      Color_t lineColor, Int_t lineWidth, Float_t hTitleXaxisOffset = 1.3, Float_t hTitleYaxisOffset = 1.3,
                      Float_t hTitleXaxisSize = 0.045, Float_t hTitleYaxisSize = 0.045, Float_t hLabelXaxisSize = 0.045, Float_t hLabelYaxisSize = 0.045,
                      Bool_t centerXaxisTitle = false, Bool_t centerYaxisTitle = false)
{

  histo->SetTitle(hTitle.Data());
  histo->GetXaxis()->SetTitle(hXaxisTitle.Data());
  histo->GetYaxis()->SetTitle(hYaxisTitle.Data());
  histo->SetMarkerStyle(markerStyle);
  histo->SetMarkerColor(markerColor);
  histo->SetMarkerSize(markerSize);
  histo->SetLineColor(lineColor);
  histo->SetLineWidth(lineWidth);
  histo->GetXaxis()->SetTitleOffset(hTitleXaxisOffset);
  histo->GetYaxis()->SetTitleOffset(hTitleYaxisOffset);
  histo->GetXaxis()->SetTitleSize(hTitleXaxisSize);
  histo->GetYaxis()->SetTitleSize(hTitleYaxisSize);
  histo->GetXaxis()->SetLabelSize(hLabelXaxisSize);
  histo->GetYaxis()->SetLabelSize(hLabelYaxisSize);
  histo->GetXaxis()->CenterTitle(centerXaxisTitle);
  histo->GetYaxis()->CenterTitle(centerYaxisTitle);

  return;
}
