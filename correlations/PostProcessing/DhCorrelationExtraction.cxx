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

#include "DhCorrelationExtraction.h"
ClassImp(DhCorrelationExtraction);

#include <TCanvas.h>
#include <TDirectoryFile.h>
#include <TF1.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <THnSparse.h>
#include <TString.h>

#include <RtypesCore.h>

#include <cstdio>
#include <iostream>

DhCorrelationExtraction::DhCorrelationExtraction() : // default constructor
  // Copyable properties-----------------------------------------------------------------------------------------------------//
  fDmesonSpecies(kDsToKKPi),
  fDmesonLabel("Ds"),
  fFileNameSE(""),
  fFileNameME(""),
  fDirNameSE(""),
  fDirNameME(""),
  fMassSparseName(""),
  fCorrelSparseNameSE(""),
  fCorrelSparseNameME(""),
  fDeltaEtaGap(""),
  fNpools(10),
  fDebug(0),
  fDoPoolByPool(kTRUE),
  fDeltaEtaIntegrated(kTRUE),
  fDeltaEtaLeftMin(-1.),
  fDeltaEtaLeftMax(1.),
  fDeltaEtaRightMin(-1.),
  fDeltaEtaRightMax(1.),

  // Non-copyable properties-----------------------------------------------------------------------------------//
  fFileSE(0x0),
  fFileME(0x0),
  fDirSE(0x0),
  fDirME(0x0),
  fPtCandBins(),
  fPtHadBins(),
  fInvMassBins(),
  fDeltaPhiBins(),
  fFactorsNormME(),
  fMethod(kMassBinning),
  fRebinAxisDeltaEta(1),
  fRebinAxisDeltaPhi(1),

  // Results---------------------------------------------------------------------------------------------------------------//
  fMassVsPt_2D(0x0), // copyable properties, for trigger normalization
  // --- final results, debug level 0 ---
  fCorrectedCorrel(0x0),
  fNormalizedCorrectedCorrel(0x0),
  fCorrectedPairsMass(0x0),
  fCorrectionRatio(0x0),
  // --- intermediate results, debug level 1 ---
  fCorrel_SE_2D(0x0),
  fCorrel_ME_2D(0x0),
  fCorrectedCorrel_2D(0x0),
  fNormalizedCorrel_ME_2D(0x0),
  fOriginalCorrel_SE_2D(0x0),
  fOriginalCorrel_ME_2D(0x0),
  fOriginalMassVsDeltaEta_2D(0x0),

  // --- original data histograms, debug level 2 ---
  fPoolVec_OriginalCorrel_SE_2D(),
  fPoolVec_OriginalCorrel_ME_2D(),
  fPoolVec_RawCorrel_SE_2D(),
  fPoolVec_RawCorrel_ME_2D(),
  fPoolVec_NormalizedCorrel_ME_2D(),
  fPoolVec_CorrectedCorrel_2D(),
  fPoolVec_RawMassVsDeltaEta_2D(),
  fPoolVec_CorrectedMass(),
  fPoolVec_CorrectionRatio()
{
}

DhCorrelationExtraction::DhCorrelationExtraction(const DhCorrelationExtraction& source) : // copy constructor
  TObject(source),
  // Copyable properties-----------------------------------------------------------------------------------------------------//
  fDmesonSpecies(source.fDmesonSpecies),
  fDmesonLabel(source.fDmesonLabel),
  fFileNameSE(source.fFileNameSE),
  fFileNameME(source.fFileNameME),
  fDirNameSE(source.fDirNameSE),
  fDirNameME(source.fDirNameME),
  fMassSparseName(source.fMassSparseName),
  fCorrelSparseNameSE(source.fCorrelSparseNameSE),
  fCorrelSparseNameME(source.fCorrelSparseNameME),
  fDeltaEtaGap(source.fDeltaEtaGap),
  fNpools(source.fNpools),
  fDebug(source.fDebug),
  fDoPoolByPool(source.fDoPoolByPool),
  fDeltaEtaIntegrated(source.fDeltaEtaIntegrated),
  fDeltaEtaLeftMin(source.fDeltaEtaLeftMin),
  fDeltaEtaLeftMax(source.fDeltaEtaLeftMax),
  fDeltaEtaRightMin(source.fDeltaEtaRightMin),
  fDeltaEtaRightMax(source.fDeltaEtaRightMax),

  // Non-copyable properties-----------------------------------------------------------------------------------------------------//
  fFileSE(0x0),
  fFileME(0x0),
  fDirSE(0x0),
  fDirME(0x0),
  fPtCandBins(),
  fPtHadBins(),
  fInvMassBins(),
  fDeltaPhiBins(),
  fFactorsNormME(),
  fMethod(kMassBinning),
  fRebinAxisDeltaEta(1),
  fRebinAxisDeltaPhi(1),

  // Results---------------------------------------------------------------------------------------------------------------//
  fMassVsPt_2D(source.fMassVsPt_2D), // copyable properties, for trigger normalization
  // --- final results, debug level 0 ---
  fCorrectedCorrel(0x0),
  fNormalizedCorrectedCorrel(0x0),
  fCorrectedPairsMass(0x0),
  fCorrectionRatio(0x0),
  // --- intermediate results, debug level 1 ---
  fCorrel_SE_2D(0x0),
  fCorrel_ME_2D(0x0),
  fCorrectedCorrel_2D(0x0),
  fNormalizedCorrel_ME_2D(0x0),
  fOriginalCorrel_SE_2D(0x0),
  fOriginalCorrel_ME_2D(0x0),
  fOriginalMassVsDeltaEta_2D(0x0),
  // --- original data histograms, debug level 2 ---
  fPoolVec_OriginalCorrel_SE_2D(),
  fPoolVec_OriginalCorrel_ME_2D(),
  fPoolVec_RawCorrel_SE_2D(),
  fPoolVec_RawCorrel_ME_2D(),
  fPoolVec_NormalizedCorrel_ME_2D(),
  fPoolVec_CorrectedCorrel_2D(),
  fPoolVec_RawMassVsDeltaEta_2D(),
  fPoolVec_CorrectedMass(),
  fPoolVec_CorrectionRatio()
{
}

DhCorrelationExtraction::~DhCorrelationExtraction()
// destructor
{
  // if (fDirSE) {
  //   fDirSE->Close();
  //   delete fDirSE;
  // }
  // if (fDirME) {
  //   fDirME->Close();
  //   delete fDirME;
  // }
  // if (fFileSE) {
  //   fFileSE->Close();
  //   delete fFileSE;
  // }
  // if (fFileME) {
  //   fFileME->Close();
  //   delete fFileME;
  // }
}

DhCorrelationExtraction DhCorrelationExtraction::CreateDefault() {
    static DhCorrelationExtraction instance;
    instance.SetDirNameSE("hf-correlator-flow-charm-hadrons-reduced");
    instance.SetDirNameME("hf-correlator-flow-charm-hadrons-reduced");
    instance.SetCorrelSparseNameSE("hSparseCorrelationsSECharmHad");
    instance.SetCorrelSparseNameME("hSparseCorrelationsMECharmHad");
    instance.SetMassSparseName("hf-correlator-flow-charm-hadrons-reduced/hSparseTrigCandsCharm");
    return instance;
}

DhCorrelationExtraction DhCorrelationExtraction::CreateCustom(const TString& dirNameSE, const TString& dirNameME, const TString& massSparseName) {
    DhCorrelationExtraction instance;
    instance.SetDirNameSE(dirNameSE);
    instance.SetDirNameME(dirNameME);
    instance.SetCorrelSparseNameSE("hSparseCorrelationsSECharmHad");
    instance.SetCorrelSparseNameME("hSparseCorrelationsMECharmHad");
    instance.SetMassSparseName(massSparseName);
    return instance;
}

DhCorrelationExtraction* DhCorrelationExtraction::CreateCopy(const DhCorrelationExtraction& source) {
    return new DhCorrelationExtraction(source);
}

Bool_t DhCorrelationExtraction::SetDmesonSpecie(DmesonSpecie k)
{

  if (k < 0 || k > 3) {
    printf("[ERROR] D meson specie not correctly set!\n");
    return kFALSE;
  } else if (k == 0) {
    fDmesonLabel = "D^{0}";
  } else if (k == 1) {
    fDmesonLabel = "D^{+}";
  } else if (k == 2) {
    fDmesonLabel = "D^{s}";
  } else {
    fDmesonLabel = "D^{*}";
  }
  fDmesonSpecies = k;
  return kTRUE;
}

Bool_t DhCorrelationExtraction::Init()
{
  // todo: secondary particle contamination
  if (!ReadInputSEandME()) {
    return kFALSE;
  }
  // fTitleCorrel = Form("%s-h correlation", fDmesonLabel.Data());
  fDeltaEtaGap = Form("|#Delta#eta| > %.1f", fDeltaEtaRightMin);
  return kTRUE;
}

Bool_t DhCorrelationExtraction::ExtractCorrelations()
{
  TH1::AddDirectory(kFALSE);
  if (Init() == kFALSE) { // todo: reload file every time
    return kFALSE;
  }

  if (!fDoPoolByPool)
    fNpools = 1; // single histogram with integrated pools

  fFactorsNormME.resize(fNpools, 1.0); // initialize normalization factors for ME histograms

  // Histograms definition
  std::vector<TH2D*> hSE_2D_Raw(fNpools, nullptr);
  std::vector<TH2D*> hME_2D_Raw(fNpools, nullptr);
  std::vector<TH2D*> hME_2D_Normalized(fNpools, nullptr);
  std::vector<TH2D*> hME_Sign_SoftPi(fNpools, nullptr); // TO BE DONE

  std::vector<TH2D*> hCorrectedCorrel_2D(fNpools, nullptr);
  std::vector<TH1D*> hCorrectedPairsMass(fNpools, nullptr);

  TH2D* h2D_SE = nullptr;
  TH2D* h2D_ME = nullptr;
  TH2D* h2D_ME_norm = nullptr;
  TH2D* h2D_CorrectedCorrel = nullptr;
  TH1D* h1D_CorrectedPairsMass = nullptr;
  TH1D* h1D_correctedRatioVsDeltaEta = nullptr;

  TH1D* h1D_CorrectedCorrel = nullptr;
  TH1D* h1D_NormalizedCorrectedCorrel = nullptr;

  for (int iPool = 0; iPool < fNpools; iPool++) {
    // Retrieve 2D plots for SE and ME, signal and bkg regions, for each pTbin and pool
    hSE_2D_Raw[iPool] = ProjCorrelHisto(kSE, iPool);
    hME_2D_Raw[iPool] = ProjCorrelHisto(kME, iPool);

    hME_2D_Normalized[iPool] = reinterpret_cast<TH2D*>(hME_2D_Raw[iPool]->Clone(Form("hNormalizedCorrel_ME_2D_Pool%d", iPool)));
    // Normalize ME plots for the entries in (deltaEta, deltaPhi) = (0, 0)
    NormalizeMEplot(hME_2D_Normalized[iPool], hME_Sign_SoftPi[iPool], iPool);

    // Apply Event Mixing Correction
    hCorrectedCorrel_2D[iPool] = reinterpret_cast<TH2D*>(hSE_2D_Raw[iPool]->Clone(Form("hCorrectedCorrel_2D_Pool%d", iPool)));
    // hCorrectedCorrel_2D[iPool]->Sumw2();
    hCorrectedCorrel_2D[iPool]->Divide(hME_2D_Normalized[iPool]);

    // Apply the ME correction on the Mass by the ratio of SE/ME integrated over deltaPhi bins for each deltaEta bin
    if (fMethod == kDeltaPhiBinning) {
      hCorrectedPairsMass[iPool] = CorrectedPairsMassDistr(hSE_2D_Raw[iPool], hCorrectedCorrel_2D[iPool], iPool);
    }

    // Set proper number of entries after ME correction
    hSE_2D_Raw[iPool]->SetEntries(hSE_2D_Raw[iPool]->Integral());
    hCorrectedCorrel_2D[iPool]->SetEntries(hCorrectedCorrel_2D[iPool]->Integral());

    // debug: normalized ME and corrected correlation histos pool by pool
    if (fDebug > 0 && fDoPoolByPool) {
      fPoolVec_CorrectedCorrel_2D.push_back(SetTH2HistoStyle(
        reinterpret_cast<TH2D*>(hCorrectedCorrel_2D[iPool]->Clone(Form("hCorrected_Correl_2D_Pool%s", fDoPoolByPool ? Form("%d", iPool) : "All"))),
        Form("Corrected %s-h correlation with %s", fDmesonLabel.Data(), fDeltaEtaGap.Data()), "#Delta#eta", "#Delta#phi (rad)", AxisLabels::kRawYieldRad
      ));
    }

    // Pools integration
    if (iPool == 0) {
      h2D_SE = reinterpret_cast<TH2D*>(hSE_2D_Raw[0]->Clone("h2D_SE"));
      h2D_ME = reinterpret_cast<TH2D*>(hME_2D_Raw[0]->Clone("h2D_ME"));
      h2D_ME_norm = reinterpret_cast<TH2D*>(hME_2D_Normalized[0]->Clone("h2D_ME_norm"));
      h2D_CorrectedCorrel = reinterpret_cast<TH2D*>(hCorrectedCorrel_2D[0]->Clone("h2D_CorrectedCorrel"));
      if (fMethod == kDeltaPhiBinning) {
        if (!hCorrectedPairsMass[0] || fPoolVec_CorrectionRatio.empty()) {
          std::cerr << "[WARNING] Empty SE for pool 0, skipping task." << std::endl;
          return kFALSE;
        }
        h1D_CorrectedPairsMass = reinterpret_cast<TH1D*>(hCorrectedPairsMass[0]->Clone("h1D_CorrectedPairsMass"));
        h1D_correctedRatioVsDeltaEta = reinterpret_cast<TH1D*>(fPoolVec_CorrectionRatio[0]->Clone("h1D_correctedRatioVsDeltaEta"));
      }
    } else {
      h2D_SE->Add(hSE_2D_Raw[iPool]);
      h2D_ME->Add(hME_2D_Raw[iPool]);
      h2D_ME_norm->Add(hME_2D_Normalized[iPool]);
      h2D_CorrectedCorrel->Add(hCorrectedCorrel_2D[iPool]);
      if (fMethod == kDeltaPhiBinning) {
        if (!hCorrectedPairsMass[iPool] || iPool >= (Int_t)fPoolVec_CorrectionRatio.size()) {
          std::cerr << "[WARNING] Empty SE for pool " << iPool << ", skipping task." << std::endl;
          return kFALSE;
        }
        h1D_CorrectedPairsMass->Add(hCorrectedPairsMass[iPool]);
        h1D_correctedRatioVsDeltaEta->Add(fPoolVec_CorrectionRatio[iPool]);
      }
    }
  } // end pool loop

  if (fMethod == kDeltaPhiBinning) {
    fCorrectedPairsMass = SetTH1HistoStyle(reinterpret_cast<TH1D*>(h1D_CorrectedPairsMass->Clone("hCorrectedPairsMass")), 
      Form("Corrected pairs mass distribution with %s", fDeltaEtaGap.Data()), "Invariant Mass (GeV/c^{2})", "Corrected pairs / GeV/c^{2}");
    fCorrectionRatio = SetTH1HistoStyle(reinterpret_cast<TH1D*>(h1D_correctedRatioVsDeltaEta->Clone("hCorrectionRatio")),
      Form("Correction ratio with %s", fDeltaEtaGap.Data()), "#Delta#eta", "Ratio (corrected SE / SE)");
  }

  // clean up pool histos
  for (int iPool = 0; iPool < fNpools; iPool++) {
    delete hSE_2D_Raw[iPool];           hSE_2D_Raw[iPool] = nullptr;
    delete hME_2D_Raw[iPool];           hME_2D_Raw[iPool] = nullptr;
    delete hME_2D_Normalized[iPool];    hME_2D_Normalized[iPool] = nullptr;
    delete hME_Sign_SoftPi[iPool];      hME_Sign_SoftPi[iPool] = nullptr;
    delete hCorrectedCorrel_2D[iPool];  hCorrectedCorrel_2D[iPool] = nullptr;
    delete hCorrectedPairsMass[iPool];       hCorrectedPairsMass[iPool] = nullptr;
  }

  // debug: integrated SE, ME, normalized ME and corrected correlation histos
  if (fDebug > 0) {
    fCorrel_SE_2D = SetTH2HistoStyle(reinterpret_cast<TH2D*>(h2D_SE->Clone("hCorrel_SE_2D")), Form("SE %s-h correlation with %s", fDmesonLabel.Data(), fDeltaEtaGap.Data()), "#Delta#eta", "#Delta#phi (rad)", AxisLabels::kRawYieldRad);
    fCorrel_ME_2D = SetTH2HistoStyle(reinterpret_cast<TH2D*>(h2D_ME->Clone("hCorrel_ME_2D")), Form("ME %s-h correlation with %s", fDmesonLabel.Data(), fDeltaEtaGap.Data()), "#Delta#eta", "#Delta#phi (rad)", AxisLabels::kRawYieldRad);
    fCorrectedCorrel_2D = SetTH2HistoStyle(reinterpret_cast<TH2D*>(h2D_CorrectedCorrel->Clone("hCorrectedCorrel_2D")), Form("Corrected %s-h correlation with %s", fDmesonLabel.Data(), fDeltaEtaGap.Data()), "#Delta#eta", "#Delta#phi (rad)", AxisLabels::kRawYieldRad);
    fNormalizedCorrel_ME_2D = SetTH2HistoStyle(reinterpret_cast<TH2D*>(h2D_ME_norm->Clone("hNormalizedCorrel_ME_2D")), Form("Normalized ME %s-h correlation with %s", fDmesonLabel.Data(), fDeltaEtaGap.Data()), "#Delta#eta", "#Delta#phi (rad)", "ME correction ratio");
  }

  // clean up integrated SE, ME and normalized ME histos
  delete h2D_SE;
  h2D_SE = nullptr;
  delete h2D_ME;
  h2D_ME = nullptr;
  delete h2D_ME_norm;
  h2D_ME_norm = nullptr;
  delete h1D_CorrectedPairsMass;
  h1D_CorrectedPairsMass = nullptr;
  delete h1D_correctedRatioVsDeltaEta;
  h1D_correctedRatioVsDeltaEta = nullptr;

  //==========================================================================================================================
  // 1D projection
  h1D_CorrectedCorrel = h2D_CorrectedCorrel->ProjectionY("hCorrectedCorrel_1D", 1, -1, "e"); // projection on deltaPhi axis

  fCorrectedCorrel = SetTH1HistoStyle(reinterpret_cast<TH1D*>(h1D_CorrectedCorrel->Clone("hCorrectedCorrel")),
    Form("Corrected %s-h correlation with %s", fDmesonLabel.Data(), fDeltaEtaGap.Data()), "#Delta#phi (rad)", AxisLabels::kRawYieldRad_DP);

  // Apply normalization to number of triggers - NOT DONE
  h1D_NormalizedCorrectedCorrel = reinterpret_cast<TH1D*>(h1D_CorrectedCorrel->Clone("h1D_NormalizedCorrectedCorrel"));
  if (!fMassVsPt_2D) {
    std::cout << "[WARNING] Mass vs pT histogram not found, skipping trigger normalization." << std::endl;
    ProjMassVsPt();
  }
  Double_t N_triggers = CalculateTriggerNormalizationFactor(fMassVsPt_2D, fPtCandBins[0], fPtCandBins[1], fInvMassBins[0], fInvMassBins[1]);
  h1D_NormalizedCorrectedCorrel->Scale(1. / N_triggers);

  fNormalizedCorrectedCorrel = SetTH1HistoStyle(reinterpret_cast<TH1D*>(h1D_NormalizedCorrectedCorrel->Clone("hNormalizedCorrectedCorrel")),
    Form("Normalized corrected %s-h correlation with %s", fDmesonLabel.Data(), fDeltaEtaGap.Data()), "#Delta#phi (rad)", AxisLabels::kPerTrigYield_DP);

  // clean up
  delete h2D_CorrectedCorrel;
  h2D_CorrectedCorrel = nullptr;
  delete h1D_CorrectedCorrel;
  h1D_CorrectedCorrel = nullptr;

  // clean up
  delete h1D_NormalizedCorrectedCorrel;
  h1D_NormalizedCorrectedCorrel = nullptr;

  return kTRUE;
}

Bool_t DhCorrelationExtraction::ReadInputSEandME()
{
  fFileSE = TFile::Open(fFileNameSE.Data());
  if (!fFileSE) {
    std::cerr << "[ERROR] File " << fFileNameSE << " cannot be opened! check your file path!";
    return kFALSE;
  }

  fFileME = TFile::Open(fFileNameME.Data());
  if (!fFileME) {
    std::cerr << "[ERROR] File " << fFileNameME << " cannot be opened! check your file path!";
    return kFALSE;
  }

  fDirSE = reinterpret_cast<TDirectoryFile*>(fFileSE->Get(fDirNameSE.Data()));
  fDirME = reinterpret_cast<TDirectoryFile*>(fFileME->Get(fDirNameME.Data()));
  return kTRUE;
}

TH2D* DhCorrelationExtraction::ProjCorrelHisto(Int_t SEorME, Int_t pool)
{
  // TODO: Subtraction of softpion
  TH2D* h2D = nullptr;
  TH2D* hFinal = nullptr;
  TH2D* hFinalMass = nullptr;
  TString poolStr = fDoPoolByPool ? Form("%d", pool) : "All";

  // get the THnSparse from the corresponding directory
  THnSparseF* hSparse = 0x0;
  if (SEorME == kSE) { // Same Event
    hSparse = reinterpret_cast<THnSparseF*>(fDirSE->Get(fCorrelSparseNameSE.Data()));
  } else { // Mixed Event
    hSparse = reinterpret_cast<THnSparseF*>(fDirME->Get(fCorrelSparseNameME.Data()));
  }

  // Check pointer
  if (!hSparse) {
    std::cerr << "[ERROR] hSparse is null! Check that the object name exists in the directory and the file is open." << std::endl;
    throw std::runtime_error("hSparse is null");
  }

  // get bin range for pool selection
  Int_t binExtPoolMin;
  Int_t binExtPoolMax;
  if (fDoPoolByPool) {
    binExtPoolMin = (Int_t)hSparse->GetAxis(kPool)->FindBin(pool + 0.01); // axis1: pool bin
    binExtPoolMax = (Int_t)hSparse->GetAxis(kPool)->FindBin(pool + 0.99);
  } else { // merge all pools in one
    binExtPoolMin = 1;
    binExtPoolMax = (Int_t)hSparse->GetAxis(kPool)->GetNbins();
  }

  // adjust deltaEta range if it's out of the histogram range
  if (fDeltaEtaLeftMin  < hSparse->GetAxis(kDeltaEta)->GetXmin()) fDeltaEtaLeftMin  = hSparse->GetAxis(kDeltaEta)->GetXmin();
  if (fDeltaEtaRightMax > hSparse->GetAxis(kDeltaEta)->GetXmax()) fDeltaEtaRightMax = hSparse->GetAxis(kDeltaEta)->GetXmax();

  // set ranges
  hSparse->GetAxis(kPool)->SetRangeUser(pool+0.01, fDoPoolByPool ? pool+0.99 : hSparse->GetAxis(kPool)->GetXmax()); // axis0: pool bin
  hSparse->GetAxis(kPtCand)->SetRangeUser(fPtCandBins[0], fPtCandBins[1]);    // axis1: ptCand
  hSparse->GetAxis(kPtHad)->SetRangeUser(fPtHadBins[0], fPtHadBins[1]);       // axis2: ptHad

  // debug: get original histogram before any operations
  if (fDebug > 0) {
    TH2D* h2D_Original = static_cast<TH2D*>(hSparse->Projection(kDeltaPhi, kDeltaEta));
    TH2D* h2D_Original_MassVsDeltaEta = nullptr;
    if (SEorME == kSE && fMethod == kDeltaPhiBinning) {
        h2D_Original_MassVsDeltaEta = static_cast<TH2D*>(hSparse->Projection(kMass, kDeltaEta));
    }

    if (fDebug > 1  && fDoPoolByPool) {
      if (SEorME == kSE) {
          TString title_OriginalCorrel_SE_2D = Form("hOriginal_Correl_SE_2D_Pool%s", poolStr.Data());
          fPoolVec_OriginalCorrel_SE_2D.push_back(SetTH2HistoStyle(static_cast<TH2D*>(h2D_Original->Clone(title_OriginalCorrel_SE_2D.Data())), 
            Form("Original Correlation from SE THnSparse projection for Pool %s", poolStr.Data()), 
            "#Delta#eta", "#Delta#phi (rad)", AxisLabels::kRawYieldRad));

        if (fMethod == kDeltaPhiBinning) {
          TString title_OriginalMassVsDeltaEta = Form("hOriginal_MassVsDeltaEta_SE_2D_Pool%s", poolStr.Data());
          fPoolVec_OriginalMassVsDeltaEta_2D.push_back(SetTH2HistoStyle(static_cast<TH2D*>(h2D_Original_MassVsDeltaEta->Clone(title_OriginalMassVsDeltaEta.Data())), 
            Form("Original Mass vs DeltaEta from SE THnSparse projection for Pool %s", poolStr.Data()), 
            "#Delta#eta", "Mass (GeV/#it{c}^{2})", "Counts"));
        }
      } else if (fDoPoolByPool) { // ME
        TString title_OriginalCorrel_ME_2D = Form("hOriginal_Correl_ME_2D_Pool%s", poolStr.Data());
        fPoolVec_OriginalCorrel_ME_2D.push_back(SetTH2HistoStyle(static_cast<TH2D*>(h2D_Original->Clone(title_OriginalCorrel_ME_2D.Data())), 
          Form("Original Correlation from ME THnSparse projection for Pool %s", poolStr.Data()), 
          "#Delta#eta", "#Delta#phi (rad)", AxisLabels::kRawYieldRad));
      }
    }

    if (SEorME == kSE) {
      if (pool == 0) {
        fOriginalCorrel_SE_2D = SetTH2HistoStyle(static_cast<TH2D*>(h2D_Original->Clone("hOriginalCorrel_SE_2D")), 
          "Original Correlation from SE THnSparse projection", "#Delta#eta", "#Delta#phi (rad)", AxisLabels::kRawYieldRad);
      } else {
        fOriginalCorrel_SE_2D->Add(h2D_Original);
      }
      
      if (fMethod == kDeltaPhiBinning) {
        if (pool == 0) {
          fOriginalMassVsDeltaEta_2D = SetTH2HistoStyle(static_cast<TH2D*>(h2D_Original_MassVsDeltaEta->Clone("hOriginalMassVsDeltaEta_SE_2D")), 
            "Original Mass vs DeltaEta from SE THnSparse projection", "#Delta#eta", "Mass (GeV/#it{c}^{2})", "Counts");
        } else {
          fOriginalMassVsDeltaEta_2D->Add(h2D_Original_MassVsDeltaEta);
        }
      }
    } else { // ME
      if (pool == 0) {
        fOriginalCorrel_ME_2D = SetTH2HistoStyle(static_cast<TH2D*>(h2D_Original->Clone("hOriginalCorrel_ME_2D")), 
          "Original Correlation from ME THnSparse projection", "#Delta#eta", "#Delta#phi (rad)", AxisLabels::kRawYieldRad);
      } else {
        fOriginalCorrel_ME_2D->Add(h2D_Original);
      }
    }

    delete h2D_Original;
    h2D_Original = nullptr;
    if (SEorME == kSE && fMethod == kDeltaPhiBinning) {
      delete h2D_Original_MassVsDeltaEta;
      h2D_Original_MassVsDeltaEta = nullptr;
    }
  }

  // mass selecton for kMassBinning method, but whole range will be applied for kDeltaPhiBinning method
  hSparse->GetAxis(kMass)->SetRangeUser(fInvMassBins[0]*1.001, fInvMassBins[1]*0.999); // axis5: invMass

  // set outer deltaEta range
  hSparse->GetAxis(kDeltaEta)->SetRangeUser(fDeltaEtaLeftMin+0.01, fDeltaEtaRightMax-0.01); // axis3: deltaEta

  // Project to 2D histogram for correlation, and 2D histogram for mass vs deltaEta if needed for kDeltaPhiBinning method
  hFinal = (TH2D*)hSparse->Projection(kDeltaPhi, kDeltaEta);            // axis4: deltaPhi, axis3: deltaEta Y X

  // rebin axes deltaEta and deltaPhi
  if (fRebinAxisDeltaEta > 1 || fRebinAxisDeltaPhi > 1) {
    hFinal->Rebin2D(fRebinAxisDeltaEta, fRebinAxisDeltaPhi); // X Y
  }

  if (SEorME == kME) CalculateNormaliztionFactorME(hFinal, pool);
  if(fMethod == kDeltaPhiBinning) {
    hSparse->GetAxis(kDeltaPhi)->SetRangeUser(fDeltaPhiBins[0], fDeltaPhiBins[1]); // axis4: deltaPhi
  }
  hFinalMass = (TH2D*)hSparse->Projection(kMass, kDeltaEta);            // axis5: invMass, axis3: deltaEta
  if (fMethod == kDeltaPhiBinning) hFinalMass->RebinY(fRebinAxisDeltaEta);

  // set to 0 for the inner deltaEta gap
  if (fDeltaEtaLeftMax != 0 && fDeltaEtaRightMin != 0) {
    Int_t leftBinMax = hFinal->GetXaxis()->FindBin(fDeltaEtaLeftMax - 0.01);
    Int_t rightBinMin = hFinal->GetXaxis()->FindBin(fDeltaEtaRightMin + 0.01);
    Int_t nBinsY = hFinal->GetNbinsY();
    bool doFinalMass = (fMethod == kDeltaPhiBinning && SEorME == kSE);
    Int_t nBinsYMass = doFinalMass ? hFinalMass->GetNbinsY() : 0;
    for (Int_t ix = leftBinMax + 1; ix < rightBinMin; ++ix) {
        for (Int_t iy = 1; iy <= nBinsY; ++iy) {
            hFinal->SetBinContent(ix, iy, 0);
            hFinal->SetBinError(ix, iy, 0);
        }
        if (doFinalMass) {
            for (Int_t iy = 1; iy <= nBinsYMass; ++iy) {
                hFinalMass->SetBinContent(ix, iy, 0);
                hFinalMass->SetBinError(ix, iy, 0);
            }
        }
    }
  }

  if (fMethod == kDeltaPhiBinning) {
    TString titleMass = Form("Raw Mass vs DeltaEta with |#Delta#eta| > %.1f for Pool %s", fDeltaEtaRightMin, poolStr.Data());
    fPoolVec_RawMassVsDeltaEta_2D.push_back(SetTH2HistoStyle(reinterpret_cast<TH2D*>(hFinalMass->Clone(titleMass)),
      Form("hRaw_MassVsDeltaEta_SE_2D_Pool%s", poolStr.Data()), "#Delta#eta", "Mass (GeV/#it{c}^{2})", "Counts"));
  }

  if (fDebug > 1) {
    TString titleCorrel = Form("Raw %s-h correlation %s with |#Delta#eta| > %.1f for Pool %s", fDmesonLabel.Data(), SEorME ? "SE" : "ME", fDeltaEtaRightMin, poolStr.Data());
    if (SEorME == kSE) {
      fPoolVec_RawCorrel_SE_2D.push_back(SetTH2HistoStyle(reinterpret_cast<TH2D*>(hFinal->Clone(titleCorrel)),
        Form("hRaw_Correl_SE_2D_Pool%s", poolStr.Data()), "#Delta#eta", "#Delta#phi (rad)", AxisLabels::kRawYieldRad));
    } else {
      fPoolVec_RawCorrel_ME_2D.push_back(SetTH2HistoStyle(reinterpret_cast<TH2D*>(hFinal->Clone(titleCorrel)), 
        Form("hRaw_Correl_ME_2D_Pool%s", poolStr.Data()), "#Delta#eta", "#Delta#phi (rad)", AxisLabels::kRawYieldRad));
    }
  }

  h2D = static_cast<TH2D*>(hFinal->Clone(Form("hCorrel_%s_2D_Pool%s", (SEorME == kSE) ? "SE" : "ME", fDoPoolByPool ? Form("%d", pool) : "All")));

  // clean up
  delete hFinal;
  hFinal = nullptr;
  delete hFinalMass;
  hFinalMass = nullptr;

  return h2D;
}

void DhCorrelationExtraction::CalculateNormaliztionFactorME(TH2D*& histoME, Int_t pool)
{

  Int_t bin0phi = histoME->GetYaxis()->FindBin(0.);
  Int_t bin0eta = histoME->GetXaxis()->FindBin(0.);

  // evaluate the normalization (from ALL tracks, including possible fake softpions) -> **histoME indeed includes bin1+bin2 of THnSparse, i.e. all the tracks**
  Double_t factorNorm = 0;
  for (int in = -1; in <= 0; in++) {
    factorNorm += histoME->GetBinContent(bin0eta, bin0phi + in);
  }
  for (int in = -1; in <= 0; in++) {
    factorNorm += histoME->GetBinContent(bin0eta - 1, bin0phi + in);
  }
  factorNorm /= 4.;

  if (factorNorm == 0) {
    throw std::runtime_error("Normalization factor is zero (Only bin (0,0) is used for normalization)");
  }

  fFactorsNormME[pool] = factorNorm;
}

void DhCorrelationExtraction::NormalizeMEplot(TH2D*& histoME, TH2D*& histoMEsoftPi, Int_t pool)
{
  // apply the normalization
  histoME->Scale(1. / fFactorsNormME[pool]);
  if (fDebug > 0) {
    fPoolVec_NormalizedCorrel_ME_2D.push_back(SetTH2HistoStyle(reinterpret_cast<TH2D*>(histoME->Clone(Form("NormalizedCorrel_ME_2D_Pool%s", fDoPoolByPool ? Form("%d", pool) : "All"))), 
      Form("Correction ratio normalized from ME for Pool %s", fDoPoolByPool ? Form("%d", pool) : "All"), "#Delta#eta", "#Delta#phi (rad)", "ME correction ratio"));
  }
}

TH1D* DhCorrelationExtraction::CorrectedPairsMassDistr(TH2D* hRawSE, TH2D* hCorrectedCorrel, Int_t iPool)
{
  TH1D* hCorrectedMass = nullptr;

  // find the bin index
  Int_t binDeltaEtaLeftMin = hRawSE->GetXaxis()->FindBin(fDeltaEtaLeftMin + 0.01);
  Int_t binDeltaEtaRightMax = hRawSE->GetXaxis()->FindBin(fDeltaEtaRightMax - 0.01);
  Int_t nBinDeltaEta = hCorrectedCorrel->GetXaxis()->GetNbins();

  Int_t binDeltaPhiMin = hRawSE->GetYaxis()->FindBin(fDeltaPhiBins.front());
  Int_t binDeltaPhiMax = hRawSE->GetYaxis()->FindBin(fDeltaPhiBins.back());

  if (fDeltaEtaIntegrated) {
    // --- Integrated ---
    Double_t original_content = hRawSE->Integral(binDeltaEtaLeftMin, binDeltaEtaRightMax, binDeltaPhiMin, binDeltaPhiMax);
    if (original_content == 0) return nullptr;

    Double_t corrected_content = hCorrectedCorrel->Integral(binDeltaEtaLeftMin, binDeltaEtaRightMax, binDeltaPhiMin, binDeltaPhiMax);
    Double_t ratio_correction = corrected_content / original_content;

    hCorrectedMass = fPoolVec_RawMassVsDeltaEta_2D[iPool]->ProjectionY(Form("hCorrectedMass_ProjY_Pool%d", iPool), binDeltaEtaLeftMin, binDeltaEtaRightMax, "e");
    hCorrectedMass->Scale(ratio_correction);
    hCorrectedMass->SetEntries(hCorrectedMass->Integral());

    TH1D* correctionRatio = new TH1D(Form("hTempRatio_Pool%d", iPool), Form("Correction ratio vs #Delta#eta, Pool %d", iPool), 1, fDeltaEtaLeftMin, fDeltaEtaRightMax);
    correctionRatio->SetBinContent(1, ratio_correction);
    TString titleCorrectionRatio = Form("#Delta#phi integrated correction ratio with %s for Pool %s", fDeltaEtaGap.Data(), fDoPoolByPool ? Form("%d", iPool) : "All");
    fPoolVec_CorrectionRatio.push_back(SetTH1HistoStyle(reinterpret_cast<TH1D*>(correctionRatio->Clone(titleCorrectionRatio.Data())), 
      Form("hCorrectionRatio_Pool%d", iPool), "#Delta#eta", "Ratio"));
    delete correctionRatio;

    // --- Debug ---
    if (fDebug > 1) {
      TString titleCorrectedMass = Form("Corrected pairs' mass distribution with %s for Pool %s", fDeltaEtaGap.Data(), fDoPoolByPool ? Form("%d", iPool) : "All");
      fPoolVec_CorrectedMass.push_back(SetTH1HistoStyle(reinterpret_cast<TH1D*>(hCorrectedMass->Clone(titleCorrectedMass.Data())),
        Form("hCorrectedMass_Pool%d", iPool), "Mass (GeV/#it{c}^{2})", "Counts"));
    }

  } else {
    hCorrectedMass = fPoolVec_RawMassVsDeltaEta_2D[iPool]->ProjectionY(Form("hCorrectedMass_ProjY_Pool%d", iPool), binDeltaEtaLeftMin, binDeltaEtaRightMax, "e");
    hCorrectedMass->Reset();

    TH1D* hCorrectionRatio = nullptr;
    hCorrectionRatio = hRawSE->ProjectionX(Form("hTempRatio_BinByBin_Pool%d", iPool));
    hCorrectionRatio->Reset();

    for (int iEtaBin = 1; iEtaBin <= nBinDeltaEta; iEtaBin++) {
      Double_t original_content = hRawSE->Integral(iEtaBin, iEtaBin, binDeltaPhiMin, binDeltaPhiMax);
      if (original_content == 0) continue;

      Double_t corrected_content = hCorrectedCorrel->Integral(iEtaBin, iEtaBin, binDeltaPhiMin, binDeltaPhiMax);
      Double_t ratio_correction = corrected_content / original_content;

      hCorrectionRatio->SetBinContent(iEtaBin, ratio_correction);

      TH1D* hTempCorrectedMass = fPoolVec_RawMassVsDeltaEta_2D[iPool]->ProjectionY(Form("hCorrectedMassPairs_ProjY_Pool%d_EtaBin%d", iPool, iEtaBin), iEtaBin, iEtaBin, "e");
      hTempCorrectedMass->Scale(ratio_correction);
      hCorrectedMass->Add(hTempCorrectedMass);

      delete hTempCorrectedMass;
    }
    hCorrectedMass->SetEntries(hCorrectedMass->Integral());

    TString titleCorrectionRatio = Form("Correction ratio (Bin-by-bin) with %s for Pool %s", fDeltaEtaGap.Data(), fDoPoolByPool ? Form("%d", iPool) : "All");
    fPoolVec_CorrectionRatio.push_back(SetTH1HistoStyle(reinterpret_cast<TH1D*>(hCorrectionRatio->Clone(titleCorrectionRatio.Data())), 
      Form("hCorrectionRatio_Pool%d", iPool), "#Delta#eta", "Ratio"));      
    delete hCorrectionRatio;

    // --- Debug ---
    if (fDebug > 1) {
      TString titleCorrectedMass = Form("Corrected pairs' mass distribution (Bin-by-bin) with %s for Pool %s", fDeltaEtaGap.Data(), fDoPoolByPool ? Form("%d", iPool) : "All");
      fPoolVec_CorrectedMass.push_back(SetTH1HistoStyle(reinterpret_cast<TH1D*>(hCorrectedMass->Clone(titleCorrectedMass.Data())), 
        Form("hCorrectedMass_Pool%d", iPool), "Mass (GeV/#it{c}^{2})", "Counts"));
    }
  }

  return hCorrectedMass;
}

TH1D* DhCorrelationExtraction::SetTH1HistoStyle(TH1D* histo, TString hTitle, TString hXaxisTitle, TString hYaxisTitle,
                                               Style_t markerStyle, Color_t markerColor, Double_t markerSize,
                                               Color_t lineColor, Int_t lineWidth, Float_t hTitleXaxisOffset, Float_t hTitleYaxisOffset,
                                               Float_t hTitleXaxisSize, Float_t hTitleYaxisSize, Float_t hLabelXaxisSize, Float_t hLabelYaxisSize,
                                               Bool_t centerXaxisTitle, Bool_t centerYaxisTitle)
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

  return histo;
}

TH2D* DhCorrelationExtraction::SetTH2HistoStyle(TH2D* histo, TString hTitle, TString hXaxisTitle, TString hYaxisTitle, TString hZaxisTitle,
                                               Float_t hTitleXaxisOffset, Float_t hTitleYaxisOffset, Float_t hTitleZaxisOffset,
                                               Float_t hTitleXaxisSize, Float_t hTitleYaxisSize, Float_t hTitleZaxisSize,
                                               Float_t hLabelXaxisSize, Float_t hLabelYaxisSize, Float_t hLabelZaxisSize,
                                               Bool_t centerXaxisTitle, Bool_t centerYaxisTitle)
{

  histo->SetTitle(hTitle.Data());
  histo->GetXaxis()->SetTitle(hXaxisTitle.Data());
  histo->GetYaxis()->SetTitle(hYaxisTitle.Data());
  histo->GetZaxis()->SetTitle(hZaxisTitle.Data());
  histo->GetXaxis()->SetTitleOffset(hTitleXaxisOffset);
  histo->GetYaxis()->SetTitleOffset(hTitleYaxisOffset);
  histo->GetZaxis()->SetTitleOffset(hTitleZaxisOffset);
  histo->GetXaxis()->SetTitleSize(hTitleXaxisSize);
  histo->GetYaxis()->SetTitleSize(hTitleYaxisSize);
  histo->GetZaxis()->SetTitleSize(hTitleZaxisSize);
  histo->GetXaxis()->SetLabelSize(hLabelXaxisSize);
  histo->GetYaxis()->SetLabelSize(hLabelYaxisSize);
  histo->GetZaxis()->SetLabelSize(hLabelZaxisSize);
  histo->GetXaxis()->CenterTitle(centerXaxisTitle);
  histo->GetYaxis()->CenterTitle(centerYaxisTitle);

  return histo;
}

// load and project mass THnSparse to get mass vs pt histogram
void DhCorrelationExtraction::ProjMassVsPt() {
  TFile* fileMass = TFile::Open(fFileNameMass.Data());
  if (!fileMass || fileMass->IsZombie()) {
    std::cerr << "[ERROR] Could not open file: " << fFileNameMass.Data() << std::endl;
    return;
  }

  THnSparseF* sparseMass = reinterpret_cast<THnSparseF*>(fileMass->Get(fMassSparseName.Data()));
  TH2D* hMassVsPt = reinterpret_cast<TH2D*>(sparseMass->Projection(1,0));
  hMassVsPt -> SetDirectory(0);
  hMassVsPt -> SetName("hMassVsPt");
  fMassVsPt_2D = hMassVsPt;

  fileMass -> Close();
  delete sparseMass;
}

// calculate the number of trigger D mesons in given pt range from mass vs pt histogram
Double_t DhCorrelationExtraction::CalculateTriggerNormalizationFactor(TH2D* hMassVsPt, Double_t ptMin, Double_t ptMax, Double_t massMin, Double_t massMax) {
  if (!hMassVsPt) {
    std::cerr << "[ERROR] hMassVsPt is null!" << std::endl;
    return 0.0;
  }

  Int_t ptBinMin = hMassVsPt->GetYaxis()->FindBin(ptMin + 1e-6);
  Int_t ptBinMax = hMassVsPt->GetYaxis()->FindBin(ptMax - 1e-6);

  TH1D* hMassProj = hMassVsPt->ProjectionX("hMassProj", ptBinMin, ptBinMax);

  Int_t massBinMin = hMassProj->GetXaxis()->FindBin(massMin);
  Int_t massBinMax = hMassProj->GetXaxis()->FindBin(massMax);

  Double_t triggerCount = hMassProj->Integral(massBinMin, massBinMax);
  delete hMassProj;

  return triggerCount;
}
