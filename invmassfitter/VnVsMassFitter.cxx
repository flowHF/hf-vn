#include "VnVsMassFitter.h"

#include <TROOT.h>
#include <TMath.h>
#include <TF1.h>
#include <TGraphErrors.h>
#include <TColor.h>
#include <TLegend.h>
#include <TList.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TFitResult.h>
#include <TVirtualPad.h>
#include <TDatabasePDG.h>
#include <TPaveText.h>
#include <TVirtualFitter.h>
#include "Fit/BinData.h"
#include "HFitInterface.h"
#include <vector>

/// \cond CLASSIMP
ClassImp(VnVsMassFitter);
/// \endcond

//________________________________________________________________
VnVsMassFitter::VnVsMassFitter()
  :TObject()
  ,fName("")
  ,fMassHisto(0x0)
  ,fVnVsMassHisto(0x0)
  ,fPrefitParsHisto(0x0)
  ,fSignalParsHisto(0x0)
  ,fSimFitParsHisto(0x0)
  ,fMassSgnFuncType(kGaus)
  ,fMassBkgFuncType(kExpo)
  ,fVnBkgFuncType(kLin)
  ,fMassFuncFromPrefit(0x0)
  ,fMassBkgFunc(0x0)
  ,fMassSgnFunc(0x0)
  ,fMassTemplFunc(0x0)
  ,fMassTotFunc(0x0)
  ,fVnBkgFunc(0x0)
  ,fVnTotFunc(0x0)
  ,fMassMin(1.69)
  ,fMassMax(2.05)
  ,fVn(0.)
  ,fVnUncertainty(0.)
  ,fSigma(0.)
  ,fSigmaUncertainty(0.)
  ,fMean(0.)
  ,fMeanUncertainty(0.)
  ,fRawYield(0.)
  ,fRawYieldUncertainty(0.)
  ,fChiSquare(0.)
  ,fNDF(0)
  ,fProb(0.)
  ,fSBVnPrefitChiSquare(0.)
  ,fSBVnPrefitNDF(0)
  ,fSBVnPrefitProb(0.)
  ,fMassPrefitChiSquare(0.)
  ,fMassPrefitNDF(0)
  ,fMassPrefitProb(0.)
  ,fNSigmaForSB(3.)
  ,fSigmaInit(0.012)
  ,fMeanInit(1.870)
  ,fSigma2GausInit(0.012)
  ,fFrac2GausInit(0.2)
  ,fMeanFixedFromMassFit(kFALSE)
  ,fSigmaFixedFromMassFit(kFALSE)
  ,fSigma2GausFixedFromMassFit(kFALSE)
  ,fFrac2GausFixedFromMassFit(kFALSE)
  ,fMassParticle(1.870)
  ,fNParsMassSgn(3)
  ,fNParsMassBkg(2)
  ,fNParsVnBkg(2)
  ,fNParsVnSgn(1)
  ,fNParsVnSecPeak(0)
  ,fNParsVnRfl(0)
  ,fSigmaFixed(0)
  ,fMeanFixed(0)
  ,fSigma2GausFixed(0)
  ,fFrac2GausFixed(0)
  ,fPolDegreeBkg(3)
  ,fPolDegreeVnBkg(3)
  ,fReflections(kFALSE)
  ,fNParsRfl(0)
  ,fNParsTotMass(0)
  ,fNParsTotVn(0)
  ,fNVnParsSgn(0)
  ,fTemplates(kFALSE)
  ,fNParsTempls(0)
  ,fRflOverSig(0.)
  ,fFixRflOverSig(kFALSE)
  ,fHistoTemplRfl(0x0)
  ,fHistoTemplRflInit(0x0)
  ,fMassRflFunc(0x0)
  ,fMassBkgRflFunc(0x0)
  ,fRflOpt("1gaus")
  ,fMinRefl(0.)
  ,fMaxRefl(0.)
  ,fSmoothRfl(kFALSE)
  ,fRawYieldHelp(0.)
  ,fVnRflOpt(0)
  ,fVnRflLimited(kFALSE)
  ,fVnRflMin(-1.)
  ,fVnRflMax(1.)
  ,fSecondPeak(kFALSE)
  ,fMassSecPeakFunc(0x0)
  ,fMassRangeMinSecPeak(0.)
  ,fMassRangeMaxSecPeak(0.)
  ,fMinCountsForSecPeak(0)
  ,fVnSecPeakFunc(0x0)
  ,fNParsSec(0)
  ,fSecMass(-999.)
  ,fSecWidth(9999.)
  ,fFixSecMass(kFALSE)
  ,fFixSecWidth(kFALSE)
  ,fDoSecondPeakVn(kFALSE)
  ,fFixVnSecPeakToSgn(kFALSE)
  ,fHarmonic(2)
  ,fSuppressOutput(kFALSE)
  ,fAnchorTemplsMode(AnchorToFirst)
  ,fHistoSgnPrefit(0x0)
  ,fFixSgnFromMCPrefit(kFALSE)
  ,fSecWidthFrac(9999.)
  ,fFixFracSecWidth(kFALSE)
  ,fIsMassSidebandFit(kFALSE)
  ,fIsVnSidebandFit(kFALSE)
  // ,fInitFuncPars()
  {
    //default constructor
}

//________________________________________________________________
VnVsMassFitter::VnVsMassFitter(std::string name, TH1F* hMass, TH1F* hvn, Double_t min, Double_t max, Int_t funcMassBkg, Int_t funcMassSgn, Int_t funcVnBkg)
  :TObject()
  ,fName(name)
  ,fPrefitParsHisto(0x0)
  ,fSignalParsHisto(0x0)
  ,fSimFitParsHisto(0x0)
  ,fMassSgnFuncType(funcMassSgn)
  ,fMassBkgFuncType(funcMassBkg)
  ,fVnBkgFuncType(funcVnBkg)
  ,fMassFuncFromPrefit(0x0)
  ,fMassBkgFunc(0x0)
  ,fMassSgnFunc(0x0)
  ,fMassTemplFunc(0x0)
  ,fMassTotFunc(0x0)
  ,fVnBkgFunc(0x0)
  ,fVnTotFunc(0x0)
  ,fMassMin(min)
  ,fMassMax(max)
  ,fVn(0.)
  ,fVnUncertainty(0.)
  ,fSigma(0.)
  ,fSigmaUncertainty(0.)
  ,fMean(0.)
  ,fMeanUncertainty(0.)
  ,fRawYield(0.)
  ,fRawYieldUncertainty(0.)
  ,fChiSquare(0.)
  ,fNDF(0)
  ,fProb(0.)
  ,fSBVnPrefitChiSquare(0.)
  ,fSBVnPrefitNDF(0)
  ,fSBVnPrefitProb(0.)
  ,fMassPrefitChiSquare(0.)
  ,fMassPrefitNDF(0)
  ,fMassPrefitProb(0.)
  ,fNSigmaForSB(3.)
  ,fSigmaInit(0.012)
  ,fMeanInit(1.870)
  ,fSigma2GausInit(0.012)
  ,fFrac2GausInit(0.2)
  ,fMeanFixedFromMassFit(kFALSE)
  ,fSigmaFixedFromMassFit(kFALSE)
  ,fSigma2GausFixedFromMassFit(kFALSE)
  ,fFrac2GausFixedFromMassFit(kFALSE)
  ,fMassParticle(1.870)
  ,fNParsMassSgn(3)
  ,fNParsMassBkg(2)
  ,fNParsVnBkg(2)
  ,fNParsVnSgn(1)
  ,fNParsVnSecPeak(0)
  ,fNParsVnRfl(0)
  ,fSigmaFixed(0)
  ,fMeanFixed(0)
  ,fSigma2GausFixed(0)
  ,fFrac2GausFixed(0)
  ,fPolDegreeBkg(3)
  ,fPolDegreeVnBkg(3)
  ,fReflections(kFALSE)
  ,fNParsRfl(0)
  ,fNParsTotMass(0)
  ,fNParsTotVn(0)
  ,fNVnParsSgn(0)
  ,fTemplates(kFALSE)
  ,fNParsTempls(0)
  ,fRflOverSig(0.)
  ,fFixRflOverSig(kFALSE)
  ,fHistoTemplRfl(0x0)
  ,fHistoTemplRflInit(0x0)
  ,fMassRflFunc(0x0)
  ,fMassBkgRflFunc(0x0)
  ,fRflOpt("1gaus")
  ,fMinRefl(0.)
  ,fMaxRefl(0.)
  ,fSmoothRfl(kFALSE)
  ,fRawYieldHelp(0.)
  ,fVnRflOpt(0)
  ,fVnRflLimited(kFALSE)
  ,fVnRflMin(-1.)
  ,fVnRflMax(1.)
  ,fSecondPeak(kFALSE)
  ,fMassSecPeakFunc(0x0)
  ,fMassRangeMinSecPeak(1.)
  ,fMassRangeMaxSecPeak(3.)
  ,fMinCountsForSecPeak(0)
  ,fVnSecPeakFunc(0x0)
  ,fNParsSec(0)
  ,fSecMass(-999.)
  ,fSecWidth(9999.)
  ,fFixSecMass(kFALSE)
  ,fFixSecWidth(kFALSE)
  ,fDoSecondPeakVn(kFALSE)
  ,fFixVnSecPeakToSgn(kFALSE)
  ,fHarmonic(2)
  ,fSuppressOutput(kFALSE)
  ,fAnchorTemplsMode(AnchorToFirst)
  ,fHistoSgnPrefit(0x0)
  ,fFixSgnFromMCPrefit(kFALSE)
  ,fSecWidthFrac(9999.)
  ,fFixFracSecWidth(kFALSE)
  ,fIsMassSidebandFit(kFALSE)
  ,fIsVnSidebandFit(kFALSE)
  // ,fInitFuncPars()
  {

    //standard constructor
    fMassHisto = (TH1F*)hMass->Clone(Form("fMassHisto_%s", fName.c_str()));
    fMassHisto->SetDirectory(0);
    fVnVsMassHisto = (TH1F*)hvn->Clone(Form("fHistoV%dVsMass_%s",fHarmonic, fName.c_str()));
    fVnVsMassHisto->SetDirectory(0);

    DefineNumberOfParameters();
    SetParInitValsAndNames();
}

//________________________________________________________________
VnVsMassFitter::~VnVsMassFitter() {

  //destructor
  if(fMassHisto)          delete fMassHisto;
  if(fVnVsMassHisto)      delete fVnVsMassHisto;
  if(fMassBkgFunc)        delete fMassBkgFunc;
  if(fMassBkgRflFunc)     delete fMassBkgRflFunc;
  if(fMassSgnFunc)        delete fMassSgnFunc;
  if(fMassTemplFunc)      delete fMassTemplFunc;
  if(fMassTotFunc)        delete fMassTotFunc;
  if(fVnBkgFunc)          delete fVnBkgFunc;
  if(fVnTotFunc)          delete fVnTotFunc;
  if(fHistoTemplRfl)      delete fHistoTemplRfl;
  if(fHistoTemplRflInit)  delete fHistoTemplRflInit;
  if(fMassRflFunc)        delete fMassRflFunc;
  if(fMassSecPeakFunc)    delete fMassSecPeakFunc;
  if(fVnSecPeakFunc)      delete fVnSecPeakFunc;
  if(fHistoSgnPrefit)     delete fHistoSgnPrefit;
  if(fPrefitParsHisto)    delete fPrefitParsHisto;
  if(fSignalParsHisto)    delete fSignalParsHisto;
  if(fSimFitParsHisto)    delete fSimFitParsHisto;
}

//________________________________________________________________
void VnVsMassFitter::InitFunctionPars(std::string func) {
  // std::cout << "\n\n Initializing " << func << ", length of init pars size: " << fInitFuncPars.size() << std::endl;
  if (func == "MassSgn") {
    // std::cout << "Setting parameters for Mass Signal function" << std::endl;
    for (auto& [name, par] : fInitFuncPars) {
      for (int iPar=0; iPar<fNParsMassSgn; ++iPar) {
        if (name == fMassSgnFunc->GetParName(iPar)) {
          fMassSgnFunc->SetParName(iPar, name.c_str());
          fMassSgnFunc->SetParameter(iPar, par.value);
          fMassSgnFunc->SetParLimits(iPar, par.low, par.high);
          // std::cout << "---> Setting Mass Signal param: " << name << " to " << par.value << " [" << par.low << ", " << par.high << "]" << std::endl;
        }
      }
    }
  } else if (func == "MassSecPeak") {
    // std::cout << "Setting parameters for Mass Secondary Peak function" << std::endl;
    for (auto& [name, par] : fInitFuncPars) {
      for (int iPar=0; iPar<fNParsSec; ++iPar) {
        if (name == fMassSecPeakFunc->GetParName(iPar)) {
          fMassSecPeakFunc->SetParName(iPar, name.c_str());
          fMassSecPeakFunc->SetParameter(iPar, par.value);
          fMassSecPeakFunc->SetParLimits(iPar, par.low, par.high);
          // std::cout << "---> Setting Mass Secondary Peak param: " << name << " to " << par.value << " [" << par.low << ", " << par.high << "]" << std::endl;
        }
      }
    }
  } else if (func == "MassBkg") {
    // std::cout << "Setting parameters for Mass Bkg function" << std::endl;
    for (auto& [name, par] : fInitFuncPars) {
      for (int iPar=0; iPar<fNParsMassBkg; ++iPar) {
        if (name == fMassBkgFunc->GetParName(iPar)) {
          fMassBkgFunc->SetParName(iPar, name.c_str());
          fMassBkgFunc->SetParameter(iPar, par.value);
          fMassBkgFunc->SetParLimits(iPar, par.low, par.high);
          // std::cout << "---> Setting Mass Bkg param: " << name << " to " << par.value << " [" << par.low << ", " << par.high << "]" << std::endl;
        }
      }
    }
  } else if (func == "VnBkg") {
    // std::cout << "Setting parameters for Vn Bkg function" << std::endl;
    for (auto& [name, par] : fInitFuncPars) {
      for (int iPar=0; iPar<fNParsVnBkg; ++iPar) {
        if (name == fVnBkgFunc->GetParName(iPar)) {
          fVnBkgFunc->SetParName(iPar, name.c_str());
          fVnBkgFunc->SetParameter(iPar, par.value);
          fVnBkgFunc->SetParLimits(iPar, par.low, par.high);
          // std::cout << "---> Setting Vn Bkg param: " << name << " to " << par.value << " [" << par.low << ", " << par.high << "]" << std::endl;
        }
      }
    }
  } else if (func == "MassFull") {
    // std::cout << "Setting parameters for Mass Full function, fInitFuncPars.size(): " << fInitFuncPars.size() << ", fNParsTotMass: " << fNParsTotMass << std::endl;
    for (auto& [name, par] : fInitFuncPars) {
      for (int iPar=0; iPar<fNParsTotMass; ++iPar) {
        if (name == fMassTotFunc->GetParName(iPar)) {
          fMassTotFunc->SetParName(iPar, name.c_str());
          fMassTotFunc->SetParameter(iPar, par.value);
          fMassTotFunc->SetParLimits(iPar, par.low, par.high);
          // std::cout << "---> Setting Mass Full param: " << name << " to " << par.value << " [" << par.low << ", " << par.high << "]" << std::endl;
        }
      }
    }
  } else if (func == "VnFull") {
    // std::cout << "Setting parameters for Vn Full function, fInitFuncPars.size(): " << fInitFuncPars.size() << ", fNParsTotVn: " << fNParsTotVn << std::endl;
    for (auto& [name, par] : fInitFuncPars) {
      for (int iPar=0; iPar<fNParsTotVn; ++iPar) {
        if (name == fVnTotFunc->GetParName(iPar)) {
          fVnTotFunc->SetParName(iPar, name.c_str());
          fVnTotFunc->SetParameter(iPar, par.value);
          fVnTotFunc->SetParLimits(iPar, par.low, par.high);
          // std::cout << "---> Setting Vn Full param: " << name << " to " << par.value << " [" << par.low << ", " << par.high << "]" << std::endl;
        }
      }
    }
  }
}

//________________________________________________________________
Int_t VnVsMassFitter::RunPrefits() {

  // Prefit the signal peak on MC template if available
  if (fHistoSgnPrefit) {
    std::cout << "\n--- [" << fName << "] Prefitting signal from MC ---" << std::endl;
    Bool_t prefitsgn=PrefitSignal();
    if(!prefitsgn) {printf("Impossible to perform the prefit of signal shape on MC template\n"); return kFALSE;}
    // Init fMassTotFunc and fVnTotFunc parameters from signal prefit, skip normalization
  }

  // Prefit the combinatorial background
  std::cout << "\n--- [" << fName << "] Prefitting combinatorial background from data sidebands ---" << std::endl;
  Bool_t prefitbkg=PrefitCombBkg();
  if(!prefitbkg) {printf("Impossible to perform the prefit of comb. bkg.\n"); return kFALSE;}
  // Prefit the invariant mass spectrum to get initial values for the simultaneous fit

  // std::cout << "fNParsTotMass: " << fNParsTotMass << ", fNParsMassBkg: " << fNParsMassBkg << ", fNParsMassSgn: " << fNParsMassSgn << ", fNParsTotVn: " << fNParsTotVn << std::endl;
  double parLowLim{0.}, parUpLim{0.};
  for (Int_t iPar = 0; iPar < fNParsMassBkg; iPar++) {
    fMassTotFunc->SetParName(iPar, fMassBkgFunc->GetParName(iPar));
    fVnTotFunc->SetParName(iPar, fMassBkgFunc->GetParName(iPar));
    // std::cout << "[" << iPar << "] Setting mass bkg par name: " << fMassBkgFunc->GetParName(iPar) << std::endl;
  }
  for (Int_t iPar = 0; iPar < fNParsMassSgn; iPar++) {
    fMassTotFunc->SetParName(iPar+fNParsMassBkg, fMassSgnFunc->GetParName(iPar));
    fVnTotFunc->SetParName(iPar+fNParsMassBkg, fMassSgnFunc->GetParName(iPar));
    // std::cout << "[" << iPar+fNParsMassBkg << "] Setting mass sgn par name: " << fMassSgnFunc->GetParName(iPar) << std::endl;
  }
  if (fSecondPeak) {
    for (Int_t iPar = 0; iPar < fNParsSec; iPar++) {
      fMassTotFunc->SetParName(iPar+fNParsMassBkg+fNParsMassSgn, fMassSecPeakFunc->GetParName(iPar));
      fVnTotFunc->SetParName(iPar+fNParsMassBkg+fNParsMassSgn, fMassSecPeakFunc->GetParName(iPar));
      // std::cout << "[" << iPar+fNParsMassBkg+fNParsMassSgn << "] Setting mass sec peak par name: " << fMassSecPeakFunc->GetParName(iPar) << std::endl;
    }
  }
  if(fReflections) {
    for (Int_t iPar = 0; iPar < fNParsRfl; iPar++) {
      fMassTotFunc->SetParName(iPar+fNParsMassBkg+fNParsMassSgn+fNParsSec, fMassRflFunc->GetParName(iPar));
      fVnTotFunc->SetParName(iPar+fNParsMassBkg+fNParsMassSgn+fNParsSec, fMassRflFunc->GetParName(iPar));
      // std::cout << "[" << iPar+fNParsMassBkg+fNParsMassSgn+fNParsSec << "] Setting mass refl par name: " << fMassRflFunc->GetParName(iPar) << std::endl;
    }
  }
  if(fTemplates) {
    for (Int_t iPar = 0; iPar < fNParsTempls; iPar++) {
      fMassTotFunc->SetParName(iPar+fNParsMassBkg+fNParsMassSgn+fNParsSec+fNParsRfl, fMassTemplFunc->GetParName(iPar));
      fVnTotFunc->SetParName(iPar+fNParsMassBkg+fNParsMassSgn+fNParsSec+fNParsRfl, fMassTemplFunc->GetParName(iPar));
      // std::cout << "[" << iPar+fNParsMassBkg+fNParsMassSgn+fNParsSec+fNParsRfl << "] Setting mass templ par name: " << fMassTemplFunc->GetParName(iPar) << std::endl;
    }
  }
  for (Int_t iPar = 0; iPar < fVnBkgFunc->GetNpar(); iPar++) {
    fVnTotFunc->SetParName(iPar+fNParsTotMass, fVnBkgFunc->GetParName(iPar));
    // std::cout << "Setting vn bkg par " << iPar << " from " << fVnTotFunc->GetParameter(iPar+fNParsTotMass) << " to " << fVnBkgFunc->GetParameter(iPar) << ", low lim: " << parLowLim << ", upp lim: " << parUpLim << std::endl;
  }
  fVnTotFunc->SetParName(fNParsMassBkg+fNParsMassSgn+fNParsRfl+fNParsSec+fNParsTempls+fNParsVnBkg,Form("v%dSgn",fHarmonic));
  if(fSecondPeak && fDoSecondPeakVn) {fVnTotFunc->SetParName(fNParsMassBkg+fNParsMassSgn+fNParsRfl+fNParsSec+fNParsTempls+fNParsVnBkg+1,Form("v%dSecPeak",fHarmonic));}
  if(fReflections && fVnRflOpt==kFreePar) {fVnTotFunc->SetParName(fNParsMassBkg+fNParsMassSgn+fNParsRfl+fNParsSec+fNParsVnBkg+fNParsTempls+1+fNParsVnSecPeak,Form("v%dRefl",fHarmonic));}

  InitFunctionPars("MassFull");

  // Prefit the invariant mass spectrum
  std::cout << "--- [" << fName << "] Prefitting Inv. Mass spectrum ---" << std::endl;
  Int_t massprefit=PrefitMass();
  if(massprefit == 0) {printf("Impossible to perform the mass prefit\n"); return kFALSE;}
  if(massprefit == 2) {return 2;}

  // Prefit the vn sidebands
  std::cout << "\n--- [" << fName << "] Prefitting Vn of sidebands ---" << std::endl;
  Bool_t vnprefit=PrefitVnSidebands();
  if(!vnprefit) {printf("Impossible to perform the bkg vn prefit\n"); return kFALSE;}

  return kTRUE;
}

//________________________________________________________________
void VnVsMassFitter::DefineFunctions() {
  fVnBkgFunc = new TF1(Form("fVnBkgFunc_%s", fName.c_str()),this,&VnVsMassFitter::vnBkgFunc,fMassMin,fMassMax,fNParsVnBkg,"VnVsMassFitter","vnBkgFunc");
  fMassBkgFunc = new TF1(Form("fMassBkgFunc_%s", fName.c_str()),this,&VnVsMassFitter::MassBkg,fMassMin,fMassMax,fNParsMassBkg,"VnVsMassFitter","MassBkg");
  fMassSgnFunc = new TF1(Form("fMassSgnFunc_%s", fName.c_str()),this,&VnVsMassFitter::MassSignal,fMassMin,fMassMax,fNParsMassSgn,"VnVsMassFitter","MassSignal");
  if(fReflections) {fMassRflFunc = new TF1(Form("fMassRflFunc_%s", fName.c_str()),this,&VnVsMassFitter::MassRfl,fMassMin,fMassMax,fNParsRfl,"VnVsMassFitter","MassRfl");}
  if(fReflections) {fMassBkgRflFunc = new TF1(Form("fMassBkgRflFunc_%s", fName.c_str()),this,&VnVsMassFitter::MassBkgRfl,fMassMin,fMassMax,fNParsMassBkg+fNParsRfl,"VnVsMassFitter","MassBkgRfl");}
  if(fSecondPeak)  {fMassSecPeakFunc = new TF1(Form("fMassSecPeakFunc_%s", fName.c_str()),this,&VnVsMassFitter::MassSecondPeak,fMassMin,fMassMax,fNParsSec+fFixFracSecWidth,"VnVsMassFitter","MassSecondPeak");}
  fMassTotFunc = new TF1(Form("fMassTotFunc_%s", fName.c_str()),this,&VnVsMassFitter::MassFunc,fMassMin,fMassMax,fNParsTotMass,"VnVsMassFitter","MassFunc");
  fVnTotFunc = new TF1(Form("fVnTotFunc_%s", fName.c_str()),this,&VnVsMassFitter::vnFunc,fMassMin,fMassMax,fNParsTotVn,"VnVsMassFitter","vnFunc");

  // SetParInitValsAndNames();
  SetFuncParNames();
  InitFunctionPars("VnBkg");
  InitFunctionPars("MassBkg");
  InitFunctionPars("MassSgn");
  if(fReflections) {InitFunctionPars("MassRfl");}
  if(fReflections) {InitFunctionPars("MassBkgRfl");}
  if(fSecondPeak)  {InitFunctionPars("MassSecPeak");}
}

//________________________________________________________________
Int_t VnVsMassFitter::SimultaneousFit() {
  // Create map of fit parameter initial values and limits
  std::cout << "\n--- [" << fName << "] Performing simultaneous fit of mass and vn vs mass ---" << std::endl;
  if (fSuppressOutput) {
    std::cout << "Suppressing output for simultaneous fit" << std::endl;
    gErrorIgnoreLevel = kFatal;
  }
  if(!fMassHisto || !fVnVsMassHisto) {printf("Histograms not set! Exit."); return kFALSE;}
  // DefineNumberOfParameters();
  DefineFunctions();
  Int_t prefitsStatus = RunPrefits();

  if (prefitsStatus == kFALSE) {
    std::cout << "Prefits failed, exiting simultaneous fit." << std::endl;
    return kFALSE;
  } else if (prefitsStatus == 2) {
    std::cout << "Mass prefit indicated to remove second peak, restarting the chain ..." << std::endl;
    fMassSecPeakFunc = 0x0;
    fSecondPeak = kFALSE;
    DefineNumberOfParameters();
    DefineFunctions();
    Int_t prefitsWithoutSecPeakStatus = RunPrefits();
    if (prefitsWithoutSecPeakStatus != kTRUE) {
      std::cout << "Prefits without second peak failed, exiting simultaneous fit." << std::endl;
      return kFALSE;
    }
  }

  // Initialize the total vn function for the simultaneous fit
  InitFunctionPars("VnFull");
  // for (Int_t iPar = 0; iPar < fMassTotFunc->GetNpar(); iPar++) {
  //   fVnTotFunc->SetParName(iPar, fMassTotFunc->GetParName(iPar));
  //   fVnTotFunc->SetParameter(iPar, fMassTotFunc->GetParameter(iPar));
  //   std::cout << "Setting mass par " << iPar << " from " << fVnTotFunc->GetParameter(iPar) << " to " << fMassTotFunc->GetParameter(iPar) << std::endl;
  // }

  std::cout << "\n--- [" << fName << "] Starting simultaneous fit ---" << std::endl;
  ROOT::Math::WrappedMultiTF1 wfTotMass(*fMassTotFunc,1);
  ROOT::Math::WrappedMultiTF1 wfTotVn(*fVnTotFunc,1);

  // set data options and ranges
  ROOT::Fit::DataOptions opt;
  ROOT::Fit::DataRange rangeMass; //same range for two functions

  rangeMass.SetRange(fMassMin,fMassMax);
  ROOT::Fit::BinData dataMass(opt,rangeMass);
  ROOT::Fit::FillData(dataMass, fMassHisto);
  ROOT::Fit::BinData dataVn(opt,rangeMass);
  ROOT::Fit::FillData(dataVn, fVnVsMassHisto);

  //define the 2 chi squares
  ROOT::Fit::Chi2Function chi2Mass(dataMass, wfTotMass);
  ROOT::Fit::Chi2Function chi2Vn(dataVn, wfTotVn);

  //define the global chi square
  GlobalChi2 globalChi2(chi2Mass, chi2Vn);

  //define fitter
  ROOT::Fit::Fitter fitter;
  std::cout << "\n--- [" << fName << "] Configuring fitter ---" << std::endl;

  // create before the parameter settings in order to fix or set range on them
  double parLowLim{0.}, parUpLim{0.};
  std::vector<Double_t> initpars;
  for(Int_t iPar=0; iPar<fMassTotFunc->GetNpar(); iPar++) {
    initpars.push_back(fMassTotFunc->GetParameter(iPar));
  }
  for(Int_t iVnBkgPar=0; iVnBkgPar<fNParsVnBkg; iVnBkgPar++) {
    initpars.push_back(fVnBkgFunc->GetParameter(iVnBkgPar));
    // if(vnprefit) {initpars.push_back(fVnBkgFunc->GetParameter(iVnBkgPar));}
    // else {initpars.push_back(0.05);}
  }
  initpars.push_back(0.10);                                       // initial parameter for signal vn
  if(fSecondPeak && fDoSecondPeakVn) {initpars.push_back(0.10);}  // initial parameter for second peak vn

  fitter.Config().SetParamsSettings(fNParsTotVn,initpars.data());     // set initial parameters from prefits
  for (int iPar = 0; iPar < fNParsTotVn; ++iPar) {
    fVnTotFunc->GetParLimits(iPar, parLowLim, parUpLim);
    fitter.Config().ParSettings(iPar).SetValue(fVnTotFunc->GetParameter(iPar));
    fitter.Config().ParSettings(iPar).SetLimits(parLowLim, parUpLim);
  }

  if(fMeanFixed==2 || fMeanFixedFromMassFit) {fitter.Config().ParSettings(fNParsMassBkg+1).Fix();}
  if(fSigmaFixed==2 || fSigmaFixedFromMassFit) {fitter.Config().ParSettings(fNParsMassBkg+2).Fix();}
  if(fMassSgnFuncType==k2Gaus) {
    if(fFrac2GausFixed==2 || fFrac2GausFixedFromMassFit) {fitter.Config().ParSettings(fNParsMassBkg+3).Fix();}
    if(fSigma2GausFixed==2 || fSigma2GausFixedFromMassFit) {fitter.Config().ParSettings(fNParsMassBkg+4).Fix();}
  }
  if(fSecondPeak) {
    if(fFixSecMass) {fitter.Config().ParSettings(fNParsMassBkg+fNParsMassSgn+1).Fix();}
    if(fFixSecWidth) {fitter.Config().ParSettings(fNParsMassBkg+fNParsMassSgn+2).Fix();}
    fitter.Config().ParSettings(fNParsMassBkg+fNParsMassSgn).SetLimits(0, 1000000);
    fitter.Config().ParSettings(fNParsMassBkg+fNParsMassSgn+1).SetLimits(1.99, 2.04);
  }
  if(fReflections) {
    if(fFixRflOverSig) fitter.Config().ParSettings(fNParsMassBkg+fNParsMassSgn+fNParsSec).Fix();
    if(fVnRflLimited) fitter.Config().ParSettings(fNParsTotMass+fNParsVnBkg+fNVnParsSgn-1).SetLimits(fVnRflMin,fVnRflMax);
  }
  std::cout << "\n--- [" << fName << "] Parameter fixing and limits set ---" << std::endl;
  // When sgn func is Double CrystalBall, the sgnInt, alpha and N parameters are fixed 
  // to the ones obtained from the mass prefit (alpha and N fixed to the MC from the config)
  if (fMassSgnFuncType==ETypeOfSgn::kDoubleCBSymm) {
    if (fFixSgnFromMCPrefit) {
    fitter.Config().ParSettings(fNParsMassBkg+3).Fix();   // alpha
    fitter.Config().ParSettings(fNParsMassBkg+4).Fix();   // N
    } else {
      fitter.Config().ParSettings(fNParsMassBkg+3).SetLimits(initpars[fNParsMassBkg+3] - (initpars[fNParsMassBkg+3] / 5), initpars[fNParsMassBkg+3] + (initpars[fNParsMassBkg+3] / 5));   // alpha
      fitter.Config().ParSettings(fNParsMassBkg+4).SetLimits(initpars[fNParsMassBkg+4] - (initpars[fNParsMassBkg+4] / 5), initpars[fNParsMassBkg+4] + (initpars[fNParsMassBkg+4] / 5));   // N
    }
  }
  if (fMassSgnFuncType==ETypeOfSgn::kDoubleCBAsymm) {
    if (fFixSgnFromMCPrefit) {
      fitter.Config().ParSettings(fNParsMassBkg+3).Fix();   // alpha1
      fitter.Config().ParSettings(fNParsMassBkg+4).Fix();   // N1
      fitter.Config().ParSettings(fNParsMassBkg+5).Fix();   // alpha2
      fitter.Config().ParSettings(fNParsMassBkg+6).Fix();   // N2
    } else {
      fitter.Config().ParSettings(fNParsMassBkg+3).SetLimits(initpars[fNParsMassBkg+3] - (initpars[fNParsMassBkg+3] / 5), initpars[fNParsMassBkg+3] + (initpars[fNParsMassBkg+3] / 5));   // alpha1
      fitter.Config().ParSettings(fNParsMassBkg+4).SetLimits(initpars[fNParsMassBkg+4] - (initpars[fNParsMassBkg+4] / 5), initpars[fNParsMassBkg+4] + (initpars[fNParsMassBkg+4] / 5));   // N1
      fitter.Config().ParSettings(fNParsMassBkg+5).SetLimits(initpars[fNParsMassBkg+5] - (initpars[fNParsMassBkg+5] / 5), initpars[fNParsMassBkg+5] + (initpars[fNParsMassBkg+5] / 5));   // alpha2
      fitter.Config().ParSettings(fNParsMassBkg+6).SetLimits(initpars[fNParsMassBkg+6] - (initpars[fNParsMassBkg+6] / 5), initpars[fNParsMassBkg+6] + (initpars[fNParsMassBkg+6] / 5));   // N2
    }
  }
  std::cout << "\n--- [" << fName << "] Parameter limits and fixing for signal shape applied ---" << std::endl;

  fitter.Config().MinimizerOptions().SetPrintLevel(0);
  fitter.Config().SetMinimizer("Minuit2","Migrad");
  for(Int_t iPar=0; iPar<fNParsTotVn; iPar++) {fitter.Config().ParSettings(iPar).SetName(fVnTotFunc->GetParName(iPar));}
  // fit FCN function directly
  // (specify optionally data size and flag to indicate that is a chi2 fit
  std::cout << "\n--- [" << fName << "] Running fit ---" << std::endl;
  std::cout << "Debugging of fit parameters before fit:" << std::endl;
  for(Int_t iPar=0; iPar<fNParsTotVn; iPar++) {
    std::cout << "Par " << iPar << " (" << fitter.Config().ParSettings(iPar).Name() << "): ";
    std::cout << fitter.Config().ParSettings(iPar).Value() << " [";
    std::cout << fitter.Config().ParSettings(iPar).LowerLimit() << ", ";
    std::cout << fitter.Config().ParSettings(iPar).UpperLimit() << "]" << std::endl;
  }
  Bool_t isFitOk = fitter.FitFCN(fNParsTotVn,globalChi2,0,dataMass.Size()+dataVn.Size(),kFALSE);
  if(!isFitOk) return kFALSE;
  ROOT::Fit::FitResult result = fitter.Result();
  if (!fSuppressOutput) {
    result.Print(std::cout);
  }
  if(fTemplates) {
    printf("\n ---> Templates share the vn parameter with the signal! \n");
  }
  for(Int_t iPar=0; iPar<fNParsTotVn; iPar++) {
    fVnTotFunc->SetParameter(iPar,result.Parameter(iPar));
    fVnTotFunc->SetParError(iPar,result.ParError(iPar));
    if(iPar<fNParsTotMass) {
      fMassTotFunc->SetParameter(iPar,result.Parameter(iPar));
      fMassTotFunc->SetParError(iPar,result.ParError(iPar));
    }
    if(iPar>=fNParsTotMass && iPar<fNParsTotVn-fNVnParsSgn) {
      fVnBkgFunc->SetParameter(iPar-fNParsTotMass,result.Parameter(iPar));
      fVnBkgFunc->SetParError(iPar-fNParsTotMass,result.ParError(iPar));
    }
    if(iPar>=fNParsMassBkg && iPar<fNParsMassBkg+fNParsMassSgn) {
      fMassSgnFunc->SetParameter(iPar-fNParsMassBkg,result.Parameter(iPar));
    }
    if(iPar<fNParsMassBkg) {
      fMassBkgFunc->SetParameter(iPar,result.Parameter(iPar));
      fMassBkgFunc->SetParError(iPar,result.ParError(iPar));
      if(fReflections) {
        fMassBkgRflFunc->SetParameter(iPar,result.Parameter(iPar));
        fMassBkgRflFunc->SetParError(iPar,result.ParError(iPar));
      }
    }
    if(fReflections && (iPar>=fNParsMassBkg+fNParsMassSgn+fNParsSec && iPar<fNParsMassBkg+fNParsMassSgn+fNParsSec+fNParsRfl)) {
      fMassRflFunc->SetParameter(iPar-(fNParsMassBkg+fNParsMassSgn+fNParsSec),result.Parameter(iPar));
      fMassRflFunc->SetParError(iPar-(fNParsMassBkg+fNParsMassSgn+fNParsSec),result.ParError(iPar));
      fMassBkgRflFunc->SetParameter(iPar-(fNParsMassSgn+fNParsSec),result.Parameter(iPar));
      fMassBkgRflFunc->SetParError(iPar-(fNParsMassSgn+fNParsSec),result.ParError(iPar));
    }
    if(fSecondPeak && (iPar>=fNParsMassBkg+fNParsMassSgn && iPar<fNParsMassBkg+fNParsMassSgn+fNParsSec)) {
      fMassSecPeakFunc->SetParameter(iPar-(fNParsMassBkg+fNParsMassSgn),result.Parameter(iPar));
      fMassSecPeakFunc->SetParError(iPar-(fNParsMassBkg+fNParsMassSgn),result.ParError(iPar));

      if (fFixFracSecWidth) {
        fMassSecPeakFunc->SetParameter(2+iPar-(fNParsMassBkg+fNParsMassSgn),fSecWidthFrac*result.Parameter(fNParsMassBkg+2));
        fMassSecPeakFunc->SetParError(2+iPar-(fNParsMassBkg+fNParsMassSgn),fSecWidthFrac*result.ParError(fNParsMassBkg+2));
      }
    }
  }

  // Save parameters to histogram
  fSimFitParsHisto = new TH1F(Form("fSimFitParsHisto_%s", fName.c_str()),";Parameter;Value",fNParsTotVn,0,fNParsTotVn);
  for (size_t iPar = 0; iPar < (size_t)fNParsTotVn; iPar++) {
    fSimFitParsHisto->SetBinContent(iPar+1, result.Parameter(iPar));
    fSimFitParsHisto->SetBinError(iPar+1, result.ParError(iPar));
    fSimFitParsHisto->GetXaxis()->SetBinLabel(iPar+1, result.ParName(iPar).c_str());
  }
  fSignalParsHisto = new TH1F(Form("fSignalParsHisto_%s", fName.c_str()),";Parameter;Value",fNParsMassSgn,0,fNParsMassSgn);
  for (size_t iPar = 0; iPar < (size_t)fNParsMassSgn; iPar++) {
    fSignalParsHisto->SetBinContent(iPar+1, fMassSgnFunc->GetParameter(iPar));
    fSignalParsHisto->SetBinError(iPar+1, fMassSgnFunc->GetParError(iPar));
    fSignalParsHisto->GetXaxis()->SetBinLabel(iPar+1, fMassSgnFunc->GetParName(iPar));
  }

  if(fTemplates) {
    switch (fAnchorTemplsMode) {
      case TemplAnchorMode::AnchorToFirst:
        fMassTemplFunc = new TF1("fMassTemplFunc",this,&VnVsMassFitter::MassTemplates,fMassMin,fMassMax,1,"VnVsMassFitter","MassTemplates");
        fMassTemplFunc->SetParameter(0, result.Parameter(fNParsTotMass - fNParsTempls));
        break;
      case TemplAnchorMode::AnchorToSgn:
        fMassTemplFunc = new TF1("fMassTemplFunc",this,&VnVsMassFitter::MassTemplates,fMassMin,fMassMax,1,"VnVsMassFitter","MassTemplates");
        fMassTemplFunc->SetParameter(0, result.Parameter(fNParsMassBkg));
        break;
      default:
        std::cerr << "Error: Invalid fAnchorTemplsMode value!" << std::endl;
        break;
    }
  }

  fVn = fVnTotFunc->GetParameter(fVnTotFunc->GetNpar()-fNVnParsSgn);
  fVnUncertainty = fVnTotFunc->GetParError(fVnTotFunc->GetNpar()-fNVnParsSgn);
  fRawYield = fVnTotFunc->GetParameter(fNParsMassBkg)/fMassHisto->GetBinWidth(10);
  fRawYieldUncertainty = fVnTotFunc->GetParError(fNParsMassBkg)/fMassHisto->GetBinWidth(10);
  fMean = fVnTotFunc->GetParameter(fNParsMassBkg+1);
  fMeanUncertainty = fVnTotFunc->GetParError(fNParsMassBkg+1);
  fSigma = fVnTotFunc->GetParameter(fNParsMassBkg+2);
  fSigmaUncertainty = fVnTotFunc->GetParError(fNParsMassBkg+2);
  fChiSquare = result.MinFcnValue();
  fNDF = result.Ndf();
  fProb = result.Prob();

  // Get Vn components to be drawn
  int idxParMassTemplsScaling = fNParsMassBkg+fNParsMassSgn+fNParsSec+fNParsRfl;
  int idxParVnSgn = idxParMassTemplsScaling+fNParsTempls+fNParsVnBkg;
  double vnSgn = result.Parameter(idxParVnSgn);
  fVnCompsDraw.push_back(new TF1(Form("vnSgn_%s", fName.c_str()),
                      [this, vnSgn] (double *x, double *par) {
                      return (vnSgn * this->fMassSgnFunc->Eval(x[0])) / (this->fMassTotFunc->Eval(x[0]));
                    }, fMassMin, fMassMax, 0));
  fVnCompsDraw.push_back(new TF1(Form("vnBkg_%s", fName.c_str()),
                      [this] (double *x, double *par) {
                      return (this->fVnBkgFunc->Eval(x[0]) * this->fMassBkgFunc->Eval(x[0])) / (this->fMassTotFunc->Eval(x[0]));
                    }, fMassMin, fMassMax, 0));
  if(fDoSecondPeakVn) {
    fVnSecPeak = fVnTotFunc->GetParameter(fVnTotFunc->GetNpar()-1);
    fVnSecPeakUncertainty = fVnTotFunc->GetParError(fVnTotFunc->GetNpar()-1);
    fVnSecPeakFunc = new TF1(Form("vnSecPeak_%s", fName.c_str()),
                        [this] (double *x, double *par) {
                        return (this->fVnSecPeak * this->fMassSecPeakFunc->Eval(x[0])) / (this->fMassTotFunc->Eval(x[0]));
                      }, fMassMin, fMassMax, 0);
    fVnCompsDraw.push_back(fVnSecPeakFunc); 
  }
  if(fTemplates) {
    double templScalingPar{0.};
    switch (fAnchorTemplsMode) {
      case TemplAnchorMode::AnchorToFirst:
        templScalingPar = result.Parameter(idxParMassTemplsScaling);
        break;
      case TemplAnchorMode::AnchorToSgn:
        templScalingPar = result.Parameter(this->fNParsMassBkg);
        break;
      default:
        std::cerr << "Error: Invalid fAnchorTemplsMode value!" << std::endl;
        break;
    }
    for(size_t iTempl=0; iTempl<fHistoTemplates.size(); iTempl++) {
      fMassTemplatesDraw.push_back(new TF1(Form("fTempl_%li", iTempl),
            [this, iTempl, templScalingPar] (double *x, double *par) {
        double xval = x[0];
        double pdfVal = fHistoTemplates[iTempl]->Interpolate(xval);
          // std::cout << "PDF value at mass = " << xval << " is: " << pdfVal << std::endl;
          return templScalingPar * fRelWeights[iTempl] * pdfVal;
        }, fMassMin, fMassMax, 0));
      fVnCompsDraw.push_back(new TF1(Form("fVnTempl_%li", iTempl),
                          [this, iTempl, vnSgn] (double *x, double *par) {
                          return (vnSgn * this->fMassTemplatesDraw[iTempl]->Eval(x[0])) / (this->fMassTotFunc->Eval(x[0]));
                  }, fMassMin, fMassMax, 0));
    }
  }
  if (prefitsStatus == 2) {
    return 2;
  } else {
    return kTRUE;
  }
}

//________________________________________________________________
TH1F* VnVsMassFitter::GetPullDistribution() {
  if(!fMassTotFunc) {
      throw std::invalid_argument("Fit not performed, pulls cannot be calculated!");
  }
  std::vector<double> pulls;
  for(int iBin=0; iBin<this->fMassHisto->GetNbinsX(); iBin++) {    
      if(this->fMassHisto->GetBinCenter(iBin+1) >= this->fMassMin &&
         this->fMassHisto->GetBinCenter(iBin+1) <= this->fMassMax) {
              pulls.push_back( (this->fMassHisto->GetBinContent(iBin+1) - this->GetMassTotFitFunc()->Eval(this->fMassHisto->GetBinCenter(iBin+1))) /         
                                this->fMassHisto->GetBinError(iBin+1));
      }
  }
  TH1F *histoPulls = new TH1F("hPulls", "hPulls;M (GeV/c); Data - fit", pulls.size(), this->fMassMin, this->fMassMax);
  for(int iBin=0; iBin<this->fMassHisto->GetNbinsX(); iBin++) {
      histoPulls->SetBinContent(iBin+1, pulls[iBin]);
  }

  return histoPulls;
}

//________________________________________________________________
Bool_t VnVsMassFitter::PrefitVnSidebands() {

  fIsVnSidebandFit = kTRUE;
  TString opt = "R,S,N";     // Range + return TFitResultPtr
  if (fSuppressOutput) {opt += ",Q";}   // Quiet
  TFitResultPtr res = fVnVsMassHisto->Fit(Form("fVnBkgFunc_%s", fName.c_str()), opt.Data());
  fIsVnSidebandFit = kFALSE;

  if (!res.Get() || !res->IsValid()) {
    printf("PrefitVnSidebands failed\n");
    return kFALSE;
  }
  for (size_t iPar = 0; iPar < (size_t)fNParsVnBkg; iPar++) {
    double parLowLim{-1.}, parUpLim{-1.};
    fVnBkgFunc->GetParLimits(iPar, parLowLim, parUpLim);
    fInitFuncPars[res->ParName(iPar)] = {res->Parameter(iPar), parLowLim, parUpLim};
  }
  return kTRUE;
}

//________________________________________________________________
void VnVsMassFitter::DefineNumberOfParameters() {

  switch(fMassSgnFuncType) {
    case 0: //single gaus
      fNParsMassSgn=3;
      break;
    case 1: //double gaus
      fNParsMassSgn=5;
      break;
    case 3:
      fNParsMassSgn=7;
      break;
    case 4:
      fNParsMassSgn=5;
      break;
    default:
      printf("Error in computing fMassSgnFuncType: check fMassSgnFuncType");
      break;
  }

  switch(fMassBkgFuncType) {
    case 0: //expo
      fNParsMassBkg=2;
      break;
    case 1: //lin
      fNParsMassBkg=2;
      break;
    case 2: //pol2
      fNParsMassBkg=3;
      break;
    case 3: //no bkg
      fNParsMassBkg=1;
      break;
    case 4: //power law
      fNParsMassBkg=2;
      break;
    case 5: //power expo
      fNParsMassBkg=2;
      break;
    case 6: //high degree pol
      fNParsMassBkg=fPolDegreeBkg+1;
      break;
    default:
      printf("Error in computing fNParsMassBkg: check fMassBkgFuncType");
      break;
  }

  switch(fVnBkgFuncType) {
    case 0: //expo
      fNParsVnBkg=2;
      break;
    case 1: //lin
      fNParsVnBkg=2;
      break;
    case 2: //pol2
      fNParsVnBkg=3;
      break;
    case 6: //high degree pol
      fNParsVnBkg=fPolDegreeVnBkg+1;
      break;
    default:
      printf("Error in computing fNParsVnBkg: check fVnBkgFuncType");
      break;
  }

  fNParsVnSgn=1;

  if(fReflections) {
    fNParsRfl=1;
    if(fVnRflOpt==3) fNParsVnRfl=1;
    else fNParsVnRfl=0;
  }
  else {
    fNParsRfl=0;
    fNParsVnRfl=0;
  }

  if (fTemplates) {
    switch (fAnchorTemplsMode) {
      case TemplAnchorMode::AnchorToFirst:
        fNParsTempls = 1;
        break;
      case TemplAnchorMode::AnchorToSgn:
        fNParsTempls = 0;
        break;
      default:
        std::cerr << "Error: Invalid fAnchorTemplsMode value!" << std::endl;
    }
  }

  if(fSecondPeak) {
    fNParsSec=3;
    if (fFixFracSecWidth) {
      fNParsSec-=1;
    }
    if (fFixVnSecPeakToSgn) {
      fNParsVnSecPeak=0;
    }
    else {
      fNParsVnSecPeak=1;
    }
  }
  else {
    fNParsSec=0;
    fNParsVnSecPeak=0;
  }
  fNParsTotMass = fNParsMassSgn+fNParsMassBkg+fNParsSec+fNParsRfl+fNParsTempls;
  fNVnParsSgn = 1; // One Vn par for the signal
  std::cout << "fFixVnSecPeakToSgn: " << fFixVnSecPeakToSgn << std::endl;
  if(fSecondPeak && fDoSecondPeakVn && !fFixVnSecPeakToSgn) {fNVnParsSgn+=1;} // One more Vn par for the second peak
  if(fReflections && fVnRflOpt==kFreePar) {fNVnParsSgn+=1;}  // One more Vn par for the reflections
  fNParsTotVn = fNParsTotMass+fNParsVnBkg+fNVnParsSgn;  // 1 for signal Vn
  std::cout << "[Parameter summary]" << std::endl;
  std::cout << "---> fNParsMassSgn: " << fNParsMassSgn << std::endl;
  std::cout << "---> fNParsMassBkg: " << fNParsMassBkg << std::endl;
  std::cout << "---> fNParsSec: " << fNParsSec << std::endl;
  std::cout << "---> fNParsRfl: " << fNParsRfl << std::endl;
  std::cout << "---> fNParsTempls: " << fNParsTempls << std::endl;
  std::cout << "---> fNParsTotMass: " << fNParsTotMass << std::endl;
  std::cout << "---> fNParsVnBkg: " << fNParsVnBkg << std::endl;
  std::cout << "---> fNVnParsSgn: " << fNVnParsSgn << std::endl;
  std::cout << "---> fNParsTotVn: " << fNParsTotVn << std::endl;
  std::cout << std::endl;
}

//_________________________________________________________________________
void VnVsMassFitter::Signal(Double_t nOfSigma,Double_t &signal,Double_t &errsignal) const {
  /// Return signal integral in mean +- n sigma

  Double_t minMass=fMean-nOfSigma*fSigma;
  Double_t maxMass=fMean+nOfSigma*fSigma;
  Signal(minMass,maxMass,signal,errsignal);
  return;
}

//_________________________________________________________________________
void VnVsMassFitter::Signal(Double_t min, Double_t max, Double_t &signal,Double_t &errsignal) const {
  /// Return signal integral in a range

  if(!fMassSgnFunc) {signal=-1; errsignal=0; return;}

  signal=fMassSgnFunc->Integral(min, max)/(Double_t)fMassHisto->GetBinWidth(1);
  errsignal=(fRawYieldUncertainty/fRawYield)*signal;/*assume relative error is the same as for total integral*/

  return;
}

//___________________________________________________________________________
void VnVsMassFitter::Background(Double_t nOfSigma,Double_t &background,Double_t &errbackground) const {
  /// Return background integral in mean +- n sigma

  Double_t minMass=fMean-nOfSigma*fSigma;
  Double_t maxMass=fMean+nOfSigma*fSigma;
  Background(minMass,maxMass,background,errbackground);

  return;
}

//___________________________________________________________________________
void VnVsMassFitter::Background(Double_t min, Double_t max, Double_t &background,Double_t &errbackground) const {
  /// Return background integral in a range

  if(!fMassBkgFunc) {background=-1; errbackground=0; return;}

  Double_t intB=fMassBkgFunc->GetParameter(0);
  Double_t intBerr=fMassBkgFunc->GetParError(0);
  //relative error evaluation: from histo

  Int_t leftBand=fMassHisto->FindBin(fMean-4*fSigma);
  Int_t rightBand=fMassHisto->FindBin(fMean+4*fSigma);
  intB=fMassHisto->Integral(1,leftBand)+fMassHisto->Integral(rightBand,fMassHisto->GetNbinsX());
  Double_t sum2=0;
  for(Int_t iBin=1; iBin<=leftBand; iBin++){
    sum2+=fMassHisto->GetBinError(iBin)*fMassHisto->GetBinError(iBin);
  }
  for(Int_t iBin=rightBand; iBin<=fMassHisto->GetNbinsX(); iBin++){
    sum2+=fMassHisto->GetBinError(iBin)*fMassHisto->GetBinError(iBin);
  }

  intBerr=TMath::Sqrt(sum2);

  background=fMassBkgFunc->Integral(min,max)/(Double_t)fMassHisto->GetBinWidth(1);
  errbackground=intBerr/intB*background;

  return;
}

//__________________________________________________________________________
void VnVsMassFitter::Significance(Double_t nOfSigma,Double_t &significance,Double_t &errsignificance) const  {
  /// Return significance in mean +- n sigma

  Double_t minMass=fMean-nOfSigma*fSigma;
  Double_t maxMass=fMean+nOfSigma*fSigma;
  Significance(minMass, maxMass, significance, errsignificance);

  return;
}

//__________________________________________________________________________
void VnVsMassFitter::Significance(Double_t min, Double_t max, Double_t &significance,Double_t &errsignificance) const {
  /// Return significance integral in a range

  Double_t background,errbackground;
  Background(min,max,background,errbackground);

  if (fRawYield+background <= 0.){
    significance=-1;
    errsignificance=0;
    return;
  }

  Double_t errSigSq=fRawYieldUncertainty*fRawYieldUncertainty;
  Double_t errBkgSq=errbackground*errbackground;
  Double_t sigPlusBkg=fRawYield+background;
  if (sigPlusBkg>0. && fRawYield>0.) {
    significance =  fRawYield/TMath::Sqrt(fRawYield+background);
    errsignificance = significance*TMath::Sqrt((errSigSq+errBkgSq)/(4.*sigPlusBkg*sigPlusBkg)+(background/sigPlusBkg)*errSigSq/fRawYield/fRawYield);
  } else {
    significance=0.;
    errsignificance=0.;
  }

  return;
}

//__________________________________________________________________________
Double_t VnVsMassFitter::DoubleSidedCBAsymm(double x, double mu, double sigma, double a1, double n1, double a2, double n2) {
  double t = (x - mu) / sigma;
  double absAlphaL = std::abs(a1);
  double absAlphaR = std::abs(a2);

  // 1. Calculate the raw (unnormalized) value
  double val = 0;
  if (t < -absAlphaL) {
      double a = std::pow(n1 / absAlphaL, n1) * std::exp(-0.5 * absAlphaL * absAlphaL);
      double b = n1 / absAlphaL - absAlphaL;
      val = a / std::pow(b - t, n1);
  } 
  else if (t > absAlphaR) {
      double a = std::pow(n2 / absAlphaR, n2) * std::exp(-0.5 * absAlphaR * absAlphaR);
      double b = n2 / absAlphaR - absAlphaR;
      val = a / std::pow(b + t, n2);
  } 
  else {
      val = std::exp(-0.5 * t * t);
  }

  // 2. Calculate the Normalization Factor (The total integral)
  // Gaussian core integral: sigma * sqrt(pi/2) * [erf(aL/sqrt2) + erf(aR/sqrt2)]
  double term_gauss = sigma * std::sqrt(M_PI / 2.0) * (std::erf(absAlphaL / std::sqrt(2.0)) + std::erf(absAlphaR / std::sqrt(2.0)));

  // Left tail integral: sigma * (n1/|a1|) * exp(-0.5*a1^2) / (n1 - 1)
  double term_left = sigma * (n1 / absAlphaL) * (1.0 / (n1 - 1.0)) * std::exp(-0.5 * absAlphaL * absAlphaL);

  // Right tail integral: sigma * (n2/|a2|) * exp(-0.5*a2^2) / (n2 - 1)
  double term_right = sigma * (n2 / absAlphaR) * (1.0 / (n2 - 1.0)) * std::exp(-0.5 * absAlphaR * absAlphaR);

  return val / (term_gauss + term_left + term_right);
}

//__________________________________________________________________________
Double_t VnVsMassFitter::DoubleSidedCBSymm(double x, double mu, double sigma, double a, double n) {
  // t is the distance from the mean in units of sigma
  double t = (x - mu) / sigma;
  double absAlpha = std::abs(a);

  // 1. Calculate the raw (unnormalized) value
  double val = 0;
  if (std::abs(t) <= absAlpha) {
      // Gaussian Core
      val = std::exp(-0.5 * t * t);
  } else {
      // Power-law Tails (Symmetric)
      double a = std::pow(n / absAlpha, n) * std::exp(-0.5 * absAlpha * absAlpha);
      double b = n / absAlpha - absAlpha;
      val = a / std::pow(b + std::abs(t), n);
  }

  // 2. Analytical Normalization Factor
  // Gaussian core integral: sigma * sqrt(pi/2) * 2 * erf(alpha/sqrt2)
  double term_gauss = sigma * std::sqrt(M_PI / 2.0) * (2.0 * std::erf(absAlpha / std::sqrt(2.0)));

  // Tails integral: 2 * [sigma * (n/|alpha|) * exp(-0.5*alpha^2) / (n - 1)]
  // We multiply by 2 because the left and right tails are identical
  double term_tails = 2.0 * sigma * (n / absAlpha) * (1.0 / (n - 1.0)) * std::exp(-0.5 * absAlpha * absAlpha);

  return val / (term_gauss + term_tails);
}

//________________________________________________________________
Double_t VnVsMassFitter::GetGausPDF(Double_t x, Double_t mean, Double_t sigma) {

  return TMath::Gaus(x,mean,sigma,kTRUE);
}

//________________________________________________________________
Double_t VnVsMassFitter::GetExpoPDF(Double_t x, Double_t coeff, Bool_t isnorm) {

  Double_t shiftedX = x - fMassMin;
  Double_t shiftedMax = fMassMax - fMassMin;
  if(isnorm) {
      Double_t norm = (TMath::Exp(coeff * shiftedMax) - 1.0) / coeff;
      return TMath::Exp(coeff * shiftedX) / norm;
  }
  else return TMath::Exp(coeff*x);
}

//________________________________________________________________
Double_t VnVsMassFitter::GetPolPDF(Double_t x, Double_t *pars, Int_t order, Bool_t isnorm) {
  Double_t xMid = (fMassMax + fMassMin) / 2.0;
  Double_t xShift = x - xMid;
  Double_t delta = (fMassMax - fMassMin) / 2.0; 

  if (isnorm) {
    Double_t norm = 0;
    Double_t funcValue = 0;

    switch(order) {
      case 0: // Constant: pars[0] is the total integral
        norm = 2.0 * delta;
        return pars[0] / norm;

      case 1: // Linear: f(x) = pars[0] * [1 + pars[1]*(x-xMid)] / Norm
        norm = 2.0 * delta;
        funcValue = 1.0 + pars[1] * xShift;
        return (pars[0] / norm) * funcValue;

      case 2: // Quadratic: f(x) = pars[0] * [1 + pars[1]*x' + pars[2]*x'^2] / Norm
        norm = 2.0 * delta + (2.0/3.0) * pars[2] * TMath::Power(delta, 3);
        funcValue = 1.0 + pars[1] * xShift + pars[2] * xShift * xShift;
        return (pars[0] / norm) * funcValue;
    }
  } else {
    // Standard unnormalized: pars[0] is the intercept at xMid
    switch(order) {
      case 0: return pars[0];
      case 1: return pars[0] + pars[1] * xShift;
      case 2: return pars[0] + pars[1] * xShift + pars[2] * xShift * xShift;
    }
  }
  return 0;
}

//________________________________________________________________
Double_t VnVsMassFitter::GetPowerFuncPDF(Double_t x, Double_t *pars) {

  Double_t mpi = TDatabasePDG::Instance()->GetParticle(211)->Mass();
  return pars[0]*(pars[1]+1.)/(TMath::Power(fMassMax-mpi,pars[1]+1.)-TMath::Power(fMassMin-mpi,pars[1]+1.))*TMath::Power(x-mpi,pars[1]);
}

//________________________________________________________________
Double_t VnVsMassFitter::GetPowerExpoPDF(Double_t x, Double_t *pars) {

  Double_t mpi = TDatabasePDG::Instance()->GetParticle(211)->Mass();
  return pars[0]*TMath::Sqrt(x - mpi)*TMath::Exp(-1.*pars[1]*(x-mpi));
}

//________________________________________________________________
Double_t VnVsMassFitter::GetHigherPolFuncPDF(Double_t x, Double_t *pars, Int_t Ndeg, Bool_t isnorm) {

  Double_t total=pars[0];
  for(Int_t iT=1; iT<=Ndeg; iT++){
    if(isnorm) total+=pars[iT]*TMath::Power(x-fMassParticle,iT)/TMath::Factorial(iT);
    else total+=pars[iT]*TMath::Power(x,iT);
  }
  return total;
}

//__________________________________________________________________________
Int_t VnVsMassFitter::PrefitMass(){
  /// Prefit the combinatorial background
  /// returns 0 if the fit fails
  /// returns 1 if the fit succeeds
  /// returns 2 if the second peak is removed due to low counts

  TString opt = "R,S,+,0,N";
  if (fSuppressOutput) opt += ",Q";   // Quiet
  TFitResultPtr res = fMassHisto->Fit(Form("fMassTotFunc_%s", fName.c_str()),opt.Data());
  std::cout << "PrefitMass fit done" << std::endl;

  if (!res.Get() || !res->IsValid()) {
    printf("PrefitMass failed\n");
    return 0;
  }

  // If second peak is present, quantify bin counting between fMassRangeMinSecPeak
  // and fMassRangeMaxSecPeak and, if too low, remove it
  if (fSecondPeak) {
    for (Int_t iPar = 0; iPar < fNParsMassBkg; iPar++) {
      std::cout << "Setting prefit parameter " << iPar << " from " << fMassBkgFunc->GetParameter(iPar) << " to " << fMassBkgFunc->GetParameter(iPar) << std::endl;
      fMassBkgFunc->SetParameter(iPar, fMassTotFunc->GetParameter(iPar));
    }
    Int_t binMin = fMassHisto->FindBin(fMassRangeMinSecPeak);
    Int_t binMax = fMassHisto->FindBin(fMassRangeMaxSecPeak);
    Double_t countsInSecPeakRegion = fMassHisto->Integral(binMin, binMax);
    // Subtract the background counts in the same region
    Double_t bkgCountsInSecPeakRegion = fMassBkgFunc->Integral(fMassRangeMinSecPeak, fMassRangeMaxSecPeak) / fMassHisto->GetBinWidth(1);
    Double_t netCounts = countsInSecPeakRegion - bkgCountsInSecPeakRegion;
    std::cout << "Counts in second peak region: " << countsInSecPeakRegion << ", background counts: " << bkgCountsInSecPeakRegion
              << ", net counts: " << netCounts << std::endl;
    if (netCounts < fMinCountsForSecPeak) {
      printf("PrefitMass: Second peak will be removed due to low counts in the specified region (%.2f < %.2i)\n",
             netCounts, fMinCountsForSecPeak);
      return 2; // Indicate that the second peak has to be removed
    }
  }

  for (size_t iPar = 0; iPar < (size_t)fNParsTotMass; iPar++) {
    double parLowLim{-1.}, parUpLim{-1.};
    fMassTotFunc->GetParLimits(iPar, parLowLim, parUpLim);
    fInitFuncPars[res->ParName(iPar)] = {res->Parameter(iPar), parLowLim, parUpLim};
  }
  return 1;
}

//__________________________________________________________________________
Int_t VnVsMassFitter::PrefitCombBkg(){
  /// Prefit the combinatorial background
  /// returns 0 if the fit fails
  /// returns 1 if the fit succeeds
  fIsMassSidebandFit = kTRUE;
  TString opt = "R,S,+,0,N";     // Range + return TFitResultPtr
  if (fSuppressOutput) opt += ",Q";   // Quiet
  TFitResultPtr res = fMassHisto->Fit(Form("fMassBkgFunc_%s", fName.c_str()), opt.Data());
  fIsMassSidebandFit = kFALSE;

  if (!res.Get() || !res->IsValid()) {
    printf("PrefitCombBkg failed\n");
    return kFALSE;
  }
  for (size_t iPar = 0; iPar < (size_t)fNParsMassBkg; iPar++) {
    double parLowLim{-1.}, parUpLim{-1.};
    fMassBkgFunc->GetParLimits(iPar, parLowLim, parUpLim);
    fInitFuncPars[res->ParName(iPar)] = {res->Parameter(iPar), parLowLim, parUpLim};
  }
  return kTRUE;
}

//__________________________________________________________________________
Int_t VnVsMassFitter::PrefitSignal(){
  /// Prefit the signal from MC
  /// returns 0 if the fit fails
  /// returns 1 if the fit succeeds

  TString opt = "R,S,+,0,N";     // Range + return TFitResultPtr
  if (fSuppressOutput) opt += ",Q";   // Quiet

  Double_t integralHisto=fHistoSgnPrefit->Integral(fHistoSgnPrefit->FindBin(fMassMin),fHistoSgnPrefit->FindBin(fMassMax),"width");
  fMassSgnFunc->SetParameter(0, integralHisto);
  fMassSgnFunc->SetParLimits(0, 0, 10000);
  TFitResultPtr res = fHistoSgnPrefit->Fit(Form("fMassSgnFunc_%s", fName.c_str()), opt.Data());

  if (!res.Get() || !res->IsValid()) {
    printf("PrefitSignal failed\n");
    return kFALSE;
  }
  // Update the init parameters map
  fPrefitParsHisto = new TH1F(Form("fPrefitParsHisto_%s",fName.c_str()),"Prefit parameters histogram",fNParsMassSgn,0,fNParsMassSgn);
  for (size_t iPar = 0; iPar < (size_t)fNParsMassSgn; iPar++) {
    fPrefitParsHisto->SetBinContent(iPar+1, res->Parameter(iPar));
    fPrefitParsHisto->SetBinError(iPar+1, res->ParError(iPar));
    fPrefitParsHisto->GetXaxis()->SetBinLabel(iPar+1, res->ParName(iPar).c_str());
    double parLowLim{res->Parameter(iPar)}, parUpLim{res->Parameter(iPar)};
    if (iPar <= 2 || !fFixSgnFromMCPrefit) {
      fMassSgnFunc->GetParLimits(iPar, parLowLim, parUpLim);  // Update the parameter limits from the function
    }
    fInitFuncPars[res->ParName(iPar)] = {res->Parameter(iPar), parLowLim, parUpLim};
  }

  return kTRUE;
}

//________________________________________________________________
Double_t VnVsMassFitter::MassSignal(Double_t *m, Double_t *pars) {

  switch(fMassSgnFuncType) {
    case 0:
      return pars[0]*GetGausPDF(m[0],pars[1],pars[2]);
      break;
    case 1:
      return pars[0]*(pars[3]*GetGausPDF(m[0],pars[1],pars[2])+(1-pars[3])*GetGausPDF(m[0],pars[1],pars[4]));
      break;
    case 3:
      return pars[0]*DoubleSidedCBAsymm(m[0],pars[1],pars[2],pars[3],pars[4],pars[5],pars[6]);
      break;
    case 4:
      return pars[0]*DoubleSidedCBSymm(m[0],pars[1],pars[2],pars[3],pars[4]);
      break;
  }

  return 0;
}

//________________________________________________________________
Double_t VnVsMassFitter::MassBkg(Double_t *m, Double_t *pars) {

  if (fIsMassSidebandFit) {
    Double_t peakMean = fMassSgnFunc->GetParameter(1);
    Double_t peakSigma = fMassSgnFunc->GetParameter(2);
    if (m[0] > peakMean - fNSigmaForSB*peakSigma && m[0] < peakMean + fNSigmaForSB*peakSigma) {
      TF1::RejectPoint();
    }
  }

  switch(fMassBkgFuncType) {
    case 0: //exponential
      return pars[0]*GetExpoPDF(m[0],pars[1],kTRUE);
      break;
    case 1: //linear
      return GetPolPDF(m[0],pars,1,kTRUE);
      break;
    case 2: //parabolic
      return GetPolPDF(m[0],pars,2,kTRUE);
      break;
    case 3: //constant
      return GetPolPDF(m[0],pars,0,kTRUE);
      break;
    case 4: //power law
      return GetPowerFuncPDF(m[0],pars);
      break;
    case 5: //power law expo
      return GetPowerExpoPDF(m[0],pars);
      break;
    case 6: //higher order (>=3) polinomial
      return GetHigherPolFuncPDF(m[0],pars,fPolDegreeBkg,kTRUE);
      break;
  }
  return 0;
}

//_________________________________________________________________________
Double_t VnVsMassFitter::MassRfl(Double_t *m,Double_t *pars){
  /// Fit function for reflections:
  /// D0->Kpi decays with swapped mass assignment to pion and kaon decay tracks
  if(!fHistoTemplRfl) return 0;

  Int_t bin =fHistoTemplRfl->FindBin(m[0]);
  Double_t value=fHistoTemplRfl->GetBinContent(bin);
  Int_t binmin=fHistoTemplRfl->FindBin(fMassMin*1.00001);
  Int_t binmax=fHistoTemplRfl->FindBin(fMassMax*0.99999);
  Double_t norm=fHistoTemplRfl->Integral(binmin,binmax)*fHistoTemplRfl->GetBinWidth(bin);
  if(TMath::Abs(value)<1.e-14 && fSmoothRfl){// very rough, assume a constant trend, much better would be a pol1 or pol2 over a broader range
    value+=fHistoTemplRfl->GetBinContent(bin-1)+fHistoTemplRfl->GetBinContent(bin+1);
    value/=3.;
  }

  return pars[0]*value/norm*fRawYieldHelp*fMassHisto->GetBinWidth(1);
}

//_________________________________________________________________________
Double_t VnVsMassFitter::MassTemplates(Double_t *m,Double_t *pars){
  Double_t totalTemplates = 0.;
  switch(fAnchorTemplsMode) {
    case TemplAnchorMode::AnchorToFirst:
      for (size_t iTempl=0; iTempl<fHistoTemplates.size(); iTempl++) {
        totalTemplates += pars[0]*fRelWeights[iTempl]*fHistoTemplates[iTempl]->Interpolate(m[0]);
      }
      break;
    case TemplAnchorMode::AnchorToSgn:
      for (size_t iTempl=0; iTempl<fHistoTemplates.size(); iTempl++) {
        totalTemplates += pars[0]*fRelWeights[iTempl]*fHistoTemplates[iTempl]->Interpolate(m[0]);
      }
      break;
    default:
      std::cerr << "Error: Invalid fAnchorTemplsMode value!" << std::endl;
      break;
  }
  return totalTemplates;
}

//_________________________________________________________________________
Double_t VnVsMassFitter::MassBkgRfl(Double_t *m,Double_t *pars){

  if(!fHistoTemplRfl) {return MassBkg(m,pars);}
  else {
    //bkg mass parameters
    const Int_t nBkgPars = fNParsMassBkg;
    Double_t bkgpars[nBkgPars];
    for(Int_t iPar=0; iPar<fNParsMassBkg; iPar++) {bkgpars[iPar] = pars[iPar];}
    //reflection parameters
    Double_t rflpars[1]; //maximum number of parameters for rfl = 1 for the implemented functions
    for(Int_t iPar=0; iPar<fNParsRfl; iPar++) {rflpars[iPar] = pars[iPar+fNParsMassBkg];}
    return MassBkg(m,bkgpars)+MassRfl(m,rflpars);
  }
}

//_________________________________________________________________________
Double_t VnVsMassFitter::MassSecondPeak(Double_t *m,Double_t *pars){
  /// Fit function for a second gaussian peak
  /// To be used, e.g., for D+->KKpi in the Ds mass spectrum

  return pars[0]*GetGausPDF(m[0],pars[1],pars[2]);
}

//________________________________________________________________
Double_t VnVsMassFitter::vnBkgFunc(Double_t *m, Double_t *pars) {

  if (fIsVnSidebandFit) {
    Double_t peakMean = fMassSgnFunc->GetParameter(1);
    Double_t peakSigma = fMassSgnFunc->GetParameter(2);
    if (m[0] > peakMean - fNSigmaForSB*peakSigma && m[0] < peakMean + fNSigmaForSB*peakSigma) {
      TF1::RejectPoint();
    }
  }

  switch(fVnBkgFuncType) {
    case 0: //expo
      return pars[0]*GetExpoPDF(m[0],pars[1],kFALSE);
      break;
    case 1: //linear
      return GetPolPDF(m[0],pars,1,kFALSE);
      break;
    case 2: //parabolic
      return GetPolPDF(m[0],pars,2,kFALSE);
      break;
    case 6: //higher order (>=3) polinomial
      return GetHigherPolFuncPDF(m[0],pars,fPolDegreeVnBkg,kFALSE);
      break;
  }
  return 0;
}

//________________________________________________________________
Double_t VnVsMassFitter::MassFunc(Double_t *m, Double_t *pars) {

  //bkg mass parameters
  const Int_t nBkgPars = fNParsMassBkg;
  Double_t bkgpars[nBkgPars];
  for(Int_t iPar=0; iPar<fNParsMassBkg; iPar++) {bkgpars[iPar] = pars[iPar];}
  //signal mass parameters
  Double_t sgnpars[7]; //maximum number of parameters for sgn = 7 for the implemented functions
  for(Int_t iPar=0; iPar<fNParsMassSgn; iPar++) {sgnpars[iPar] = pars[iPar+fNParsMassBkg];}
  //second peak parameters
  Double_t secpeakpars[3]; //maximum number of parameters for second peak = 3 for the implemented functions
  for(Int_t iPar=0; iPar<fNParsSec; iPar++) {
    secpeakpars[iPar] = pars[iPar+fNParsMassBkg+fNParsMassSgn];
  }
  if (fFixFracSecWidth) {
    secpeakpars[2] = sgnpars[2]*fSecWidthFrac; // sigma * frac
  }
  //reflection parameters
  Double_t rflpars[1]; //maximum number of parameters for rfl = 1 for the implemented functions
  for(Int_t iPar=0; iPar<fNParsRfl; iPar++) {rflpars[iPar] = pars[iPar+fNParsMassBkg+fNParsMassSgn+fNParsSec];}

  Double_t total = MassSignal(m,sgnpars)+MassBkg(m,bkgpars);
  if(fSecondPeak) {total += MassSecondPeak(m,secpeakpars);}
  if(fReflections) {total += MassRfl(m,rflpars);}
  if(fTemplates) {
    switch (fAnchorTemplsMode) {
      case TemplAnchorMode::AnchorToFirst:
        total += MassTemplates(m,&pars[fNParsMassBkg+fNParsMassSgn+fNParsSec+fNParsRfl]);
        break;
      case TemplAnchorMode::AnchorToSgn:
        total += MassTemplates(m,sgnpars);
        break;
      default:
        std::cerr << "Error: Invalid fAnchorTemplsMode value!" << std::endl;
        break;
    }
  }
  return total;
}

//________________________________________________________________
Double_t VnVsMassFitter::vnFunc(Double_t *m, Double_t *pars) {

  //bkg mass parameters
  const Int_t nBkgPars = fNParsMassBkg;
  Double_t massbkgpars[nBkgPars];
  for(Int_t iPar=0; iPar<fNParsMassBkg; iPar++) {massbkgpars[iPar] = pars[iPar];}
  //signal mass parameters
  Double_t masssgnpars[7]; //maximum number of parameters for mass sgn = 7 for the implemented functions
  for(Int_t iPar=0; iPar<fNParsMassSgn; iPar++) {masssgnpars[iPar] = pars[iPar+fNParsMassBkg];}
  //second peak parameters
  Double_t secpeakpars[3]; //maximum number of parameters for second peak = 3 for the implemented functions
  for(Int_t iPar=0; iPar<fNParsSec; iPar++) {
    secpeakpars[iPar] = pars[iPar+fNParsMassBkg+fNParsMassSgn];
  }
  if (fFixFracSecWidth) {
    secpeakpars[2] = masssgnpars[2]*fSecWidthFrac; // sigma * frac
  }
  //reflection parameters
  Double_t rflpars[1]; //maximum number of parameters for rfl = 1 for the implemented functions
  for(Int_t iPar=0; iPar<fNParsRfl; iPar++) {rflpars[iPar] = pars[iPar+fNParsMassBkg+fNParsMassSgn+fNParsSec];}
  //templates parameters
  Double_t templpars[fNParsTempls]; //one parameter for each template (i.e. its scaling parameter)
  for(Int_t iPar=0; iPar<fNParsTempls; iPar++) {templpars[iPar] = pars[iPar+fNParsMassBkg+fNParsMassSgn+fNParsSec+fNParsRfl];}

  //bkg vn parameters
  const Int_t nVnBkgPars = fNParsVnBkg;
  Double_t vnbkgpars[nVnBkgPars];
  for(Int_t iPar=0; iPar<fNParsVnBkg; iPar++) {
    vnbkgpars[iPar] = pars[iPar+fNParsMassSgn+fNParsMassBkg+fNParsSec+fNParsRfl+fNParsTempls];
  }
  //signal vn parameter
  Double_t vnSgn = pars[fNParsMassSgn+fNParsMassBkg+fNParsSec+fNParsRfl+fNParsTempls+fNParsVnBkg];
  //second peak vn parameter
  Double_t vnSecPeak = 0;
  if(fSecondPeak && fDoSecondPeakVn && !fFixVnSecPeakToSgn) {vnSecPeak = pars[fNParsMassSgn+fNParsMassBkg+fNParsSec+fNParsRfl+fNParsTempls+fNParsVnBkg+fNParsVnSgn];}
  if(fSecondPeak && fDoSecondPeakVn && fFixVnSecPeakToSgn) {vnSecPeak = pars[fNParsMassSgn+fNParsMassBkg+fNParsSec+fNParsRfl+fNParsTempls+fNParsVnBkg];}
  //refl vn parameter
  Double_t vnRefl = 0;
  if(fReflections) {
    switch(fVnRflOpt) {
      case 0:
        vnRefl = pars[fNParsMassSgn+fNParsMassBkg+fNParsSec+fNParsRfl+fNParsVnBkg];
        break;
      case 1:
        vnRefl = -pars[fNParsMassSgn+fNParsMassBkg+fNParsSec+fNParsRfl+fNParsVnBkg];
        break;
      case 2:
        vnRefl = 0; //not used
        break;
      case 3:
        vnRefl = pars[fNParsMassSgn+fNParsMassBkg+fNParsSec+fNParsRfl+fNParsVnBkg+fNParsVnSgn+fNParsVnSecPeak];
        break;
      default:
        printf("Error in setting reflection vn option: check fVnRflOpt");
        break;
    }
  }
  Double_t vnBkg = vnBkgFunc(m,vnbkgpars);
  Double_t Sgn = MassSignal(m,masssgnpars);
  Double_t Bkg = MassBkg(m,massbkgpars);
  Double_t SecPeak = 0;
  if(fSecondPeak) {
    if(fDoSecondPeakVn) SecPeak += MassSecondPeak(m,secpeakpars);
    else Bkg += MassSecondPeak(m,secpeakpars);
  }
  Double_t Refl=0;
  if(fReflections) {
    if(fVnRflOpt==kSameVnBkg) Bkg += MassRfl(m,rflpars);
    else Refl += MassRfl(m,rflpars);
  }
  Double_t TemplatesVn = 0;
  Double_t TemplatesMass = 0;
  if (fTemplates) {
    switch (fAnchorTemplsMode) {
      case TemplAnchorMode::AnchorToFirst:
        TemplatesMass += MassTemplates(m,&pars[fNParsMassBkg+fNParsMassSgn+fNParsSec+fNParsRfl]);
        break;
      case TemplAnchorMode::AnchorToSgn:
        TemplatesMass += MassTemplates(m,masssgnpars);
        break;
      default:
        std::cerr << "Error: Invalid fAnchorTemplsMode value!" << std::endl;
    }
    TemplatesVn += vnSgn*TemplatesMass; // Templates have vn fixed to the one of the signal
  }
  return (vnSgn*Sgn+vnBkg*Bkg+vnSecPeak*SecPeak+vnRefl*Refl+TemplatesVn)/(Sgn+Bkg+SecPeak+Refl+TemplatesMass);
}

//________________________________________________________________
void VnVsMassFitter::SetFuncParNames() {

  // For mass comb bkg parameter initialization
  Double_t integral = fMassHisto->Integral("width");
  // Initial slope estimate: difference between last and first bin
  Double_t slopeEst = (fMassHisto->GetBinContent(fMassHisto->GetNbinsX()) - fMassHisto->GetBinContent(1)) / (fMassMax - fMassMin);

  fMassSgnFunc->SetParName(0, "SgnNorm");
  fMassSgnFunc->SetParName(1, "Mean");

  switch(fMassSgnFuncType) {
    case 0: //single gaus
      fMassSgnFunc->SetParName(2, "Sigma");
      break;
    case 1: //double gaus
      fMassSgnFunc->SetParName(2, "Sigma1");
      fMassSgnFunc->SetParName(4, "Sigma2");
      fMassSgnFunc->SetParName(3, "Frac");
      break;
    case 3: //asymmetric crystalball
      fMassSgnFunc->SetParName(2, "Sigma");
      fMassSgnFunc->SetParName(3, "Alpha1");
      fMassSgnFunc->SetParName(4, "N1");
      fMassSgnFunc->SetParName(5, "Alpha2");
      fMassSgnFunc->SetParName(6, "N2");
      break;
    case 4: //symmetric crystalball
      fMassSgnFunc->SetParName(2, "Sigma");
      fMassSgnFunc->SetParName(3, "Alpha");
      fMassSgnFunc->SetParName(4, "N");
      break;

    default:
      printf("Error in setting signal par names: check fMassSgnFuncType");
      break;
  }

  std::cout << "Setting background parameter names..." << std::endl;
  switch(fMassBkgFuncType) {
    case 0: //expo
      fMassBkgFunc->SetParName(0, "BkgNorm");
      fMassBkgFunc->SetParName(1, "BkgExpLambda");
      break;
    case 1: //lin
      fMassBkgFunc->SetParName(0, "BkgNorm");
      fMassBkgFunc->SetParName(1, "BkgPolCoef1");
      break;
    case 2: //pol2
      fMassBkgFunc->SetParName(0, "BkgNorm");
      fMassBkgFunc->SetParName(1, "BkgPolCoef1");
      fMassBkgFunc->SetParName(2, "BkgPolCoef2");
      break;
    case 3: //no bkg
      fMassBkgFunc->SetParName(0, "BkgNorm");
      break;
    case 4: //power law
      fMassBkgFunc->SetParName(0, "BkgNorm");
      fMassBkgFunc->SetParName(1, "BkgPowCoef1");
      break;
    case 5: //power expo
      fMassBkgFunc->SetParName(0, "BkgNorm");
      fMassBkgFunc->SetParName(1, "BkgPowExpCoef1");
      fMassBkgFunc->SetParName(2, "BkgPowExpCoef2");
      break;
    case 6: //high degree pol
      fMassBkgFunc->SetParName(0, "BkgNorm");
      for(Int_t iPar=1; iPar<fNParsMassBkg; iPar++) {
        fMassBkgFunc->SetParName(iPar, Form("BkgPolNCoef%d",iPar));
      }
      break;
    default:
      printf("Error in setting signal par names: check fMassBkgFuncType");
      break;
  }

  if(fReflections) {
    fVnTotFunc->SetParName(fNParsMassBkg+fNParsMassSgn+fNParsSec,"ReflOverS");
  }

  if(fSecondPeak) {
    fMassSecPeakFunc->SetParName(0, "SecPeakNorm");
    fMassSecPeakFunc->SetParName(1, "SecPeakMean");
    if (!fFixFracSecWidth) {
      fMassSecPeakFunc->SetParName(2, "SecPeakSigma");
    }
  }

  // Setting Vn parameter names
  switch(fVnBkgFuncType) {
    case 1:
      fVnBkgFunc->SetParName(0, "ConstVnBkg");
      fVnBkgFunc->SetParName(1, "SlopeVnBkg");
      break;
    case 2:
      fVnBkgFunc->SetParName(0, "ConstVnBkg");
      fVnBkgFunc->SetParName(1, "Coef1VnBkg");
      fVnBkgFunc->SetParName(2, "Coef2VnBkg");
      break;
    case 3:
      fVnBkgFunc->SetParName(0, "ConstVnBkg");
    default:
      printf("Error in setting signal par names: check fVnBkgFuncType");
      break;
  }
  // fVnTotFunc->SetParName(fNParsMassBkg+fNParsMassSgn+fNParsRfl+fNParsSec+fNParsTempls+fNParsVnBkg,Form("v%dSgn",fHarmonic));
  fInitFuncPars[Form("v%dSgn",fHarmonic)] = {0.05, -0.3, 0.5};

  // if(fSecondPeak && fDoSecondPeakVn) {fVnTotFunc->SetParName(fNParsMassBkg+fNParsMassSgn+fNParsRfl+fNParsSec+fNParsTempls+fNParsVnBkg+1,Form("v%dSecPeak",fHarmonic));}
  if(fSecondPeak && fDoSecondPeakVn) {
    fInitFuncPars[Form("v%dSecPeak",fHarmonic)] = {0.05, -0.3, 0.5};
  }
  // if(fReflections && fVnRflOpt==kFreePar) {fVnTotFunc->SetParName(fNParsMassBkg+fNParsMassSgn+fNParsRfl+fNParsSec+fNParsVnBkg+fNParsTempls+1+fNParsVnSecPeak,Form("v%dRefl",fHarmonic));}
  if(fReflections && fVnRflOpt==kFreePar) {
    fInitFuncPars[Form("v%dRefl",fHarmonic)] = {0.05, -0.3, 0.5};
  }
  std::cout << "Parameter names set." << std::endl;
  std::cout << "Length of fInitFuncPars: " << fInitFuncPars.size() << std::endl;
}

//________________________________________________________________
void VnVsMassFitter::SetParInitValsAndNames() {

  // For mass comb bkg parameter initialization
  Double_t integral = fMassHisto->Integral("width");
  // Initial slope estimate: difference between last and first bin
  Double_t slopeEst = (fMassHisto->GetBinContent(fMassHisto->GetNbinsX()) - fMassHisto->GetBinContent(1)) / (fMassMax - fMassMin);

  fInitFuncPars["SgnNorm"] = {integral/10, 0, integral};
  fInitFuncPars["Mean"] = {1.869, 1.8, 2.0};

  switch(fMassSgnFuncType) {
    case 0: //single gaus
      fInitFuncPars["Sigma"] = {0.015, 0, 0.2};
      break;
    case 1: //double gaus
      fInitFuncPars["Sigma1"] = {0.015, 0, 0.2};
      fInitFuncPars["Sigma2"] = {0.015, 0, 0.2};
      fInitFuncPars["Frac"] = {0.1, 0, 1};
      break;
    case 3: //asymmetric crystalball
      fInitFuncPars["Sigma"] = {0.015, 0, 0.2};
      fInitFuncPars["Alpha1"] = {2, 1, 10};
      fInitFuncPars["N1"] = {2, 1.05, 10};
      fInitFuncPars["Alpha2"] = {2, 1, 10};
      fInitFuncPars["N2"] = {2, 1.05, 10};
      break;
    case 4: //symmetric crystalball
      fInitFuncPars["Sigma"] = {0.015, 0, 0.2};
      fInitFuncPars["Alpha"] = {2, 1, 10};
      fInitFuncPars["N"] = {2, 1.05, 10};
      break;

    default:
      printf("Error in setting signal par names: check fMassSgnFuncType");
      break;
  }

  std::cout << "Setting background parameter names..." << std::endl;
  switch(fMassBkgFuncType) {
    case 0: //expo
      fInitFuncPars["BkgNorm"] = {integral/10, 0, integral};
      fInitFuncPars["BkgExpLambda"] = {-1, -100, 100};
      break;
    case 1: //lin
      fInitFuncPars["BkgNorm"] = {integral/10, 0, integral};
      fInitFuncPars["BkgPolCoef1"] = {slopeEst, -1e4, 1e4};
      break;
    case 2: //pol2
      fInitFuncPars["BkgNorm"] = {integral/10, 0, integral};
      fInitFuncPars["BkgPolCoef1"] = {slopeEst, -1e4, 1e4};
      fInitFuncPars["BkgPolCoef2"] = {0.0, -1e4, 1e4};
      break;
    case 3: //no bkg
      fInitFuncPars["BkgNorm"] = {integral/10, 0, integral};
      break;
    case 4: //power law
      fInitFuncPars["BkgNorm"] = {integral/10, 0, integral};
      fInitFuncPars["BkgPowCoef1"] = {0.0, -1e4, 1e4};
      break;
    case 5: //power expo
      fInitFuncPars["BkgNorm"] = {integral/10, 0, integral};
      fInitFuncPars["BkgPowExpCoef1"] = {fMassHisto->GetMaximum(), 0, fMassHisto->GetMaximum() * 10};
      fInitFuncPars["BkgPowExpCoef2"] = {2.0, 0.01, 200};
      break;
    case 6: //high degree pol
      fInitFuncPars["BkgNorm"] = {integral/10, 0, integral};
      for(Int_t iPar=1; iPar<fNParsMassBkg; iPar++) {
        fInitFuncPars[Form("BkgPolNCoef%d",iPar)] = {0.0, -1e4, 1e4};
      }
      break;
    default:
      printf("Error in setting signal par names: check fMassBkgFuncType");
      break;
  }

  if(fReflections) {
    fVnTotFunc->SetParName(fNParsMassBkg+fNParsMassSgn+fNParsSec,"ReflOverS");
  }

  if(fSecondPeak) {
    fInitFuncPars["SecPeakNorm"] = {integral/20, 0, integral};
    fInitFuncPars["SecPeakMean"] = {(fMassRangeMaxSecPeak + fMassRangeMinSecPeak)/2, fMassRangeMinSecPeak, fMassRangeMaxSecPeak};
    if (!fFixFracSecWidth) {
      fInitFuncPars["SecPeakSigma"] = {0.015, 0.005, 0.05};
    }
  }

  // Setting Vn parameter names
  switch(fVnBkgFuncType) {
    case 1:
      fInitFuncPars["ConstVnBkg"] = {0.05, 0, 1};
      fInitFuncPars["SlopeVnBkg"] = {0.0, -1, 1};
      break;
    case 2:
      fInitFuncPars["ConstVnBkg"] = {0.05, 0, 1};
      fInitFuncPars["Coef1VnBkg"] = {0.0, -1, 1};
      fInitFuncPars["Coef2VnBkg"] = {0.0, -1, 1};
      break;
    case 3:
      fInitFuncPars["ConstVnBkg"] = {0.05, 0, 1};
    default:
      printf("Error in setting signal par names: check fVnBkgFuncType");
      break;
  }
  // fVnTotFunc->SetParName(fNParsMassBkg+fNParsMassSgn+fNParsRfl+fNParsSec+fNParsTempls+fNParsVnBkg,Form("v%dSgn",fHarmonic));
  fInitFuncPars[Form("v%dSgn",fHarmonic)] = {0.05, -0.3, 0.5};

  // if(fSecondPeak && fDoSecondPeakVn) {fVnTotFunc->SetParName(fNParsMassBkg+fNParsMassSgn+fNParsRfl+fNParsSec+fNParsTempls+fNParsVnBkg+1,Form("v%dSecPeak",fHarmonic));}
  if(fSecondPeak && fDoSecondPeakVn) {
    fInitFuncPars[Form("v%dSecPeak",fHarmonic)] = {0.05, -0.3, 0.5};
  }
  // if(fReflections && fVnRflOpt==kFreePar) {fVnTotFunc->SetParName(fNParsMassBkg+fNParsMassSgn+fNParsRfl+fNParsSec+fNParsVnBkg+fNParsTempls+1+fNParsVnSecPeak,Form("v%dRefl",fHarmonic));}
  if(fReflections && fVnRflOpt==kFreePar) {
    fInitFuncPars[Form("v%dRefl",fHarmonic)] = {0.05, -0.3, 0.5};
  }
  std::cout << "Parameter names set." << std::endl;
  std::cout << "Length of fInitFuncPars: " << fInitFuncPars.size() << std::endl;
}
