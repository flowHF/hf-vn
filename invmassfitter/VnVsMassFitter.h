#ifndef VNVSMASSFITTER_H
#define VNVSMASSFITTER_H
  /// \class VnVsMassFitter
  /// \class that performs the vn vs mass simultaneus fit for D mesons
  
#include <TObject.h>
#include <Riostream.h>
#include <TVirtualPad.h>
#include <TH1F.h>
#include <TH1F.h>
#include <TFile.h>
#include "Fit/Fitter.h"
#include "Fit/Chi2FCN.h"
#include "Math/WrappedMultiTF1.h"

struct ParInit {
  double value;
  double low;
  double high;
};

class VnVsMassFitter : public TObject {

public:
  VnVsMassFitter();
  VnVsMassFitter(std::string name, TH1F* hMass, TH1F* hvn, Double_t min, Double_t max, Int_t funcMassBkg, Int_t funcMassSgn, Int_t funcvnBkg);
  ~VnVsMassFitter();

  enum ETypeOfBkg{kExpo=0, kLin=1, kPol2=2, kNoBk=3, kPow=4, kPowEx=5, kPoln=6};
  enum ETypeOfSgn{kGaus=0, k2Gaus=1, kDoubleCBAsymm=3, kDoubleCBSymm=4};
  enum ETypeOfVnRfl{kSameVnSignal=0, kOppVnSignal=1, kSameVnBkg=2, kFreePar=3};
  enum TemplAnchorMode{AnchorToFirst=1, AnchorToSgn=2};

  Int_t SimultaneousFit();
  void DrawHere(TVirtualPad* c);

  //setters
  void SetInitialGaussianSigma(Double_t sigma, Int_t opt) {fSigmaInit=sigma; fSigmaFixed=opt;}
  void SetInitialGaussianMean(Double_t mean, Int_t opt) {fMeanInit=mean; fMeanFixed=opt;}
  void SetInitialGaussianSigma2Gaus(Double_t sigma, Int_t opt) {fSigma2GausInit=sigma; fSigma2GausFixed=opt;}
  void SetInitialFrac2Gaus(Double_t frac, Int_t opt) {fFrac2GausInit=frac; fFrac2GausFixed=opt;}
  void SetParticlePdgMass(Double_t mass){fMassParticle=mass;}
  void SetMassSgnFunc(Int_t functype) {fMassSgnFuncType=functype;}
  void SetMassBkgFunc(Int_t functype) {fMassBkgFuncType=functype;}
  void SetVnBkgFunc(Int_t functype) {fVnBkgFuncType=functype;}
  void FixSigmaFromMassFit() {fSigmaFixedFromMassFit=kTRUE;}
  void FixMeanFromMassFit() {fMeanFixedFromMassFit=kTRUE;}
  void FixSigma2GausFromMassFit() {fSigma2GausFixedFromMassFit=kTRUE;}
  void FixFrac2GausFromMassFit() {fFrac2GausFixedFromMassFit=kTRUE;}
  void SetNSigmaForVnSB(Int_t nsigma=4) {fNSigmaForSB=nsigma;}
  void SetPolDegreeForBackgroundFit(Int_t deg){
    if(fMassBkgFuncType!=6) printf("fMassBkgFuncType should be set to 6 to use higher order polynomials\n");
    fPolDegreeBkg=deg;
  }
  void SetPolDegreeForVnBackgroundFit(Int_t deg){
    if(fVnBkgFuncType!=6) printf("fVnBkgFuncType should be set to 6 to use higher order polynomials\n");
    fPolDegreeVnBkg=deg;
  }
  void SetTemplateReflections(const TH1 *h, TString opt, Double_t minRange, Double_t maxRange) {
    fHistoTemplRflInit=(TH1F*)h->Clone();
    /// option could be:
    ///    "template"                use MC histograms
    ///    "1gaus" ot "singlegaus"   single gaussian function fit to MC templates
    ///    "2gaus" ot "doublegaus"   double gaussian function fit to MC templates
    ///    "pol3"                    3rd order polynomial fit to MC templates
    ///    "pol6"                    6th order polynomial fit to MC templates
    fRflOpt=opt;
    fMinRefl=minRange;
    fMaxRefl=maxRange;
    fReflections=kTRUE;
  }

  void SetTemplatesHisto(const std::vector<const TH1*>& histopdfs,
                        const std::vector<double>& relweights,
                        int anchorMode)
  {
    fTemplates = kTRUE;

    if (histopdfs.size() != relweights.size()) {
      std::cerr << "ERROR: Number of templates and weights does not match!" << std::endl;
      return;
    }

    for (size_t i = 0; i < histopdfs.size(); ++i) {
      if (!histopdfs[i]) {
          std::cerr << "ERROR: histopdfs[" << i << "] is nullptr!" << std::endl;
      }
    }

    for (size_t i = 0; i < histopdfs.size(); ++i) {
      // Clone the histogram
      TH1F* hNorm = (TH1F*)histopdfs[i]->Clone(Form("hTemplInternal_%zu", i));
      hNorm->SetDirectory(nullptr); // Important: disconnect from global files

      // Store in a vector of TH1*
      fHistoTemplates.push_back(hNorm);
      fRelWeights.push_back(relweights[i]);
    }

    // Set anchoring mode
    if (anchorMode == (int)TemplAnchorMode::AnchorToSgn) {
      fAnchorTemplsMode = TemplAnchorMode::AnchorToSgn;
    } else if (anchorMode == (int)TemplAnchorMode::AnchorToFirst) {
      fAnchorTemplsMode = TemplAnchorMode::AnchorToFirst;
    } else {
      std::cerr << "WARNING: Unknown anchorMode, defaulting to AnchorToSgn" << std::endl;
      fAnchorTemplsMode = TemplAnchorMode::AnchorToSgn;
    }

    std::cout << "WARNING: Vn parameter of templates will be the same as the signal!" << std::endl;
  }

  void SetHistoPrefitSgn(TH1F* h, bool fixtoPrefit) {
    fHistoSgnPrefit=(TH1F*)h->Clone("fHistoSgnPrefit");
    fHistoSgnPrefit->SetDirectory(0);
    fFixSgnFromMCPrefit = fixtoPrefit;
    std::cout << "Histo for signal prefit set!" << std::endl;
  }

  void InitFunctionPars(std::string funcType);

  void SetInitialReflOverS(Double_t rovers){fRflOverSig=rovers;}
  void SetFixReflOverS(Double_t rovers){
    SetInitialReflOverS(rovers);
    fFixRflOverSig=kTRUE;
  }
  void SetReflVnOption(Int_t opt) {fVnRflOpt=opt;}
  void SetReflVnParLimits(Double_t min, Double_t max) {
    fVnRflLimited=kTRUE;
    fVnRflMin=min;
    fVnRflMax=max;
  }
  void IncludeSecondGausPeak(Double_t mass, Bool_t fixm, Double_t width, Bool_t fixw,
                             Double_t massmin, Double_t massmax, Int_t mincounts,
                             Double_t fracw, Bool_t fixfracw,
                             Bool_t doVn, Bool_t fixtosgn){
    fSecondPeak=kTRUE; fSecMass=mass; fSecWidth=width; fSecWidthFrac=fracw;
    fFixSecMass=fixm;  fFixSecWidth=fixw;
    fFixFracSecWidth=fixfracw;
    fDoSecondPeakVn=doVn;
    fFixVnSecPeakToSgn=fixtosgn;
    fMassRangeMinSecPeak=massmin;
    fMassRangeMaxSecPeak=massmax;
    fMinCountsForSecPeak=mincounts;
    DefineNumberOfParameters();
    SetParInitValsAndNames();
  }
  void ExcludeSecondGausPeak() {
    fSecondPeak=kFALSE; fSecMass=-999.; fSecWidth=9999.; fSecWidthFrac=9999.;
    fFixSecMass=kFALSE;  fFixSecWidth=kFALSE;
    fDoSecondPeakVn=kFALSE;
    fFixVnSecPeakToSgn=kFALSE;
  }
  void SetInitPar(std::string name, Double_t val, Double_t min, Double_t max) {
    fInitFuncPars[name] = {val, min, max};
  }
  void ApplyInitPars();
  // void SetInitPars(std::string parname, double val, double min, double max) {
  //   fInitFuncPars[parname] = {val, min, max};
  // }
  void SetHarmonic(Int_t harmonic=2) {fHarmonic=harmonic;}
  void SetSuppressOutput(Bool_t suppress) {fSuppressOutput=suppress;}

  // Double-sided crystal ball functions
  Double_t DoubleSidedCBAsymm(double x, double mu, double sigma, double a1, double n1, double a2, double n2);
  Double_t DoubleSidedCBSymm(double x, double mu, double sigma, double a, double n);

  TH1F *GetPullDistribution();

  //getters
  Double_t GetVn() const {return fVn;}
  Double_t GetVnUncertainty() const {return fVnUncertainty;}
  Double_t GetMean() const {return fMean;}
  Double_t GetMeanUncertainty() const {return fMeanUncertainty;}
  Double_t GetSigma() const {return fSigma;}
  Double_t GetSigmaUncertainty() const {return fSigmaUncertainty;}
  Double_t GetRawYield() const {return fRawYield;}
  Double_t GetRawYieldUncertainty() const {return fRawYieldUncertainty;}
  Double_t GetChiSquare() const {return fChiSquare;}
  Int_t GetNDF() const {return fNDF;}
  Double_t GetReducedChiSquare() const {return fChiSquare/fNDF;}
  Double_t GetFitProbability() const {return fProb;}
  Double_t GetSBVnPrefitChiSquare() const {return fSBVnPrefitChiSquare;}
  Int_t GetSBVnPrefitNDF() const {return fSBVnPrefitNDF;}
  Double_t GetSBVnPrefitReducedChiSquare() const {return fSBVnPrefitChiSquare/fSBVnPrefitNDF;}
  Double_t GetSBVnPrefitProbability() const {return fSBVnPrefitProb;}
  Double_t GetMassPrefitChiSquare() const {return fMassPrefitChiSquare;}
  Int_t GetMassPrefitNDF() const {return fMassPrefitNDF;}
  Double_t GetMassPrefitReducedChiSquare() const {return fMassPrefitChiSquare/fMassPrefitNDF;}
  Double_t GetMassPrefitProbability() const {return fMassPrefitProb;}
  Double_t GetParticlePdgMass() const {return fMassParticle;}
  TH1F* GetTemplateReflections() {
    if(fHistoTemplRfl) {return (TH1F*)fHistoTemplRfl->Clone("fHistoTemplRfl");}
    else if(fHistoTemplRflInit) {return (TH1F*)fHistoTemplRflInit->Clone("fHistoTemplRflInit");}
    else {return 0;}
  }
  std::vector<Double_t> GetBkgPars() const {
    std::vector<Double_t> bkgpars;
    for (Int_t iBkgPar=0; iBkgPar<fNParsMassBkg; iBkgPar++) {
      bkgpars.push_back(fVnBkgFunc->GetParameter(iBkgPar));
    }
    for (Int_t iBkgPar=0; iBkgPar<fNParsMassBkg; iBkgPar++) {
      bkgpars.push_back(fVnBkgFunc->GetParError(iBkgPar));
    }
    return bkgpars;
  }
  void Signal(Double_t nOfSigma,Double_t &signal,Double_t &errsignal) const;
  void Signal(Double_t min,Double_t max,Double_t &signal,Double_t &errsignal) const;
  void Background(Double_t nOfSigma, Double_t &background,Double_t &errbackground) const;
  void Background(Double_t min, Double_t max, Double_t &background,Double_t &errbackground) const;
  void Significance(Double_t nOfSigma, Double_t &significance,Double_t &errsignificance) const;
  void Significance(Double_t min, Double_t max, Double_t &significance,Double_t &errsignificance) const;
  TF1* GetMassTotFitFunc() const {
    if(fMassTotFunc) return fMassTotFunc;
    else return nullptr;
  }
  Int_t GetNMassBkgPars() const {
    return fNParsMassBkg;
  }
  Int_t GetNMassSgnPars() const {
    return fNParsMassSgn;
  }
  Int_t GetNMassSecPeakPars() const {
    // only for gaussian case
    return 3;
  }
  Int_t GetNMassReflPars() const {
    // only for gaussian case
    return fNParsRfl;
  }
  Int_t GetNVnBkgPars() const {
    return fNParsVnBkg;
  }
  Int_t GetNVnSgnPars() const {
    return fNParsVnSgn;
  }
  Int_t GetNVnSecPeakPars() const {
    return fNParsVnSecPeak;
  }
  Int_t GetNVnReflPars() const {
    return fNParsVnRfl;
  }
  TF1* GetMassSignalFitFunc() const {
    if(fMassSgnFunc) return fMassSgnFunc;
    else return nullptr;
  }
  TF1* GetMassBkgFitFunc() const {
    if(fMassBkgFunc) return fMassBkgFunc;
    else return nullptr;
  }
  TF1* GetMassTemplFitFunc() const {
    if(fMassTemplFunc) return fMassTemplFunc;
    else return nullptr;
  }
  TF1* GetVnVsMassTotFitFunc() const {
    if(fVnTotFunc) return fVnTotFunc;
    else return nullptr;
  }
  TF1* GetVnVsMassBkgFitFunc() const {
    if(fVnBkgFunc) return fVnBkgFunc;
    else return nullptr;
  }
  TF1* GetMassRflFunc() const {
    if(fReflections) return fMassRflFunc;
    else return nullptr;
  }
  TF1* GetMassBkgRflFunc() const {
    if(fReflections) return fMassBkgRflFunc;
    else return nullptr;
  }
  TF1* GetMassSecPeakFunc() const {
    if(fSecondPeak) return fMassSecPeakFunc;
    else return nullptr;
  }
  TF1* GetVnSecPeakFunc() const {
    if(fSecondPeak) return fVnSecPeakFunc;
    else return nullptr;
  }
  std::vector<TF1*> GetMassTemplFuncts() const {
    if(fTemplates) return fMassTemplatesDraw;
    else return {};
  }
  double GetTemplOverSig() const {
    if(fMassTemplFunc && fMassSgnFunc) return fMassTemplFunc->Integral(this->fMassMin, this->fMassMax) / fMassSgnFunc->Integral(this->fMassMin, this->fMassMax);
    else return 0;
  }
  std::vector<TF1*> GetVnCompsFuncts() const {
    return fVnCompsDraw;
  }
  std::vector<double> GetVnTemplates() const {
    std::cout << "The vn of templates is equal to the one of signal: " << GetVn() << std::endl;
    std::vector<double> vnPars;
    for(size_t iFunc=0; iFunc<fHistoTemplates.size(); iFunc++) {
      vnPars.push_back(GetVn());
    }
    return vnPars;
  }
  std::vector<double> GetVnTemplatesUncertainties() const {
    std::cout << "The vn of templates is equal to the one of signal: " << GetVn() << std::endl;
    std::vector<double> vnPars;
    for(size_t iFunc=0; iFunc<fHistoTemplates.size(); iFunc++) {
      vnPars.push_back(GetVnUncertainty());
    }
    return vnPars;
  }
  TH1F* GetPrefitParsHisto() const {
    return fPrefitParsHisto;
  }
  TH1F* GetSignalParsHisto() const {
    return fSignalParsHisto;
  }
  TH1F* GetSimFitParsHisto() const {
    return fSimFitParsHisto;
  }
  //struct for global chi2 (for simultaneus fit)
  struct GlobalChi2 {
    GlobalChi2(ROOT::Math::IMultiGenFunction & f1,ROOT::Math::IMultiGenFunction & f2) : fChi2_1(&f1), fChi2_2(&f2) {}

    double operator() (const double *par) const {
        return (*fChi2_1)(par) + (*fChi2_2)(par);
    }
    const  ROOT::Math::IMultiGenFunction * fChi2_1;
    const  ROOT::Math::IMultiGenFunction * fChi2_2;
  };

private:

    ///fit functions
  Double_t GetGausPDF(Double_t x, Double_t mean, Double_t sigma);
  Double_t GetExpoPDF(Double_t x, Double_t slope, Bool_t isnorm=kTRUE);
  Double_t GetPolPDF(Double_t x, Double_t *pars, Int_t order, Bool_t isnorm=kTRUE);
  Double_t GetPowerFuncPDF(Double_t x, Double_t *pars);
  Double_t GetPowerExpoPDF(Double_t x, Double_t *pars);
  Double_t GetHigherPolFuncPDF(Double_t x, Double_t *pars, Int_t Ndeg, Bool_t isnorm=kTRUE);
  Double_t MassSignal(Double_t *m, Double_t *pars);
  Double_t MassBkg(Double_t *m, Double_t *pars);
  Double_t MassRfl(Double_t *m,Double_t *par);
  Double_t MassBkgRfl(Double_t *m,Double_t *par);
  Double_t MassTemplates(Double_t *m,Double_t *pars);
  Double_t MassSecondPeak(Double_t *m,Double_t *par);
  Double_t MassFunc(Double_t *m, Double_t *pars);
  Double_t vnBkgFunc(Double_t *m, Double_t *pars);
  Double_t vnFunc(Double_t *m, Double_t *pars);

    ///private methods
  void DefineNumberOfParameters();
  void DefineFunctions();
  Int_t RunPrefits();
  Int_t PrefitSignal();
  Int_t PrefitMass();
  Int_t PrefitCombBkg();
  Bool_t PrefitVnSidebands();
  void SetParInitValsAndNames();
  void SetFuncParNames();

    ///data members
  std::string           fName;                          /// name of the fitter
  TH1F*                 fMassHisto;                     /// mass histogram to fit
  TH1F*                 fPrefitParsHisto;               /// histogram to store MC fit parameters
  TH1F*                 fSignalParsHisto;               /// histogram to store fit parameters of signal function
  TH1F*                 fSimFitParsHisto;               /// histogram to store fit parameters of simultaneous fit
  TH1F*                 fVnVsMassHisto;                 /// vn vs. mass histogram to fit
  Int_t                 fMassSgnFuncType;               /// type of mass signal fit function
  Int_t                 fMassBkgFuncType;               /// type of mass bkg fit function
  Int_t                 fVnBkgFuncType;                 /// type of vn bkg fit function
  TF1*                  fMassFuncFromPrefit;            /// mass fit function (1st step, from prefit)
  TF1*                  fMassBkgFunc;                   /// mass bkg fit function (final, after simultaneus fit)
  TF1*                  fMassSgnFunc;                   /// mass signal fit function (final, after simultaneus fit)
  TF1*                  fMassTemplFunc;                 /// mass signal fit function (final, after simultaneus fit)
  TF1*                  fMassTotFunc;                   /// mass fit function (final, after simultaneus fit)
  TF1*                  fVnBkgFunc;                     /// vn bkg fit function (final, after simultaneus fit)
  TF1*                  fVnTotFunc;                     /// vn fit function (final, after simultaneus fit)
  Double_t              fMassMin;                       /// upper mass limit
  Double_t              fMassMax;                       /// lower mass limit
  Double_t              fVn;                            /// vn of the signal from fit
  Double_t              fVnUncertainty;                 /// uncertainty on vn of the signal from simultaneus fit
  Double_t              fSigma;                         /// mass peak width from simultaneus fit
  Double_t              fSigmaUncertainty;              /// uncertainty on mass peak width from simultaneus fit
  Double_t              fMean;                          /// mass peak position from simultaneus fit
  Double_t              fMeanUncertainty;               /// uncertainty on mass peak position from simultaneus fit
  Double_t              fRawYield;                      /// raw yield from simultaneus fit
  Double_t              fRawYieldUncertainty;           /// uncertainty raw yield from simultaneus fit
  Double_t              fChiSquare;                     /// simultaneus fit chi square
  Int_t                 fNDF;                           /// simultaneus fit number of degree of freedom
  Double_t              fProb;                          /// simultaneus fit probability
  Double_t              fSBVnPrefitChiSquare;           /// vn SB prefit chi square
  Int_t                 fSBVnPrefitNDF;                 /// vn SB prefit number of degree of freedom
  Double_t              fSBVnPrefitProb;                /// vn SB prefit probability
  Double_t              fMassPrefitChiSquare;           /// Mass prefit chi square
  Int_t                 fMassPrefitNDF;                 /// Mass prefit number of degree of freedom
  Double_t              fMassPrefitProb;                /// Mass prefit probability
  Int_t                 fNSigmaForSB;                   /// number of sigma for sidebands region (vn bkg prefit)
  Double_t              fSigmaInit;                     /// initialization for peak width
  Double_t              fMeanInit;                      /// initialization for peak position
  Double_t              fSigma2GausInit;                /// initialization for second peak width in case of k2Gaus
  Double_t              fFrac2GausInit;                 /// initialization for fraction of second gaussian in case of k2Gaus
  Bool_t                fMeanFixedFromMassFit;          /// flag to fix peak position from mass prefit
  Bool_t                fSigmaFixedFromMassFit;         /// flag to fix peak width from mass prefit
  Bool_t                fSigma2GausFixedFromMassFit;    /// flag to fix second peak width from mass prefit in case of k2Gaus
  Bool_t                fFrac2GausFixedFromMassFit;     /// flag to fix fraction of second gaussian in case of k2Gaus
  Bool_t                fIsMassSidebandFit;             /// flag to indicate if mass sideband fit is being performed
  Bool_t                fIsVnSidebandFit;               /// flag to indicate if vn sideband fit is being performed
  Double_t              fMassParticle;                  /// mass of selected particle
  Int_t                 fNParsMassSgn;                  /// number of parameters in mass signal fit function
  Int_t                 fNParsMassBkg;                  /// number of parameters in mass bkg fit function
  Int_t                 fNParsVnBkg;                    /// number of parameters in vn bkg fit function
  Int_t                 fNParsVnSgn;                    /// number of parameters in vn sgn fit function (1)
  Int_t                 fNParsVnSecPeak;                /// number of parameters in vn sec peak fit function (1 if included, 0 otherwise)
  Int_t                 fNParsVnRfl;                    /// number of parameters in vn refl fit function (1 if included, 0 otherwise)
  Int_t                 fSigmaFixed;                    /// flag to fix peak width
  Int_t                 fMeanFixed;                     /// flag to fix peak position
  Int_t                 fSigma2GausFixed;               /// flag to fix second peak width in case of k2Gaus
  Int_t                 fFrac2GausFixed;                /// flag to fix fraction of second gaussian in case of k2Gaus
  Int_t                 fPolDegreeBkg;                  /// degree of polynomial expansion for back fit (option 6 for back)
  Int_t                 fPolDegreeVnBkg;                /// degree of polynomial expansion for vn back fit (option 6 for back)
  Bool_t                fReflections;                   /// flag use/not use reflections
  Int_t                 fNParsRfl;                      /// fit parameters in reflection fit function
  Int_t                 fNParsTotMass;                  /// fit parameters in mass fit function
  Int_t                 fNParsTotVn;                    /// fit parameters in vn vs mass fit function
  Int_t                 fNVnParsSgn;                    /// minimum counts for fit
  Double_t              fRflOverSig;                    /// reflection/signal
  Bool_t                fFixRflOverSig;                 /// switch for fix refl/signal
  Bool_t                fFixSgnFromMCPrefit;            /// switch for fix signal shape from MC prefit
  TH1F*                 fHistoTemplRfl;                 /// histogram with reflection template
  TH1F*                 fHistoSgnPrefit;                /// histogram with reflection template
  TH1F*                 fHistoTemplRflInit;             /// initial histogram with reflection template
  TF1*                  fMassRflFunc;                   /// fit function for reflections
  TF1*                  fMassBkgRflFunc;                /// mass bkg fit function plus reflections (final, after simultaneus fit)
  TString               fRflOpt;                        /// refelction option
  Double_t              fMinRefl;                       /// minimum for refelction histo
  Double_t              fMaxRefl;                       /// maximum for refelction histo
  Bool_t                fSmoothRfl;                     /// switch for smoothing of reflection template
  Double_t              fRawYieldHelp;                  /// internal variable for fit with reflections
  Int_t                 fVnRflOpt;                      /// option for reflection vn type
  Bool_t                fVnRflLimited;                  /// flag to limit or not the vn of reflections
  Double_t              fVnRflMin;                      /// minimum vn of reflections
  Double_t              fVnRflMax;                      /// maximum vn of reflections
  Bool_t                fSecondPeak;                    /// switch off/on second peak (for D+->KKpi in Ds)
  TF1*                  fMassSecPeakFunc;               /// fit function for second peak
  Double_t              fMassRangeMinSecPeak;           /// Minimum mass range for second peak
  Double_t              fMassRangeMaxSecPeak;           /// Maximum mass range for second peak
  Int_t                 fMinCountsForSecPeak;           /// Minimum counts for second peak
  TF1*                  fVnSecPeakFunc;                 /// fit function for second peak
  Int_t                 fNParsSec;                      /// number of parameters in second peak fit function
  Double_t              fSecMass;                       /// position of the 2nd peak
  Double_t              fSecWidth;                      /// width of the 2nd peak
  Double_t              fSecWidthFrac;                  /// fraction of the 2nd peak width wrt the signal peak width
  Bool_t                fFixSecMass;                    /// flag to fix the position of the 2nd peak
  Bool_t                fFixSecWidth;                   /// flag to fix the width of the 2nd peak
  Bool_t                fFixFracSecWidth;               /// flag to fix the fraction of the 2nd peak width wrt the signal peak width
  Double_t              fVnSecPeak;                     /// vn of second peak from fit
  Bool_t                fDoSecondPeakVn;                /// flag to introduce second peak vn in the vn vs. mass fit
  Bool_t                fFixVnSecPeakToSgn;             /// flag to fix the vn of the second peak to the one of signal
  Double_t              fVnSecPeakUncertainty;          /// vn uncertainty of second peak from fit
  Int_t                 fHarmonic;                      /// harmonic number for drawing
  Bool_t                fTemplates;                     /// flag use/not use templates
  Int_t                 fNParsTempls;                   /// fit parameters to include templates
  std::vector<TF1 *>    fVnCompsDraw;                   /// vector to store TH1 to be added as templates to the fit function 
  std::vector<TF1 *>    fMassTemplatesDraw;             /// vector to store TH1 to be added as templates to the fit function 
  std::vector<Double_t> fRelWeights;                    /// relative weights of templates 
  std::vector<Double_t> fMassWeightsUpperLims;          /// upper limit of the templates' weights
  std::vector<Double_t> fMassWeightsLowerLims;          /// lower limit of the templates' weights
  std::vector<Double_t> fMassInitWeights;               /// init values of the templates' weights
  TemplAnchorMode       fAnchorTemplsMode;              /// init values of the templates' weights
  std::unordered_map<std::string, ParInit> fInitFuncPars;   /// init values of total fit function
  std::vector<TH1*> fHistoTemplates;                    /// vector to store TH1 to be added as templates to the fit function
  Bool_t                fSuppressOutput;                /// flag to suppress outputs (for multitrial fits)

    /// \cond CLASSDEF
  ClassDef(VnVsMassFitter,5);
    /// \endcond
};
#endif //VNVSMASSFITTER