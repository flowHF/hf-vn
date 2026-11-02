#include "TDirectoryFile.h"
#include "TFile.h"
#include "TH2D.h"
#include "TMath.h"
#include "TColor.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TSystem.h"
#include "TGraph.h"
#include "TPad.h"
#include "TF1.h"
#include "TLine.h"
#include "TROOT.h"
#include "TStyle.h"

using std::map;
using std::string;

const int nChFT0A = 96;
const int nChFT0C = 112;
const int nChFT0 = 208;
const int nChFV0 = 48;
const int nFV0Rings = 5;
const int nFV0RingSectors = 8;

const int nChnDrawTogether = 8;
int RainbowColorChnSets[nChnDrawTogether];
int RainbowColorAllFT0A[nChFT0A];
int RainbowColorAllFT0C[nChFT0C];
int RainbowColorAllFV0[nChFV0];
const int NRGBs = 5;
double stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
double red[NRGBs] = {0.00, 0.00, 0.87, 0.9 * 1.00, 0.51};
double green[NRGBs] = {0.00, 0.81, 0.9 * 1.00, 0.20, 0.00};
double blue[NRGBs] = {0.51, 0.9 * 1.00, 0.12, 0.00, 0.00};
void initColors() {
    int sysColorPalletNChnDrawTogether = TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, nChnDrawTogether);
    for (int i=0; i<nChnDrawTogether; i++) {
        RainbowColorChnSets[i] = sysColorPalletNChnDrawTogether + nChnDrawTogether - i - 1;
    }
    int sysColorPalletFT0A = TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, nChFT0A);
    for (int i=0; i<nChFT0A; i++) {
        RainbowColorAllFT0A[i] = sysColorPalletFT0A + nChFT0A - i - 1;
    }
    int sysColorPalletFT0C = TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, nChFT0C);
    for (int i=0; i<nChFT0C; i++) {
        RainbowColorAllFT0C[i] = sysColorPalletFT0C + nChFT0C - i - 1;
    }
    int sysColorPalletFV0 = TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, nChFV0);
    for (int i=0; i<nChFV0; i++) {
        RainbowColorAllFV0[i] = sysColorPalletFV0 + nChFV0 - i - 1;
    }
}

void calcRelGainFT0(std::vector<double>& RelGain,
                    std::vector<double>& mean,
                    std::vector<TH1D*>& hAmpProj,
                    TH2F* hFIT,
                    int nCh,
                    TGraph* gMean,
                    TGraph* gMeanScaled) {

    TH1D* hTotalAmpFT0C;
    TH1D* hTotalAmpFT0A;

    for (int i=0; i<nCh; i++) {
        hAmpProj[i] = (TH1D*)hFIT->ProjectionX(Form("hAmpProj_%d",i), i+1, i+1);
        if (i==0) {
            hTotalAmpFT0C = (TH1D*)hAmpProj[i]->Clone("hTotalAmpFT0C");
            hTotalAmpFT0C->Reset();
            hTotalAmpFT0A = (TH1D*)hAmpProj[i]->Clone("hTotalAmpFT0A");
            hTotalAmpFT0A->Reset();
        }
        if (i < nChFT0A) {
            hTotalAmpFT0A->Add(hAmpProj[i], 1.0);
        } else{
            hTotalAmpFT0C->Add(hAmpProj[i], 1.0);
        }

        mean[i] = hAmpProj[i]->GetMean();
        gMean->SetPoint(i, mean[i], (double)i + 0.5);
        hAmpProj[i]->SetLineColor(RainbowColorChnSets[i % nChnDrawTogether]);

        hAmpProj[i]->GetXaxis()->SetRangeUser(0, 5000);
        hAmpProj[i]->GetYaxis()->SetTitle("Counts");
        hAmpProj[i]->SetLineWidth(2);
    }

    for (int i=0; i<nCh; i++) {
        RelGain[i] = mean[i];
        if (i < nChFT0A) {
            RelGain[i] /= hTotalAmpFT0A->GetMean();
        } else{
            RelGain[i] /= hTotalAmpFT0C->GetMean();
        }
    }
}

void calcRelGainFV0(std::vector<double>& RelGain,
                    std::vector<double>& mean,
                    std::vector<TH1D*>& hAmpProj,
                    TH2F* hFIT,
                    int nCh,
                    TGraph* gMean,
                    TGraph* gMeanScaled) {

    TH1D* hTotalAmp[nFV0Rings];

    for (int i=0; i<nCh; i++) {
        hAmpProj[i] = (TH1D*)hFIT->ProjectionX(Form("hAmpProj_%d",i), i+1, i+1);
        if (i==0) {
            for (int j=0;j<nFV0Rings;j++) {
                hTotalAmp[j] = (TH1D*)hFIT->ProjectionX(Form("hAmpProj_%d_%d",j,i),i+2,i+2);
                hTotalAmp[j]->Reset();
            }
        }

        if (i<40) {  // Last ring has 16 sectors, while the others have 8
            hTotalAmp[i/nFV0RingSectors]->Add(hAmpProj[i], 1.0);
        } else {
            hTotalAmp[4]->Add(hAmpProj[i], 1.0);
        }
        mean[i] = hAmpProj[i]->GetMean();
        gMean->SetPoint(i, mean[i], (double)i + 0.5);
        hAmpProj[i]->SetLineColor(RainbowColorChnSets[i % nChnDrawTogether]);
        hAmpProj[i]->GetXaxis()->SetRangeUser(0,3000);
        hAmpProj[i]->GetYaxis()->SetTitle("Counts");
        hAmpProj[i]->SetLineWidth(2);
    }

    for (int i=0; i<nCh; i++) {
        RelGain[i] = mean[i];
        if (i<40) {  // Last ring has 16 sectors, while the others have 8
            RelGain[i] /= hTotalAmp[i/nFV0RingSectors]->GetMean();
        } else {
            RelGain[i] /= hTotalAmp[4]->GetMean();
        }
    }
}

void calibGainRun(TFile* outFile,
                  std::vector<double>& relGains,
                  std::vector<double>& means,
                  std::vector<double>& meansScaled,
                  std::string outDir,
                  TFile* inFile,
                  int nCh,
                  std::string detName,
                  int run,
                  bool draw) {

    gSystem->mkdir(Form("%s/gainCor/%s/fig_gain_%d", outDir.data(), detName.data(), run), true);

    gStyle->SetTitleFont(43,"X");
    gStyle->SetTitleFont(43,"Y");
    gStyle->SetLabelFont(43,"X");
    gStyle->SetLabelFont(43,"Y");

    gStyle->SetTitleSize(32,"X");
    gStyle->SetTitleSize(32,"Y");
    gStyle->SetLabelSize(28,"X");
    gStyle->SetLabelSize(28,"Y");

    gStyle->SetOptStat(0);

    TCanvas* c = new TCanvas("c","c",800,700);
    gPad->SetLeftMargin(0.15);
    gPad->SetBottomMargin(0.15);
    gPad->SetRightMargin(0.03);
    gPad->SetTopMargin(0.03);
    gPad->SetTicks();
    gPad->SetLogy();

    TLegend* leg = new TLegend(0.4,0.6,0.9,0.9);
    leg->SetTextFont(43);
    leg->SetTextSize(32);
    leg->SetLineWidth(0);
    leg->SetFillStyle(0);
    leg->SetNColumns(2);

    TH2F* hFIT = (TH2F*)inFile->Get(Form("q-vectors-table/%sAmp", detName.data()));
    hFIT->GetYaxis()->SetTitle("Channel ID");
    hFIT->GetXaxis()->SetTitle(Form("%s Amplitude", detName.data()));

    std::vector<TH1D*> hAmpProj(nCh);
    std::vector<double> mean(nCh);
    std::vector<double> RelGain(nCh);
    TGraph* gMean = new TGraph();
    TGraph* gMeanScaled = new TGraph();

    // Calculate relative gain
    if (detName == "FT0") {
        calcRelGainFT0(RelGain, mean, hAmpProj, hFIT, nCh, gMean, gMeanScaled);
    } else if (detName == "FV0") {
        calcRelGainFV0(RelGain, mean, hAmpProj, hFIT, nCh, gMean, gMeanScaled);
    }

    std::vector<TH1D*> hAmpProjScaled(nCh);
    TH2F* hFITScaled = (TH2F*)hFIT->Clone("hFITScaled");
    hFITScaled->Reset();

    for (int i=0; i<nCh; i++) {
        if (RelGain[i] > 1e-4) {
            hAmpProjScaled[i] = new TH1D(Form("hAmpProjScaled_%d",i),"",hAmpProj[i]->GetNbinsX(),
                                        (hAmpProj[i]->GetBinCenter(1) - hAmpProj[i]->GetBinWidth(1)/2.) / RelGain[i],
                                        (hAmpProj[i]->GetBinCenter(hAmpProj[i]->GetNbinsX()) + hAmpProj[i]->GetBinWidth(hAmpProj[i]->GetNbinsX())/2.) / RelGain[i]);
        } else {
            hAmpProjScaled[i] = new TH1D(Form("hAmpProjScaled_%d",i),"",1,0,1);
        }
        for (int j=0; j<hAmpProjScaled[ i]->GetNbinsX();j++) {
            hAmpProjScaled[i]->SetBinContent(j+1, hAmpProj[i]->GetBinContent(j+1));
            hFITScaled->SetBinContent(hFITScaled->GetXaxis()->FindBin(hAmpProjScaled[i]->GetBinCenter(j+1)),
                                      i+1, hAmpProjScaled[i]->GetBinContent(j+1));
        }
        gMeanScaled->SetPoint(i, hAmpProjScaled[i]->GetMean(), (double)i + 0.5);
        means.push_back(hAmpProj[i]->GetMean());
        meansScaled.push_back(hAmpProjScaled[i]->GetMean());
        relGains.push_back(RelGain[i]);
        hAmpProjScaled[i]->SetLineColor(RainbowColorChnSets[i % nChnDrawTogether]);
        hAmpProjScaled[i]->SetTitle(Form(";Scaled %s amp;Counts", detName.data()));
    }
    gMeanScaled->SetMarkerStyle(20);

    for (int iNDraw=0; iNDraw<nCh/nChnDrawTogether; iNDraw++) {
        hAmpProj[iNDraw]->SetMaximum(1e7); //!
        hAmpProj[iNDraw]->Draw();
        leg->Clear();
        if (detName == "FT0" && iNDraw < 12) {
            leg->SetHeader(Form("FT0A, run ID: %d",run));
        } else if (detName == "FT0" && iNDraw >= 12) {
            leg->SetHeader(Form("FT0C, run ID: %d",run));
        } else {
            leg->SetHeader(Form("FV0, run ID: %d",run));
        }
        for (int j=0; j<nChnDrawTogether; j++) {
            hAmpProj[iNDraw*nChnDrawTogether + j]->Draw("same");
            leg->AddEntry(hAmpProj[iNDraw*nChnDrawTogether + j], Form("Ch Id: %d",iNDraw*nChnDrawTogether + j), "l");
        }
        leg->Draw();
        if (draw) c->SaveAs(Form("%s/gainCor/%s/fig_gain_%d/amp_%dId.pdf", outDir.data(), detName.data(), run, iNDraw));
    }

    gPad->SetLogy(0);
    gPad->SetLogz(1);

    gPad->SetRightMargin(0.1);
    if (detName == "FV0") {
        hFIT->GetXaxis()->SetRangeUser(0, 3000);
        hFIT->GetYaxis()->SetRangeUser(-0.5, 50.0);
    }

    hFIT->Draw("colz");
    gMean->SetMarkerStyle(20);
    gMean->Draw("P");

    if (detName == "FT0") {
        TLine* lSepFT0CFT0A = new TLine(0, nChFT0A, 5000, nChFT0A);
        lSepFT0CFT0A->SetLineStyle(2);
        lSepFT0CFT0A->SetLineWidth(3);
        lSepFT0CFT0A->SetLineColor(kBlack);
        lSepFT0CFT0A->Draw("same");
    } else {
        // Draw lines at channels 8, 16, 24 and 32 to separate the rings
        for (int i=1; i<nFV0Rings; i++) {
            TLine* lFV0Ring = new TLine(0, i*nFV0RingSectors, 3000, i*nFV0RingSectors);
            lFV0Ring->SetLineStyle(2);
            lFV0Ring->SetLineWidth(3);
            lFV0Ring->SetLineColor(kBlack);
            lFV0Ring->Draw("same");
        }
    }

    // Config detector
    if (draw) c->SaveAs(Form("%s/gainCor/%s/raw_amps_%d.pdf", outDir.data(), detName.data(), run));

    hFITScaled->GetXaxis()->SetRangeUser(0, 3000);
    if (detName == "FV0") {
        hFITScaled->GetYaxis()->SetRangeUser(-0.5, 50.0);
        hFITScaled->GetXaxis()->SetRangeUser(0, 3000.0);
    }
    hFITScaled->Draw("colz");
    if (detName == "FT0") {
        TLine* lSepFT0CFT0A = new TLine(0, nChFT0A, 3000, nChFT0A);
        lSepFT0CFT0A->SetLineStyle(2);
        lSepFT0CFT0A->SetLineWidth(3);
        lSepFT0CFT0A->SetLineColor(kBlack);
        lSepFT0CFT0A->Draw("same");
    } else {
        // Draw lines at channels 8, 16, 24 and 32 to separate the rings
        for (int i=1; i<nFV0Rings; i++) {
            TLine* lFV0Ring = new TLine(0, i*nFV0RingSectors, 3000, i*nFV0RingSectors);
            lFV0Ring->SetLineStyle(2);
            lFV0Ring->SetLineWidth(3);
            lFV0Ring->SetLineColor(kBlack);
            lFV0Ring->Draw("same");
        }
    }
    gMeanScaled->Draw("P");

    if (draw) c->SaveAs(Form("%s/gainCor/%s/scaled_amps_%d.pdf", outDir.data(), detName.data(), run));

    gPad->SetLogy();
    gPad->SetRightMargin(0.03);
    for (int i=0; i<nCh/nChnDrawTogether; i++) {

        if (hAmpProjScaled[i*nChnDrawTogether]) hAmpProjScaled[i*nChnDrawTogether]->Draw();
        leg->Clear();
        if (detName == "FT0" && i*nChnDrawTogether < nChFT0A) {
            leg->SetHeader(Form("FT0A, run ID: %d",run));
        } else if (detName == "FT0" && i*nChnDrawTogether >= nChFT0A) {
            leg->SetHeader(Form("FT0C, run ID: %d",run));
        } else {
            leg->SetHeader(Form("FV0, run ID: %d",run));
        }
        for (int j=0; j<nChnDrawTogether; j++) {
            if (hAmpProjScaled[i*nChnDrawTogether + j]) hAmpProjScaled[i*nChnDrawTogether + j]->Draw("same");
            leg->AddEntry(hAmpProjScaled[i*nChnDrawTogether + j], Form("Ch Id: %d",i*nChnDrawTogether + j), "l");
        }
        leg->Draw();
        if (draw) c->SaveAs(Form("%s/gainCor/%s/fig_gain_%d/ScaledAmp_%dId.pdf", outDir.data(), detName.data(), run, i));
    }

    leg->Clear();
    // Draw all detector channels, scaled and not scaled
    if (detName == "FT0") {
        for (int i=0; i<nChFT0A; i++) {
            hAmpProjScaled[i]->SetLineColor(RainbowColorAllFT0A[i]);
            hAmpProjScaled[i]->Draw("same");
        }
        if (draw) c->SaveAs(Form("%s/gainCor/all_chns_FT0A_scaled.pdf", outDir.data()));
        for (int i=0; i<nChFT0A; i++) {
            hAmpProj[i]->SetLineColor(RainbowColorAllFT0A[i]);
            hAmpProj[i]->Draw("same");
        }
        if (draw) c->SaveAs(Form("%s/gainCor/all_chns_FT0A_raw.pdf", outDir.data()));
         for (int i=nChFT0A; i<nChFT0; i++) {
            hAmpProjScaled[i]->SetLineColor(RainbowColorAllFT0C[i-nChFT0A]);
            hAmpProjScaled[i]->Draw("same");
        }
        if (draw) c->SaveAs(Form("%s/gainCor/all_chns_FT0C_scaled.pdf", outDir.data()));
        for (int i=nChFT0A; i<nChFT0; i++) {
            hAmpProj[i]->SetLineColor(RainbowColorAllFT0C[i-nChFT0A]);
            hAmpProj[i]->Draw("same");
        }
        if (draw) c->SaveAs(Form("%s/gainCor/all_chns_FT0C_raw.pdf", outDir.data()));
    }
    if (detName == "FV0") {
        for (int i=0; i<nChFV0; i++) {
            hAmpProjScaled[i]->SetLineColor(RainbowColorAllFV0[i]);
            hAmpProjScaled[i]->Draw("same");
        }
        if (draw) c->SaveAs(Form("%s/gainCor/all_chns_FV0_scaled.pdf", outDir.data()));
        for (int i=0; i<nChFV0; i++) {
            hAmpProj[i]->SetLineColor(RainbowColorAllFV0[i]);
            hAmpProj[i]->Draw("same");
        }
        if (draw) c->SaveAs(Form("%s/gainCor/all_chns_FV0_raw.pdf", outDir.data()));
    }

    // Store all distributions to outfile
    if (detName == "FT0") {
        for (int i=0; i<nCh; i++) {
            if (i < nChFT0A) {
                outFile->cd("FT0A");
                hAmpProj[i]->SetLineColor(kBlack);
                hAmpProjScaled[i]->SetLineColor(kBlack);
                hAmpProj[i]->Write(Form("hAmpProj_chn%d", i+1));
                hAmpProjScaled[i]->Write(Form("hAmpProjScaled_chn%d", i+1));
            } else {
                outFile->cd("FT0C");
                hAmpProj[i]->SetLineColor(kBlack);
                hAmpProjScaled[i]->SetLineColor(kBlack);
                hAmpProj[i]->Write(Form("hAmpProj_chn%d", i+1));
                hAmpProjScaled[i]->Write(Form("hAmpProjScaled_chn%d", i+1));
            }
        }
    } else {
        outFile->cd("FV0");
        for (int i=0; i<nCh; i++) {
            hAmpProj[i]->SetLineColor(kBlack);
            hAmpProjScaled[i]->SetLineColor(kBlack);
            hAmpProj[i]->Write(Form("hAmpProj_chn%d", i+1));
            hAmpProjScaled[i]->Write(Form("hAmpProjScaled_chn%d", i+1));
        }
    }

    delete c;
}

int calibGain(std::string outDir,
              std::string inFileDir,
              int run,
              bool draw) {

    gErrorIgnoreLevel = kError;
    gROOT->SetBatch(kTRUE);

    TFile* inFile = new TFile(Form("%s/AnalysisResults_%d.root", inFileDir.data(), run), "read");
    if (!inFile || inFile->IsZombie()) {
        std::cout << "Cannot open file" << std::endl;
        return -1;
    }

    initColors();

    gSystem->mkdir(Form("%s/runs/%d/gainCor", outDir.data(), run), true);
    TFile* outFile = new TFile(Form("%s/runs/%d/gainCor/summary.root", outDir.data(), run), "recreate");
    outFile->mkdir("FT0A");
    outFile->mkdir("FT0C");
    outFile->mkdir("FV0");

    std::cout << "[Run " << run << "] Running gain calibration for FT0" << std::endl;
    std::vector<double> relGainsFT0, meansFT0, meansScaledFT0;
    calibGainRun(outFile, relGainsFT0, meansFT0, meansScaledFT0, Form("%s/runs/%d", outDir.data(), run), inFile, nChFT0, "FT0", run, draw);
    TH1F* histRelGainsFT0A = new TH1F("histRelGainsFT0A", "FT0A Corrections", nChFT0A, 0, nChFT0A);
    TH1F* histMeansFT0A = new TH1F("histMeansFT0A", "FT0A Means", nChFT0A, 0, nChFT0A);
    TH1F* histMeansScaledFT0A = new TH1F("histMeansScaledFT0A", "FT0A Means", nChFT0A, 0, nChFT0A);
    TH1F* histRelGainsFT0C = new TH1F("histRelGainsFT0C", "FT0C Corrections", nChFT0C, 0, nChFT0C);
    TH1F* histMeansFT0C = new TH1F("histMeansFT0C", "FT0C Means", nChFT0C, 0, nChFT0C);
    TH1F* histMeansScaledFT0C = new TH1F("histMeansScaledFT0C", "FT0C Means", nChFT0C, 0, nChFT0C);
    for (size_t iCorr=0; iCorr<relGainsFT0.size(); iCorr++) {
        if (iCorr < nChFT0A) {
            histRelGainsFT0A->SetBinContent(iCorr+1, relGainsFT0[iCorr]);
            histMeansFT0A->SetBinContent(iCorr+1, meansFT0[iCorr]);
            histMeansScaledFT0A->SetBinContent(iCorr+1, meansScaledFT0[iCorr]);
        } else {
            histRelGainsFT0C->SetBinContent(iCorr+1 - nChFT0A, relGainsFT0[iCorr]);
            histMeansFT0C->SetBinContent(iCorr+1 - nChFT0A, meansFT0[iCorr]);
            histMeansScaledFT0C->SetBinContent(iCorr+1 - nChFT0A, meansScaledFT0[iCorr]);
        }
    }
    std::cout << "[Run " << run << "] Ended FT0 gain calibration\n\n\n" << std::endl;

    std::cout << "[Run " << run << "] Running gain calibration for FV0" << std::endl;
    std::vector<double> relGainsFV0, meansFV0, meansScaledFV0;
    calibGainRun(outFile, relGainsFV0, meansFV0, meansScaledFV0, Form("%s/runs/%d", outDir.data(), run), inFile, nChFV0, "FV0", run, draw);
    TH1F* histRelGainsFV0 = new TH1F("histRelGainsFV0", "FV0 Corrections", nChFV0, 0, nChFV0);
    TH1F* histMeansFV0 = new TH1F("histMeansFV0", "FV0 Means", nChFV0, 0, nChFV0);
    TH1F* histMeansScaledFV0 = new TH1F("histMeansScaledFV0", "FV0 Means", nChFV0, 0, nChFV0);
    for (size_t iCorr=0; iCorr<relGainsFV0.size(); iCorr++) {
        histRelGainsFV0->SetBinContent(iCorr+1, relGainsFV0[iCorr]);
        histMeansFV0->SetBinContent(iCorr+1, meansFV0[iCorr]);
        histMeansScaledFV0->SetBinContent(iCorr+1, meansScaledFV0[iCorr]);
    }
    std::cout << "[Run " << run << "] Ended FV0 gain calibration\n\n\n" << std::endl;

    outFile->cd();
    histRelGainsFT0A->Write();
    histRelGainsFT0C->Write();
    histRelGainsFV0->Write();
    histMeansFT0A->Write();
    histMeansFT0C->Write();
    histMeansFV0->Write();
    histMeansScaledFT0A->Write();
    histMeansScaledFT0C->Write();
    histMeansScaledFV0->Write();
    outFile->Close();
    inFile->Close();

    return 0;
}
