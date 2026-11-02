#include <TFile.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TStyle.h>
#include <TColor.h>
#include <TLegend.h>
#include <TSystem.h>
#include <TStyle.h>
#include <TROOT.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>

std::array<std::string, 6> qVecCorrectionHistos = {
    "hParRecenterX",
    "hParRecenterY",
    "hParTwistX",
    "hParTwistY",
    "hParRescaleX",
    "hParRescaleY"
};

std::array<std::string, 6> qVecCorrectionAxesLabels = {
    "Recenter X Coordinate",
    "Recenter Y Coordinate",
    "Twist X Coordinate",
    "Twist Y Coordinate",
    "Rescale X Coordinate",
    "Rescale Y Coordinate"
};

std::array<std::string, 7> detNames = {
    "FT0C",
    "FT0A",
    "FT0M",
    "FV0",
    "TPCPOS",
    "TPCNEG",
    "TPCALL"
};

struct Range {
    double ymin =  1e30;
    double ymax = -1e30;
};

const int nChFT0A = 96;
const int nChFT0C = 112;
const int nChFT0 = 208;
const int nChFV0 = 48;
std::array<int, 3> nChns = {nChFT0A, nChFT0C, nChFV0};

std::map<std::string, Range> ranges;

void qaQVecCorrections(const std::string& outDir,
                       const std::vector<int>& runNums,
                       const std::vector<int>& runColors,
                       bool isSp)
{
    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    gStyle->SetPadTickX(1);
    gStyle->SetPadTickY(1);

    gStyle->SetTitleFont(42, "XYZ");
    gStyle->SetLabelFont(42, "XYZ");

    gStyle->SetTitleSize(0.045, "XYZ");
    gStyle->SetLabelSize(0.04, "XYZ");

    gStyle->SetTitleOffset(1.2, "Y");
    gStyle->SetTitleOffset(1.1, "X");

    gStyle->SetLegendBorderSize(0);
    gStyle->SetLegendFillColor(0);
    gStyle->SetLegendFont(42);

    gStyle->SetCanvasBorderMode(0);
    gStyle->SetFrameBorderMode(0);

    gStyle->SetPadLeftMargin(0.12);
    gStyle->SetPadRightMargin(0.05);
    gStyle->SetPadBottomMargin(0.12);
    gStyle->SetPadTopMargin(0.05);

    const std::string mode = isSp ? "SP" : "EsE";

    // -----------------------------
    // Range storage (FIRST PASS)
    // -----------------------------
    struct Range {
        double ymin =  1e30;
        double ymax = -1e30;
        bool filled = false;
    };

    std::map<std::string, Range> ranges;

    // ============================================================
    // PASS 1: compute global min/max per (det,param)
    // ============================================================
    for (size_t iRun = 0; iRun < runNums.size(); ++iRun) {

        const int run = runNums[iRun];

        TString fileName = Form("%s/runs/%d/%s/qvecCorrs/qvec_corrections.root",
                                outDir.c_str(), run, mode.c_str());

        std::unique_ptr<TFile> file(TFile::Open(fileName, "READ"));
        if (!file || file->IsZombie()) continue;

        for (const auto& det : detNames) {
            for (const auto& param : qVecCorrectionHistos) {

                TString hName = Form("%s/%s_%s",
                                    det.c_str(),
                                    param.c_str(),
                                    det.c_str());

                TH1F* h = (TH1F*)file->Get(hName);
                if (!h) continue;

                std::string key = det + "_" + param;
                auto& r = ranges[key];

                r.filled = true;

                r.ymax = std::max(r.ymax, h->GetMaximum());
                r.ymin = std::min(r.ymin, h->GetMinimum());
            }
        }
    }

    // -----------------------------
    // Plot container
    // -----------------------------
    struct PlotInfo {
        TCanvas* canvas = nullptr;
        TLegend* legend = nullptr;
        bool firstDraw = true;
        double ymin = 1e9;
        double ymax = -1e9;
    };

    std::map<std::string, PlotInfo> plots;

    // ============================================================
    // CREATE CANVASES + APPLY GLOBAL RANGES
    // ============================================================
    for (const auto& det : detNames) {
        for (const auto& param : qVecCorrectionHistos) {

            std::string key = det + "_" + param;

            auto it = ranges.find(key);
            if (it == ranges.end() || !it->second.filled) continue;

            auto& r = it->second;

            int idxHisto = std::distance(qVecCorrectionHistos.begin(),
                                         std::find(qVecCorrectionHistos.begin(),
                                         qVecCorrectionHistos.end(),
                                         param));
            PlotInfo p;
            p.canvas = new TCanvas(Form("c_%s", key.c_str()), Form("%s (%s)", qVecCorrectionAxesLabels[idxHisto].c_str(), det.c_str()), 800, 700);

            p.canvas->SetLeftMargin(0.15);
            p.canvas->SetBottomMargin(0.15);
            p.canvas->SetRightMargin(0.03);
            p.canvas->SetTopMargin(0.03);
            p.canvas->SetTicks();

            p.legend = new TLegend(0.60, 0.60, 0.88, 0.88);
            p.legend->SetBorderSize(0);
            p.legend->SetFillStyle(0);

            double margin = 0.5;
            float semiDiffMinMax = std::abs(r.ymax - r.ymin) * 0.5;
            p.ymin = r.ymin - (margin * semiDiffMinMax);
            p.ymax = r.ymax + (margin * semiDiffMinMax);
            plots[key] = p;
        }
    }

    // ============================================================
    // PASS 2: DRAW
    // ============================================================
    for (size_t iRun = 0; iRun < runNums.size(); ++iRun) {

        const int run = runNums[iRun];

        TString fileName = Form("%s/runs/%d/%s/qvecCorrs/qvec_corrections.root",
            outDir.c_str(), run, mode.c_str());

            std::unique_ptr<TFile> file(TFile::Open(fileName, "READ"));
        if (!file || file->IsZombie()) continue;

        for (const auto& det : detNames) {
            for (const auto& param : qVecCorrectionHistos) {

                TString hName = Form("%s/%s_%s",
                                    det.c_str(),
                                    param.c_str(),
                                    det.c_str());

                TH1F* h = (TH1F*)file->Get(hName);
                if (!h) continue;

                std::string key = det + "_" + param;
                auto& plot = plots[key];

                plot.canvas->cd();

                h->SetLineColor(runColors[iRun]);
                h->SetLineWidth(2);

                if (plot.firstDraw) {

                    h->GetXaxis()->SetTitle("Centrality (%)");
                    h->GetYaxis()->SetTitle(Form("%s (%s)", qVecCorrectionAxesLabels[std::distance(qVecCorrectionHistos.begin(),
                                                                                     std::find(qVecCorrectionHistos.begin(),
                                                                                               qVecCorrectionHistos.end(),
                                                                                               param))].c_str(), det.c_str()));

                    h->SetMinimum(plot.ymin);
                    h->SetMaximum(plot.ymax);

                    h->DrawCopy("hist");
                    plot.firstDraw = false;

                } else {
                    h->DrawCopy("hist same");
                }

                plot.legend->AddEntry((TObject*)nullptr,
                                      Form("%d", run),
                                      "l");
            }
        }
    }

    // ============================================================
    // SAVE OUTPUT
    // ============================================================
    for (const auto& det : detNames) {
        for (const auto& param : qVecCorrectionHistos) {

            std::string key = det + "_" + param;
            auto& plot = plots[key];

            if (!plot.canvas) continue;

            plot.canvas->cd();
            // plot.legend->Draw();

            plot.canvas->SaveAs(Form("%s/QA/%s/qvec_corrections_%s_%s.pdf",
                                     outDir.c_str(),
                                     mode.c_str(),
                                     det.c_str(),
                                     param.c_str()));
        }
    }

    // ============================================================
    // CLEANUP
    // ============================================================
    for (auto& [key, plot] : plots) {
        delete plot.legend;
        delete plot.canvas;
    }
}

void qaGainCalibration(const std::string& outDir,
                       const std::vector<int>& runNums,
                       const std::vector<int>& runColors,
                       const std::vector<int>& ft0aColors,
                       const std::vector<int>& ft0cColors,
                       const std::vector<int>& fv0Colors)
{

    gStyle->SetOptStat(0);
    gStyle->SetOptTitle(0);

    gStyle->SetPadTickX(1);
    gStyle->SetPadTickY(1);

    gStyle->SetTitleFont(42, "XYZ");
    gStyle->SetLabelFont(42, "XYZ");

    gStyle->SetTitleSize(0.045, "XYZ");
    gStyle->SetLabelSize(0.04, "XYZ");

    gStyle->SetTitleOffset(1.2, "Y");
    gStyle->SetTitleOffset(1.1, "X");

    gStyle->SetLegendBorderSize(0);
    gStyle->SetLegendFillColor(0);
    gStyle->SetLegendFont(42);

    gStyle->SetCanvasBorderMode(0);
    gStyle->SetFrameBorderMode(0);

    gStyle->SetPadLeftMargin(0.12);
    gStyle->SetPadRightMargin(0.05);
    gStyle->SetPadBottomMargin(0.12);
    gStyle->SetPadTopMargin(0.05);

    std::array<std::string, 3> detNames = {"FT0A", "FT0C", "FV0"};

    std::array<double, 3> minGains = { 1e9,  1e9,  1e9};
    std::array<double, 3> maxGains = {-1e9, -1e9, -1e9};

    std::vector<TH1F*> hFT0A(runNums.size(), nullptr);
    std::vector<TH1F*> hFT0C(runNums.size(), nullptr);
    std::vector<TH1F*> hFV0(runNums.size(), nullptr);

    // =========================
    // PASS 1: LOAD + CLONE
    // =========================
    for (size_t runIdx = 0; runIdx < runNums.size(); ++runIdx) {

        TString fileName = Form("%s/runs/%d/gainCor/summary.root",
                                outDir.c_str(), runNums[runIdx]);

        std::unique_ptr<TFile> file(TFile::Open(fileName, "READ"));
        if (!file || file->IsZombie()) continue;

        TH1F* hA = (TH1F*)file->Get("histRelGainsFT0A");
        TH1F* hC = (TH1F*)file->Get("histRelGainsFT0C");
        TH1F* hV = (TH1F*)file->Get("histRelGainsFV0");

        if (hA) {
            hFT0A[runIdx] = (TH1F*)hA->Clone(Form("hFT0A_run_%zu", runIdx));
            hFT0A[runIdx]->SetDirectory(nullptr);
            minGains[0] = std::min(minGains[0], hFT0A[runIdx]->GetMinimum());
            maxGains[0] = std::max(maxGains[0], hFT0A[runIdx]->GetMaximum());
        }

        if (hC) {
            hFT0C[runIdx] = (TH1F*)hC->Clone(Form("hFT0C_run_%zu", runIdx));
            hFT0C[runIdx]->SetDirectory(nullptr);
            minGains[1] = std::min(minGains[1], hFT0C[runIdx]->GetMinimum());
            maxGains[1] = std::max(maxGains[1], hFT0C[runIdx]->GetMaximum());
        }

        if (hV) {
            hFV0[runIdx] = (TH1F*)hV->Clone(Form("hFV0_run_%zu", runIdx));
            hFV0[runIdx]->SetDirectory(nullptr);
            minGains[2] = std::min(minGains[2], hFV0[runIdx]->GetMinimum());
            maxGains[2] = std::max(maxGains[2], hFV0[runIdx]->GetMaximum());
        }
    }

    // =========================
    // CANVAS
    // =========================

    TCanvas* c = new TCanvas("cRelGains", "Relative Gains", 1200, 400);
    c->Divide(3,1);

    for (size_t i = 0; i < detNames.size(); ++i) {

        c->cd(i+1);
        gPad->SetTicks();

        std::vector<TH1F*>* vec = nullptr;

        if (detNames[i] == "FT0A") vec = &hFT0A;
        if (detNames[i] == "FT0C") vec = &hFT0C;
        if (detNames[i] == "FV0") vec = &hFV0;

        if (!vec) continue;

        // -------------------------
        // CREATE FRAME HISTOGRAM
        // -------------------------
        int nBins = 1;
        double xMin = 0;
        double xMax = 1;

        TH1F* frame = new TH1F(
            Form("frame_%s", detNames[i].c_str()),
            Form(";Channel %s;Relative Gain", detNames[i].c_str()),
            nBins, 0, nChns[i]
        );

        frame->SetMinimum(minGains[i] - 0.1);
        frame->SetMaximum(maxGains[i] + 0.1);

        frame->Draw("axis");

        // -------------------------
        // DRAW ALL RUNS
        // -------------------------
        for (size_t runIdx = 0; runIdx < runNums.size(); ++runIdx) {

            TH1F* hRun = (*vec)[runIdx];
            if (!hRun) continue;

            hRun->SetLineColor(runColors[runIdx]);
            hRun->SetLineWidth(2);
            hRun->Draw("hist same");
        }
    }
    c->SaveAs(Form("%s/QA/GainCalib/gain_calibration_comparison.pdf", outDir.c_str()));

    // cleanup
    for (auto* h : hFT0A) delete h;
    for (auto* h : hFT0C) delete h;
    for (auto* h : hFV0) delete h;
    delete c;
}

int produceQa(std::string corrsDir,
              std::string runsFile)
{
    gROOT->SetBatch(kTRUE);
    TH1::AddDirectory(kFALSE);
    gSystem->mkdir(Form("%s/QA", corrsDir.c_str()), true);
    std::cout << "Created folder for QA plots: " << Form("%s/QA", corrsDir.c_str()) << std::endl;

    std::ifstream infile(runsFile);
    if (!infile.is_open()) {
        std::cerr << "Cannot open " << runsFile << std::endl;
        return -1;
    }

    std::vector<int> runNums;
    std::string line;
    std::getline(infile, line); // skip header

    while (std::getline(infile, line)) {

        std::istringstream iss(line);

        int run;
        long long sor;
        long long eor;

        if (iss >> run >> sor >> eor) {
            runNums.push_back(run);
        }
    }
    infile.close();

    const int nRuns = runNums.size();

    std::vector<int> runColors(nRuns);

    const int NRGBs = 5;

    double stops[NRGBs] = {0.00, 0.34, 0.61, 0.84, 1.00};
    double red[NRGBs]   = {0.00, 0.00, 0.87, 0.90, 0.51};
    double green[NRGBs] = {0.00, 0.81, 0.90, 0.20, 0.00};
    double blue[NRGBs]  = {0.51, 0.90, 0.12, 0.00, 0.00};

    int colorsForRuns =
        TColor::CreateGradientColorTable(
            NRGBs,
            stops,
            red,
            green,
            blue,
            nRuns);

    for (int i = 0; i < nRuns; ++i) {
        runColors[i] = colorsForRuns + nRuns - i - 1;
    }

    int colorsForFT0A =
        TColor::CreateGradientColorTable(
            NRGBs,
            stops,
            red,
            green,
            blue,
            nChFT0A);

    std::vector<int> ft0aColors(nChFT0A);
    for (int i = 0; i < nChFT0A; ++i) {
        ft0aColors[i] = colorsForFT0A + nChFT0A - i - 1;
    }

    int colorsForFT0C =
        TColor::CreateGradientColorTable(
            NRGBs,
            stops,
            red,
            green,
            blue,
            nChFT0C);

    std::vector<int> ft0cColors(nChFT0C);
    for (int i = 0; i < nChFT0C; ++i) {
        ft0cColors[i] = colorsForFT0C + nChFT0C - i - 1;
    }

    int colorsForFV0 =
        TColor::CreateGradientColorTable(
            NRGBs,
            stops,
            red,
            green,
            blue,
            nChFV0);

    std::vector<int> fv0Colors(nChFV0);
    for (int i = 0; i < nChFV0; ++i) {
        fv0Colors[i] = colorsForFV0 + nChFV0 - i - 1;
    }

    gSystem->mkdir(Form("%s/QA/SP", corrsDir.c_str()), true);
    qaQVecCorrections(
        corrsDir.c_str(),
        runNums,
        runColors,
        true);

    gSystem->mkdir(Form("%s/QA/EsE", corrsDir.c_str()), true);
    qaQVecCorrections(
        corrsDir.c_str(),
        runNums,
        runColors,
        false);

    gSystem->mkdir(Form("%s/QA/GainCalib", corrsDir.c_str()), true);
    qaGainCalibration(corrsDir.c_str(),
                      runNums,
                      runColors,
                      ft0aColors,
                      ft0cColors,
                      fv0Colors);

    return 0;
}