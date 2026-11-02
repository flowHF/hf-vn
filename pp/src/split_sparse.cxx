// Single-pass selection and reduction of a THnSparse. No corrections or fits.
// Loaded by preprocess.py through ROOT's CompileMacro.
#pragma once

#include <TAxis.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TH1D.h>
#include <THnSparse.h>
#include <TKey.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace prep2pc {

// Counts refer to allocated source bins, not to the number of pair entries.
// flow counts bins with under/overflow on any source axis, including dropped axes.
struct Totals {
  long long scanned = 0;
  long long flow = 0;
  long long selected = 0;
  double sumw = 0;
  double sumw2 = 0;
};

namespace detail {

constexpr double kEdgeTolerance = 1e-7;
constexpr Long64_t kProgressStep = 5000000;

// Per-output-histogram bookkeeping used for the conservation check.
struct Expected {
  double sumw = 0;
  double sumw2 = 0;
  double sumAbsW = 0;  // Stable scale for checks when signed weights cancel.
};

// Axes of the reduced (kept + rebinned) output histograms.
struct ReducedAxes {
  std::vector<int> nbins;
  std::vector<double> xmin;
  std::vector<double> xmax;
  std::vector<std::vector<double>> edges;
};

// Every requested boundary must coincide with an existing histogram edge.
inline int edgeIndex(const TAxis* axis, double value) {
  const double tol = kEdgeTolerance * std::max(1., std::abs(value));
  for (int i = 1; i <= axis->GetNbins() + 1; ++i) {
    if (std::abs(axis->GetBinLowEdge(i) - value) <= tol) {
      return i;
    }
  }
  throw std::runtime_error("Boundary " + std::to_string(value) +
                           " is not a bin edge of axis '" + axis->GetTitle() +
                           "'. Choose an existing edge.");
}

// Map regular source bins to half-open intervals [edges[j], edges[j + 1]).
// Underflow, overflow and bins outside the requested range remain unassigned (-1).
inline std::vector<int> partition(const TAxis* axis, const std::vector<double>& edges) {
  std::vector<int> binToInterval(axis->GetNbins() + 2, -1);
  for (size_t j = 0; j + 1 < edges.size(); ++j) {
    const int lo = edgeIndex(axis, edges[j]);
    const int hi = edgeIndex(axis, edges[j + 1]);
    if (hi <= lo) {
      throw std::runtime_error("Split edges must increase");
    }
    for (int b = lo; b < hi; ++b) {
      binToInterval[b] = static_cast<int>(j);
    }
  }
  return binToInterval;
}

// Read an independently owned object so its lifetime is explicit at the call site.
inline std::unique_ptr<THnSparse> readSparse(TFile& file, const std::string& objectPath) {
  const auto slash = objectPath.find_last_of('/');
  TDirectory* dir = &file;
  std::string name = objectPath;
  if (slash != std::string::npos) {
    dir = file.GetDirectory(objectPath.substr(0, slash).c_str());
    name = objectPath.substr(slash + 1);
  }
  TKey* key = dir ? dir->GetKey(name.c_str()) : nullptr;
  if (!key) {
    throw std::runtime_error("Missing sparse " + objectPath);
  }

  TObject* object = key->ReadObj();
  auto* sparse = dynamic_cast<THnSparse*>(object);
  if (!sparse) {
    delete object;
    throw std::runtime_error("Input must be a THnSparse");
  }
  return std::unique_ptr<THnSparse>(sparse);
}

inline ReducedAxes buildReducedAxes(const THnSparse& src, const std::vector<int>& keep,
                                   const std::vector<int>& rebin) {
  ReducedAxes out;
  for (size_t a = 0; a < keep.size(); ++a) {
    const TAxis* axis = src.GetAxis(keep[a]);
    if (rebin[a] < 1 || axis->GetNbins() % rebin[a] != 0) {
      throw std::runtime_error("Rebin factor must divide source bin count for axis " +
                               std::to_string(keep[a]));
    }
    if (rebin[a] > 1 && axis->GetLabels()) {
      throw std::runtime_error("Cannot rebin a labelled categorical axis");
    }

    out.nbins.push_back(axis->GetNbins() / rebin[a]);
    out.xmin.push_back(axis->GetXmin());
    out.xmax.push_back(axis->GetXmax());
    std::vector<double> edges;
    // Taking every rebin[a]-th edge also preserves variable-width source axes.
    for (int b = 1; b <= axis->GetNbins() + 1; b += rebin[a]) {
      edges.push_back(axis->GetBinLowEdge(b));
    }
    out.edges.push_back(std::move(edges));
  }
  return out;
}

inline std::unique_ptr<THnSparseD> makeOutputHist(const THnSparse& src,
                                              const std::vector<int>& keep,
                                              const std::vector<int>& rebin,
                                              const ReducedAxes& axes) {
  const int nkeep = static_cast<int>(keep.size());
  auto hist = std::make_unique<THnSparseD>("hPairs", "Uncorrected pairs", nkeep,
                                        axes.nbins.data(), axes.xmin.data(),
                                        axes.xmax.data(), 1024);
  hist->Sumw2();
  for (int a = 0; a < nkeep; ++a) {
    const TAxis* srcAxis = src.GetAxis(keep[a]);
    TAxis* dstAxis = hist->GetAxis(a);
    hist->SetBinEdges(a, axes.edges[a].data());
    dstAxis->SetTitle(srcAxis->GetTitle());
    if (rebin[a] != 1) {
      continue;  // Labels only survive without rebinning.
    }
    for (int b = 1; b <= axes.nbins[a]; ++b) {
      const char* label = srcAxis->GetBinLabel(b);
      if (label && label[0]) {
        dstAxis->SetBinLabel(b, label);
      }
    }
  }
  return hist;
}

// Verifies totals, then writes the histogram and its 1D QA projections.
inline void writeHistogram(THnSparseD& hist, const Expected& expected) {
  double sum = 0;
  double variance = 0;
  for (Long64_t b = 0; b < hist.GetNbins(); ++b) {
    sum += hist.GetBinContent(b);
    variance += hist.GetBinError2(b);
  }
  if (std::abs(sum - expected.sumw) > 1e-10 * std::max(1., expected.sumAbsW) ||
      std::abs(variance - expected.sumw2) > 1e-10 * std::max(1., expected.sumw2)) {
    throw std::runtime_error("Content/variance conservation check failed");
  }

  hist.SetEntries(0);  // Fill counts cannot be recovered from an aggregated sparse.
  if (hist.Write("hPairs") <= 0) {
    throw std::runtime_error("Histogram write failed");
  }

  for (int a = 0; a < hist.GetNdimensions(); ++a) {
    std::unique_ptr<TH1D> proj(hist.Projection(a));
    if (proj->Write(("proj_" + std::to_string(a)).c_str()) <= 0) {
      throw std::runtime_error("Projection write failed");
    }
  }
}

}  // namespace detail

// Splits the sparse into (multiplicity, pT, pT-assoc) intervals, applies per-pT
// cuts, reduces to the kept/rebinned axes and writes one file per (mult, pT).
//
// splitAxes = {multiplicity (-1 = none), pT, pT assoc}; edges has matching order.
// keep defines output axis order; rebin gives one integer factor per kept axis.
// cutLow/cutHigh hold one range per candidate pT bin for each cut axis.
// All split and cut ranges are half-open and must align with source bin edges.
// outputDir must exist. Files contain sample/pt_assoc_<index>/{hPairs, proj_<axis>}.
// axisOrder and configYaml remain unused for compatibility with the Python caller.
inline Totals split(const std::string& filename, const std::string& objectPath,
                    const std::string& outputDir, const std::string& sample,
                    const std::vector<int>& keep, const std::vector<int>& rebin,
                    const std::string& /*axisOrder*/, const std::vector<int>& splitAxes,
                    const std::vector<std::vector<double>>& edges,
                    const std::vector<int>& cutAxes,
                    const std::vector<std::vector<double>>& cutLow,
                    const std::vector<std::vector<double>>& cutHigh,
                    bool allowPoisson, const std::string& /*configYaml*/) {
  using namespace detail;

  // Validate the array shapes before opening the input.
  if (sample != "SE" && sample != "ME") {
    throw std::runtime_error("Sample must be SE or ME");
  }
  if (keep.empty() || rebin.size() != keep.size() || splitAxes.size() != 3 || edges.size() != 3) {
    throw std::runtime_error("Invalid retained/rebin/split axis arrays");
  }
  if (splitAxes[1] < 0 || splitAxes[2] < 0) {
    throw std::runtime_error("Missing pT axes");
  }
  if (cutLow.size() != cutAxes.size() || cutHigh.size() != cutAxes.size()) {
    throw std::runtime_error("Cut array mismatch");
  }

  std::unique_ptr<TFile> input(TFile::Open(filename.c_str(), "READ"));
  if (!input || input->IsZombie()) {
    throw std::runtime_error("Cannot open " + filename);
  }
  auto source = readSparse(*input, objectPath);

  const int ndim = source->GetNdimensions();
  auto checkAxis = [ndim](int a) {
    if (a < 0 || a >= ndim) {
      throw std::runtime_error("Axis index outside sparse dimensions");
    }
  };
  for (int a : keep) {
    checkAxis(a);
  }
  for (int a : splitAxes) {
    if (a != -1) {
      checkAxis(a);
    }
  }
  for (int a : cutAxes) {
    checkAxis(a);
  }

  const bool storedErrors = source->GetCalculateErrors();
  if (!storedErrors && !allowPoisson) {
    throw std::runtime_error(
        "Sparse has no stored Sumw2. Set allow_poisson_errors only for unweighted counts.");
  }
  if (!storedErrors) {
    std::cout << "[" << sample << "] No Sumw2: assuming unweighted Poisson counts" << std::endl;
  }

  // Precompute bin lookups so selection needs no edge searches during the scan.
  const int nMult = splitAxes[0] < 0 ? 1 : static_cast<int>(edges[0].size()) - 1;
  const int nPt = static_cast<int>(edges[1].size()) - 1;
  const int nAssoc = static_cast<int>(edges[2].size()) - 1;
  if (nMult < 1 || nPt < 1 || nAssoc < 1) {
    throw std::runtime_error("Empty splitting intervals");
  }

  std::array<std::vector<int>, 3> binToInterval;
  for (int s = 0; s < 3; ++s) {
    if (splitAxes[s] >= 0) {
      binToInterval[s] = partition(source->GetAxis(splitAxes[s]), edges[s]);
    }
  }

  // cutLowBin[c][p] / cutHighBin[c][p]: accepted bins are [low, high).
  std::vector<std::vector<int>> cutLowBin(cutAxes.size());
  std::vector<std::vector<int>> cutHighBin(cutAxes.size());
  for (size_t c = 0; c < cutAxes.size(); ++c) {
    if (cutLow[c].size() != static_cast<size_t>(nPt) ||
        cutHigh[c].size() != static_cast<size_t>(nPt)) {
      throw std::runtime_error("Cuts require one range per candidate pT bin");
    }
    const TAxis* axis = source->GetAxis(cutAxes[c]);
    for (int p = 0; p < nPt; ++p) {
      cutLowBin[c].push_back(edgeIndex(axis, cutLow[c][p]));
      cutHighBin[c].push_back(edgeIndex(axis, cutHigh[c][p]));
      if (cutLowBin[c].back() >= cutHighBin[c].back()) {
        throw std::runtime_error("Empty cut interval");
      }
    }
  }

  // Flatten (multiplicity, candidate pT, associated pT), with associated pT fastest.
  const auto targetIndex = [nPt, nAssoc](int multBin, int ptBin, int assocBin) {
    return (multBin * nPt + ptBin) * nAssoc + assocBin;
  };
  const int nkeep = static_cast<int>(keep.size());
  const ReducedAxes reducedAxes = buildReducedAxes(*source, keep, rebin);
  const int nTargets = nMult * nPt * nAssoc;
  std::vector<std::unique_ptr<THnSparseD>> output(nTargets);
  std::vector<Expected> expected(nTargets);
  for (auto& hist : output) {
    hist = makeOutputHist(*source, keep, rebin, reducedAxes);
  }

  std::vector<int> axisNbins(ndim);
  for (int a = 0; a < ndim; ++a) {
    axisNbins[a] = source->GetAxis(a)->GetNbins();
  }

  // THnSparse::GetNbins() counts allocated bins, not the full Cartesian grid.
  std::cout << "[" << sample << "] scanning " << source->GetNbins()
            << " allocated bins once" << std::endl;

  Totals totals;
  std::vector<int> coord(ndim);
  std::vector<int> reduced(nkeep);
  for (Long64_t i = 0; i < source->GetNbins(); ++i) {
    const double value = source->GetBinContent(i, coord.data());
    ++totals.scanned;
    if (i && i % kProgressStep == 0) {
      std::cout << "\r[" << sample << "] " << i << " bins scanned" << std::flush;
    }

    // Exclude flow on all axes before applying splits or dropping dimensions.
    bool isFlow = false;
    for (int a = 0; a < ndim && !isFlow; ++a) {
      isFlow = coord[a] < 1 || coord[a] > axisNbins[a];
    }
    if (isFlow) {
      ++totals.flow;
      continue;
    }

    const int multBin = splitAxes[0] < 0 ? 0 : binToInterval[0][coord[splitAxes[0]]];
    const int ptBin = binToInterval[1][coord[splitAxes[1]]];
    const int assocBin = binToInterval[2][coord[splitAxes[2]]];
    if (multBin < 0 || ptBin < 0 || assocBin < 0) {
      continue;
    }

    bool accepted = true;
    for (size_t c = 0; c < cutAxes.size() && accepted; ++c) {
      const int cutBin = coord[cutAxes[c]];
      accepted = cutBin >= cutLowBin[c][ptBin] && cutBin < cutHighBin[c][ptBin];
    }
    if (!accepted) {
      continue;
    }

    const double error2 = storedErrors ? source->GetBinError2(i) : value;
    if (!std::isfinite(value) || !std::isfinite(error2) || error2 < 0) {
      throw std::runtime_error("Nonfinite content or invalid variance in selected bin");
    }
    if (!storedErrors && (value < 0 || std::abs(value - std::round(value)) > 1e-7)) {
      throw std::runtime_error("Poisson fallback requires nonnegative integer contents");
    }

    const int target = targetIndex(multBin, ptBin, assocBin);
    // ROOT bin coordinates are one-based; integer division merges adjacent bins.
    for (int a = 0; a < nkeep; ++a) {
      reduced[a] = (coord[keep[a]] - 1) / rebin[a] + 1;
    }
    if (value != 0 || error2 != 0) {
      const Long64_t bin = output[target]->GetBin(reduced.data(), true);
      // Add the stored variances, rather than squaring aggregated bin contents.
      output[target]->AddBinContent(bin, value);
      output[target]->AddBinError2(bin, error2);
    }

    Expected& targetTotals = expected[target];
    targetTotals.sumw += value;
    targetTotals.sumw2 += error2;
    targetTotals.sumAbsW += std::abs(value);
    ++totals.selected;
    totals.sumw += value;
    totals.sumw2 += error2;
  }
  std::cout << "\r[" << sample << "] " << totals.scanned << " bins scanned" << std::endl;

  // Write one file per (multiplicity, candidate pT), with a directory per assoc pT.
  for (int m = 0; m < nMult; ++m) {
    for (int p = 0; p < nPt; ++p) {
      const std::string path = outputDir + "/mult_" + std::to_string(m) +
                               "_pt_" + std::to_string(p) + ".root";
      // SE is written first and recreates the file; ME appends to it.
      std::unique_ptr<TFile> file(TFile::Open(path.c_str(), sample == "SE" ? "RECREATE" : "UPDATE"));
      if (!file || file->IsZombie()) {
        throw std::runtime_error("Cannot write " + path);
      }

      TDirectory* sampleDir = file->mkdir(sample.c_str());
      sampleDir->cd();
      for (int h = 0; h < nAssoc; ++h) {
        const int target = targetIndex(m, p, h);
        sampleDir->mkdir(("pt_assoc_" + std::to_string(h)).c_str())->cd();
        writeHistogram(*output[target], expected[target]);
        output[target].reset();  // Free memory as soon as it is written.
      }
      file->Close();
      if (file->TestBit(TFile::kWriteError)) {
        throw std::runtime_error("ROOT write error: " + path);
      }
    }
  }
  return totals;
}

}  // namespace prep2pc
