#ifndef HFVN_CORR2PC_H
#define HFVN_CORR2PC_H

// Construct acceptance-corrected two-particle TH2 correlation maps.
// SE denotes same-event pairs; ME denotes mixed-event pairs.
//
// For each configured mass interval:
//   hSE   = raw SE map summed over mixing pools
//   hME   = normalized ME map summed over mixing pools (QA)
//   hCorr = sum_pool [ SE_pool(mass interval) / ME_pool_normalized ]
//
// IMPORTANT:
// hCorr is always constructed pool-by-pool before summing.
// It is NOT computed as hSE / hME.
//
// ME normalization reproduces the historical convention:
// average of the four bins around (DeltaEta,DeltaPhi)=(0,0),
// selected with FindBin(-eps) and FindBin(+eps), eps=0.001.
//
// Performance:
//   - ME THnSparse scanned once per pt_assoc
//   - SE THnSparse scanned once per pt_assoc
//   - no THnSparse::Projection calls
//   - correlation outputs are TH2 maps, with a TH1 normalization QA histogram
#include <TAxis.h>
#include <TCollection.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <THnSparse.h>
#include <TKey.h>
#include <TNamed.h>
#include <TObject.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace corr2pc {

struct Settings {
  // Axis indices in the preprocessed SE and ME sparse histograms.
  int poolAxis = 0;
  int massAxis = 1;
  int etaAxis = 2;
  int phiAxis = 3;

  // Coordinates used to select the four central ME normalization bins.
  double normEpsilon = 0.001;

  // Requested final mass-bin edges.
  // Every edge must coincide with an existing edge of the
  // preprocessed mass axis.
  std::vector<double> massBins;
};

inline void validateAxis(const THnSparseD* h, int axis, const std::string& name) {
  if (axis < 0 || axis >= h->GetNdimensions()) {
    throw std::runtime_error("Invalid " + name + "=" + std::to_string(axis) + " for THnSparseD with " +
        std::to_string(h->GetNdimensions()) + " dimensions");
  }
}

inline void validateSettings(const THnSparseD* h, const Settings& setting) {
  validateAxis(h, setting.poolAxis, "poolAxis");
  validateAxis(h, setting.massAxis, "massAxis");
  validateAxis(h, setting.etaAxis, "etaAxis");
  validateAxis(h, setting.phiAxis, "phiAxis");

  std::vector<int> axes = {setting.poolAxis, setting.massAxis, setting.etaAxis, setting.phiAxis};

  std::sort(axes.begin(), axes.end());

  if (std::adjacent_find(axes.begin(), axes.end()) != axes.end()) {
    throw std::runtime_error("poolAxis, massAxis, etaAxis and phiAxis must be distinct");
  }

  if (!std::isfinite(setting.normEpsilon) || setting.normEpsilon <= 0.) {
    throw std::runtime_error("normEpsilon must be finite and > 0");
  }

  if (setting.massBins.size() < 2) {
    throw std::runtime_error("At least two correlation.mass_bins edges are required");
  }

  for (size_t i = 0; i + 1 < setting.massBins.size(); ++i) {
    if (!(setting.massBins[i + 1] > setting.massBins[i])) {
      throw std::runtime_error("correlation.mass_bins must be strictly increasing");
    }
  }
}

inline TDirectory* mkdirChecked(TDirectory* parent, const std::string& name) {
  if (!parent) {
    throw std::runtime_error("Cannot create directory '" + name + "' from null parent");
  }

  if (auto* existing = parent->GetDirectory(name.c_str())) {
    return existing;
  }

  auto* created = parent->mkdir(name.c_str());

  if (!created) {
    throw std::runtime_error("Could not create ROOT directory '" + name + "'");
  }

  return created;
}

// Return the ROOT bin whose lower edge matches value; N+1 denotes the upper edge.
inline int edgeIndex(const TAxis* axis, double value) {
  for (int i = 1; i <= axis->GetNbins() + 1; ++i) {
    const double edge = axis->GetBinLowEdge(i);

    if (std::abs(edge - value) <= 1.e-7 * std::max(1., std::abs(value))) {
      return i;
    }
  }

  throw std::runtime_error("Requested mass boundary " + std::to_string(value) +
      " is not an existing edge of the preprocessed mass axis");
}

// Map ROOT mass-bin numbers to zero-based output intervals [low, high).
// Underflow, overflow and bins outside the requested range retain the -1 sentinel.
inline std::vector<int> buildMassMap(const TAxis* axis, const std::vector<double>& edges) {
  std::vector<int> map(axis->GetNbins() + 2, -1);

  for (size_t i = 0; i + 1 < edges.size(); ++i) {
    const int low = edgeIndex(axis, edges[i]);
    const int high = edgeIndex(axis, edges[i + 1]);

    if (high <= low) {
      throw std::runtime_error("Invalid configured mass interval");
    }

    for (int bin = low; bin < high; ++bin) {
      map[bin] = static_cast<int>(i);
    }
  }

  return map;
}

// Preserve the source bin edges, including variable-width binning.
inline std::vector<double> axisEdges(const TAxis* axis) {
  std::vector<double> edges(axis->GetNbins() + 1);

  for (int i = 1; i <= axis->GetNbins(); ++i) {
    edges[i - 1] = axis->GetBinLowEdge(i);
  }

  edges.back() = axis->GetBinUpEdge(axis->GetNbins());

  return edges;
}

inline std::unique_ptr<TH2D> makeTH2(const THnSparseD* source, const Settings& setting,
    const std::string& name) {
  const auto etaEdges = axisEdges(source->GetAxis(setting.etaAxis));
  const auto phiEdges = axisEdges(source->GetAxis(setting.phiAxis));

  auto h = std::make_unique<TH2D>(name.c_str(), name.c_str(), source->GetAxis(setting.etaAxis)->GetNbins(),
      etaEdges.data(), source->GetAxis(setting.phiAxis)->GetNbins(), phiEdges.data());

  // The unique_ptr owns this histogram, independently of the current ROOT file.
  h->SetDirectory(nullptr);
  h->Sumw2();

  h->GetXaxis()->SetTitle(source->GetAxis(setting.etaAxis)->GetTitle());

  h->GetYaxis()->SetTitle(source->GetAxis(setting.phiAxis)->GetTitle());

  return h;
}

// Flatten regular ROOT bins (one-based) into an eta-major, zero-based array.
// Flow bins must be rejected before calling this helper.
inline int flatBin(int iEta, int iPhi, int nPhi) {
  return (iEta - 1) * nPhi + (iPhi - 1);
}

// Discover SE momentum selections and require a matching ME directory for each.
inline std::vector<std::string> ptAssocDirectories(TFile* file) {
  auto* seDir = file->GetDirectory("SE");
  auto* meDir = file->GetDirectory("ME");

  if (!seDir || !meDir) {
    throw std::runtime_error("Input file must contain both SE/ and ME/");
  }

  std::vector<std::string> names;

  TIter next(seDir->GetListOfKeys());

  while (auto* object = next()) {
    auto* key = dynamic_cast<TKey*>(object);

    if (!key) {
      continue;
    }

    const std::string name = key->GetName();

    if (name.rfind("pt_assoc_", 0) != 0) {
      continue;
    }

    if (!seDir->GetDirectory(name.c_str())) {
      continue;
    }

    if (!meDir->GetDirectory(name.c_str())) {
      throw std::runtime_error("Missing ME/" + name);
    }

    names.push_back(name);
  }

  std::sort(names.begin(), names.end());

  if (names.empty()) {
    throw std::runtime_error("No SE/pt_assoc_* directories found");
  }

  return names;
}

// Write pt_assoc_*/mass_*/{hSE,hME,hCorr} and return the number of mass triplets.
inline int processFile(const std::string& inputFilename, const std::string& outputFilename,
    const Settings& setting) {
  std::unique_ptr<TFile> input(TFile::Open(inputFilename.c_str(), "READ"));

  if (!input || input->IsZombie()) {
    throw std::runtime_error("Cannot open input file " + inputFilename);
  }

  std::unique_ptr<TFile> output(TFile::Open(outputFilename.c_str(), "RECREATE"));

  if (!output || output->IsZombie()) {
    throw std::runtime_error("Cannot create output file " + outputFilename);
  }

  const auto ptDirs = ptAssocDirectories(input.get());

  int nMapsWritten = 0;

  for (const auto& ptDirName : ptDirs) {
    auto* se = dynamic_cast<THnSparseD*>(input->Get(("SE/" + ptDirName + "/hPairs").c_str()));
    auto* me = dynamic_cast<THnSparseD*>(input->Get(("ME/" + ptDirName + "/hPairs").c_str()));

    if (!se || !me) {
      throw std::runtime_error("Missing THnSparseD hPairs for " + ptDirName + " in " + inputFilename);
    }

    validateSettings(se, setting);
    validateSettings(me, setting);

    const int nPools = se->GetAxis(setting.poolAxis)->GetNbins();
    const int nEta = se->GetAxis(setting.etaAxis)->GetNbins();
    const int nPhi = se->GetAxis(setting.phiAxis)->GetNbins();
    const int nMassOut = static_cast<int>(setting.massBins.size()) - 1;
    const int nCorrBins = nEta * nPhi;

    if (me->GetAxis(setting.poolAxis)->GetNbins() != nPools || me->GetAxis(setting.etaAxis)->GetNbins() != nEta ||
        me->GetAxis(setting.phiAxis)->GetNbins() != nPhi) {
      throw std::runtime_error("SE/ME pool or correlation-axis binning mismatch");
    }

    const auto massMap = buildMassMap(se->GetAxis(setting.massAxis), setting.massBins);

    // Accumulate ME as [pool][eta, phi], integrating over mass and any other axes.
    // The integration over the mass is valid under the assumption that the ME is mass-independent.
    // Iterate over allocated sparse bins without creating projections.
    std::vector<double> meContent(static_cast<size_t>(nPools) * nCorrBins, 0.);
    std::vector<double> meVariance(static_cast<size_t>(nPools) * nCorrBins, 0.);
    std::vector<int> coordME(me->GetNdimensions());
    const bool meStoredErrors = me->GetCalculateErrors();

    for (Long64_t ibin = 0; ibin < me->GetNbins(); ++ibin) {
      const double value = me->GetBinContent(ibin, coordME.data());
      const int pool = coordME[setting.poolAxis];
      const int iEta = coordME[setting.etaAxis];
      const int iPhi = coordME[setting.phiAxis];

      if (pool < 1 || pool > nPools || iEta < 1 || iEta > nEta || iPhi < 1 || iPhi > nPhi) {
        continue;
      }

      const int corrBin = flatBin(iEta, iPhi, nPhi);
      const size_t idx = static_cast<size_t>(pool - 1) * nCorrBins + corrBin;

      // Use stored squared errors for weighted data, otherwise Poisson variance.
      const double error2 = meStoredErrors ? me->GetBinError2(ibin) : value;

      meContent[idx] += value;
      meVariance[idx] += error2;
    }

    // Normalize each pool using the historical four-bin average around zero.
    // Selections may coincide when +/-epsilon fall in the same source bin.
    const TAxis* etaAxis = me->GetAxis(setting.etaAxis);
    const TAxis* phiAxis = me->GetAxis(setting.phiAxis);
    const int ixL = etaAxis->FindBin(-setting.normEpsilon);
    const int ixR = etaAxis->FindBin(+setting.normEpsilon);
    const int iyD = phiAxis->FindBin(-setting.normEpsilon);
    const int iyU = phiAxis->FindBin(+setting.normEpsilon);

    TH1D hMENormalization("hMENormalization", "ME average around (0,0);mixing pool;average ME", nPools,
        0.5, nPools + 0.5);

    hMENormalization.SetDirectory(nullptr);

    // QA map: normalized ME summed over pools.
    std::vector<double> meSumContent(nCorrBins, 0.);
    std::vector<double> meSumVariance(nCorrBins, 0.);

    for (int pool = 1; pool <= nPools; ++pool) {
      const size_t offset = static_cast<size_t>(pool - 1) * nCorrBins;

      const double avg = (
          meContent[offset + flatBin(ixL, iyD, nPhi)] +
          meContent[offset + flatBin(ixR, iyD, nPhi)] +
          meContent[offset + flatBin(ixL, iyU, nPhi)] +
          meContent[offset + flatBin(ixR, iyU, nPhi)]) / 4.;

      hMENormalization.SetBinContent(pool, avg);

      if (avg > 0.) {
        const double scale = 1. / avg;

        // Treat the normalization as fixed: scale variances by the factor squared.
        const double scale2 = scale * scale;

        for (int icorr = 0; icorr < nCorrBins; ++icorr) {
          const size_t idx = offset + icorr;

          meContent[idx] *= scale;
          meVariance[idx] *= scale2;
          meSumContent[icorr] += meContent[idx];
          meSumVariance[icorr] += meVariance[idx];
        }
      } else {
        std::cerr << "[corr2pc] " << ptDirName << ": ME normalization not properly working "
            << "for pool " << (pool - 1) << ", average around (0,0) = " << avg << std::endl;

        // Preserve previous behavior:
        // if normalization fails, use the unscaled ME.
        for (int icorr = 0; icorr < nCorrBins; ++icorr) {
          const size_t idx = offset + icorr;

          meSumContent[icorr] += meContent[idx];
          meSumVariance[icorr] += meVariance[idx];
        }
      }
    }

    // Accumulate SE and corrected maps as [output mass interval][eta, phi].
    // Divide each sparse SE contribution by its own pool'setting ME before summing.
    std::vector<double> seContent(static_cast<size_t>(nMassOut) * nCorrBins, 0.);
    std::vector<double> seVariance(static_cast<size_t>(nMassOut) * nCorrBins, 0.);
    std::vector<double> corrContent(static_cast<size_t>(nMassOut) * nCorrBins, 0.);
    std::vector<double> corrVariance(static_cast<size_t>(nMassOut) * nCorrBins, 0.);
    std::vector<int> coordSE(se->GetNdimensions());
    const bool seStoredErrors = se->GetCalculateErrors();

    for (Long64_t ibin = 0; ibin < se->GetNbins(); ++ibin) {
      const double valSE = se->GetBinContent(ibin, coordSE.data());
      const int pool = coordSE[setting.poolAxis];
      const int sourceMass = coordSE[setting.massAxis];
      const int iEta = coordSE[setting.etaAxis];
      const int iPhi = coordSE[setting.phiAxis];

      if (pool < 1 || pool > nPools || sourceMass < 1 ||
          sourceMass >= static_cast<int>(massMap.size()) || iEta < 1 || iEta > nEta ||
          iPhi < 1 || iPhi > nPhi) {
        continue;
      }

      const int outMass = massMap[sourceMass];

      // Source mass bin is outside configured intervals.
      if (outMass < 0) {
        continue;
      }

      const int icorr = flatBin(iEta, iPhi, nPhi);
      const size_t outIdx = static_cast<size_t>(outMass) * nCorrBins + icorr;
      const double errSE2 = seStoredErrors ? se->GetBinError2(ibin) : valSE;

      // Raw SE QA map summed over pools.
      seContent[outIdx] += valSE;
      seVariance[outIdx] += errSE2;

      // Pool-specific normalized ME.
      const size_t meIdx = static_cast<size_t>(pool - 1) * nCorrBins + icorr;
      const double valME = meContent[meIdx];

      // Keep the raw SE contribution, but omit correction for an empty ME bin.
      if (valME == 0.) {
        continue;
      }

      const double errME2 = meVariance[meIdx];
      const double corrected = valSE / valME;

      // Propagate SE/ME errors independently for each sparse contribution.
      // This convention does not include covariance from shared ME denominators
      // or uncertainty in the central-bin normalization.
      const double correctedVariance =
          errSE2 / (valME * valME) +
          (valSE * valSE * errME2 / (valME * valME * valME * valME));

      corrContent[outIdx] += corrected;
      corrVariance[outIdx] += correctedVariance;
    }

    // Materialize the accumulated arrays as TH2 maps and write each mass triplet.
    auto* outPt = mkdirChecked(output.get(), ptDirName);

    outPt->cd();

    hMENormalization.Write("hMENormalization", TObject::kOverwrite);

    for (int iMass = 0; iMass < nMassOut; ++iMass) {
      auto* massDir = mkdirChecked(outPt, "mass_" + std::to_string(iMass));
      auto hSE = makeTH2(se, setting, "hSE");
      auto hME = makeTH2(me, setting, "hME");
      auto hCorr = makeTH2(se, setting, "hCorr");
      const size_t offset = static_cast<size_t>(iMass) * nCorrBins;

      for (int iEta = 1; iEta <= nEta; ++iEta) {
        for (int iPhi = 1; iPhi <= nPhi; ++iPhi) {
          const int icorr = flatBin(iEta, iPhi, nPhi);

          hSE->SetBinContent(iEta, iPhi, seContent[offset + icorr]);
          hSE->SetBinError(iEta, iPhi, std::sqrt(std::max(0., seVariance[offset + icorr])));

          // ME is mass-independent by construction.
          // It is intentionally repeated in every mass folder
          // for a self-contained SE/ME/correlation triplet.
          hME->SetBinContent(iEta, iPhi, meSumContent[icorr]);
          hME->SetBinError(iEta, iPhi, std::sqrt(std::max(0., meSumVariance[icorr])));
          hCorr->SetBinContent(iEta, iPhi, corrContent[offset + icorr]);
          hCorr->SetBinError(iEta, iPhi, std::sqrt(std::max(0., corrVariance[offset + icorr])));
        }
      }

      massDir->cd();

      hSE->Write("hSE", TObject::kOverwrite);
      hME->Write("hME", TObject::kOverwrite);
      hCorr->Write("hCorr", TObject::kOverwrite);

      const std::string range = "[" + std::to_string(setting.massBins[iMass]) + ", " +
          std::to_string(setting.massBins[iMass + 1]) + ")";

      TNamed massRange("mass_range", range.c_str());

      ++nMapsWritten;
    }

    outPt->cd();

    TNamed normalizationDefinition("me_normalization_definition",
        "ME is integrated over mass for each pool and normalized to the "
        "average of the four bins selected by FindBin(+-0.001) around "
        "(DeltaEta,DeltaPhi)=(0,0). hCorr is the sum over pools of "
        "SE_pool_mass/ME_pool_normalized. hSE is raw SE summed over pools. "
        "hME is normalized ME summed over pools and is stored for QA.");

    normalizationDefinition.Write("me_normalization_definition", TObject::kOverwrite);

    std::cout << "[corr2pc] " << ptDirName << ": wrote " << nMassOut
        << " mass intervals with hSE, hME, hCorr" << std::endl;
  }

  output->Write();

  if (output->TestBit(TFile::kWriteError)) {
    throw std::runtime_error("ROOT write error for " + outputFilename);
  }

  output->Close();
  input->Close();

  return nMapsWritten;
}

} // namespace corr2pc

#endif // HFVN_CORR2PC_H
