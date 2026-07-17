import ROOT
from ROOT import TFile, TH1D
import argparse
import numpy as np
import array

pt_bins = [1, 4, 8, 24]
pv_contributors_bins = np.arange(0, 501, 25)  # Adjust the range and step as needed

gen_axes = {
    "Pt": 0,
    "PVContributors": 2
}

reco_axes = {
    "Pt": 1,
    "ScoreBkg": 2,
    "PVContributors": 5,
}

def compute_effs_vs_pv_contrib(sparse_gen, sparse_reco):
    print("\n\n")

    hists_effs, gen_h2d_hists, reco_h2d_hists = [], [], []
    sparse_reco.GetAxis(reco_axes["ScoreBkg"]).SetRangeUser(0, 0.01)
    for pt_min, pt_max in zip(pt_bins[:-1], pt_bins[1:]):
        print("\n----> Pt Range: {} - {} GeV/c <----".format(pt_min, pt_max))

        sparse_gen.GetAxis(gen_axes["Pt"]).SetRange(0, 0)
        sparse_reco.GetAxis(reco_axes["Pt"]).SetRange(0, 0)

        # Project the sparse histograms onto the PV contributors axis for the given pt range
        sparse_gen.GetAxis(gen_axes["Pt"]).SetRangeUser(pt_min, pt_max)
        sparse_reco.GetAxis(reco_axes["Pt"]).SetRangeUser(pt_min, pt_max)

        gen_h2d_hist = sparse_gen.Projection(gen_axes["PVContributors"], gen_axes["Pt"])
        gen_h2d_hist.SetName(f"gen_pt_{pt_min}_{pt_max}")
        reco_h2d_hist = sparse_reco.Projection(reco_axes["PVContributors"], reco_axes["ScoreBkg"])
        reco_h2d_hist.SetName(f"reco_pt_{pt_min}_{pt_max}")
        gen_h2d_hists.append(gen_h2d_hist)
        reco_h2d_hists.append(reco_h2d_hist)

        # Compute efficiency
        eff_hist = TH1D(f"eff_pt_{pt_min}_{pt_max}", f"Pt {pt_min}-{pt_max} GeV/c;PV Contributors;Efficiency", len(pv_contributors_bins)-1, array.array('d', pv_contributors_bins))

        sparse_gen.GetAxis(gen_axes["PVContributors"]).SetRange(0,0)
        sparse_reco.GetAxis(reco_axes["PVContributors"]).SetRange(0,0)
        for i, (pv_low, pv_high) in enumerate(zip(pv_contributors_bins[:-1], pv_contributors_bins[1:]), start=1):

            # Set PV Range on the Sparse directly before projecting
            sparse_gen.GetAxis(gen_axes["PVContributors"]).SetRangeUser(pv_low, pv_high)
            sparse_reco.GetAxis(reco_axes["PVContributors"]).SetRangeUser(pv_low, pv_high)

            # Project to a 1D hist (just to get the integral)
            tmp_gen = sparse_gen.Projection(gen_axes["Pt"]) # Any axis works for Integral
            tmp_reco = sparse_reco.Projection(reco_axes["Pt"])
            
            gen_count = tmp_gen.Integral()
            reco_count = tmp_reco.Integral()

            # Calculate and Fill
            if gen_count > 0:
                eff = reco_count / gen_count
                err = np.sqrt(eff * (1 - eff) / gen_count)
                eff_hist.SetBinContent(i, eff)
                eff_hist.SetBinError(i, err)
            print(f"Pt {pt_min}-{pt_max} GeV/c, PV Contributors {pv_low}-{pv_high}: Gen={gen_count}, Reco={reco_count}, Eff={eff:.4f} ± {err:.4f}")

            # Clean up temp hists
            tmp_gen.Delete()
            tmp_reco.Delete()

        hists_effs.append(eff_hist)

    return hists_effs, gen_h2d_hists, reco_h2d_hists


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute efficiencies vs PV contributors')
    parser.add_argument('--input-file', '-i', type=str, required=True, help='Path to the input ROOT file containing the sparse D+ histogram')
    parser.add_argument('--output-file', '-o', type=str, required=True, help='Path to the output ROOT file to save the efficiency histograms')
    args = parser.parse_args()

    # Open the ROOT file and retrieve the sparse D+ histogram
    input_file = TFile(args.input_file)
    sparse_gen_prompt = input_file.Get("hf-task-dplus/hSparseMassGenPrompt")
    sparse_reco_prompt = input_file.Get("hf-task-dplus/hSparseMassPrompt")
    sparse_gen_fd = input_file.Get("hf-task-dplus/hSparseMassGenFD")
    sparse_reco_fd = input_file.Get("hf-task-dplus/hSparseMassFD")

    outfile = TFile(args.output_file, "RECREATE")

    # Compute efficiencies vs PV contributors
    hists_eff_prompt, gen_h2d_hists_prompt, reco_h2d_hists_prompt = compute_effs_vs_pv_contrib(sparse_gen_prompt, sparse_reco_prompt)
    outfile.mkdir("Prompt")
    outfile.cd("Prompt")
    for i_pt, (hist_eff, gen_hist, reco_hist) in enumerate(zip(hists_eff_prompt, gen_h2d_hists_prompt, reco_h2d_hists_prompt)):
        outfile.mkdir(f"Prompt/Pt_{pt_bins[i_pt]}_{pt_bins[i_pt+1]}")
        outfile.cd(f"Prompt/Pt_{pt_bins[i_pt]}_{pt_bins[i_pt+1]}")
        hist_eff.Write()
        gen_hist.Write()
        reco_hist.Write()

    eff_fd, gen_h2d_hists_fd, reco_h2d_hists_fd = compute_effs_vs_pv_contrib(sparse_gen_fd, sparse_reco_fd)
    outfile.mkdir("FD")
    outfile.cd("FD")
    for i_pt, (hist_eff, gen_hist, reco_hist) in enumerate(zip(eff_fd, gen_h2d_hists_fd, reco_h2d_hists_fd)):
        outfile.mkdir(f"FD/Pt_{pt_bins[i_pt]}_{pt_bins[i_pt+1]}")
        outfile.cd(f"FD/Pt_{pt_bins[i_pt]}_{pt_bins[i_pt+1]}")
        hist_eff.Write()
        gen_hist.Write()
        reco_hist.Write()

    input_file.Close()
    outfile.Close()
