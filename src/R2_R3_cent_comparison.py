import ROOT
import os
import array

# Set global style for professional plots
def set_global_style():
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetLegendBorderSize(0)
    ROOT.gStyle.SetLegendFillColor(0)
    ROOT.gStyle.SetLegendFont(42)
    ROOT.gStyle.SetLegendTextSize(0.035)
    ROOT.gStyle.SetLabelSize(0.04, "XYZ")
    ROOT.gStyle.SetTitleSize(0.045, "XYZ")
    ROOT.gStyle.SetTitleOffset(1.1, "Y")
    ROOT.gStyle.SetTitleOffset(1.0, "X")
    ROOT.gStyle.SetPadLeftMargin(0.15)
    ROOT.gStyle.SetPadBottomMargin(0.15)
    ROOT.gStyle.SetPadRightMargin(0.05)
    ROOT.gStyle.SetPadTopMargin(0.05)
    ROOT.gStyle.SetFrameLineWidth(2)

set_global_style()

# Open the files
f1 = ROOT.TFile.Open("resosp0_100_v3.root")   # R3: 0-100%
f2 = ROOT.TFile.Open("resospk0100_inte.root")   # R2: 0-100%

if not f1 or not f1.IsOpen():
    print("Error: Could not open resosp0_100_v3.root")
    exit()
if not f2 or not f2.IsOpen():
    print("Error: Could not open resospk0100_inte.root")
    exit()

# List of detector combinations to compare
det_combinations = [
    "FT0c_FT0a_FV0a",
    "FT0c_FT0a_TPCpos",
    "FT0c_FT0a_TPCneg",
    "FT0c_FT0a_TPCtot",
    "FT0c_FV0a_TPCpos",
    "FT0c_FV0a_TPCneg",
    "FT0c_FV0a_TPCtot",
    "FT0c_TPCpos_TPCneg",
    "FT0a_FV0a_TPCpos",
    "FT0a_FV0a_TPCneg",
    "FT0a_FV0a_TPCtot",
    "FT0a_TPCpos_TPCneg",
    "FT0m_FV0a_TPCpos",
    "FT0m_FV0a_TPCneg",
    "FT0m_FV0a_TPCtot",
    "FT0m_TPCpos_TPCneg",
    "FV0a_TPCpos_TPCneg"
]

# Colors for different resolution types
resolution_colors = {
    "R3": ROOT.kRed+1,
    "R2": ROOT.kBlue+1
}

markers = {
    "R3": 20,
    "R2": 21
}

# Create output directory if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")

# Function to get resolution histogram from a directory
def get_resolution_hist(directory):
    if not directory:
        return None
    keys = directory.GetListOfKeys()
    for key in keys:
        obj_name = key.GetName()
        # Look for resolution histogram
        if "histo_reso" in obj_name and not "delta_cent" in obj_name:
            obj = key.ReadObj()
            if "TH1" in obj.ClassName():
                return obj
    return None

# Create output file for results
output_file = ROOT.TFile("plots/resolution_comparison_results.root", "RECREATE")

# Store results for summary
results_summary = []

# Create individual plots for each detector combination
for det in det_combinations:
    print(f"\nAnalyzing {det}...")
    
    # Get directories
    d1 = f1.Get(det)  # R3
    d2 = f2.Get(det)  # R2
    
    if not d1 or not d2:
        print(f"Skipping {det}, not found in one of the files")
        continue
    
    # Get resolution histograms
    h_r3 = d1.Get("histo_reso")
    h_r2 = d2.Get("histo_reso")
    
    if not h_r3 or not h_r2:
        print(f"Skipping {det}, resolution histogram not found")
        continue
    
    # Create comparison canvas
    c = ROOT.TCanvas(f"c_{det}", f"Resolution Comparison - {det}", 800, 600)
    
    # Style the histograms
    h_r3.SetLineColor(resolution_colors["R3"])
    h_r3.SetMarkerColor(resolution_colors["R3"])
    h_r3.SetMarkerStyle(markers["R3"])
    h_r3.SetLineWidth(2)
    h_r3.SetMarkerSize(1.5)
    h_r3.SetTitle("")
    h_r3.GetXaxis().SetTitle("Centrality (%)")
    h_r3.GetYaxis().SetTitle("Resolution")
    h_r3.GetYaxis().SetTitleOffset(1.4)
    
    h_r2.SetLineColor(resolution_colors["R2"])
    h_r2.SetMarkerColor(resolution_colors["R2"])
    h_r2.SetMarkerStyle(markers["R2"])
    h_r2.SetLineWidth(2)
    h_r2.SetMarkerSize(1.5)
    
    # Determine y-axis range
    y_min = min(h_r3.GetMinimum(), h_r2.GetMinimum()) * 0.9
    y_max = max(h_r3.GetMaximum(), h_r2.GetMaximum()) * 1.1
    h_r3.GetYaxis().SetRangeUser(y_min, y_max)
    
    # Draw without uncertainties
    h_r3.Draw("P")
    h_r2.Draw("P SAME")
    
    # Add legend
    leg = ROOT.TLegend(0.65, 0.75, 0.88, 0.88)
    leg.AddEntry(h_r3, "R3 (v3 method)", "lp")
    leg.AddEntry(h_r2, "R2 (standard)", "lp")
    leg.Draw()
    
    # Add detector label
    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextSize(0.04)
    label.DrawLatex(0.2, 0.85, det.replace("_", " & "))
    
    # Add ALICE label
    alice_label = ROOT.TLatex()
    alice_label.SetNDC()
    alice_label.SetTextFont(42)
    alice_label.SetTextSize(0.04)
    # alice_label.DrawLatex(0.2, 0.92, "")
    
    c.Update()
    c.SaveAs(f"plots/resolution_comparison_{det}.pdf")
    c.SaveAs(f"plots/resolution_comparison_{det}.png")
    
    # Save histograms to output file
    output_file.cd()
    h_r3.Write(f"h_r3_{det}")
    h_r2.Write(f"h_r2_{det}")
    
    # Calculate average values for summary
    n_bins = h_r3.GetNbinsX()
    avg_r3 = 0
    avg_r2 = 0
    count = 0
    
    for bin in range(1, n_bins + 1):
        if h_r3.GetBinContent(bin) > 0 and h_r2.GetBinContent(bin) > 0:
            avg_r3 += h_r3.GetBinContent(bin)
            avg_r2 += h_r2.GetBinContent(bin)
            count += 1
    
    if count > 0:
        avg_r3 /= count
        avg_r2 /= count
    
    # Store results for summary
    results_summary.append({
        'detector': det,
        'avg_r3': avg_r3,
        'avg_r2': avg_r2,
        'count': count
    })
    
    print(f"Created comparison plot for {det}")

# Create summary comparison plot
c_summary = ROOT.TCanvas("c_summary", "R3 vs R2 Summary Comparison", 1200, 800)

# Create graphs for average values
n_det = len(results_summary)
if n_det > 0:
    x = array.array('d')
    y_r3 = array.array('d')
    y_r2 = array.array('d')
    labels = []
    
    for i, result in enumerate(results_summary):
        x.append(i + 1)
        y_r3.append(result['avg_r3'])
        y_r2.append(result['avg_r2'])
        labels.append(result['detector'].replace('_', '\n'))
    
    # Create graphs
    g_r3 = ROOT.TGraph(n_det, x, y_r3)
    g_r2 = ROOT.TGraph(n_det, x, y_r2)
    
    # Style the graphs
    g_r3.SetTitle("Average Resolution by Detector Combination")
    g_r3.GetXaxis().SetTitle("Detector Combination")
    g_r3.GetYaxis().SetTitle("Average Resolution")
    g_r3.GetYaxis().SetTitleOffset(1.4)
    g_r3.SetMarkerColor(resolution_colors["R3"])
    g_r3.SetLineColor(resolution_colors["R3"])
    g_r3.SetMarkerStyle(markers["R3"])
    g_r3.SetMarkerSize(1.5)
    g_r3.SetLineWidth(2)
    
    g_r2.SetMarkerColor(resolution_colors["R2"])
    g_r2.SetLineColor(resolution_colors["R2"])
    g_r2.SetMarkerStyle(markers["R2"])
    g_r2.SetMarkerSize(1.5)
    g_r2.SetLineWidth(2)
    
    # Draw graphs
    g_r3.Draw("APL")
    g_r2.Draw("PL SAME")
    
    # Set x-axis labels
    for i in range(n_det):
        label = ROOT.TLatex()
        label.SetTextSize(0.02)
        label.SetTextAngle(45)
        label.DrawLatex(x[i], min(y_r3 + y_r2) * 0.9, labels[i])
    
    # Add legend
    leg_summary = ROOT.TLegend(0.7, 0.8, 0.9, 0.9)
    leg_summary.AddEntry(g_r3, "R3 (v3 method)", "lp")
    leg_summary.AddEntry(g_r2, "R2 (standard)", "lp")
    leg_summary.Draw()
    
    # Add ALICE label
    alice_label = ROOT.TLatex()
    alice_label.SetNDC()
    alice_label.SetTextFont(42)
    alice_label.SetTextSize(0.04)
    alice_label.DrawLatex(0.2, 0.95, " R3 vs R2 Resolution Comparison")
    
    c_summary.Update()
    c_summary.SaveAs("plots/r3_r2_summary_comparison.pdf")
    c_summary.SaveAs("plots/r3_r2_summary_comparison.png")
    
    # Save summary results
    output_file.cd()
    g_r3.Write("summary_avg_r3")
    g_r2.Write("summary_avg_r2")

# Create text file with results summary
with open("plots/results_summary.txt", "w") as f:
    f.write("R3 vs R2 Resolution Comparison Summary\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"{'Detector Combination':<25} {'Avg R3':<10} {'Avg R2':<10} {'Bins':<5}\n")
    f.write("-" * 60 + "\n")
    
    for result in results_summary:
        f.write(f"{result['detector']:<25} {result['avg_r3']:<10.3f} {result['avg_r2']:<10.3f} {result['count']:<5}\n")
    
    f.write("\nLegend:\n")
    f.write("- R3: Resolution from v3 method (0-100% centrality)\n")
    f.write("- R2: Standard resolution (0-100% centrality)\n")

# Close files
output_file.Close()
f1.Close()
f2.Close()

print("\n" + "="*60)
print("COMPARISON COMPLETED SUCCESSFULLY")
print("="*60)
print(f"Compared {len(results_summary)} detector combinations")
print("Results saved in 'plots/' directory:")
print("- Individual comparison plots (PDF/PNG)")
print("- Summary comparison plot")
print("- ROOT file with all histograms")
print("- Text file with results summary")
print("="*60)