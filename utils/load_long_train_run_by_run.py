import os
import subprocess
import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
from ROOT import TFile, THnSparse
import yaml

def split_into_three(lst):
    # Calculate the base size of each sublist and the remainder
    n = len(lst)
    base_size = n // 3
    remainder = n % 3
    
    # Create the sublists
    sublists = []
    start = 0
    for i in range(3):
        end = start + base_size + (1 if i < remainder else 0)
        sublist = lst[start:end]
        sublists.append(sublist)
        start = end
    
    return sublists

def download_file(line, run, output_dir):
    line = line.strip()
    dest_dir = f"{output_dir}/runs/{run}"
    os.makedirs(dest_dir, exist_ok=True)
    command = f"alien_cp {line}/AnalysisResults.root file:{dest_dir}/"
    print(f"[Downloading] {command}")
    os.system(command)
    input_file = Path(dest_dir + "/AnalysisResults.root")  # check copying of file
    return command, input_file # Optionally return something for logging

def merge_files(sublist, merge_name, sparse_path, force):
    merge_command = f"hadd -f {merge_name} " + " ".join(sublist) if force else f"hadd {merge_name} " + " ".join(sublist)
    print(f"\n[Merging] {merge_command}")
    os.system(merge_command)
    merged_file = TFile.Open(merge_name, 'r')
    status = True if isinstance(merged_file.Get(sparse_path), THnSparse) else False
    return sublist, status
    
def file_downloader(output_dir, dirs, runs, suffix, num_merged, sparse_path, num_threads, force):

    print(f"Starting parallel downloads with {num_threads} threads...\n")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        downloads = [executor.submit(download_file, line, run, output_dir) for (line, run) in zip(dirs, runs)]

    # Step 2: Find downloaded files
    find_command = f'find {output_dir}/runs -wholename "*/AnalysisResults.root" | tr "\\n" " "'
    result = subprocess.run(find_command, shell=True, text=True, capture_output=True)
    output_list = result.stdout.strip().split()

    print("\nTotal files found:", len(output_list))

    # Step 3: Create sublists for merging
    total_files = len(output_list)
    files_per_merge = max(1, total_files // num_merged)
    sublists = [output_list[i:i + files_per_merge] for i in range(0, total_files, files_per_merge)]
    print(f"Created sublists for merging: {sublists}")

    # Step 4: Parallel merging of files
    print(f"\nStarting parallel merging with {num_threads} threads...\n")
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        mergers = []
        for i, sublist in enumerate(sublists):
            merge_name = f"{output_dir}/MERGED_{i}_{suffix}.root"
            mergers.append(executor.submit(merge_files, sublist, merge_name, sparse_path, force))
        
        for future in as_completed(mergers):
            future.result()  # catch exceptions and confirm merge
            
    print(f"mergers: {mergers}")
    summary_merge = []
    for i, future in enumerate(mergers):
        sublist, status = future.result()
        if status:
            print(f"\nMerged {sublist} into {output_dir}/MERGED_{i}_{suffix}.root with status: {status}.")
            summary_merge.append(f"\nMerged {sublist} into {output_dir}/MERGED_{i}_{suffix}.root with status: {status}.")
        else:
            print(f"\nError merging files into {output_dir}/MERGED_{i}_{suffix}.root. Trying to split runs")
            summary_merge.append(f"\nError merging files into {output_dir}/MERGED_{i}_{suffix}.root. Trying to split runs")
            subsublist = split_into_three(sublist)
            for list in subsublist:
                merge_command = f"hadd -f {output_dir}/MERGED_{i+num_merged}_{suffix}.root {list}" if force else f"hadd {merge_name} {list}"
                print(f"\n[Merging] {merge_command}")
                os.system(merge_command)
            num_merged += len(subsublist)
    
    # Step 5: Copy the runs.txt file
    outfile_summary = f"{output_dir}/runs_{suffix}.txt"
    summary_lines = []
    for dir, run in zip(dirs, runs):
        summary_lines.append(f"{dir} --- {run}")
    with open(outfile_summary, 'w') as f:
        f.writelines("\n".join(summary_lines))
        f.writelines("\n".join(summary_merge))
    print(f"\nDownload summary written to {output_dir}/runs_{suffix}.txt")

    print(f"\n\n")
    for future in as_completed(downloads):
        _, input_file = future.result()  # just to catch any exceptions
        if not input_file.is_file():
            print(f"    Error downloading file: {input_file} --> run merging needed!")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download files from ALIEN and merge them using hadd.")
    parser.add_argument("--output_dir", "-o", type=str, required=False, default="path/to/output", help="Path to output directory")
    parser.add_argument("--input_file_dirs", "-id", type=str, required=False, default="grid_dirs.txt", help="Path to the input file list (default: runs.txt)")
    parser.add_argument("--input_file_runs", "-ir", type=str, required=False, default="grid_runs.txt", help="Path to the input file list (default: runs.txt)")
    parser.add_argument("--train_no", "-tr", type=str, required=False, default="trainno", help="suffix (default: trainno)")
    parser.add_argument("--suffix", "-s", type=str, required=False, default="suffix", help="suffix (default: trainno)")
    parser.add_argument("--num_merged", "-n", type=int, required=False, default=1, help="Number of MERGED files to create")
    parser.add_argument("--sparse_path", "-sp", type=str, required=False, default="hf-task-flow-charm-hadrons/hSparseFlowCharm", help="Path to the sparse to be checked when merging (default: runs.txt)")
    parser.add_argument("--threads", "-t", type=int, required=False, default=4, help="Number of parallel download threads (default: 4)")
    parser.add_argument("--force", "-f", action="store_true",  default=False, help="Overwrite existing merged files")
    parser.add_argument("--config_file", "-cfg", type=str, required=False, default="", help="Path to the input file list (default: runs.txt)")
    args = parser.parse_args()
    
    if args.config_file != "":
        with open(args.config_file, 'r', encoding='utf8') as ymlfitConfigFile:
                config = yaml.load(ymlfitConfigFile, yaml.FullLoader)
                print("YAML file loaded!")
        output_dir = config['output_dir']
        suffix = config['suffix']
        num_merged = config['num_merged']
        sparse_path = config['sparse_path']
        threads = config['threads']
        force = config['force']
        train_number = config['train_number']
        dirs = config['grid_dirs']
        runs = config['grid_runs']
        output_dir = output_dir + "Train" + str(train_number) + "/"
        file_downloader(output_dir, dirs, runs, suffix, num_merged, sparse_path, threads, force)
    else:
        output_dir = args.output_dir + "Train" + args.train_no + "/"
        with open(grid_dirs, 'r') as file:
            dirs = file.readlines()
        with open(grid_runs, 'r') as file:
            runs = file.readlines()
        print(f"Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        file_downloader(output_dir, dirs, runs, args.suffix, args.num_merged, args.sparse_path, args.threads, args.force)


