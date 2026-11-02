#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
# ============================================================================
# CONFIG
# ============================================================================
ALIEN_DIRS = [
    f"/alice/cern.ch/user/a/alihyperloop/jobs/0623/hy_{job_id}"
    for job_id in range(6239678, 6239527, -1)
]
REMOTE_FILE_NAME = "AnalysisResults.root"
FILE_LIST_NAME = "input_files.txt"
SKIP_EXISTING = True
MAX_WORKERS = 8
MAX_RETRIES = 2
SAMPLE = "ME"
HISTOGRAM_DIRECTORY = "hf-correlator-flow-charm-hadrons-reduced"
SANITY_CHECK_SCRIPT = Path(__file__).with_name("check_anres_sanity.py")
OUTPUT_DIR = "/home/spolitan/alice/dplus_v2_pp/101026/inputs/me"
# ============================================================================


def run(cmd):
    # Keep transfer progress from concurrent alien.py processes from interleaving.
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, errors="replace")


def check_runtime():
    result = subprocess.run([sys.executable, str(SANITY_CHECK_SCRIPT), "--check-runtime"],
                            capture_output=True, text=True, errors="replace")

    if result.returncode != 0:
        raise RuntimeError(f"Sanity checker unavailable; no files will be deleted. "
                           f"Run in your ROOT environment. {(result.stdout + result.stderr).strip()}")


def validate(path, histograms):
    cmd = [sys.executable, str(SANITY_CHECK_SCRIPT), "--file", str(path)]
    for histogram in histograms:
        cmd.extend(["--histogram", histogram])
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")

    if result.returncode == 0:
        return True
    detail = (result.stdout + result.stderr).strip()

    if result.returncode != 1:
        raise RuntimeError(f"Sanity checker runtime failure for {path}: {detail}")
    print(f"[{Path(path).name}] Sanity check failed: {detail}", flush=True)
    return False


def download(src, dst, retries=MAX_RETRIES, histograms=None, check_only=False):
    dst = Path(dst)

    if histograms is None:
        histograms = [f"{HISTOGRAM_DIRECTORY}/hSparseCorrelations{SAMPLE}CharmHad"]

    if dst.exists():
        if not validate(dst, histograms):
            dst.unlink()
            print(f"Removed invalid file: {dst}", flush=True)
            return False

        if SKIP_EXISTING or check_only:
            return True

    if check_only:
        return False
    # Each transfer owns its temporary file, including across separate runs.
    tmp = dst.with_name(f".{dst.name}.{uuid.uuid4().hex}.part")
    for attempt in range(retries + 1):
        try:
            run([
                "alien.py", "cp", f"alien://{src}", f"file://{tmp.resolve()}",
            ])

            if not tmp.exists() or tmp.stat().st_size <= 0:
                raise OSError("Downloaded file is missing or empty")

            if not validate(tmp, histograms):
                # A missing required key is not a transient transfer failure.
                return False
            tmp.replace(dst)
            return True
        except (subprocess.CalledProcessError, OSError) as exc:
            detail = (exc.stdout or str(exc)) if isinstance(exc, subprocess.CalledProcessError) else str(exc)
            print(f"[{dst.name}] attempt {attempt + 1}/{retries + 1} failed: {detail[-2000:].strip()}", flush=True)
        finally:
            tmp.unlink(missing_ok=True)

        if attempt < retries:
            time.sleep(min(2 ** attempt, 8))
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download Hyperloop outputs concurrently.")
    parser.add_argument("--workers", "-j", type=int, default=MAX_WORKERS,
                        help=f"concurrent downloads (default: {MAX_WORKERS})")
    parser.add_argument("--retries", type=int, default=MAX_RETRIES,
                        help=f"retries per failed file (default: {MAX_RETRIES})")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--sample", choices=("SE", "ME"), default=SAMPLE)
    parser.add_argument("--config", type=Path, help="use inputs.<sample>.histogram from this YAML")
    parser.add_argument("--histogram", action="append", help="override required histogram keys; repeat as needed")
    parser.add_argument("--check-only", action="store_true", help="validate existing files and remove invalid ones without downloading")
    args = parser.parse_args(argv)

    if args.workers < 1 or args.retries < 0:
        parser.error("--workers must be positive and --retries must be nonnegative")

    if args.histogram:
        histograms = args.histogram
    elif args.config:
        import yaml
        cfg = yaml.safe_load(args.config.read_text())
        histograms = [cfg["inputs"][args.sample]["histogram"]]
    else:
        histograms = [f"{HISTOGRAM_DIRECTORY}/hSparseCorrelations{args.sample}CharmHad"]

    if any(not isinstance(name, str) or not name.strip() for name in histograms):
        parser.error("required histogram paths must be nonempty strings")
    # Distinguish an unavailable checker from a bad file before deleting anything.
    check_runtime()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    destinations = {}
    for alien_dir in dict.fromkeys(path.rstrip("/") for path in ALIEN_DIRS):
        # There are no subdirectories:
        # the ROOT file is directly inside alien_dir.
        remote = f"{alien_dir}/{REMOTE_FILE_NAME}"
        job_name = Path(alien_dir).name
        local = output_dir / f"{job_name}_{REMOTE_FILE_NAME}"

        if local in destinations and destinations[local] != remote:
            raise ValueError(f"Different remote files would overwrite {local}")
        destinations[local] = remote
        jobs.append((remote, local))
    ready = set()
    pending = jobs
    existing = sum(local.is_file() for _, local in jobs)
    started = time.monotonic()
    print(f"Files: {len(jobs)}; existing to check: {existing}; workers: {args.workers}; required keys: {histograms}", flush=True)

    if pending:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(pending))) as pool:
            futures = {pool.submit(download, remote, local, args.retries, histograms, args.check_only): (remote, local)
                       for remote, local in pending}
            for done, future in enumerate(as_completed(futures), start=1):
                remote, local = futures[future]

                if future.result():
                    ready.add(local)
                    print(f"[{done}/{len(pending)}] Valid: {local.name}", flush=True)
                else:
                    print(f"[{done}/{len(pending)}] Missing or failed: {remote}", flush=True)
    # Keep only valid local files and remove duplicates while preserving order.
    valid_files = [
        path.resolve()
        for _, path in jobs

        if path in ready and path.exists() and path.stat().st_size > 0
    ]
    valid_files = list(dict.fromkeys(valid_files))
    file_list = output_dir / FILE_LIST_NAME
    # Atomically replace the list after all transfers, preserving input order.
    list_tmp = file_list.with_name(f".{file_list.name}.{uuid.uuid4().hex}.part")
    try:
        list_tmp.write_text("".join(f"{path}\n" for path in valid_files))
        list_tmp.replace(file_list)
    finally:
        list_tmp.unlink(missing_ok=True)
    print("\n========================================")
    print(f"Valid files downloaded/found: {len(valid_files)}")
    print(f"Input list: {file_list}")
    print(f"Failed files: {len(jobs) - len(valid_files)}; elapsed: {time.monotonic() - started:.1f}s")
    print("========================================")

    if not valid_files:
        raise RuntimeError("No valid files were downloaded.")

if __name__ == "__main__":
    main()
