#!/usr/bin/env python3
"""Split SE/ME THnSparses into compact files and merge the prefiltered outputs.

Filter source files independently in worker processes, then merge the compact
products with parallel hadd calls. Each worker holds one source THnSparse in
memory, so automatic concurrency is limited by both CPUs and available RAM.
Override it with --jobs, run_preprocess(..., jobs=N), or preprocess.jobs in YAML.
"""

import argparse
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import yaml

from .dmeson_preprocess import split_dmeson_file, validate_mass_config


DEFAULT_KEEP = ("pool", "mass", "delta_eta", "delta_phi")
OUTPUT_PATTERN = "mult_*_pt_*.root"
HADD_BATCH_SIZE = 80
MEMORY_PER_FILE_FACTOR = 6  # assumed RAM of one worker, in units of its source file size
SAMPLES = ("SE", "ME")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
def increasing(values, name):
    """Convert bin edges to floats and require finite, strictly increasing values."""
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError(f"{name}: provide at least two edges")
    edges = [float(v) for v in values]
    if not all(math.isfinite(v) for v in edges) or any(
        a >= b for a, b in zip(edges, edges[1:])
    ):
        raise ValueError(f"{name}: edges must be finite and strictly increasing")
    return edges


def _read_path_list(list_path):
    """Read a plain-text file containing one ROOT path per line."""
    list_path = Path(list_path).resolve()
    if not list_path.is_file():
        raise FileNotFoundError(list_path)
    files = []
    for line in list_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line).expanduser()
        if not path.is_absolute():
            path = list_path.parent / path
        files.append(path.resolve())
    if not files:
        raise ValueError(f"Input list is empty: {list_path}")
    return files


def _resolve_input_entry(value, config_dir, sample):
    """Resolve a path, list of paths, or @file-list relative to the configuration.

    Paths inside an @file-list are relative to that list's own directory.
    Return unique absolute paths in their configured order.
    """
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        if not value:
            raise ValueError(f"{sample}: file list is empty")

        if any(not isinstance(item, str) for item in value):
            raise ValueError(f"{sample}: every file entry must be a string")
        values = value
    else:
        raise ValueError(
            f"{sample}.file must be a path, a list of paths, or '@path/to/input_files.txt'"
        )
    files = []
    for item in values:
        item = item.strip()
        if not item:
            raise ValueError(f"{sample}: empty input path")

        if item.startswith("@"):
            list_path = Path(item[1:]).expanduser()
            if not list_path.is_absolute():
                list_path = config_dir / list_path
            files.extend(_read_path_list(list_path))
            continue
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = config_dir / path
        files.append(path.resolve())
    # Remove duplicates while preserving the configured ordering.
    unique = list(dict.fromkeys(files))
    if not unique:
        raise ValueError(f"{sample}: no input files resolved")
    return unique


def read_config(path):
    """Load YAML, normalize paths/options, and validate axes, bins, cuts and jobs."""
    path = Path(path).resolve()
    with path.open() as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    # Flatten nested preprocessing options so downstream code uses one schema.
    options = cfg.pop("preprocess", {})
    if not isinstance(options, dict):
        raise ValueError("preprocess must be a mapping")
    allowed = {"keep_axes", "rebin", "cuts", "jobs"}
    if set(options) - allowed:
        raise ValueError(f"Unknown preprocess options: {sorted(set(options) - allowed)}")
    for name, value in options.items():
        if name in cfg:
            raise ValueError(
                f"{name}: configure it either under preprocess or at top level, not both"
            )
        cfg[name] = value
    if "inputs" not in cfg or not isinstance(cfg["inputs"], dict):
        raise ValueError("inputs must be a mapping")
    # SE and ME are resolved independently; they may have a different number of files.
    for sample in SAMPLES:
        if sample not in cfg["inputs"]:
            raise ValueError(f"inputs.{sample} is missing")
        item = cfg["inputs"][sample]
        if not isinstance(item, dict):
            raise ValueError(f"inputs.{sample} must be a mapping")

        if "file" not in item:
            raise ValueError(f"{sample}: missing file")
        files = _resolve_input_entry(item["file"], path.parent, sample)
        item["file"] = [str(file) for file in files]  # internally always a list
        if not item.get("histogram"):
            raise ValueError(f"{sample}: missing histogram path")
    output_dir = Path(cfg["output_dir"]).expanduser()
    if not output_dir.is_absolute():
        output_dir = path.parent / output_dir
    cfg["output_dir"] = str(output_dir.resolve())
    axes = cfg["axes"]
    keep = cfg.setdefault("keep_axes", list(DEFAULT_KEEP))
    if (
        not isinstance(keep, list)
        or not keep
        or any(not isinstance(name, str) for name in keep)
    ):
        raise ValueError("keep_axes must be a nonempty list of axis names")

    if len(set(keep)) != len(keep):
        raise ValueError("keep_axes contains duplicates")

    if set(keep) & {"multiplicity", "pt_cand", "pt_assoc"}:
        raise ValueError("Splitting axes cannot also be retained")
    used = []
    for name in (*keep, "pt_cand", "pt_assoc", "multiplicity"):
        axis = axes.get(name)
        if name == "multiplicity" and axis is None:
            continue

        if type(axis) is not int or axis < 0:
            raise ValueError(f"axes.{name}: expected a nonnegative integer")
        used.append(axis)
    if len(used) != len(set(used)):
        raise ValueError("Retained and splitting axes must be distinct")
    rebin = cfg.setdefault("rebin", {})
    if not isinstance(rebin, dict) or set(rebin) - set(keep):
        raise ValueError("rebin must map retained axis names to integer factors")
    for name in keep:
        factor = rebin.setdefault(name, 1)
        if type(factor) is not int or factor < 1:
            raise ValueError(f"rebin.{name}: expected a positive integer")

        if name == "pool" and factor != 1:
            raise ValueError("Do not merge event pools during preprocessing")
    bins = cfg["bins"]
    for name in ("pt_cand", "pt_assoc"):
        bins[name] = increasing(bins[name], name)
    if axes.get("multiplicity") is None:
        if bins.get("multiplicity") is not None:
            raise ValueError("No multiplicity axis: bins.multiplicity must be null")

        if not cfg.get("multiplicity_label"):
            raise ValueError("Provide multiplicity_label for an already selected input")
    else:
        bins["multiplicity"] = increasing(bins.get("multiplicity"), "multiplicity")
    # Normalize cuts to numeric axes and one interval per candidate-pT bin.
    npt = len(bins["pt_cand"]) - 1
    split_indices = {axes.get(name) for name in ("multiplicity", "pt_cand", "pt_assoc")}
    cut_axes = set()
    for cut in cfg.setdefault("cuts", []):
        axis = cut["axis"]
        if isinstance(axis, str):
            if axis not in axes:
                raise ValueError(f"Unknown cut axis: {axis}")
            axis = axes[axis]
        if type(axis) is not int or axis < 0 or axis in split_indices or axis in cut_axes:
            raise ValueError("Each cut needs a distinct non-splitting axis")
        cut["axis"] = axis
        cut_axes.add(axis)
        ranges = cut["ranges"]
        if len(ranges) == 2 and all(isinstance(value, (int, float)) for value in ranges):
            ranges = [ranges] * npt
        if len(ranges) != npt:
            raise ValueError("Cut ranges need one [low, high] pair per candidate pT bin")
        cut["ranges"] = [increasing(pair, "cut range") for pair in ranges]
        if any(len(pair) != 2 for pair in cut["ranges"]):
            raise ValueError("Cut ranges must have exactly two edges")

    if type(cfg.setdefault("allow_poisson_errors", False)) is not bool:
        raise ValueError("allow_poisson_errors must be true or false")
    jobs = cfg.get("jobs")
    if jobs is not None and (type(jobs) is not int or jobs < 1):
        raise ValueError("jobs must be a positive integer")
    validate_mass_config(cfg)
    return cfg


# --------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------
def _available_cpus():
    """Return the CPU count allowed by affinity, when supported."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on every platform
        return os.cpu_count() or 1


def _available_memory():
    """Available RAM in bytes (Linux), or None."""
    try:
        with open("/proc/meminfo") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _auto_jobs(paths, n_tasks):
    """Workers: limited by tasks, CPUs and (roughly) by the memory one worker needs."""
    jobs = max(1, min(n_tasks, _available_cpus()))
    memory = _available_memory()
    if memory:
        biggest = max(Path(path).stat().st_size for path in paths)
        jobs = min(jobs, max(1, int(memory // (MEMORY_PER_FILE_FACTOR * biggest))))
    return jobs


# --------------------------------------------------------------------------
# Splitting (one source file per task)
# --------------------------------------------------------------------------
# Each process owns its ROOT objects; only plain settings cross process boundaries.
_WORKER = {}
def _vector(ROOT, kind, values):
    """Build a typed C++ vector for the compiled splitter."""
    result = ROOT.std.vector(kind)()
    for value in values:
        result.push_back(value)
    return result


def _matrix(ROOT, rows):
    """Build the nested double vectors used for bin edges and cut intervals."""
    result = ROOT.std.vector("std::vector<double>")()
    for row in rows:
        result.push_back(_vector(ROOT, "double", row))
    return result


def _init_worker(settings):
    """Per-process setup: load ROOT, the compiled splitter and the shared arguments.

    Idempotent and cheap when the macro is already compiled and loaded (the parent
    compiles it before any worker is started, so workers never compile concurrently).
    """
    import ROOT

    ROOT.gROOT.SetBatch(True)
    if not ROOT.gSystem.CompileMacro(settings["source"], "kO"):
        raise RuntimeError(f"Cannot compile {settings['source']}")
    _WORKER.clear()
    _WORKER.update(
        ROOT=ROOT,
        settings=settings,
        keep=_vector(ROOT, "int", settings["keep"]),
        rebin=_vector(ROOT, "int", settings["rebin"]),
        split_axes=_vector(ROOT, "int", settings["split_axes"]),
        edges=_matrix(ROOT, settings["edges"]),
        cut_axes=_vector(ROOT, "int", settings["cut_axes"]),
        low=_matrix(ROOT, settings["low"]),
        high=_matrix(ROOT, settings["high"]),
    )


def _split_task(task):
    """Filter one source file; returns (sample, index, result, generated files)."""
    sample, index, input_file, histogram, part_dir = task
    worker, settings = _WORKER, _WORKER["settings"]
    Path(part_dir).mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        totals = worker["ROOT"].prep2pc.split(
            input_file,
            histogram,
            part_dir,
            sample,
            worker["keep"],
            worker["rebin"],
            settings["axis_order"],
            worker["split_axes"],
            worker["edges"],
            worker["cut_axes"],
            worker["low"],
            worker["high"],
            settings["allow_poisson_errors"],
            settings["serialized"],
        )
    except Exception as exc:
        raise RuntimeError(f"{sample}: splitting {input_file} failed: {exc}") from exc
    result = {
        "file": input_file,
        "scanned_bins": int(totals.scanned),
        "excluded_flow_bins": int(totals.flow),
        "selected_source_bins": int(totals.selected),
        "sum_weights": float(totals.sumw),
        "sum_squared_errors": float(totals.sumw2),
        "seconds": time.monotonic() - started,
    }
    generated = sorted(str(p) for p in Path(part_dir).glob(OUTPUT_PATTERN))
    if result["selected_source_bins"] > 0 and not generated:
        raise RuntimeError(
            f"{sample}: splitter selected {result['selected_source_bins']} bins "
            f"from {input_file}, but produced no {OUTPUT_PATTERN} files"
        )
    return sample, index, result, generated


def _run_splits(tasks, settings, jobs):
    """Run all split tasks; returns {(sample, index): (result, generated files)}."""
    results = {}
    total = len(tasks)

    def report(sample, index, result):
        print(
            f"[{len(results)}/{total}] {sample} #{index + 1}: "
            f"{Path(result['file']).name}  ({result['seconds']:.0f} s, "
            f"{result['selected_source_bins']} bins selected)",
            flush=True,
        )

    if jobs == 1:  # in-process: simplest to debug, no process overhead
        _init_worker(settings)
        for task in tasks:
            sample, index, result, generated = _split_task(task)
            results[(sample, index)] = (result, generated)
            report(sample, index, result)
        return results
    # fork is safe here (nothing multi-threaded exists yet) and avoids re-importing
    # the calling script; other platforms fall back to spawn.
    method = "fork" if sys.platform.startswith("linux") else "spawn"
    context = multiprocessing.get_context(method)
    # Largest files first: the long tasks start early, the short ones fill the gaps.
    order = sorted(tasks, key=lambda t: Path(t[2]).stat().st_size, reverse=True)
    with ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=context,
        initializer=_init_worker,
        initargs=(settings,),
    ) as pool:
        futures = [pool.submit(_split_task, task) for task in order]
        try:
            for future in as_completed(futures):
                sample, index, result, generated = future.result()
                results[(sample, index)] = (result, generated)
                report(sample, index, result)
        except BrokenProcessPool as exc:
            raise RuntimeError(
                "A worker process died (crash or out of memory); "
                "retry with fewer workers (--jobs)"
            ) from exc
        finally:
            for future in futures:
                future.cancel()  # do not start queued files after a failure
    return results


# --------------------------------------------------------------------------
# Merging
# --------------------------------------------------------------------------
def _run_hadd(output, inputs):
    """Run one merge and include the end of hadd's log if it fails."""
    command = ["hadd", "-f", str(output), *[str(path) for path in inputs]]
    process = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"hadd failed (exit {process.returncode}) for {output}:\n"
            f"{process.stdout[-3000:]}"
        )


def _merge_root_files(input_files, output_file, jobs=1, batch_size=HADD_BATCH_SIZE):
    """Merge compact ROOT files with hadd (batched for very long input lists).

    The source AnalysisResults files are never merged, only the already prefiltered
    mult_*_pt_*.root products. Batches of one stage run in parallel.
    """
    input_files = [Path(path).resolve() for path in input_files]
    output_file = Path(output_file).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not input_files:
        raise ValueError(f"No inputs to merge into {output_file}")
    temporary = []
    try:
        current, stage = input_files, 0
        while len(current) > batch_size:
            chunks = [
                current[i : i + batch_size]
                for i in range(0, len(current), batch_size)
            ]
            partials = [
                output_file.parent / f".{output_file.stem}_merge_{stage}_{k}.root"
                for k in range(len(chunks))
            ]
            temporary.extend(partials)
            with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(chunks)))) as pool:
                for future in [pool.submit(_run_hadd, p, c) for p, c in zip(partials, chunks)]:
                    future.result()
            current, stage = partials, stage + 1
        _run_hadd(output_file, current)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def _merge_all(products, out, jobs):
    """Merge SE and ME parts of every output file; independent files in parallel."""
    names = sorted(products)
    # Every final file must contain both samples: fail here rather than in the
    # correlation stage.
    for name in names:
        for sample in SAMPLES:
            if not products[name][sample]:
                raise RuntimeError(f"{name}: no {sample} contribution")
    outer = max(1, min(jobs, len(names)))
    # Divide concurrency between output files and batches within each file.
    inner = max(1, jobs // outer)
    merged = []

    def merge_one(name):
        parts = products[name]["SE"] + products[name]["ME"]
        final_file = out / name
        _merge_root_files(parts, final_file, jobs=inner)
        if not final_file.is_file() or final_file.stat().st_size <= 0:
            raise RuntimeError(f"Invalid merged output: {final_file}")
        for part in parts:  # free disk space as soon as a file is complete
            Path(part).unlink(missing_ok=True)
        return name, len(products[name]["SE"]), len(products[name]["ME"])

    with ThreadPoolExecutor(max_workers=outer) as pool:
        futures = [pool.submit(merge_one, name) for name in names]
        try:
            for future in as_completed(futures):
                name, n_se, n_me = future.result()
                merged.append(str(out / name))
                print(f"merged {name}: {n_se} SE + {n_me} ME part(s)", flush=True)
        finally:
            for future in futures:
                future.cancel()
    return sorted(merged)


def _sum_totals(items):
    """Aggregate file counters; seconds is cumulative worker time, not wall time."""
    keys = (
        "scanned_bins",
        "excluded_flow_bins",
        "selected_source_bins",
        "sum_weights",
        "sum_squared_errors",
        "seconds",
    )
    return {key: sum(item[key] for item in items) for key in keys}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def run_preprocess(config_path, jobs=None):
    """Split and merge configured inputs, write reports, and return the output path.

    An explicit jobs argument overrides the YAML setting. Each run replaces the
    preprocessed output directory and, when enabled, the D-candidate directory.
    """
    cfg = read_config(config_path)
    jobs = jobs if jobs is not None else cfg.pop("jobs", None)
    cfg.pop("jobs", None)  # a runtime setting, not part of the recorded configuration
    if jobs is not None and (type(jobs) is not int or jobs < 1):
        raise ValueError("jobs must be a positive integer")
    # Fail before the long scan, not after it.
    if shutil.which("hadd") is None:
        raise RuntimeError("hadd was not found in PATH")
    for sample in SAMPLES:
        for input_file in cfg["inputs"][sample]["file"]:
            if not Path(input_file).is_file():
                raise FileNotFoundError(input_file)
    out = Path(cfg["output_dir"]) / "preprocessed"
    # A run produces the complete compact input set. Remove old products first,
    # otherwise stale mult/pt files could survive configuration changes.
    if out.exists():
        print(
            f"Warning: output directory already exists and will be overwritten: {out}",
            flush=True,
        )
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    part_root = out / ".parts"
    d_enabled = bool(cfg["inputs"]["SE"].get("hsparsemass"))
    d_out = Path(cfg["output_dir"]) / "preprocessed_dmeson"
    if d_enabled:
        if d_out.exists():
            shutil.rmtree(d_out)
        d_out.mkdir(parents=True, exist_ok=True)
    import ROOT

    ROOT.gROOT.SetBatch(True)
    source = Path(__file__).resolve().with_name("split_sparse.cxx")
    # Compile once here, before any worker exists, so that workers only load it.
    if not ROOT.gSystem.CompileMacro(str(source), "kO"):
        raise RuntimeError(f"Cannot compile {source}")
    axes, bins = cfg["axes"], cfg["bins"]
    serialized = yaml.safe_dump(cfg, sort_keys=False)
    (out / "config.yaml").write_text(serialized)
    # Keep axis/edge ordering aligned with the C++ splitter's argument contract.
    # A multiplicity axis of -1 means the input was already selected.
    settings = {
        "source": str(source),
        "keep": [axes[name] for name in cfg["keep_axes"]],
        "rebin": [cfg["rebin"][name] for name in cfg["keep_axes"]],
        "axis_order": ";".join(f"{i}={name}" for i, name in enumerate(cfg["keep_axes"])),
        "split_axes": [
            axes.get("multiplicity") if axes.get("multiplicity") is not None else -1,
            axes["pt_cand"],
            axes["pt_assoc"],
        ],
        "edges": [bins.get("multiplicity") or [], bins["pt_cand"], bins["pt_assoc"]],
        "cut_axes": [cut["axis"] for cut in cfg["cuts"]],
        "low": [[r[0] for r in cut["ranges"]] for cut in cfg["cuts"]],
        "high": [[r[1] for r in cut["ranges"]] for cut in cfg["cuts"]],
        "allow_poisson_errors": cfg["allow_poisson_errors"],
        "serialized": serialized,
    }
    # Isolated part directories prevent source files from overwriting one another.
    tasks = [
        (
            sample,
            index,
            input_file,
            cfg["inputs"][sample]["histogram"],
            str(part_root / sample / f"{index:04d}"),
        )
        for sample in SAMPLES
        for index, input_file in enumerate(cfg["inputs"][sample]["file"])
    ]
    all_files = [task[2] for task in tasks]
    if jobs is None:
        jobs = _auto_jobs(all_files, len(tasks))
    jobs = max(1, min(jobs, len(tasks)))
    print(f"\n{'=' * 70}\n PREPROCESS: {len(tasks)} INPUT FILE(S), {jobs} WORKER(S)\n{'=' * 70}")
    for sample in SAMPLES:
        print(f"{sample}: {len(cfg['inputs'][sample]['file'])} file(s)")
    print(flush=True)
    started = time.monotonic()
    results = _run_splits(tasks, settings, jobs)
    split_seconds = time.monotonic() - started
    # Deterministic order (by sample, then input index) regardless of completion order.
    summary = {}
    products = defaultdict(lambda: {"SE": [], "ME": []})
    for sample in SAMPLES:
        n_files = len(cfg["inputs"][sample]["file"])
        sample_results = []
        for index in range(n_files):
            result, generated = results[(sample, index)]
            sample_results.append(result)
            for path in generated:
                products[Path(path).name][sample].append(Path(path))
        summary[sample] = {
            "n_input_files": n_files,
            "total": _sum_totals(sample_results),
            "files": sample_results,
        }
    if not products:
        raise RuntimeError("Preprocessing produced no compact ROOT files")

    if d_enabled:
        # Dedicated D-candidate histograms use SE only and have separate outputs.
        print("\nPREFILTER D CANDIDATES", flush=True)
        d_parts = d_out / ".parts"
        d_products = defaultdict(list)
        source_hist = cfg["inputs"]["SE"]["hsparsemass"]
        d_reports = []
        for index, input_file in enumerate(cfg["inputs"]["SE"]["file"]):
            report, generated = split_dmeson_file(
                input_file, source_hist, d_parts / f"{index:04d}", cfg
            )
            d_reports.append(report)
            for part in generated:
                d_products[Path(part).name].append(part)
            print(
                f"[D] {index + 1}/{len(cfg['inputs']['SE']['file'])}: "
                f"{Path(input_file).name} ({report['selected_source_bins']} selected bins)",
                flush=True,
            )
        if not d_products:
            raise RuntimeError("D-candidate prefilter produced no files")
        for name, parts in sorted(d_products.items()):
            _merge_root_files(parts, d_out / name, jobs=jobs)
        (d_out / "config.yaml").write_text(serialized)
        summary["D"] = {
            "n_input_files": len(d_reports),
            "n_output_files": len(d_products),
            "output_dir": str(d_out),
            "selected_source_bins": sum(r["selected_source_bins"] for r in d_reports),
        }
        shutil.rmtree(d_parts, ignore_errors=True)
    print(f"\n{'=' * 70}\n MERGE PREFILTERED OUTPUT\n{'=' * 70}", flush=True)
    started = time.monotonic()
    merged_files = _merge_all(products, out, jobs)
    merge_seconds = time.monotonic() - started
    summary["merged"] = {"n_files": len(merged_files), "files": merged_files}
    summary["run"] = {
        "jobs": jobs,
        "split_wall_seconds": split_seconds,
        "merge_wall_seconds": merge_seconds,
    }
    (out / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    # The individual parts are implementation details of the merge.
    shutil.rmtree(part_root, ignore_errors=True)
    print(f"\n{'=' * 70}\n PREPROCESS COMPLETED\n{'=' * 70}")
    print(f"Input SE files : {summary['SE']['n_input_files']}")
    print(f"Input ME files : {summary['ME']['n_input_files']}")
    print(f"Merged outputs : {summary['merged']['n_files']}")
    print(f"Workers        : {jobs}")
    print(f"Wall time      : split {split_seconds:.0f} s, merge {merge_seconds:.0f} s")
    print(f"Output         : {out}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="number of parallel workers (default: from CPUs and free memory)",
    )
    args = parser.parse_args()
    run_preprocess(args.config, jobs=args.jobs)
