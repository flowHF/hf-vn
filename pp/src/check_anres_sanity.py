#!/usr/bin/env python3
"""Check ROOT readability and required histogram keys; exit 0=valid, 1=bad, 2=setup error."""
import argparse
import sys
from pathlib import Path


def check_file(path, histograms, root):
    path = Path(path)

    if not path.is_file() or path.stat().st_size == 0:
        return False, f"Missing or empty file: {path}"
    try:
        file = root.TFile.Open(str(path), "READ")
    except OSError as exc:
        return False, f"Cannot open ROOT file: {path} ({exc})"
    try:
        if not file or file.IsZombie():
            return False, f"Cannot open ROOT file: {path}"
        for histogram in histograms:
            directory, _, name = histogram.rpartition("/")
            parent = file.GetDirectory(directory) if directory else file

            if not parent or not parent.GetKey(name):
                return False, f"Missing {histogram}: {path}"
        return True, f"Valid: {path}"
    finally:
        if file:
            file.Close()


def resolve_files(value, base):
    result = []
    for item in [value] if isinstance(value, str) else value:
        if item.startswith("@"):
            list_path = Path(item[1:]).expanduser()

            if not list_path.is_absolute():
                list_path = base / list_path
            entries = [line.strip() for line in list_path.read_text().splitlines()
                       if line.strip() and not line.lstrip().startswith("#")]
            result.extend(resolve_files(entries, list_path.parent))
        else:
            path = Path(item).expanduser()
            result.append(path if path.is_absolute() else base / path)
    return list(dict.fromkeys(result))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="pipeline YAML (checks SE and ME)")
    parser.add_argument("--file", type=Path, help="check a single downloaded ROOT file")
    parser.add_argument("--histogram", action="append", default=[], help="required histogram key; repeat as needed")
    parser.add_argument("--check-runtime", action="store_true", help="verify PyROOT availability without opening files")
    args = parser.parse_args(argv)
    try:
        import ROOT
        ROOT.gROOT.SetBatch(True)

        if args.check_runtime:
            return 0

        if args.file:
            if not args.histogram:
                parser.error("--file requires at least one --histogram")
            valid, message = check_file(args.file, args.histogram, ROOT)
            print(message, flush=True)
            return 0 if valid else 1

        if not args.config:
            parser.error("provide a config or --file")
        import yaml
        config = Path(args.config).resolve()
        cfg = yaml.safe_load(config.read_text())
        failures = 0
        for sample in ("SE", "ME"):
            entry = cfg["inputs"][sample]
            for path in resolve_files(entry["file"], config.parent):
                valid, message = check_file(path, [entry["histogram"]], ROOT)
                print(f"[{sample}] {message}", flush=True)
                failures += not valid
        print(f"\nFiles failing the check: {failures}")
        return 1 if failures else 0
    except Exception as exc:
        print(f"Sanity checker setup/runtime error: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())
