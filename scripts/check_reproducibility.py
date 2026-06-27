#!/usr/bin/env python3
"""
Check that a single hermes-3 binary produces bit-identical output on repeated runs.

Runs the binary --nruns times and compares all pairs of dump files.  Any
variable that ever differs between two runs is non-deterministic (caused by
MPI reduction ordering or other run-to-run variation); variables that are
identical in every pair are reproducible.

This is used to distinguish pre-existing non-determinism from regressions
introduced by a code change.  If a variable shows large differences in
compare_builds.py (old vs new) but is also non-deterministic within old-vs-old
runs, the anomaly existed before the change.

Usage:
    python3 scripts/check_reproducibility.py \\
        --binary /path/to/hermes-3 \\
        --test tests/integrated/2D-production \\
        --test tests/integrated/2D-recycling \\
        --nruns 3 \\
        --mpirun "mpirun -np"
"""

import argparse
import itertools
import pathlib
import sys
import tempfile

import numpy

try:
    import xhermes
except ImportError:
    sys.exit("xhermes is required: pip install xhermes")

# Reuse helpers from compare_builds in the same directory to avoid duplication.
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from compare_builds import (
    _ulp_diff,
    _BOUT_TIMING_VARS,
    detect_nproc,
    missing_input_files,
    run_hermes,
    setup_work_dir,
)


def _compare_pair(data_a: pathlib.Path, data_b: pathlib.Path) -> dict[str, int]:
    """Return {var: max_ulps} for every variable that differs between the two dumps."""
    ds_a = ds_b = None
    try:
        ds_a = xhermes.open(data_a, unnormalise=False)
        ds_b = xhermes.open(data_b, unnormalise=False)

        a_last = ds_a.isel(t=-1)
        b_last = ds_b.isel(t=-1)

        common = sorted(
            (set(a_last.data_vars) & set(b_last.data_vars)) - _BOUT_TIMING_VARS
        )
        diffs: dict[str, int] = {}

        for v in common:
            a = a_last[v].values.ravel().astype(numpy.float64)
            b = b_last[v].values.ravel().astype(numpy.float64)

            mask = numpy.isfinite(a) & numpy.isfinite(b)
            if not mask.any():
                continue

            max_ulps = int(_ulp_diff(a[mask], b[mask]).max())
            if max_ulps > 0:
                diffs[v] = max_ulps

        return diffs

    finally:
        for ds in (ds_a, ds_b):
            if ds is not None:
                try:
                    ds.close()
                except Exception:
                    pass


def check_test(test_dir: pathlib.Path, binary: pathlib.Path,
               nruns: int, mpirun: str) -> bool:
    """
    Run binary nruns times in isolated work dirs, compare all C(nruns, 2) pairs.
    Returns True if the binary is reproducible on this test (all variables identical).
    """
    name = test_dir.name
    nproc = detect_nproc(test_dir)
    print(f"\n{'='*70}")
    print(f"Test: {name}  (nproc={nproc}, {nruns} runs, "
          f"{nruns * (nruns - 1) // 2} comparison pair(s))")
    print('='*70)

    if not (test_dir / "data" / "BOUT.inp").is_file():
        print("  [!] No data/BOUT.inp — skipping")
        return True

    missing = missing_input_files(test_dir)
    if missing:
        print(f"  [!] Skipping: missing input file(s): {missing}")
        return True

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        work_dirs = [tmp / f"run{i}" for i in range(nruns)]

        for wd in work_dirs:
            setup_work_dir(test_dir, wd, binary)

        for i, wd in enumerate(work_dirs):
            print(f"  Run {i + 1}/{nruns}...", end="", flush=True)
            try:
                elapsed = run_hermes(wd, nproc, mpirun)
                print(f" {elapsed:.1f}s")
            except RuntimeError as e:
                print(f"\n  [!] Run {i + 1} failed: {e}")
                return False

        # Aggregate worst-case ULP difference per variable across all pairs.
        worst: dict[str, int] = {}
        for i, j in itertools.combinations(range(nruns), 2):
            pair_diffs = _compare_pair(
                work_dirs[i] / "data",
                work_dirs[j] / "data",
            )
            for v, ulps in pair_diffs.items():
                worst[v] = max(worst.get(v, 0), ulps)

        if not worst:
            print("\n  => REPRODUCIBLE: all variables bit-identical across all runs.")
            return True

        print(f"\n  => NON-DETERMINISTIC: {len(worst)} variable(s) differ between runs:")
        print(f"\n  {'Variable':<32} {'max ULPs (worst pair)':>22}")
        print("  " + "-" * 56)
        for v, ulps in sorted(worst.items(), key=lambda x: -x[1]):
            print(f"  {v:<32} {ulps:>22}")
        print()
        print("  These variables differ between runs of the same binary.")
        print("  Any anomaly in compare_builds.py for these variables is pre-existing.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--binary", required=True, type=pathlib.Path,
        help="hermes-3 executable to test for reproducibility",
    )
    parser.add_argument(
        "--test", action="append", dest="tests",
        type=pathlib.Path, metavar="DIR", required=True,
        help="Integrated test directory (repeatable)",
    )
    parser.add_argument(
        "--nruns", type=int, default=3,
        help="Number of independent runs to compare (default: 3, giving 3 pairs)",
    )
    parser.add_argument(
        "--mpirun", default="mpirun -np",
        help='MPI launch prefix including the -n flag (default: "mpirun -np")',
    )
    args = parser.parse_args()

    if not args.binary.is_file():
        sys.exit(f"Binary not found: {args.binary}")

    reproducible = []
    non_deterministic = []

    for test in args.tests:
        ok = check_test(
            test.resolve(), args.binary.resolve(),
            nruns=args.nruns, mpirun=args.mpirun,
        )
        (reproducible if ok else non_deterministic).append(test.name)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    if reproducible:
        print(f"  REPRODUCIBLE      ({len(reproducible)}): {', '.join(reproducible)}")
    if non_deterministic:
        print(f"  NON-DETERMINISTIC ({len(non_deterministic)}): {', '.join(non_deterministic)}")

    if not non_deterministic:
        print("\n  All tested binaries are reproducible.")
        print("  Differences seen in compare_builds.py come from the code change, not noise.")
    else:
        print("\n  Non-deterministic variables are NOT reliable regression indicators.")
        sys.exit(1)


if __name__ == "__main__":
    main()
