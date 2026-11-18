"""Run the six unchanged RL algorithms for seed42/seed45 sequentially."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS = ROOT / "proposed"
METHODS = {
    "proposed": "Proposed.py",
    # "routing": "Routing.py",
    # "pruning": "pruning.py",
    # "fix9": "FIxpriuning9.py",
    # "withoutpruning": "withoutpriuning.py",
    # "fix5": "FIxpriuning5.py",
}
EXPECTED = ("reward.txt", "r_m.txt", "one_hop_error", "pathbest.npy")


def build_jobs(seeds, methods, results_root):
    jobs = []
    for seed in seeds:
        if seed not in (42, 49,48,50,51):
            raise ValueError("This continuation package only runs seed42 and seed45; seed48 is excluded.")
        for method in methods:
            directory = results_root / f"job_1{seed}" / method
            jobs.append((seed, method, ROOT / METHODS[method], directory))
    if len({str(job[3]) for job in jobs}) != len(jobs):
        raise ValueError("Duplicate job_1/method job")
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        choices=[42, 48, 49, 50, 51],
        default=[42, 48, 49, 50, 51]
    )
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS,
                        help="The proposed/ directory containing seed42/Q_table.npy and seed45/Q_table.npy")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true", help="Check inputs and list jobs; do not execute or create output folders")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Rerun all jobs even if outputs exist"
    )

    parser.add_argument("--show-output", action="store_true", help="Print original per-episode/per-hop output (very verbose)")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    results_root = args.results_root.expanduser()
    if not results_root.is_absolute():
        results_root = ROOT / results_root
    results_root = results_root.resolve()
    jobs = build_jobs(args.seeds, args.methods, results_root)
    # Validate all input tables before starting any long run. No copying seed48 Q.
    for seed in dict.fromkeys(args.seeds):
        path = results_root / f"job_1{seed}" / "Q_table.npy"
        if not path.is_file():
            parser.error(f"Missing job_1-specific Q table: {path}")
        table = np.load(path, allow_pickle=False, mmap_mode="r")
        if table.shape != (10, 10, 10, 10) or not np.issubdtype(table.dtype, np.number):
            parser.error(f"Invalid Q table shape/type for job_1{seed}: {table.shape}, {table.dtype}")
        print(f"Q input: {path}", flush=True)
        del table
    for _, _, script, _ in jobs:
        if not script.is_file():
            parser.error(f"Missing script: {script}")
        compile(script.read_text(encoding="utf-8-sig"), str(script), "exec")
    print(f"Plan: {len(jobs)} jobs; seed48 is excluded; original 50000 episodes retained.", flush=True)
    failures, finished, skipped = [], 0, 0
    for index, (seed, method, script, directory) in enumerate(jobs, start=1):
        present = [name for name in EXPECTED if (directory / name).is_file()]
        print(f"[{index}/{len(jobs)}] job_1{seed} / {method} -> {directory}", flush=True)
        if len(present) == len(EXPECTED) and not args.overwrite:
            print("  SKIP: all four output files already exist.", flush=True)
            skipped += 1
            continue
        if present and not args.overwrite:
            message = f"job_1{seed}/{method}: partial outputs exist; inspect them or explicitly use --overwrite"
            print("  BLOCKED: " + message, flush=True)
            failures.append(message)
            if args.stop_on_error:
                break
            continue
        if args.dry_run:
            print(f"  READY: {args.python} {script.name}; RL_SEED={seed}", flush=True)
            continue
        environment = os.environ.copy()
        environment.update(RL_SEED=str(seed), RL_RESULTS_ROOT=str(results_root), PYTHONIOENCODING="utf-8")
        # Each original script creates its own output directory. Seed-specific
        # Q stays at the job_1 root; each method writes to an independent child.
        try:
            completed = subprocess.run(
                [args.python, "-u", str(script)], cwd=ROOT, env=environment,
                stdout=None if args.show_output else subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            )
            missing = [name for name in EXPECTED if not (directory / name).is_file()]
            if completed.returncode != 0 or missing:
                message = (f"job_1{seed}/{method}: returncode={completed.returncode}; missing outputs={missing}\n"
                           + completed.stderr[-16000:])
                failures.append(message)
                print("  FAILED: " + message, flush=True)
                if args.stop_on_error:
                    break
            else:
                finished += 1
                print("  DONE", flush=True)
        except OSError as exc:
            failures.append(f"job_1{seed}/{method}: {exc}")
            print(f"  FAILED: {exc}", flush=True)
            if args.stop_on_error:
                break
    print(f"Finished={finished}, skipped={skipped}, failed/blocked={len(failures)}, dry_run={args.dry_run}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
