import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build and audit the Q2 EDF baseline.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="results/q2_baseline")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    sys.path.insert(0, str(project_root / "src"))
    from dproblem.q2.baseline import write_q2_baseline_results

    summary = write_q2_baseline_results(project_root, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
