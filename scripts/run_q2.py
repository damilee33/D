import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Solve Q2 with multi-seed ALNS.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="results/q2")
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    sys.path.insert(0, str(project_root / "src"))
    from dproblem.q2.solver import write_q2_results

    summary = write_q2_results(
        project_root, output_dir=args.output_dir, iterations=args.iterations
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
