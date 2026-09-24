import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Solve Q1 and write audited results.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="results/q1")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    sys.path.insert(0, str(project_root / "src"))
    from dproblem.q1.solver import write_q1_results

    summary = write_q1_results(project_root, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
