import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run the minimal Q1 physical-model smoke test.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="results/p1_smoke/q1_single_box_smoke.json")
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    sys.path.insert(0, str(project_root / "src"))
    from dproblem.q1.smoke import write_single_box_smoke

    print(write_single_box_smoke(project_root, args.output))


if __name__ == "__main__":
    main()
