import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="results/q3")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root / "src"))
    from dproblem.q3.joint import run_q3_joint

    summary = run_q3_joint(root, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
