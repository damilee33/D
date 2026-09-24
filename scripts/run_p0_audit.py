import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run the read-only P0 input audit.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="results/00_audit")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    sys.path.insert(0, str(project_root / "src"))
    from dproblem.validation.audit import write_audit

    outputs = write_audit(project_root, args.output_dir)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

