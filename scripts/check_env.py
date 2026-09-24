"""Project-local runtime check used before every full computation."""

import argparse
import importlib
import json
import sys
from pathlib import Path


FEATURE_MODULES = {
    "data": ["numpy"],
    "excel": ["dproblem.io.excel"],
    "optimization": ["scipy"],
    "visualization": ["matplotlib"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features", nargs="+", default=["data", "excel", "optimization"]
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from dproblem.compat import patch_numpy_legacy_aliases

    patch_numpy_legacy_aliases()
    modules = []
    for feature in args.features:
        if feature not in FEATURE_MODULES:
            raise SystemExit("unknown feature: {}".format(feature))
        modules.extend(FEATURE_MODULES[feature])

    installed = {}
    missing = []
    for name in sorted(set(modules)):
        try:
            module = importlib.import_module(name)
            if name == "dproblem.io.excel":
                openpyxl = importlib.import_module("openpyxl")
                installed["openpyxl"] = getattr(openpyxl, "__version__", "unknown")
            else:
                installed[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # environment diagnostic must report exact failure
            missing.append({"module": name, "error": repr(exc)})
    result = {
        "ok": not missing,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "features": args.features,
        "installed": installed,
        "missing": missing,
        "optional_exclusions": {
            "pandas": "not used by this pipeline; openpyxl optional pandas integration is disabled"
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
