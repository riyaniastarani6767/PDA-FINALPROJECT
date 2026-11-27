from __future__ import annotations

import argparse
import json
import subprocess

from utils.model_verification import atomic_write_marker, invalidate_marker, run_verification


def main() -> int:
    parser = argparse.ArgumentParser(description="Formal verification for existing RF and SVM model artifacts")
    parser.add_argument("--no-marker", action="store_true", help="Run checks without writing models/verified_models.json")
    parser.add_argument("--skip-regression", action="store_true", help="Skip complete pytest after formal artifact checks")
    args = parser.parse_args()

    marker = run_verification(write_marker=not args.no_marker)
    if marker["verification_status"] != "verified":
        print(json.dumps(marker, indent=2, ensure_ascii=False))
        return 1

    if args.no_marker:
        print(json.dumps(marker, indent=2, ensure_ascii=False))
        return 0

    if not args.skip_regression:
        regression = subprocess.run(
            [".venv/bin/python", "-m", "pytest", "tests", "-q"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        output = regression.stdout.strip()
        print(output)
        if regression.returncode != 0:
            invalidate_marker()
            marker["verification_status"] = "failed"
            marker["verification_error"] = "Complete pytest suite gagal."
            print(json.dumps(marker, indent=2, ensure_ascii=False))
            return 1
        marker["test_summary"]["complete_pytest"] = "passed"
        marker["test_summary"]["complete_pytest_output"] = output

    atomic_write_marker(marker)
    print(json.dumps(marker, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
