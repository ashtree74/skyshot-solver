#!/usr/bin/env python3
"""python examples/python_cli.py /path/to/skyshot /path/to/index image.fits"""
import json
import subprocess
import sys


def solve(binary, index, image):
    result = subprocess.run(
        [binary, "solve", image, "--index", index],
        capture_output=True, text=True, check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.strip() or f"solver exited {result.returncode}")
    return json.loads(result.stdout)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit("usage: python_cli.py SOLVER INDEX IMAGE")
    print(json.dumps(solve(*sys.argv[1:]), indent=2))
