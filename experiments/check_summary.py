"""Compare a regenerated summary.md with the committed one, number by number.

Exact text equality is too strict across platforms: Platt scaling is fit by gradient descent and
its last digit can differ between NumPy builds. Every number must match within ``--tol``.
"""

from __future__ import annotations

import argparse
import re
import sys

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("expected")
    ap.add_argument("actual")
    ap.add_argument("--tol", type=float, default=0.03)
    a = ap.parse_args()
    with open(a.expected) as f:
        exp = f.read().splitlines()
    with open(a.actual) as f:
        act = f.read().splitlines()
    if len(exp) != len(act):
        print(f"row count differs: {len(exp)} vs {len(act)}")
        return 1
    bad = 0
    for e, c in zip(exp, act, strict=True):
        ne, nc = NUM.findall(e), NUM.findall(c)
        if e.split("|")[1] != c.split("|")[1] or len(ne) != len(nc):
            print(f"row differs:\n  {e}\n  {c}")
            bad += 1
            continue
        for x, y in zip(ne, nc, strict=True):
            if abs(float(x) - float(y)) > a.tol:
                print(f"value differs ({x} vs {y}):\n  {e}\n  {c}")
                bad += 1
                break
    print("summary reproduced" if not bad else f"{bad} rows differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
