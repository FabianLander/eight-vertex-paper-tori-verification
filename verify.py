"""The verification suite of the paper, one command:

    python3 verify.py

Every line below is recomputed from the coordinates in data.py, in exact
arithmetic, and the run exits nonzero if any check fails.
"""
import sys
import time

import combinatorics
import development
import field
import separation

MODULES = [
    (field, 'the ordered field Q(sqrt3)'),
    (combinatorics, 'both face lists are 8-vertex tori'),
    (separation, 'both lifts embedded, 240 face pairs'),
    (development, 'both bases flat, moduli exact, det M != 0'),
]


def main():
    t0, rows = time.time(), []
    for mod, what in MODULES:
        print()
        start = time.time()
        try:
            ok, n = mod.main()
        except Exception as e:
            print(f'  FAIL  {type(e).__name__}: {e}')
            ok, n = False, 0
        rows.append((mod.__name__, ok, n, time.time() - start, what))

    print('\nverify.py   what was tested\n')
    for name, ok, n, secs, what in rows:
        print(f'  {"PASS" if ok else "FAIL"}  {name:<15}{n:>3} checks'
              f'{secs:>6.1f} s   {what}')
    bad = [r for r in rows if not r[1]]
    print(f'\nverify.py: {"PASS" if not bad else "FAIL"}   {len(rows)} modules, '
          f'{sum(r[2] for r in rows)} checks, {len(bad)} failed, '
          f'{time.time() - t0:.1f} s')
    return not bad


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
