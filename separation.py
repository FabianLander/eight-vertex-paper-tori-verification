"""The separation checks, all 120 face pairs of both tori.

For each pair of faces of a folded base this checks the case of the
separation criterion that its shared simplex dictates.  A pair sharing
nothing has the gap evaluated at every corner of the overlap of its two
shadows: the corners are the corners of one shadow lying in the other
together with the crossings of an edge of one with an edge of the other,
each of them exact, a membership being three sign evaluations and a
crossing one 2 x 2 solve, and one strict sign at all of them puts one
sheet above the other over the whole overlap.  A pair sharing a vertex
needs two positive weights on two edge functions through the shared
vertex, one 2 x 2 solve per choice of two of the four, and a pair sharing
an edge needs a nonzero slope difference across it, a single ratio.

As corroboration, for a pair sharing nothing whose shadows genuinely
overlap, the smallest gap value at the corners is compared with the
constant of a positive-combination identity sigma*g = alpha + mu*l_i +
nu*l_j derived independently by the certificate search below, the two
being equal because an affine function attains its minimum over the
overlap at a corner.  Every sign is the sign of an exact field element;
the floating point values printed alongside are for reading only.

Input: only what the paper prints, taken from data.py, the planar bases
of Table 1, the lift directions zeta and zeta', and the two face lists.
The development plays no role here.

Field layer: Q(sqrt3) with the rationals as the b = 0 slice, the sign of
a + b sqrt3 decided from the signs of a, b and a^2 - 3b^2, in field.py.
"""
from itertools import combinations
import sys

import data

from field import K as N, k as mk


def less(x, y):
    return (x - y).sign() < 0


def solve(M, rhs):
    """Exact Gauss-Jordan over the field. Returns None if the matrix is singular."""
    n = len(M)
    A = [[mk(M[i][j]) for j in range(n)] + [mk(rhs[i])] for i in range(n)]
    for c in range(n):
        p = next((r for r in range(c, n) if not A[r][c].iszero()), None)
        if p is None:
            return None
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        A[c] = [x / pv for x in A[c]]
        for r in range(n):
            if r != c and not A[r][c].iszero():
                f = A[r][c]
                A[r] = [A[r][j] - f * A[c][j] for j in range(n + 1)]
    return [A[i][n] for i in range(n)]


# ------------------------------------------- affine functions (c0, cx, cy) ---
def aff_eval(f, P):
    return f[0] + f[1] * P[0] + f[2] * P[1]


def aff_sub(f, g):
    return (f[0] - g[0], f[1] - g[1], f[2] - g[2])


def aff_neg(f):
    return (-f[0], -f[1], -f[2])


def interp(Q, face, zeta):
    """The affine h with h(Q_v) = zeta_v at the three vertices of the face."""
    M = [[N(1), Q[v][0], Q[v][1]] for v in face]
    s = solve(M, [mk(zeta[v]) for v in face])
    if s is None:
        raise ValueError(f'degenerate shadow at face {face}')
    return tuple(s)


def edge_form(Q, j, k, opp):
    """The form of the edge (j,k) of a face, signed positive at the third vertex."""
    Pj, Pk = Q[j], Q[k]
    dx, dy = Pk[0] - Pj[0], Pk[1] - Pj[1]
    f = (dy * Pj[0] - dx * Pj[1], -dy, dx)
    val = aff_eval(f, Q[opp])
    if val.sign() == 0:
        raise ValueError(f'degenerate face at edge {(j, k)}')
    return f if val.sign() > 0 else aff_neg(f)


def face_forms(Q, face):
    """The three signed forms of a face, each with the vertex pair it vanishes at."""
    out = []
    for i in range(3):
        opp = face[i]
        j, k = face[(i + 1) % 3], face[(i + 2) % 3]
        out.append((edge_form(Q, j, k, opp), {j, k}))
    return out


# ------------------------------------------------------------ the certificates
def certificate(Q, faces, H, FF, i, j):
    """Return (case, record, error). record holds the multipliers of the chosen
    certificate; error is None on success and a message on failure."""
    Fi, Fj = set(faces[i]), set(faces[j])
    shared = Fi & Fj
    g = aff_sub(H[i], H[j])
    forms = [f for f, _ in FF[i]] + [f for f, _ in FF[j]]
    vanish = [s for _, s in FF[i]] + [s for _, s in FF[j]]

    # ---- case (c): the two faces share an edge
    if len(shared) == 2:
        a, b = sorted(shared)
        lab = next(f for f, s in FF[i] if s == {a, b})
        c = (Fi - shared).pop()
        denom = aff_eval(lab, Q[c])
        mu = aff_eval(g, Q[c]) / denom
        if any(not (g[t] - mu * lab[t]).iszero() for t in range(3)):
            return 'c', None, 'g is not a multiple of the shared edge form'
        if mu.iszero():
            return 'c', None, 'mu = 0, the two lifted faces are coplanar'
        return 'c', {'mu': mu}, None

    # ---- case (b): the two faces share exactly one vertex
    if len(shared) == 1:
        s = next(iter(shared))
        cand = [f for f, vs in zip(forms, vanish) if s in vs]
        if len(cand) != 4:
            return 'b', None, 'not four forms through the shared vertex'
        best = None
        for sg in (1, -1):
            gg = g if sg == 1 else aff_neg(g)
            for p, q in combinations(range(4), 2):
                fp, fq = cand[p], cand[q]
                sol = solve([[fp[1], fq[1]], [fp[2], fq[2]]], [gg[1], gg[2]])
                if sol is None:
                    continue                      # linear parts dependent
                m1, m2 = sol
                if m1.sign() <= 0 or m2.sign() <= 0:
                    continue
                if any(not (gg[t] - m1 * fp[t] - m2 * fq[t]).iszero() for t in range(3)):
                    continue
                lo = m1 if less(m1, m2) else m2
                if best is None or less(best['low'], lo):
                    best = {'low': lo}
        if best is None:
            return 'b', None, 'no two-form certificate with positive multipliers'
        return 'b', best, None

    # ---- case (a): the two faces share nothing
    best = None
    for sg in (1, -1):
        gg = g if sg == 1 else aff_neg(g)
        for p, q in combinations(range(6), 2):
            fp, fq = forms[p], forms[q]
            M = [[N(1), fp[0], fq[0]],
                 [N(0), fp[1], fq[1]],
                 [N(0), fp[2], fq[2]]]
            sol = solve(M, [gg[0], gg[1], gg[2]])
            if sol is None:
                continue
            al, m1, m2 = sol
            if al.sign() <= 0 or m1.sign() < 0 or m2.sign() < 0:
                continue
            if any(not (gg[t] - (al if t == 0 else N(0)) - m1 * fp[t] - m2 * fq[t]).iszero()
                   for t in range(3)):
                continue
            if best is None or less(best['alpha'], al):
                best = {'sigma': sg, 'alpha': al}
    if best is None:
        return 'a', None, 'no certificate alpha + two forms'
    return 'a', best, None


CHECKS = []


def check(ok, label, value=''):
    """One aligned PASS/FAIL line, collected for the closing summary."""
    CHECKS.append(bool(ok))
    print(f'  {"PASS" if ok else "FAIL"}  {label:<50} {value}'.rstrip())
    return ok


def note(label):
    print(f'        {label}')


# ------------------------------------------------------------- the corners
def edges_of(face):
    return [(face[k], face[(k + 1) % 3]) for k in range(3)]


def line_of(Q, p, q):
    """The line through Q_p and Q_q as an affine function, unsigned."""
    dx, dy = Q[q][0] - Q[p][0], Q[q][1] - Q[p][1]
    return (dy * Q[p][0] - dx * Q[p][1], -dy, dx)


def inside(x, forms):
    """Membership in a shadow: the three edge functions nonnegative."""
    return all(aff_eval(f, x).sign() >= 0 for f, _ in forms)


def key(x):
    a, b = x
    return (a.a, a.b, b.a, b.b)


def corners(Q, fi, gj, faces, FF):
    """Every corner of the overlap of the two shadows, as a superset: the
    corners of one triangle lying in the other and the crossings of an edge
    of one with an edge of the other, all of them points of the overlap."""
    out, seen = [], set()

    def push(x):
        if key(x) not in seen:
            seen.add(key(x))
            out.append(x)

    for a, b in ((fi, gj), (gj, fi)):
        for v in faces[a]:
            if inside(Q[v], FF[b]):
                push(Q[v])
    for p, q in edges_of(faces[fi]):
        e1 = line_of(Q, p, q)
        for r, u in edges_of(faces[gj]):
            e2 = line_of(Q, r, u)
            sol = solve([[e1[1], e1[2]], [e2[1], e2[2]]],
                            [aff_neg(e1)[0], aff_neg(e2)[0]])
            if sol is None:
                continue                      # parallel or collinear lines
            x = (sol[0], sol[1])
            if inside(x, FF[fi]) and inside(x, FF[gj]):
                push(x)
    return out


def on_segment(x, Q, a, b):
    """Exact: x on the closed segment from Q_a to Q_b."""
    ax, ay = x[0] - Q[a][0], x[1] - Q[a][1]
    bx, by = x[0] - Q[b][0], x[1] - Q[b][1]
    ex, ey = Q[b][0] - Q[a][0], Q[b][1] - Q[a][1]
    if (ex * ay - ey * ax).sign() != 0:
        return False
    return (ax * bx + ay * by).sign() <= 0


# ------------------------------------------------------------------ the run
def run(name, Qraw, faces, zeta, split):
    Q = {v: (mk(p[0]), mk(p[1])) for v, p in Qraw.items()}
    print(f'\n{name}')

    H = [interp(Q, f, zeta) for f in faces]
    FF = [face_forms(Q, f) for f in faces]

    counts = {'a': 0, 'b': 0, 'c': 0}
    overlapping = 0
    ok_signs = ok_agree = ok_bc = True
    min_a = min_b = min_c = min_side = None

    def keep_min(cur, val):
        return val if cur is None or less(val, cur) else cur

    for i, j in combinations(range(16), 2):
        shared = set(faces[i]) & set(faces[j])
        case = {0: 'a', 1: 'b', 2: 'c'}[len(shared)]
        counts[case] += 1
        g = aff_sub(H[i], H[j])
        pts = corners(Q, i, j, faces, FF)

        if case == 'a':
            vals = [aff_eval(g, x) for x in pts]
        elif case == 'b':
            s = shared.pop()
            vals = [aff_eval(g, x) for x in pts if key(x) != key(Q[s])]
        else:
            a, b = sorted(shared)
            vals = [aff_eval(g, x) for x in pts
                    if not on_segment(x, Q, a, b)]

        signs = {v.sign() for v in vals}
        strict = signs in (set(), {1}, {-1})
        if not strict:
            ok_signs = False
            note(f'pair {faces[i]}, {faces[j]} case {case}: '
                 f'gap signs {sorted(signs)} at the corners')
        if vals and strict:
            m = None
            for v in vals:
                w = v if v.sign() > 0 else mk(0) - v
                m = w if m is None or less(w, m) else m
            if case == 'a':
                min_a = keep_min(min_a, m)
            else:
                min_side = keep_min(min_side, m)

        cert_case, rec, err = certificate(Q, faces, H, FF, i, j)
        if cert_case != case or rec is None:
            ok_bc = False
            note(f'pair {faces[i]}, {faces[j]}: identity route failed, {err}')
            continue

        if case == 'a':
            if pts:
                overlapping += 1
                agree = all((aff_eval(g, x) * rec['sigma']).sign() > 0
                            for x in pts)
                low = None
                for x in pts:
                    v = aff_eval(g, x) * rec['sigma']
                    low = v if low is None or less(v, low) else low
                if not agree or not (low - rec['alpha']).iszero():
                    ok_agree = False
                    note(f'pair {faces[i]}, {faces[j]}: corner minimum '
                         f'{low} against identity alpha {rec["alpha"]}')
        elif case == 'b':
            lo = rec['low']
            min_b = keep_min(min_b, lo)
        else:
            mu = rec['mu']
            mu = mu if mu.sign() > 0 else mk(0) - mu
            min_c = keep_min(min_c, mu)

    check(counts == dict(zip('abc', split)),
          f'the 120 pairs split {split[0]} / {split[1]} / {split[2]}',
          f"{counts['a']} share nothing, {counts['b']} a vertex, "
          f"{counts['c']} an edge")
    check(ok_signs, 'one strict gap sign at every corner off the shared simplex')
    check(ok_bc, 'weights and slopes exist, the identity route agrees on cases')
    check(ok_agree,
          'case (a): corner minimum equals the identity constant',
          f"{overlapping} of {counts['a']} pairs have overlapping shadows")

    for label, val in (('smallest |gap| at a corner, case (a)', min_a),
                       ('smallest weight, case (b)', min_b),
                       ('smallest |slope difference|, case (c)', min_c),
                       ('smallest |gap| at a corner off a shared simplex',
                        min_side)):
        note(f'{label}: {val}')
        note(f'  ~ {val.to_float():.10e}')


def main():
    print('separation   the gap at the overlap corners, weights, slopes')
    run('SQUARE TORUS, field Q', data.Q_SQUARE, data.FACES_SQUARE,
        data.ZETA_SQUARE, (24, 72, 24))
    run('HEXAGONAL TORUS, field Q(sqrt3)', data.hex_base(N), data.FACES_HEX,
        data.ZETA_HEX, (21, 75, 24))
    return all(CHECKS), len(CHECKS)


if __name__ == '__main__':
    sys.exit(0 if main()[0] else 1)
