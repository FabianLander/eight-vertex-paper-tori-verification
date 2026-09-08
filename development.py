"""Face signs, gluings, vertex holonomies, tau, and det M != 0, both tori.

The correction map

    Phi = (delta_0, ..., delta_6, Re tau - Re tau_0, Im tau - Im tau_0)

is a function of the sixteen horizontal coordinates of the eight vertices, with
the heights frozen at t*zeta.  Every squared edge length of the lift is its
planar value plus t^2 (zeta_a - zeta_b)^2, and Phi sees the configuration only
through squared edge lengths, so Phi is smooth in x and t jointly, and the base is (x^0, 0).
This assembles the 9 x 16 Jacobian of Phi at the base in exact arithmetic and
eliminates over the field.

Input.  The run consumes the printed bases, the square and hexagonal columns
of Table 1, from data.py, together with the face lists, the sign strings, the
lift directions and the markings.  Everything that follows is computed from
that printed data.

Development.  The development tree is breadth first in the dual graph, rooted at
face 0, the neighbours of a face enqueued in the order the three edges appear in
its boundary cycle and the faces of each level processed in the order reached.
The root face (a,b,c) is placed at D(a) = (0,0), D(b) = (1,0),
D(c) = (p/L_ab, S/L_ab), the metric development scaled by 1/|ab|; every child is
placed from its parent by

    D(r) = D(q) + ( m (D(p)-D(q)) + n J(D(p)-D(q)) ) / L_qp,

with (q,p,r) the child's own positive cycle, m = <qr,qp>, n = S of the child,
and J the quarter turn.  Both formulas are rational in the squared edge lengths
and in the doubled areas S, so the whole development is rational; tau = v_2/v_1
is similarity invariant, so the scaling does not touch it.

Doubled areas.  S_f = 2 Area(f) = sqrt(alpha beta - <,>^2).  At the planar base
S_f is the absolute value of the planar cross product, an exact field element,
and along horizontal directions with the heights at zero S_f is the signed cross
product times the fold sign, so its exact derivative needs no square root.  That
derivative is cross-checked against Lagrange's identity
2 S dS = beta d alpha + alpha d beta - 2 <,> d<,>.

Angle derivatives.  At the corner A of ABC, with alpha = |AB|^2, beta = |AC|^2,
<AB,AC> = (alpha + beta - |BC|^2)/2 and S = 2 Area,

    d theta_A = ( <AB,AC> dS - S d<AB,AC> ) / (alpha beta),

since <AB,AC>^2 + S^2 = alpha beta.  Rational at the base.  No angle is ever
evaluated, only its derivative.

Vertex holonomies.  For each vertex the faces of its star are walked around the
link cycle and the transition maps of the crossed edges composed, each
transition read off the committed development as the orientation preserving
isometry carrying the chart of the face entered to the chart of the face left.
A tree edge contributes the identity, a non-tree edge its gluing.
The composite must be the identity in both parts, which says the developed star
closes up exactly.  Every transition fixes the shared edge and hence the
developed image of the vertex, so the transported vertex position agrees across
all charts of the star; that is checked too.

Periods.  Each marking loop is an edge loop and its period is the sum of its
developed edge vectors, one developed triangle committed per loop edge, the
earlier of the two faces of that edge in the printed face order.

The finite-difference control at the end compares the exact Jacobian against an
independent floating point evaluation of the same F, arctangents and square
roots and all.  It validates the symbolic assembly and is not a certificate.
"""
import itertools
import math
import sys
from fractions import Fraction as Fr

import data
from field import K


# ---------------------------------------------------------------- reporting
CHECKS = []


def check(ok, label, value=''):
    """One aligned PASS/FAIL line, collected for the closing summary."""
    CHECKS.append(bool(ok))
    print(f'  {"PASS" if ok else "FAIL"}  {label:<50} {value}'.rstrip())
    return ok


def note(label, value=''):
    """A reported quantity, indented past the check column."""
    print(f'        {label}{value}')


# ---------------------------------------------------------------- the fields
class FieldQ:
    name = 'Q'
    ZERO = Fr(0)
    ONE = Fr(1)

    @staticmethod
    def c(x):
        return x if isinstance(x, Fr) else Fr(x)

    @staticmethod
    def fl(x):
        return float(x)

    @staticmethod
    def show(x):
        return str(x)


class FieldK:
    name = 'Q(sqrt3)'
    ZERO = K(0, 0)
    ONE = K(1, 0)

    @staticmethod
    def c(x):
        return x if isinstance(x, K) else K(x, 0)

    @staticmethod
    def fl(x):
        return x.to_float()

    @staticmethod
    def show(x):
        return str(x)


# ---------------------------------------------------------------- dual numbers
class Dual:
    """value plus a gradient of length N over the ambient field."""
    __slots__ = ('v', 'd')
    Z = None
    N = 0
    C = None

    def __init__(self, v, d):
        self.v = v
        self.d = d

    @classmethod
    def setup(cls, fld, n):
        cls.Z, cls.N, cls.C = fld.ZERO, n, fld.c

    @classmethod
    def const(cls, v):
        return cls(cls.C(v), [cls.Z]*cls.N)

    @classmethod
    def var(cls, v, i):
        d = [cls.Z]*cls.N
        d[i] = cls.C(1)
        return cls(cls.C(v), d)

    @classmethod
    def _c(cls, o):
        return o if isinstance(o, Dual) else cls.const(o)

    def __add__(s, o):
        o = Dual._c(o)
        return Dual(s.v + o.v, [a + b for a, b in zip(s.d, o.d)])
    __radd__ = __add__

    def __sub__(s, o):
        o = Dual._c(o)
        return Dual(s.v - o.v, [a - b for a, b in zip(s.d, o.d)])

    def __rsub__(s, o):
        return Dual._c(o).__sub__(s)

    def __neg__(s):
        return Dual(-s.v, [-a for a in s.d])

    def __mul__(s, o):
        o = Dual._c(o)
        sv, ov = s.v, o.v
        return Dual(sv*ov, [sv*b + ov*a for a, b in zip(s.d, o.d)])
    __rmul__ = __mul__

    def __truediv__(s, o):
        o = Dual._c(o)
        ov = o.v
        q = s.v/ov
        return Dual(q, [(a - q*b)/ov for a, b in zip(s.d, o.d)])

    def __rtruediv__(s, o):
        return Dual._c(o).__truediv__(s)


# ---------------------------------------------------------------- linear algebra
def pivot_columns(rows):
    """exact Gaussian elimination with column pivots; returns (rank, columns)."""
    A = [r[:] for r in rows]
    nr, nc = len(A), len(A[0])
    piv, r = [], 0
    for c in range(nc):
        p = next((i for i in range(r, nr) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        pr = A[r][c]
        A[r] = [x/pr for x in A[r]]
        for i in range(nr):
            if i != r and A[i][c] != 0:
                fac = A[i][c]
                A[i] = [x - fac*y for x, y in zip(A[i], A[r])]
        piv.append(c)
        r += 1
        if r == nr:
            break
    return r, piv


def det_exact(M, one):
    A = [r[:] for r in M]
    n = len(A)
    det = one
    for c in range(n):
        p = next((i for i in range(c, n) if A[i][c] != 0), None)
        if p is None:
            return one - one
        if p != c:
            A[c], A[p] = A[p], A[c]
            det = -det
        det = det*A[c][c]
        piv = A[c][c]
        for i in range(c+1, n):
            if A[i][c] != 0:
                fac = A[i][c]/piv
                A[i] = [x - fac*y for x, y in zip(A[i], A[c])]
    return det


# ------------------------------------------------------------ vertex holonomy
# An orientation preserving isometry of the plane is carried as (c, s, tx, ty),
# acting by z |-> (c + i s) z + (tx + i ty) with c^2 + s^2 = 1.  Everything here
# stays in the ambient field, so the composites are exact over Q and over
# Q(sqrt3) alike.


def iso_compose(A, B):
    """A after B."""
    ca, sa, ax, ay = A
    cb, sb, bx, by = B
    return (ca*cb - sa*sb, ca*sb + sa*cb,
            ca*bx - sa*by + ax, sa*bx + ca*by + ay)


def iso_apply(A, z):
    c, s, tx, ty = A
    return (c*z[0] - s*z[1] + tx, s*z[0] + c*z[1] + ty)


def transition(Dv, f, g, p, q):
    """The orientation preserving isometry carrying the chart of g to the chart
    of f, pinned by the two developed endpoints of the shared edge {p, q}.  It
    is the identity across a tree edge and the gluing of the paper
    across a non-tree edge, read in the direction f <- g."""
    gx = Dv[(g, q)][0] - Dv[(g, p)][0]
    gy = Dv[(g, q)][1] - Dv[(g, p)][1]
    fx = Dv[(f, q)][0] - Dv[(f, p)][0]
    fy = Dv[(f, q)][1] - Dv[(f, p)][1]
    n = gx*gx + gy*gy                      # nonzero, the faces being nondegenerate
    c = (gx*fx + gy*fy)/n                  # the complex quotient (f edge)/(g edge)
    s = (gx*fy - gy*fx)/n
    px, py = Dv[(g, p)]
    return (c, s,
            Dv[(f, p)][0] - (c*px - s*py),
            Dv[(f, p)][1] - (s*px + c*py))


def link_cycle(FACES, edge2faces, v):
    """The faces of the star of v in cyclic order, each paired with the edge at
    v through which the walk leaves it.  Raises if the link is not a single
    cycle through every face at v."""
    star = [fi for fi, f in enumerate(FACES) if v in f]
    f0 = star[0]
    i = FACES[f0].index(v)
    e_start = frozenset((v, FACES[f0][(i + 1) % 3]))
    walk, cur, e = [], f0, e_start
    while True:
        walk.append((cur, e))
        nxt = [x for x in edge2faces[e] if x != cur][0]
        if nxt == f0:
            break
        j = FACES[nxt].index(v)
        a, b = FACES[nxt][(j + 1) % 3], FACES[nxt][(j + 2) % 3]
        e = frozenset((v, b)) if frozenset((v, a)) == e else frozenset((v, a))
        cur = nxt
        if len(walk) > len(star):
            raise ValueError(f'link of vertex {v} is not a simple cycle')
    j = FACES[f0].index(v)
    e_back = frozenset((v, FACES[f0][(j + 2) % 3]))
    if walk[-1][1] != e_back or len(walk) != len(star):
        raise ValueError(f'link of vertex {v} misses part of its star, '
                         f'{len(walk)} faces walked of {len(star)}')
    return walk, star


def vertex_holonomies(FACES, edge2faces, Dv, tree_set, gluings, FLD):
    """Compose the transition maps around each vertex link and check that the
    holonomy is the identity in both parts, on the actual field elements,
    together with the link being a single cycle through the whole star, the tree
    edges gluing by the identity, and each crossed non-tree edge carrying the
    translation of its gluing."""
    one, zero = FLD.ONE, FLD.ZERO
    ident = (one, zero, zero, zero)
    gluing_by_edge = {frozenset(e): t for e, t in gluings}
    sizes, crossings, ok, pts_ok = [], [], True, True
    for v in range(8):
        walk, star = link_cycle(FACES, edge2faces, v)
        sizes.append(len(star))
        A, ncross = ident, 0
        base = Dv[(walk[0][0], v)]
        for step, (f, e) in enumerate(walk):
            g = [x for x in edge2faces[e] if x != f][0]
            p, q = sorted(e)
            T = transition(Dv, f, g, p, q)
            if T[0]*T[0] + T[1]*T[1] != one:
                ok = False
                note(f'vertex {v}: transition across {p}{q} is not an isometry')
            if e in tree_set:
                if T != ident:
                    ok = False
                    note(f'vertex {v}: tree edge {p}{q} moves the chart')
            else:
                ncross += 1
                t = (T[2], T[3])
                d = gluing_by_edge.get(frozenset(e))
                if d is None or (T[0], T[1]) != (one, zero) or \
                        (t != d and t != (-d[0], -d[1])):
                    ok = False
                    note(f'vertex {v}: non-tree edge {p}{q} disagrees with the '
                         f'translation of its gluing')
            A = iso_compose(A, T)
            # the vertex, transported back into the chart of the first face
            if iso_apply(A, Dv[(g, v)]) != base:
                pts_ok = False
                note(f'vertex {v}: chart of face {FACES[g]} places it elsewhere '
                     f'at step {step}')
        crossings.append(ncross)
        if A != ident:
            ok = False
            note(f'vertex {v}: holonomy is (c,s,tx,ty) = '
                 f'({FLD.show(A[0])}, {FLD.show(A[1])}, '
                 f'{FLD.show(A[2])}, {FLD.show(A[3])}), not the identity')
    check(ok, 'eight vertex holonomies are the identity',
          f'stars {sizes}, sum {sum(sizes)} of 48')
    check(pts_ok, 'every chart of a star places its vertex alike',
          f'non-tree edges crossed {crossings}')


# ---------------------------------------------------------------- exact assembly
def assemble(FACES, P, ZETA, LOOPS, TARGET, FLD, sign_string, free_printed):
    Dual.setup(FLD, 16)
    HALF = FLD.c(Fr(1, 2))
    zero = Dual.const(0)
    one = Dual.const(1)

    PD = {v: (Dual.var(P[v][0], 2*v), Dual.var(P[v][1], 2*v + 1)) for v in range(8)}
    note('planar base Q^0, as the tables print it:')
    for v in range(8):
        note(f'  Q{v} = ({FLD.show(P[v][0])}, {FLD.show(P[v][1])})')
    note(f'lift direction zeta = {[str(x) for x in ZETA]}')

    edge2faces = {}
    for fi, f in enumerate(FACES):
        for j in range(3):
            edge2faces.setdefault(frozenset((f[j], f[(j+1) % 3])), []).append(fi)
    assert len(edge2faces) == 24 and all(len(x) == 2 for x in edge2faces.values())

    Lsq = {}
    for e in edge2faces:
        a, b = sorted(e)
        dx = PD[a][0] - PD[b][0]
        dy = PD[a][1] - PD[b][1]
        Lsq[(a, b)] = dx*dx + dy*dy

    def LL(a, b):
        return Lsq[(a, b)] if a < b else Lsq[(b, a)]

    # ---- fold signs and doubled areas -------------------------------------
    S, signs = {}, ''
    for fi, f in enumerate(FACES):
        a, b, c = f
        cr = ((PD[b][0]-PD[a][0])*(PD[c][1]-PD[a][1])
              - (PD[b][1]-PD[a][1])*(PD[c][0]-PD[a][0]))
        assert cr.v != 0, f'planar face {fi} degenerate'
        pos = cr.v > 0
        signs += '+' if pos else '-'
        S[fi] = cr if pos else -cr
    check(signs == sign_string, 'sixteen planar fold signs match the printed',
          signs)

    # ---- angle derivatives, with the Lagrange cross-check ------------------
    dtheta = {}
    lagrange_ok = True
    for fi, f in enumerate(FACES):
        Sf = S[fi]
        for A, B, C in ((f[0], f[1], f[2]), (f[1], f[2], f[0]), (f[2], f[0], f[1])):
            al, be, ga = LL(A, B), LL(A, C), LL(B, C)
            p = (al + be - ga)*HALF
            # Lagrange:  <,>^2 + S^2 = alpha beta, hence 2 S dS = ...
            if al.v*be.v - p.v*p.v != Sf.v*Sf.v:
                lagrange_ok = False
            dS_lag = [(be.v*a + al.v*b - 2*p.v*c)/(2*Sf.v)
                      for a, b, c in zip(al.d, be.d, p.d)]
            if dS_lag != Sf.d:
                lagrange_ok = False
            dtheta[(fi, A)] = [(p.v*ds - Sf.v*dp)/(al.v*be.v)
                               for ds, dp in zip(Sf.d, p.d)]
    check(lagrange_ok, "dS agrees with Lagrange's identity", '48 of 48 corners')

    # ---- development -------------------------------------------------------
    ROOT = 0
    a, b, c = FACES[ROOT]
    Lab, Lac, Lbc = LL(a, b), LL(a, c), LL(b, c)
    pA = (Lab + Lac - Lbc)*HALF
    D = {(ROOT, a): (zero, zero), (ROOT, b): (one, zero),
         (ROOT, c): (pA/Lab, S[ROOT]/Lab)}

    placements, nontree, tree = [], [], []
    visited, frontier, parent_edge = {ROOT}, [ROOT], {}
    while frontier:
        nxt = []
        for fi in frontier:
            f = FACES[fi]
            for j in range(3):
                pv, qv = f[j], f[(j+1) % 3]
                e = frozenset((pv, qv))
                gj = [x for x in edge2faces[e] if x != fi][0]
                if gj in visited:
                    if parent_edge.get(gj) != e and parent_edge.get(fi) != e:
                        nontree.append((fi, gj, (pv, qv)))
                    continue
                g = FACES[gj]
                q0 = p0 = r0 = None
                for kk in range(3):
                    if {g[kk], g[(kk+1) % 3]} == {pv, qv}:
                        q0, p0, r0 = g[kk], g[(kk+1) % 3], g[(kk+2) % 3]
                        break
                Bq, Bp = D[(fi, q0)], D[(fi, p0)]
                vx, vy = Bp[0] - Bq[0], Bp[1] - Bq[1]
                m = (LL(q0, r0) + LL(q0, p0) - LL(p0, r0))*HALF
                n = S[gj]
                Lqp = LL(q0, p0)
                D[(gj, q0)], D[(gj, p0)] = Bq, Bp
                D[(gj, r0)] = (Bq[0] + (m*vx - n*vy)/Lqp,
                               Bq[1] + (m*vy + n*vx)/Lqp)
                placements.append((fi, gj, q0, p0, r0))
                parent_edge[gj] = e
                visited.add(gj)
                tree.append(e)
                nxt.append(gj)
        frontier = nxt
    assert len(visited) == 16
    seen, nt = set(), []
    for fi, gj, ab in nontree:
        if frozenset(ab) not in seen:
            seen.add(frozenset(ab))
            nt.append((fi, gj, ab))
    nontree = nt
    check(len(tree) == 15 and len(nontree) == 9,
          'development tree, 15 tree and 9 non-tree edges',
          f'{len(tree)} and {len(nontree)}, root face {ROOT} = {FACES[ROOT]}')

    # developed edge lengths and orientations (the development is faithful)
    dev_ok = True
    for fi, f in enumerate(FACES):
        pts = [D[(fi, v)] for v in f]
        cr = ((pts[1][0]-pts[0][0])*(pts[2][1]-pts[0][1])
              - (pts[1][1]-pts[0][1])*(pts[2][0]-pts[0][0]))
        if not cr.v > 0:
            dev_ok = False
        for j in range(3):
            u_, w_ = f[j], f[(j+1) % 3]
            dx = D[(fi, u_)][0].v - D[(fi, w_)][0].v
            dy = D[(fi, u_)][1].v - D[(fi, w_)][1].v
            if dx*dx + dy*dy != LL(u_, w_).v/Lab.v:
                dev_ok = False
    check(dev_ok, '16 developed faces positive, side lengths right')

    # ---- the nine gluings ---------------------------------------------
    rot_free, gluings = True, []
    for fi, gj, (pv, qv) in nontree:
        ef = (D[(fi, pv)][0].v - D[(fi, qv)][0].v,
              D[(fi, pv)][1].v - D[(fi, qv)][1].v)
        eg = (D[(gj, pv)][0].v - D[(gj, qv)][0].v,
              D[(gj, pv)][1].v - D[(gj, qv)][1].v)
        if ef != eg:
            rot_free = False
            note(f'rotation in the gluing at edge {(pv, qv)}')
            continue
        tp = (D[(fi, pv)][0].v - D[(gj, pv)][0].v,
              D[(fi, pv)][1].v - D[(gj, pv)][1].v)
        tq = (D[(fi, qv)][0].v - D[(gj, qv)][0].v,
              D[(fi, qv)][1].v - D[(gj, qv)][1].v)
        assert tp == tq
        gluings.append(((pv, qv), tp))
    check(rot_free and len(gluings) == 9,
          'the nine gluings are translations')

    # one gluing in full, the worked example of the text
    fi, gj, (pv, qv) = nontree[1]
    note(f'worked gluing, non-tree edge ({pv},{qv}) between faces '
         f'{FACES[fi]} and {FACES[gj]}:')
    for lab, ff in ((' from', fi), ('   in', gj)):
        note(f'  {lab} {FACES[ff]}: D({pv}) = ({FLD.show(D[(ff, pv)][0].v)}, '
             f'{FLD.show(D[(ff, pv)][1].v)}), '
             f'D({qv}) = ({FLD.show(D[(ff, qv)][0].v)}, '
             f'{FLD.show(D[(ff, qv)][1].v)})')
    ev = (D[(fi, pv)][0].v - D[(fi, qv)][0].v, D[(fi, pv)][1].v - D[(fi, qv)][1].v)
    tr = (D[(fi, pv)][0].v - D[(gj, pv)][0].v, D[(fi, pv)][1].v - D[(gj, pv)][1].v)
    note(f'  common unfolded edge vector: ({FLD.show(ev[0])}, {FLD.show(ev[1])})')
    note(f'  translation of the gluing:   ({FLD.show(tr[0])}, {FLD.show(tr[1])})')

    # ---- vertex holonomies -------------------------------------------------
    Dv = {key: (z[0].v, z[1].v) for key, z in D.items()}
    assert len(Dv) == 48
    vertex_holonomies(FACES, edge2faces, Dv, set(tree), gluings, FLD)

    # ---- periods of the marking -------------------------------------------
    loop_faces = []
    for lp in LOOPS:
        lf = []
        for k in range(len(lp)-1):
            lf.append(min(fi for fi, f in enumerate(FACES)
                          if lp[k] in f and lp[k+1] in f))
        loop_faces.append(lf)

    def period(lp, lf):
        tx, ty = zero, zero
        for k in range(len(lp)-1):
            fi, a_, b_ = lf[k], lp[k], lp[k+1]
            tx = tx + D[(fi, b_)][0] - D[(fi, a_)][0]
            ty = ty + D[(fi, b_)][1] - D[(fi, a_)][1]
        return (tx, ty)

    per = [period(lp, lf) for lp, lf in zip(LOOPS, loop_faces)]
    n1 = per[0][0]*per[0][0] + per[0][1]*per[0][1]
    tim0 = (per[0][0]*per[1][1] - per[0][1]*per[1][0])/n1
    swapped = False
    if not tim0.v > 0:
        LOOPS = [LOOPS[1], LOOPS[0]]
        loop_faces = [loop_faces[1], loop_faces[0]]
        per = [per[1], per[0]]
        swapped = True
    v1, v2 = per
    n1 = v1[0]*v1[0] + v1[1]*v1[1]
    tre = (v1[0]*v2[0] + v1[1]*v2[1])/n1
    tim = (v1[0]*v2[1] - v1[1]*v2[0])/n1
    check(tre.v == TARGET[0] and tim.v == TARGET[1],
          'tau equals the target exactly',
          f'tau = {FLD.show(tre.v)} + i {FLD.show(tim.v)}')
    note(f'marking loops {LOOPS[0]} and {LOOPS[1]}'
         f'{"  (order swapped for positive orientation)" if swapped else ""}, '
         f'det(v1, v2) > 0 in the listed order: {not swapped}')
    note(f'committed faces per loop edge: {loop_faces}')
    note(f'v1  = ({FLD.show(v1[0].v)}, {FLD.show(v1[1].v)})')
    note(f'v2  = ({FLD.show(v2[0].v)}, {FLD.show(v2[1].v)})')
    note(f'tau ~ {FLD.fl(tre.v):.15f} + i {FLD.fl(tim.v):.15f}')

    # ---- the period lattice ------------------------------------------------
    det12 = v1[0].v*v2[1].v - v1[1].v*v2[0].v
    coords = []
    integral = True
    for (e, w) in gluings:
        n_1 = (w[0]*v2[1].v - w[1]*v2[0].v)/det12
        n_2 = (v1[0].v*w[1] - v1[1].v*w[0])/det12
        cc = []
        for n in (n_1, n_2):
            if isinstance(n, K):
                if n.b != 0 or n.a.denominator != 1:
                    integral = False
                    cc.append(None)
                else:
                    cc.append(int(n.a))
            else:
                if n.denominator != 1:
                    integral = False
                    cc.append(None)
                else:
                    cc.append(int(n))
        coords.append((e, tuple(cc)))
    g = 0
    if integral:
        for (_, x), (_, y) in itertools.combinations(coords, 2):
            g = math.gcd(g, abs(x[0]*y[1] - x[1]*y[0]))
    check(integral, 'the nine translations integral in (v1, v2)')
    check(g == 1, 'gcd of their 2x2 minors is 1', f'gcd = {g}')
    for e, cc in coords:
        note(f'  edge {e[0]}{e[1]}: {cc}')

    # ---- the 9 x 16 Jacobian ----------------------------------------------
    rows = []
    for v in range(7):
        row = [FLD.ZERO]*16
        for fi, f in enumerate(FACES):
            if v in f:
                row = [x - y for x, y in zip(row, dtheta[(fi, v)])]
        rows.append(row)
    rows.append(list(tre.d))
    rows.append(list(tim.d))

    rank, piv = pivot_columns(rows)
    names = [f'{"xy"[k]}{v}' for v in range(8) for k in range(2)]
    check(rank == 9, 'Jacobian 9 x 16 has exact rank 9',
          f'rank {rank}, kernel dimension {16-rank}')
    Msq = [[rows[i][c] for c in piv] for i in range(9)]
    dM = det_exact(Msq, FLD.ONE)
    check(dM != 0, 'det M nonvanishing', f'~ {FLD.fl(dM):.10e}')
    picked = tuple(names[c] for c in piv)
    check(picked == tuple(free_printed),
          'the nine columns pivoting selects are the printed nine',
          ' '.join(picked))
    note(f'det M = {FLD.show(dM)}')

    return dict(FACES=FACES, ROOT=ROOT, placements=placements, LOOPS=LOOPS,
                loop_faces=loop_faces, rows=rows, ZETA=ZETA, P=P, FLD=FLD,
                det=dM)


# ---------------------------------------------------------------- float twin
def phi_float(xy, z, FACES, ROOT, placements, LOOPS, loop_faces, target):
    """an independent floating-point evaluation of the same F, with real
    arctangents and real square roots, used only as a control."""
    Q = [(xy[2*v], xy[2*v+1], z[v]) for v in range(8)]

    def L(a, b):
        return sum((Q[a][i]-Q[b][i])**2 for i in range(3))

    S = {}
    for fi, f in enumerate(FACES):
        a, b, c = f
        al, be, ga = L(a, b), L(a, c), L(b, c)
        p = (al + be - ga)/2
        S[fi] = math.sqrt(al*be - p*p)

    out = []
    for v in range(7):
        tot = 0.0
        for fi, f in enumerate(FACES):
            if v in f:
                i = f.index(v)
                A, B, C = v, f[(i+1) % 3], f[(i+2) % 3]
                al, be, ga = L(A, B), L(A, C), L(B, C)
                tot += math.atan2(S[fi], (al + be - ga)/2)
        out.append(2*math.pi - tot)

    a, b, c = FACES[ROOT]
    Lab, Lac, Lbc = L(a, b), L(a, c), L(b, c)
    D = {(ROOT, a): (0.0, 0.0), (ROOT, b): (1.0, 0.0),
         (ROOT, c): ((Lab + Lac - Lbc)/2/Lab, S[ROOT]/Lab)}
    for (fi, gj, q0, p0, r0) in placements:
        Bq, Bp = D[(fi, q0)], D[(fi, p0)]
        vx, vy = Bp[0]-Bq[0], Bp[1]-Bq[1]
        m = (L(q0, r0) + L(q0, p0) - L(p0, r0))/2
        n, Lqp = S[gj], L(q0, p0)
        D[(gj, q0)], D[(gj, p0)] = Bq, Bp
        D[(gj, r0)] = (Bq[0] + (m*vx - n*vy)/Lqp, Bq[1] + (m*vy + n*vx)/Lqp)

    per = []
    for lp, lf in zip(LOOPS, loop_faces):
        tx = ty = 0.0
        for k in range(len(lp)-1):
            fi, a_, b_ = lf[k], lp[k], lp[k+1]
            tx += D[(fi, b_)][0] - D[(fi, a_)][0]
            ty += D[(fi, b_)][1] - D[(fi, a_)][1]
        per.append((tx, ty))
    (v1x, v1y), (v2x, v2y) = per
    nn = v1x*v1x + v1y*v1y
    out.append((v1x*v2x + v1y*v2y)/nn - target[0])
    out.append((v1x*v2y - v1y*v2x)/nn - target[1])
    return out


def fd_control(res, target_float):
    FLD = res['FLD']
    xy0 = [FLD.fl(res['P'][v][k]) for v in range(8) for k in (0, 1)]
    z0 = [0.0]*8
    args = (res['FACES'], res['ROOT'], res['placements'], res['LOOPS'],
            res['loop_faces'], target_float)

    base = phi_float(xy0, z0, *args)
    note(f'float control, not part of the proof: |F| at the base = '
         f'{max(abs(x) for x in base):.3e}')

    h = 1e-6
    worst = 0.0
    for j in range(16):
        xp, xm = xy0[:], xy0[:]
        xp[j] += h
        xm[j] -= h
        fp = phi_float(xp, z0, *args)
        fm = phi_float(xm, z0, *args)
        for i in range(9):
            fd = (fp[i] - fm[i])/(2*h)
            ex = FLD.fl(res['rows'][i][j])
            worst = max(worst, abs(fd - ex)/max(1.0, abs(ex)))
    note(f'central differences against the exact Jacobian, 144 entries: '
         f'worst relative deviation {worst:.3e}')

    # the residual is quadratic in t
    zeta = [FLD.fl(FLD.c(x)) if not isinstance(x, float) else x
            for x in res['ZETA']]
    note('residual of the lift, showing the t^2 law:')
    for t in (1e-2, 1e-3, 1e-4):
        zz = [t*x for x in zeta]
        r = max(abs(x) for x in phi_float(xy0, zz, *args))
        note(f'  t = {t:.0e}   |F| = {r:.6e}   |F|/t^2 = {r/t**2:.6f}')
    return worst


# ------------------------------------------------------- the two printed bases
PRINTED_SQUARE = data.Q_SQUARE          # Table 1, square column
PRINTED_HEX = data.hex_base(K)          # Table 1, hexagonal column


def main():
    print('development   face signs, gluings, holonomies, tau, det M != 0')

    print("\nSQUARE TORUS, target tau = i   field Q")
    res_sq = assemble(data.FACES_SQUARE, PRINTED_SQUARE, data.ZETA_SQUARE, data.LOOPS_SQUARE,
                      (Fr(0), Fr(1)), FieldQ, data.SIGNS_SQUARE,
                      data.FREE_SQUARE)
    fd_control(res_sq, (0.0, 1.0))

    print("\nHEXAGONAL TORUS, target tau = rho   field Q(sqrt3)")
    res_hx = assemble(data.FACES_HEX, PRINTED_HEX, data.ZETA_HEX, data.LOOPS_HEX,
                      (K(Fr(1, 2), 0), K(0, Fr(1, 2))),
                      FieldK, data.SIGNS_HEX, data.FREE_HEX)
    fd_control(res_hx, (0.5, math.sqrt(3.0)/2.0))

    print(f'\ndet M (square)     = {res_sq["det"]}')
    print(f'                   ~ {float(res_sq["det"]):.10e}')
    print(f'det M (hexagonal)  = {res_hx["det"]}')
    print(f'                   ~ {res_hx["det"].to_float():.10e}')
    return all(CHECKS), len(CHECKS)


if __name__ == '__main__':
    sys.exit(0 if main()[0] else 1)
