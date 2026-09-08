"""The real quadratic field K = Q(sqrt 3) as an ordered field.

An element is a + b*sqrt(3) with a, b exact Fractions.  Every comparison is
decided by rational arithmetic alone; `to_float` is for logging and for the
random self-test below, and is never read by an accepted claim.

Sign rule.  For x = a + b sqrt3, not both zero:
  a >= 0 and b >= 0            -> +1
  a <= 0 and b <= 0            -> -1
  otherwise (a, b of strictly opposite signs, both nonzero)
                               -> sign(a) * sign(a^2 - 3 b^2).
The last line is the correct one: with a > 0 > b, x > 0 iff a > |b| sqrt3 iff
a^2 > 3 b^2; with a < 0 < b, x > 0 iff b sqrt3 > |a| iff 3 b^2 > a^2, which is
sign(a) * sign(a^2 - 3b^2) again since sign(a) = -1.  And a^2 = 3 b^2 with a, b
rational forces a = b = 0, so the middle case never returns 0.

Running this file directly runs the self-test at the bottom: random trials of
the ring laws, the sign rule and the order against floats with a guard band, and
then a block of exact identities no float can decide.
"""
import math
import random
from fractions import Fraction as Fr

class K:
    __slots__ = ('a', 'b')

    def __init__(self, a=0, b=0):
        self.a = a if type(a) is Fr else Fr(a)
        self.b = b if type(b) is Fr else Fr(b)

    # ---------- coercion ----------
    @staticmethod
    def _c(o):
        if isinstance(o, K):
            return o
        if isinstance(o, (int, Fr)):
            return K(o, 0)
        return NotImplemented

    # ---------- printing ----------
    def __repr__(self):
        return f'K({self.a})' if self.b == 0 else f'K({self.a} + {self.b}*sqrt3)'

    def __str__(self):
        return str(self.a) if self.b == 0 else f'({self.a} + {self.b}*sqrt3)'

    # ---------- ring operations ----------
    def __add__(s, o):
        o = K._c(o)
        return NotImplemented if o is NotImplemented else K(s.a + o.a, s.b + o.b)
    __radd__ = __add__

    def __sub__(s, o):
        o = K._c(o)
        return NotImplemented if o is NotImplemented else K(s.a - o.a, s.b - o.b)

    def __rsub__(s, o):
        o = K._c(o)
        return NotImplemented if o is NotImplemented else K(o.a - s.a, o.b - s.b)

    def __mul__(s, o):
        o = K._c(o)
        if o is NotImplemented:
            return NotImplemented
        return K(s.a*o.a + 3*s.b*o.b, s.a*o.b + s.b*o.a)
    __rmul__ = __mul__

    def inv(s):
        """1/(a + b sqrt3) = (a - b sqrt3)/(a^2 - 3 b^2); the norm vanishes only
        at zero because sqrt 3 is irrational."""
        d = s.a*s.a - 3*s.b*s.b
        if d == 0:
            raise ZeroDivisionError('K division by zero')
        return K(s.a/d, -s.b/d)

    def __truediv__(s, o):
        o = K._c(o)
        return NotImplemented if o is NotImplemented else s*o.inv()

    def __rtruediv__(s, o):
        o = K._c(o)
        return NotImplemented if o is NotImplemented else o*s.inv()

    def __neg__(s):
        return K(-s.a, -s.b)

    # ---------- order ----------
    def sign(s):
        a, b = s.a, s.b
        if a == 0 and b == 0:
            return 0
        if a >= 0 and b >= 0:
            return 1
        if a <= 0 and b <= 0:
            return -1
        n = a*a - 3*b*b
        if n == 0:
            raise ArithmeticError('a^2 == 3 b^2 with (a,b) != 0 is impossible over Q')
        sa = 1 if a > 0 else -1
        sn = 1 if n > 0 else -1
        return sa*sn

    def __eq__(s, o):
        o = K._c(o)
        return NotImplemented if o is NotImplemented else (s.a == o.a and s.b == o.b)

    def __ne__(s, o):
        r = s.__eq__(o)
        return r if r is NotImplemented else (not r)

    def __lt__(s, o):
        o = K._c(o)
        return NotImplemented if o is NotImplemented else (s - o).sign() < 0

    def __le__(s, o):
        o = K._c(o)
        return NotImplemented if o is NotImplemented else (s - o).sign() <= 0

    def __gt__(s, o):
        o = K._c(o)
        return NotImplemented if o is NotImplemented else (s - o).sign() > 0

    def __ge__(s, o):
        o = K._c(o)
        return NotImplemented if o is NotImplemented else (s - o).sign() >= 0

    def __abs__(s):
        return -s if s.sign() < 0 else s

    def __bool__(s):
        return not (s.a == 0 and s.b == 0)

    def __hash__(s):
        return hash((s.a, s.b))

    # ---------- floats: logging and sanity only ----------
    def to_float(s):
        return float(s.a) + float(s.b)*math.sqrt(3.0)

    def iszero(s):
        return s.a == 0 and s.b == 0


def k(x):
    """coerce int / Fraction / K to K."""
    return x if isinstance(x, K) else K(x, 0)


ONE = K(1, 0)
SQRT3 = K(0, 1)
HALF_SQRT3 = K(0, Fr(1, 2))


# ---------------------------------------------------------------- self-test
def _selftest(trials=20000, seed=20260814):
    rng = random.Random(seed)
    S3 = math.sqrt(3.0)

    def rnd():
        # small denominators plus occasional zeros, to hit every sign branch
        num = rng.randint(-30, 30)
        den = rng.randint(1, 12)
        return Fr(num, den)

    bad = 0
    for _ in range(trials):
        x = K(rnd(), rnd())
        y = K(rnd(), rnd())
        fx = float(x.a) + float(x.b)*S3
        fy = float(y.a) + float(y.b)*S3

        # sign against the float, with a guard band so ties never decide
        sx = x.sign()
        if abs(fx) > 1e-9 and sx != (1 if fx > 0 else -1):
            bad += 1
            print(f'        sign mismatch {x} float {fx}')
        if x.a == 0 and x.b == 0 and sx != 0:
            bad += 1

        # ring laws against floats
        for got, want in ((x + y, fx + fy), (x - y, fx - fy), (x*y, fx*fy)):
            if abs(got.to_float() - want) > 1e-9*max(1.0, abs(want)):
                bad += 1
                print(f'        arithmetic mismatch {got} vs {want}')
        if y:
            q = x/y
            if abs(q.to_float() - fx/fy) > 1e-9*max(1.0, abs(fx/fy)):
                bad += 1
                print(f'        division mismatch {q} vs {fx/fy}')
            if (q*y) != x:
                bad += 1
                print('        (x/y)*y != x')

        # order against the float
        if abs(fx - fy) > 1e-9 and (x < y) != (fx < fy):
            bad += 1
            print(f'        order mismatch {x} {y}')

    # exact identities floats cannot decide; an assertion here aborts the run
    assert SQRT3*SQRT3 == K(3, 0)
    assert (K(2, 1)*K(2, -1)) == K(1, 0)          # (2+sqrt3)(2-sqrt3) = 1
    assert K(2, -1).inv() == K(2, 1)
    assert abs(K(-1, 0)) == ONE
    assert SQRT3 > K(Fr(17, 10), 0)               # sqrt3 > 1.7
    assert SQRT3 < K(Fr(174, 100), 0)             # sqrt3 < 1.74
    assert HALF_SQRT3*HALF_SQRT3 == K(Fr(3, 4), 0)
    assert K(Fr(-97, 56), 1).sign() == -1         # sqrt3 < 97/56
    assert K(Fr(-265, 153), 1).sign() == 1        # sqrt3 > 265/153
    assert hash(K(Fr(2, 4), Fr(3, 6))) == hash(K(Fr(1, 2), Fr(1, 2)))

    print(f'  {"PASS" if not bad else "FAIL"}  '
          f'{"ring laws, sign rule and order against floats":<50}'
          f'{trials} trials, {bad} mismatches')
    print(f'  PASS  {"exact identities no float can decide":<50}'
          f'10 assertions')
    return bad == 0


def main():
    print('field   the ordered field Q(sqrt3), self-tested')
    return _selftest(), 2


if __name__ == '__main__':
    raise SystemExit(0 if main()[0] else 1)
