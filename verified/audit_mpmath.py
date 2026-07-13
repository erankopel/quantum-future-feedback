#!/usr/bin/env python3
"""Second-stack audit of the machine-verified certificate.

Independent re-verification of every check in verified_certificate.py using a
different verified-arithmetic stack and different algorithms for the delicate
steps:

  arithmetic   mpmath.iv directed-rounded interval arithmetic (pure Python;
               no Arb / FLINT / python-flint anywhere in this script);
  spectra      self-contained enclosures: an approximate diagonalizer Q is
               taken from numpy, converted ENTRY-EXACTLY to rational numbers,
               inverted EXACTLY over the rationals, and the interval matrix
               G = Q^{-1} H Q (similar to H, hence with the same eigenvalues)
               is bounded by Gershgorin discs. A connected component formed
               by k discs contains exactly k eigenvalues, which handles the
               doubly degenerate zero eigenvalue of rho_FL directly - no
               rank-2 Gram reduction and no Fannes slack are needed here,
               making this an independent treatment of that subtlety;
  linear solve interval Gaussian elimination (3x3) for the fixed point;
  equalities   exact rational arithmetic (fractions.Fraction).

numpy is used ONLY to propose non-certified approximate diagonalizers; all
certified statements are established in interval or rational arithmetic.
Interval comparisons via endpoints are certified: x.b < y.a implies x < y for
all values in the intervals.

Usage: python audit_mpmath.py   (exit code 0 iff all checks pass)
Requires: mpmath, numpy.
"""

import json
import hashlib
import platform
import sys
import datetime
from fractions import Fraction

import mpmath
from mpmath import iv
import numpy as np

iv.prec = 192

checks = []


def check(cid, statement, result):
    checks.append({"id": cid, "statement": statement, "passed": bool(result)})
    print(f"  [{'PASS' if result else 'FAIL'}] {cid}: {statement}")
    return bool(result)


# ------------------------------------------------------- certified compares
def lt(x, y):
    """Certified x < y for intervals/numbers (True only if provable)."""
    x, y = iv.mpf(x), iv.mpf(y)
    return bool(x.b < y.a)


def within(x, lo, hi):
    """Certified: interval x strictly inside [lo, hi] (decimal strings)."""
    return lt(iv.mpf(lo), x) and lt(x, iv.mpf(hi))


IVZ = iv.mpf(0)
TINY = iv.mpf(10) ** -45          # provably-negligible scale at prec 192
LOG2 = iv.log(iv.mpf(2))


def log2iv(x):
    return iv.log(x) / LOG2


# ------------------------------------------------------- complex intervals
class C:
    __slots__ = ("re", "im")

    def __init__(self, re=0, im=0):
        self.re = re if isinstance(re, type(IVZ)) else iv.mpf(re)
        self.im = im if isinstance(im, type(IVZ)) else iv.mpf(im)

    def __add__(self, o):
        o = _c(o); return C(self.re + o.re, self.im + o.im)

    def __sub__(self, o):
        o = _c(o); return C(self.re - o.re, self.im - o.im)

    def __mul__(self, o):
        o = _c(o)
        return C(self.re * o.re - self.im * o.im,
                 self.re * o.im + self.im * o.re)

    def __neg__(self):
        return C(-self.re, -self.im)

    def conj(self):
        return C(self.re, -self.im)

    def abs2(self):
        return self.re * self.re + self.im * self.im


def _c(x):
    return x if isinstance(x, C) else C(x)


# ------------------------------------------------------- matrix operations
def zeros(n, m):
    return [[C() for _ in range(m)] for _ in range(n)]


def eye(n):
    return [[C(1) if i == j else C() for j in range(n)] for i in range(n)]


def mmul(A, B):
    n, k, m = len(A), len(B), len(B[0])
    out = zeros(n, m)
    for i in range(n):
        for j in range(m):
            s = C()
            for t in range(k):
                s = s + A[i][t] * B[t][j]
            out[i][j] = s
    return out


def madd(A, B, sa=None, sb=None):
    sa = C(1) if sa is None else _c(sa)
    sb = C(1) if sb is None else _c(sb)
    return [[sa * A[i][j] + sb * B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))]


def dagger(A):
    return [[A[j][i].conj() for j in range(len(A))] for i in range(len(A[0]))]


def kron(A, B):
    na, ma, nb, mb = len(A), len(A[0]), len(B), len(B[0])
    return [[A[i // nb][j // mb] * B[i % nb][j % mb]
             for j in range(ma * mb)] for i in range(na * nb)]


def trace(A):
    s = C()
    for i in range(len(A)):
        s = s + A[i][i]
    return s


def frob_sq(A):
    s = IVZ
    for row in A:
        for x in row:
            s = s + x.abs2()
    return s


def ptrace(A, keep, dims=(2, 2, 2)):
    n = len(dims)
    keep = tuple(keep)
    dkeep = 1
    for i in keep:
        dkeep *= dims[i]
    out = zeros(dkeep, dkeep)
    D = 1
    for d in dims:
        D *= d

    def unindex(flat):
        idx = []
        for d in reversed(dims):
            idx.append(flat % d)
            flat //= d
        return list(reversed(idx))

    for r in range(D):
        ir = unindex(r)
        for c_ in range(D):
            ic = unindex(c_)
            if any(ir[t] != ic[t] for t in range(n) if t not in keep):
                continue
            rr = cc = 0
            for t in keep:
                rr = rr * dims[t] + ir[t]
                cc = cc * dims[t] + ic[t]
            out[rr][cc] = out[rr][cc] + A[r][c_]
    return out


# ------------------------------------- exact rational linear algebra (Q^-1)
class RC:
    """Rational complex number (exact)."""
    __slots__ = ("re", "im")

    def __init__(self, re=0, im=0):
        self.re, self.im = Fraction(re), Fraction(im)

    def __add__(s, o): return RC(s.re + o.re, s.im + o.im)
    def __sub__(s, o): return RC(s.re - o.re, s.im - o.im)
    def __mul__(s, o): return RC(s.re * o.re - s.im * o.im,
                                 s.re * o.im + s.im * o.re)

    def __truediv__(s, o):
        d = o.re * o.re + o.im * o.im
        return RC((s.re * o.re + s.im * o.im) / d,
                  (s.im * o.re - s.re * o.im) / d)

    def absf(s):
        return float(s.re) ** 2 + float(s.im) ** 2


def rc_inv(M):
    """Exact inverse of a rational-complex matrix via Gaussian elimination."""
    n = len(M)
    A = [[RC(M[i][j].re, M[i][j].im) for j in range(n)] for i in range(n)]
    I = [[RC(1 if i == j else 0) for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: A[r][col].absf())
        A[col], A[piv] = A[piv], A[col]
        I[col], I[piv] = I[piv], I[col]
        p = A[col][col]
        for j in range(n):
            A[col][j] = A[col][j] / p
            I[col][j] = I[col][j] / p
        for r in range(n):
            if r != col:
                f = A[r][col]
                for j in range(n):
                    A[r][j] = A[r][j] - f * A[col][j]
                    I[r][j] = I[r][j] - f * I[col][j]
    return I


def frac_to_iv(f):
    """Exact conversion Fraction -> interval (numerator/denominator exact,
    division outward-rounded)."""
    return iv.mpf(f.numerator) / iv.mpf(f.denominator)


def rc_to_C(M):
    return [[C(frac_to_iv(x.re), frac_to_iv(x.im)) for x in row] for row in M]


# --------------------------------------------- Gershgorin spectral enclosure
def sqrt_upper(x):
    """Upper bound (degenerate interval) for sqrt of a nonneg-valued interval."""
    hi = abs(x).b
    return iv.sqrt(iv.mpf([hi, hi])).b


def eig_enclosures(H):
    """Rigorous eigenvalue enclosures with multiplicities for a Hermitian
    interval matrix H (containing a true Hermitian matrix H').

    Returns a list of (lo, hi, count) clusters covering all eigenvalues.
    Method: exact rational similarity by an approximate diagonalizer,
    followed by Gershgorin discs; a connected component of k discs contains
    exactly k eigenvalues (with multiplicity)."""
    n = len(H)
    Hf = np.array([[complex(mpmath.mpf(H[i][j].re.mid), mpmath.mpf(H[i][j].im.mid))
                    for j in range(n)] for i in range(n)])
    _, Qf = np.linalg.eigh((Hf + Hf.conj().T) / 2)
    Qr = [[RC(Fraction(float(Qf[i, j].real)), Fraction(float(Qf[i, j].imag)))
           for j in range(n)] for i in range(n)]
    Qinv = rc_inv(Qr)
    Qi, Qc = rc_to_C(Qinv), rc_to_C(Qr)
    G = mmul(Qi, mmul(H, Qc))
    discs = []
    for i in range(n):
        R = IVZ
        for j in range(n):
            if j != i:
                R = R + iv.mpf([sqrt_upper(G[i][j].abs2())] * 2)
        c = G[i][i].re  # true eigenvalues are real
        discs.append((c.a - R.b, c.b + R.b))
    discs.sort(key=lambda d: d[0])
    clusters = []
    for lo, hi in discs:
        if clusters and not (clusters[-1][1] < lo):  # overlap
            plo, phi, k = clusters[-1]
            clusters[-1] = (plo, max(phi, hi), k + 1)
        else:
            clusters.append((lo, hi, 1))
    return clusters


def cluster_to_iv(cl):
    lo, hi, _ = cl
    return iv.mpf([lo, hi])


# ----------------------------------------------- interval Gaussian solve 3x3
def iv_solve3(Areal, breal):
    """Interval Gaussian elimination for a 3x3 real interval system."""
    A = [[Areal[i][j] for j in range(3)] + [breal[i]] for i in range(3)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(float(mpmath.mpf(A[r][col].mid))))
        A[col], A[piv] = A[piv], A[col]
        for r in range(col + 1, 3):
            f = A[r][col] / A[col][col]
            for j in range(col, 4):
                A[r][j] = A[r][j] - f * A[col][j]
    x = [IVZ] * 3
    for i in (2, 1, 0):
        s = A[i][3]
        for j in range(i + 1, 3):
            s = s - A[i][j] * x[j]
        x[i] = s / A[i][i]
    return x


# ----------------------------------------------------------------- entropy
def entropy_from_clusters(clusters):
    """Enclosure of -sum lam log2 lam over rigorous eigenvalue clusters of a
    density matrix (true eigenvalues in [0,1])."""
    S = IVZ
    for lo, hi, k in clusters:
        hi_iv = iv.mpf([hi, hi])
        if IVZ.b < lo:  # strictly positive cluster
            lam = iv.mpf([lo, hi])
            S = S + k * (-(lam * log2iv(lam)))
        else:
            # cluster touches 0: true eigenvalues in [0, hi]; on [0,1/e]
            # the map -x log2 x is increasing, so each eigenvalue contributes
            # a value in [0, -hi log2 hi]
            assert lt(hi_iv, iv.mpf("0.3678")), "near-zero cluster too wide"
            top = (-(hi_iv * log2iv(hi_iv))).b
            S = S + k * iv.mpf([0, top])
    return S


def h2(x):
    return -(x * log2iv(x)) - ((1 - x) * log2iv(1 - x))


# ------------------------------------------------------------- model setup
PI = iv.pi
I2 = eye(2)
X = [[C(), C(1)], [C(1), C()]]
Y = [[C(), C(0, -1)], [C(0, 1), C()]]
Z = [[C(1), C()], [C(), C(-1)]]
P0 = [[C(1), C()], [C(), C()]]
P1 = [[C(), C()], [C(), C(1)]]
PAULI = [X, Y, Z]


def Ry(a):
    c, s = iv.cos(a / 2), iv.sin(a / 2)
    return [[C(c), C(-s)], [C(s), C(c)]]


SWAP_ML = zeros(8, 8)
for m_ in range(2):
    for f_ in range(2):
        for l_ in range(2):
            SWAP_ML[4 * l_ + 2 * f_ + m_][4 * m_ + 2 * f_ + l_] = C(1)


def U_W(th):
    return madd(kron(P0, kron(Ry(PI - 2 * th), I2)),
                kron(P1, kron(Ry(2 * th), I2)))


def U_weak(k):
    zy = kron(I2, kron(Z, Y))
    return madd(eye(8), zy, C(iv.cos(k / 2)), C(IVZ, -iv.sin(k / 2)))


def U_fb(ph):
    return madd(eye(8), SWAP_ML, C(iv.cos(ph)), C(IVZ, -iv.sin(ph)))


def U_round(th, k, ph, b):
    return mmul(kron(Ry(b), kron(I2, I2)),
                mmul(U_fb(ph), mmul(U_weak(k), U_W(th))))


P00 = zeros(4, 4)
P00[0][0] = C(1)


def channel_apply(U, Xop):
    return ptrace(mmul(U, mmul(kron(Xop, P00), dagger(U))), (0,))


def bloch_data(U):
    def rtr(out, P):
        t = trace(mmul(P, out))
        assert lt(abs(t.im), TINY)
        return t.re

    half = [[C(iv.mpf(1) / 2), C()], [C(), C(iv.mpf(1) / 2)]]
    out_half = channel_apply(U, half)
    c = [rtr(out_half, P) for P in PAULI]
    A = [[IVZ] * 3 for _ in range(3)]
    for j, Pj in enumerate(PAULI):
        out_j = channel_apply(U, Pj)
        for i, Pi in enumerate(PAULI):
            A[i][j] = rtr(out_j, Pi) / 2
    return A, c


# ================================================================== AUDIT ==
print("Second-stack audit (mpmath.iv intervals + rational Gershgorin, prec = %d bits)"
      % iv.prec)
print("=" * 74)

theta0 = iv.mpf("0.16345853") * PI
kappa0 = iv.mpf("0.43230980") * PI
phi0 = iv.mpf("0.20061939") * PI
beta0 = iv.mpf("0.23903823") * PI

U0 = U_round(theta0, kappa0, phi0, beta0)

# A1 unitarity
check("A1", "||U'U - I||_F^2 < 1e-90 (unitarity of U_round)",
      lt(frob_sq(madd(mmul(dagger(U0), U0), eye(8), C(1), C(-1))), TINY * TINY))

A, c = bloch_data(U0)
IA = [[(iv.mpf(1) if i == j else IVZ) - A[i][j] for j in range(3)] for i in range(3)]

AtA = [[C(sum((A[t][i] * A[t][j] for t in range(3)), IVZ)) for j in range(3)] for i in range(3)]
MtM = [[C(sum((IA[t][i] * IA[t][j] for t in range(3)), IVZ)) for j in range(3)] for i in range(3)]
nA = iv.sqrt(cluster_to_iv(eig_enclosures(AtA)[-1]))
Rnorm = 1 / iv.sqrt(cluster_to_iv(eig_enclosures(MtM)[0]))

v = iv_solve3(IA, c)
vnorm = iv.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
coh = iv.sqrt(v[0] * v[0] + v[1] * v[1])

# rho* = (I + v.sigma)/2
rho_star = [[C((1 + v[2]) / 2), C(v[0] / 2, -v[1] / 2)],
            [C(v[0] / 2, v[1] / 2), C((1 - v[2]) / 2)]]

V = mmul(U_weak(kappa0), U_W(theta0))
big = mmul(V, mmul(kron(rho_star, P00), dagger(V)))
rho_FL = ptrace(big, (1, 2))
rho_F = ptrace(big, (1,))
rho_L = ptrace(big, (2,))

clFL = eig_enclosures(rho_FL)
nzero = sum(k for lo, hi, k in clFL if not (IVZ.b < lo))
print(f"  [info] rho_FL Gershgorin clusters: "
      + ", ".join(f"[{mpmath.nstr(lo.mid, 8)},{mpmath.nstr(hi.mid, 8)}]x{k}"
                  for lo, hi, k in clFL))
check("A2", "rho_FL has a doubly degenerate near-zero cluster (count = 2), "
            "handled by cluster counting (independent of the Gram reduction)",
      nzero == 2)

Iq = entropy_from_clusters(eig_enclosures(rho_F)) \
    + entropy_from_clusters(eig_enclosures(rho_L)) \
    - entropy_from_clusters(clFL)

J = zeros(4, 4)
for i in range(2):
    for j in range(2):
        E = zeros(2, 2)
        E[i][j] = C(1)
        J = madd(J, kron(E, channel_apply(U0, E)), C(1), C(iv.mpf(1) / 2))

# CPTP checks
tp = lt(abs((trace(J) - C(1)).re) + abs((trace(J)).im), TINY + TINY)
herm = lt(frob_sq(madd(J, dagger(J), C(1), C(-1))), TINY * TINY)
tro = True
for a in range(2):
    for ap in range(2):
        s = C()
        for b in range(2):
            s = s + J[2 * a + b][2 * ap + b]
        target = iv.mpf(1) / 2 if a == ap else IVZ
        tro = tro and lt(abs(s.re - target) + abs(s.im), TINY + TINY)
Jpos = all(IVZ.b < lo for lo, hi, k in eig_enclosures(J))
check("A3", "CPTP: Tr J = 1, Hermitian, Tr_out J = I/2, J > 0 (certified)",
      tp and herm and tro and Jpos)

H = zeros(4, 4)
for a in range(2):
    for b in range(2):
        for ap in range(2):
            for bp in range(2):
                H[2 * a + b][2 * ap + bp] = J[2 * ap + b][2 * a + bp]
clH = eig_enclosures(H)
assert all(k == 1 for _, _, k in clH), "Choi PT spectrum should be simple"
Hev = [cluster_to_iv(cl) for cl in clH]
mu = -Hev[0]

# A4..A9: Table II containment
ok = True
ok &= check("A4", "||A0||_2 in [0.71330, 0.71333]", within(nA, "0.71330", "0.71333"))
ok &= check("A5", "||(I-A0)^-1||_2 in [2.3872, 2.3874]", within(Rnorm, "2.3872", "2.3874"))
ok &= check("A6", "||v0*||_2 in [0.5850, 0.5852]", within(vnorm, "0.5850", "0.5852"))
ok &= check("A7", "C0 in [0.5710, 0.5712]", within(coh, "0.5710", "0.5712"))
ok &= check("A8", "I0(F:L) in [1.5685, 1.5689] bits", within(Iq, "1.5685", "1.5689"))
ok &= check("A9", "mu_NPT in [0.2051, 0.2054]", within(mu, "0.2051", "0.2054"))

# A10..A13: Choi PT spectrum intervals
ok &= check("A10", "lam1(H0) in [-0.2054, -0.2051]", within(Hev[0], "-0.2054", "-0.2051"))
ok &= check("A11", "lam2(H0) in [0.3123, 0.3127]", within(Hev[1], "0.3123", "0.3127"))
ok &= check("A12", "lam3(H0) in [0.4199, 0.4203]", within(Hev[2], "0.4199", "0.4203"))
ok &= check("A13", "lam4(H0) in [0.4725, 0.4729]", within(Hev[3], "0.4725", "0.4729"))

# A14: Prop 3 boundary
delta3 = iv.mpf("0.01632312") * PI
check("A14", "8 sin(delta/2) < 0.2051 at delta/pi = 0.01632312 (Prop. 3)",
      lt(8 * iv.sin(delta3 / 2), iv.mpf("0.2051")))

# A15..A19: Prop 4 margins from unfavorable Table II endpoints
Aup, Rup, vup = iv.mpf("0.71333"), iv.mpf("2.3874"), iv.mpf("0.5852")
Clo, Ilo, mulo = iv.mpf("0.5710"), iv.mpf("1.5685"), iv.mpf("0.2051")
sq = 2 * iv.sin((iv.mpf("0.0013") * PI) / 2)
Ds = Rup * (4 * iv.sqrt(iv.mpf(3)) * sq + 12 * sq * vup) / (1 - 12 * sq * Rup)
tau = 2 * sq + Ds / 2
check("A15", "stability margin 1-(0.71333+12s) > 0.23766",
      lt(iv.mpf("0.23766"), 1 - (Aup + 12 * sq)))
check("A16", "coherence margin 0.5710 - D(s) > 0.41695",
      lt(iv.mpf("0.41695"), Clo - Ds))
check("A17", "tau(s) < 1/2 (Fannes-Audenaert regime)", lt(tau, iv.mpf(1) / 2))
check("A18", "information margin 1.5685 - [3 h2(tau) + tau log2 3] > 0.17283",
      lt(iv.mpf("0.17283"), Ilo - (3 * h2(tau) + tau * log2iv(iv.mpf(3)))))
check("A19", "NPT margin 0.2051 - 4s > 0.18876",
      lt(iv.mpf("0.18876"), mulo - 4 * sq))

# A20..A22: threshold safety (equality case in exact rational arithmetic)
check("A20", "0.71333 + 12*0.02388916 < 1",
      lt(Aup + 12 * iv.mpf("0.02388916"), iv.mpf(1)))
sc = iv.mpf("0.01149723")
check("A21", "D(0.01149723) < 0.5710",
      lt(Rup * (4 * iv.sqrt(iv.mpf(3)) * sc + 12 * sc * vup) / (1 - 12 * sc * Rup), Clo))
si = iv.mpf("0.00471753")
Dsi = Rup * (4 * iv.sqrt(iv.mpf(3)) * si + 12 * si * vup) / (1 - 12 * si * Rup)
ti = 2 * si + Dsi / 2
check("A22", "3 h2(tau) + tau log2 3 < 1.5685 at s = 0.00471753",
      lt(3 * h2(ti) + ti * log2iv(iv.mpf(3)), Ilo))
check("A23", "4*0.05127500 <= 0.2051 (exact rational arithmetic)",
      4 * Fraction(5127500, 10 ** 8) <= Fraction(2051, 10 ** 4))

# A24: closed-form spot check
ok24 = True
for kf, bf in [("0.30", "0.20"), ("0.43230980", "0.23903823")]:
    k, b = iv.mpf(kf) * PI, iv.mpf(bf) * PI
    Ac, cc = bloch_data(U_round(iv.mpf(0), k, PI / 2, b))
    Aan = [[IVZ, IVZ, -(iv.cos(b) * iv.sin(k))], [IVZ] * 3,
           [IVZ, IVZ, iv.sin(b) * iv.sin(k)]]
    can = [iv.sin(b) * iv.cos(k), IVZ, iv.cos(b) * iv.cos(k)]
    for i in range(3):
        ok24 = ok24 and lt(abs(cc[i] - can[i]), TINY)
        for j in range(3):
            ok24 = ok24 and lt(abs(Ac[i][j] - Aan[i][j]), TINY)
check("A24", "closed-form A(kappa,beta), c(kappa,beta) of Prop. 1 (2 samples)", ok24)

# ================================================================ REPORT ==
all_pass = all(ch["passed"] for ch in checks)
with open(__file__, "rb") as f:
    script_sha = hashlib.sha256(f.read()).hexdigest()
audit = {
    "audit_of": "verified_certificate.py (Arb/python-flint stack)",
    "stack": "mpmath.iv directed-rounded intervals + exact rational "
             "similarity + Gershgorin cluster counting",
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "environment": {
        "python": sys.version,
        "mpmath": mpmath.__version__,
        "numpy": np.__version__,
        "iv_precision_bits": iv.prec,
        "platform": platform.platform(),
    },
    "enclosures": {
        "norm_A0": str(nA), "resolvent_norm": str(Rnorm),
        "norm_v0": str(vnorm), "coherence_C0": str(coh),
        "info_I0_bits": str(Iq), "mu_NPT": str(mu),
        "choi_pt_eigs": [str(e) for e in Hev],
    },
    "checks": checks,
    "all_passed": all_pass,
    "script_sha256": script_sha,
}
with open("audit.json", "w") as f:
    json.dump(audit, f, indent=2)
print("=" * 74)
print(f"RESULT: {'ALL %d CHECKS PASSED' % len(checks) if all_pass else '*** FAILURES PRESENT ***'}")
print("audit.json written (sha256 of this script: %s)" % script_sha[:16])
sys.exit(0 if all_pass else 1)
