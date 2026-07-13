#!/usr/bin/env python3
"""Machine-verified interval certificate.

Paper: "Stable, informative, coherent, and non-entanglement-breaking fixed
points of multi-time quantum processes" (E. Kopel).

This script executes the directed-rounding workflow of Appendix B using
rigorous ball arithmetic (Arb, via python-flint). Every quantity is computed
as a ball (midpoint +/- radius) that provably contains the exact real number,
and every assertion below uses certified ball comparisons: with Arb, a
comparison such as `x < y` evaluates True only when it holds for ALL numbers
in the respective balls. Decimal input parameters are converted to balls of
radius ~2^-prec containing the exact decimal values.

Checks performed
  C1  unitarity of U_round (residual Frobenius norm < 1e-60)
  C2  CPTP structure of the induced channel (Choi trace, Hermiticity,
      trace preservation, positive semidefiniteness up to enclosure width)
  C3  containment of all six reference quantities in the working enclosures
      of Table II of the paper
  C4  containment of the four Choi partial-transpose eigenvalues in the
      quoted intervals, and the NPT margin mu >= 0.2051
  C5  Proposition 3 boundary: 8 sin(delta/2) < 0.2051 at delta/pi = 0.01632312
  C6  Proposition 4 margins on the square of half-width 0.0013*pi, computed
      from the unfavorable Table II endpoints exactly as in the paper's proof
  C7  safety of the four thresholds quoted in Eq. (22) of the paper
  C8  spot-check of the closed-form family of Proposition 1 against the
      full unitary construction at three sample points

Exit code 0 if and only if every check passes.

Requires: python-flint >= 0.9 (pip install python-flint)
"""

import json
import hashlib
import platform
import sys
import datetime

import flint
from flint import ctx, arb, acb, arb_mat

ctx.prec = 256  # bits of working precision for all ball arithmetic

TINY = arb(10) ** -60   # generic "provably negligible" scale
ZERO_EIG = arb(10) ** -50  # threshold for structurally-zero eigenvalues

LOG2 = arb(2).log()

checks = []


def check(cid, statement, result):
    checks.append({"id": cid, "statement": statement, "passed": bool(result)})
    print(f"  [{'PASS' if result else 'FAIL'}] {cid}: {statement}")
    return bool(result)


# ---------------------------------------------------------------- basic ops
def amat(rows):
    """Complex ball matrix as list-of-lists of acb."""
    return [[x if isinstance(x, acb) else acb(x) for x in r] for r in rows]


def zeros(n, m):
    return [[acb(0) for _ in range(m)] for _ in range(n)]


def eye(n):
    return [[acb(1) if i == j else acb(0) for j in range(n)] for i in range(n)]


def mmul(A, B):
    n, k, m = len(A), len(B), len(B[0])
    return [[sum((A[i][t] * B[t][j] for t in range(k)), acb(0)) for j in range(m)]
            for i in range(n)]


def madd(A, B, sa=acb(1), sb=acb(1)):
    return [[sa * A[i][j] + sb * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def dagger(A):
    return [[A[j][i].conjugate() for j in range(len(A))] for i in range(len(A[0]))]


def kron(A, B):
    na, ma, nb, mb = len(A), len(A[0]), len(B), len(B[0])
    return [[A[i // nb][j // mb] * B[i % nb][j % mb]
             for j in range(ma * mb)] for i in range(na * nb)]


def trace(A):
    return sum((A[i][i] for i in range(len(A))), acb(0))


def frob_sq(A):
    """Ball enclosure of the SQUARED Frobenius norm.

    We work with the square throughout: for a ball matrix whose entries
    contain zero, the sum of entry squares is a ball containing zero with a
    slightly negative lower endpoint, on which sqrt is undefined; comparing
    squared norms against squared tolerances avoids the issue. (Same reason
    products x*x are used instead of x**2: arb pow is not defined for balls
    containing zero or negative numbers.)"""
    s = arb(0)
    for row in A:
        for x in row:
            s += x.real * x.real + x.imag * x.imag
    return s


def eig2_hermitian(M):
    """Closed-form rigorous eigenvalues of a 2x2 Hermitian ball matrix."""
    a, d, b = M[0][0].real, M[1][1].real, M[0][1]
    assert abs(M[0][0].imag) < TINY and abs(M[1][1].imag) < TINY
    half_diff = (a - d) / 2
    disc = (half_diff * half_diff + b.real * b.real + b.imag * b.imag).sqrt()
    mid = (a + d) / 2
    return [mid - disc, mid + disc]


def ptrace(A, keep, dims=(2, 2, 2)):
    """Partial trace over the subsystems not in `keep` (A is square, dim 8)."""
    n = len(dims)
    keep = tuple(keep)
    dkeep = 1
    for i in keep:
        dkeep *= dims[i]
    out = zeros(dkeep, dkeep)

    def unindex(flat):
        idx = []
        for d in reversed(dims):
            idx.append(flat % d)
            flat //= d
        return list(reversed(idx))

    def index(idx):
        flat = 0
        for i, d in zip(idx, dims):
            flat = flat * d + i
        return flat

    D = 1
    for d in dims:
        D *= d
    for r in range(D):
        for c in range(D):
            ir, ic = unindex(r), unindex(c)
            ok = all(ir[t] == ic[t] for t in range(n) if t not in keep)
            if not ok:
                continue
            rr = 0
            cc = 0
            for t in keep:
                rr = rr * dims[t] + ir[t]
                cc = cc * dims[t] + ic[t]
            out[rr][cc] += A[r][c]
    return out


def eig_hermitian(A):
    """Rigorous eigenvalue enclosures of a Hermitian ball matrix.

    Symmetrizes (exact for a Hermitian-by-construction ball matrix), calls
    Arb's certified eigenvalue solver (acb_mat.eig with multiplicity
    handling), asserts the imaginary enclosures are negligible, and returns
    real balls sorted by midpoint.
    """
    from flint import acb_mat
    H = madd(A, dagger(A), acb(arb(1) / 2), acb(arb(1) / 2))
    M = acb_mat([[H[i][j] for j in range(len(H))] for i in range(len(H))])
    ev = M.eig(multiple=True)
    out = []
    for e in ev:
        assert abs(e.imag) < TINY, "eigenvalue enclosure has non-negligible imaginary part"
        out.append(e.real)
    return sorted(out, key=lambda x: float(x.mid()))


def inside(x, lo_str, hi_str):
    """Certified: ball x strictly inside [lo, hi] given as decimal strings."""
    return bool(arb(lo_str) < x) and bool(x < arb(hi_str))


def log2ball(x):
    return x.log() / LOG2


def neg_xlog2x(lam):
    """Enclosure of -x log2 x for the true eigenvalue contained in ball lam,
    knowing mathematically that the true eigenvalue lies in [0, 1]."""
    if lam > ZERO_EIG:
        return -(lam * log2ball(lam))
    if abs(lam) < ZERO_EIG:
        # true eigenvalue in [0, 1e-50]; -x log2 x is monotone there and
        # bounded by -1e-50*log2(1e-50) < 1.7e-49
        return arb(0).union(arb(10) ** -48)
    raise AssertionError("eigenvalue ball neither positive nor negligibly small")


def entropy(eigs):
    return sum((neg_xlog2x(l) for l in eigs), arb(0))


def h2(x):
    return -(x * log2ball(x)) - ((1 - x) * log2ball(1 - x))


# ---------------------------------------------------------- model unitaries
PI = arb.pi()

I2 = eye(2)
X = amat([[0, 1], [1, 0]])
Y = amat([[0, acb(0, -1)], [acb(0, 1), 0]])
Z = amat([[1, 0], [0, -1]])
P0 = amat([[1, 0], [0, 0]])
P1 = amat([[0, 0], [0, 1]])
PAULI = [X, Y, Z]


def Ry(a):
    c, s = (a / 2).cos(), (a / 2).sin()
    return amat([[acb(c), acb(-s)], [acb(s), acb(c)]])


SWAP_ML = zeros(8, 8)
for m_ in range(2):
    for f_ in range(2):
        for l_ in range(2):
            SWAP_ML[4 * l_ + 2 * f_ + m_][4 * m_ + 2 * f_ + l_] = acb(1)


def U_W(th):
    return madd(kron(P0, kron(Ry(PI - 2 * th), I2)),
                kron(P1, kron(Ry(2 * th), I2)))


def U_weak(k):
    zy = kron(I2, kron(Z, Y))
    c, s = (k / 2).cos(), (k / 2).sin()
    return madd(eye(8), zy, acb(c), acb(arb(0), -s))  # cos(k/2) I - i sin(k/2) Z_F Y_L


def U_fb(ph):
    return madd(eye(8), SWAP_ML, acb(ph.cos()), acb(arb(0), -ph.sin()))


def U_round(th, k, ph, b):
    R = kron(Ry(b), kron(I2, I2))
    return mmul(R, mmul(U_fb(ph), mmul(U_weak(k), U_W(th))))


P00 = zeros(4, 4)
P00[0][0] = acb(1)


def channel_apply(U, Xop):
    rho = mmul(U, mmul(kron(Xop, P00), dagger(U)))
    return ptrace(rho, (0,))


def bloch_data(U):
    """Real ball matrices A (3x3) and c (3) of the induced qubit channel."""
    def real_trace(Xop, P):
        t = trace(mmul(P, channel_apply(U, Xop)))
        assert abs(t.imag) < TINY
        return t.real

    half = amat([[acb(arb(1) / 2), 0], [0, acb(arb(1) / 2)]])
    c = [real_trace(half, P) for P in PAULI]
    A = [[arb(0)] * 3 for _ in range(3)]
    for j, Pj in enumerate(PAULI):
        col = [real_trace(Pj, Pi) / 2 for Pi in PAULI]
        for i in range(3):
            A[i][j] = col[i]
    return A, c


# ================================================================== MAIN ==
print("Machine-verified certificate (Arb ball arithmetic, prec = %d bits)" % ctx.prec)
print("=" * 74)

theta0 = arb("0.16345853") * PI
kappa0 = arb("0.43230980") * PI
phi0 = arb("0.20061939") * PI
beta0 = arb("0.23903823") * PI

U0 = U_round(theta0, kappa0, phi0, beta0)

# ---- C1: unitarity -------------------------------------------------------
res_sq = frob_sq(madd(mmul(dagger(U0), U0), eye(8), acb(1), acb(-1)))
check("C1", "||U'U - I||_F < 1e-60 (unitarity of U_round)", res_sq < TINY * TINY)

# ---- channel data --------------------------------------------------------
A, c = bloch_data(U0)
IA = [[(arb(1) if i == j else arb(0)) - A[i][j] for j in range(3)] for i in range(3)]

Aab = arb_mat([[A[i][j] for j in range(3)] for i in range(3)])
IAab = arb_mat([[IA[i][j] for j in range(3)] for i in range(3)])
cab = arb_mat([[c[0]], [c[1]], [c[2]]])

# spectral norms via eigenvalues of A^T A and (I-A)^T (I-A)
AtA = [[acb(sum((A[t][i] * A[t][j] for t in range(3)), arb(0))) for j in range(3)] for i in range(3)]
MtM = [[acb(sum((IA[t][i] * IA[t][j] for t in range(3)), arb(0))) for j in range(3)] for i in range(3)]
nA = eig_hermitian(AtA)[-1].sqrt()
Rnorm = 1 / eig_hermitian(MtM)[0].sqrt()

vstar = IAab.solve(cab)  # rigorous linear solve, certifies nonsingularity
vx, vy, vz = vstar[0, 0], vstar[1, 0], vstar[2, 0]
vnorm = (vx * vx + vy * vy + vz * vz).sqrt()
coh = (vx * vx + vy * vy).sqrt()

# ---- pre-feedback state and mutual information ---------------------------
rho_star = madd(madd(eye(2), X, acb(arb(1) / 2), acb(vx / 2)),
                madd(Y, Z, acb(vy / 2), acb(vz / 2)))
V = mmul(U_weak(kappa0), U_W(theta0))
big = mmul(V, mmul(kron(rho_star, P00), dagger(V)))
rho_FL = ptrace(big, (1, 2))
rho_F = ptrace(big, (1,))
rho_L = ptrace(big, (2,))

# S(rho_FL): rho_FL is structurally a two-term mixture
#   rho_FL = p0 |Psi_0><Psi_0| + p1 |Psi_1><Psi_1|,
#   |Psi_m> = U_weak^{FL} (|psi_m>_F |0>_L),  p_m = <m|rho_M*|m>,
# because U_W is controlled in the M basis and U_weak acts only on FL, so
# tracing out M kills all cross terms. We do NOT need to trust this argument:
# we build R2 explicitly, certify || rho_FL - R2 ||_F < 1e-55 in ball
# arithmetic, and add a Fannes-Audenaert slack for the entropy. The nonzero
# spectrum of the rank-<=2 mixture R2 is that of its 2x2 Gram matrix.
psi0 = [acb((PI / 2 - theta0).cos()), acb((PI / 2 - theta0).sin())]  # Ry(pi-2θ)|0>
psi1 = [acb(theta0.cos()), acb(theta0.sin())]                        # Ry(2θ)|0>
ck, sk = (kappa0 / 2).cos(), (kappa0 / 2).sin()
ZY = kron(Z, Y)
UwFL = madd(eye(4), ZY, acb(ck), acb(arb(0), -sk))
Psi = []
for psi in (psi0, psi1):
    vec4 = [psi[0], acb(0), psi[1], acb(0)]  # |psi>_F tensor |0>_L
    Psi.append([sum((UwFL[i][j] * vec4[j] for j in range(4)), acb(0)) for i in range(4)])
p0 = (1 + vz) / 2
p1 = (1 - vz) / 2
R2 = zeros(4, 4)
for pm, Ps in ((p0, Psi[0]), (p1, Psi[1])):
    for i in range(4):
        for j in range(4):
            R2[i][j] += acb(pm) * Ps[i] * Ps[j].conjugate()
diff_sq = frob_sq(madd(rho_FL, R2, acb(1), acb(-1)))
assert diff_sq < arb(10) ** -110, "rho_FL does not match its rank-2 form"
g = sum((Psi[0][i].conjugate() * Psi[1][i] for i in range(4)), acb(0))
disc = (((p0 - p1) / 2) * ((p0 - p1) / 2) + p0 * p1 * (g.real * g.real + g.imag * g.imag)).sqrt()
lamFL = [arb(1) / 2 - disc, arb(1) / 2 + disc]
# Fannes-Audenaert slack for trace distance <= ||diff||_1/2 <= ||diff||_F < 1e-55
S_FL = entropy(lamFL) + arb(0, 1e-53)
Iq = entropy(eig2_hermitian(rho_F)) + entropy(eig2_hermitian(rho_L)) - S_FL

# ---- Choi matrix and partial transpose -----------------------------------
J = zeros(4, 4)
for i in range(2):
    for j in range(2):
        E = zeros(2, 2)
        E[i][j] = acb(1)
        J = madd(J, kron(E, channel_apply(U0, E)), acb(1), acb(arb(1) / 2))

# C2: CPTP checks
tp_ok = abs(trace(J) - 1) < TINY
herm = frob_sq(madd(J, dagger(J), acb(1), acb(-1))) < TINY * TINY
# trace preservation: Tr_out J = I/2  (output = second factor)
Jr = [[[[J[2 * a + b][2 * ap + bp] for bp in range(2)] for ap in range(2)]
       for b in range(2)] for a in range(2)]
tro_ok = True
for a in range(2):
    for ap in range(2):
        s = sum((Jr[a][b][ap][b] for b in range(2)), acb(0))
        target = arb(1) / 2 if a == ap else arb(0)
        tro_ok = tro_ok and bool(abs(s - target) < TINY)
Jev = eig_hermitian(J)
psd_ok = all(bool(l > -TINY) for l in Jev)
check("C2", "Choi state: Tr J = 1, Hermitian, Tr_out J = I/2, PSD (up to 1e-60)",
      tp_ok and herm and tro_ok and psd_ok)

# partial transpose on the reference (first) qubit: J[(a,b),(a',b')] -> J[(a',b),(a,b')]
H = zeros(4, 4)
for a in range(2):
    for b in range(2):
        for ap in range(2):
            for bp in range(2):
                H[2 * a + b][2 * ap + bp] = J[2 * ap + b][2 * a + bp]
Hev = eig_hermitian(H)
mu = -Hev[0]

# ---- C3: Table II containment --------------------------------------------
t3 = True
t3 &= check("C3a", "||A0||_2 in [0.71330, 0.71333]", inside(nA, "0.71330", "0.71333"))
t3 &= check("C3b", "||(I-A0)^-1||_2 in [2.3872, 2.3874]", inside(Rnorm, "2.3872", "2.3874"))
t3 &= check("C3c", "||v0*||_2 in [0.5850, 0.5852]", inside(vnorm, "0.5850", "0.5852"))
t3 &= check("C3d", "C0 in [0.5710, 0.5712]", inside(coh, "0.5710", "0.5712"))
t3 &= check("C3e", "I0(F:L) in [1.5685, 1.5689] bits", inside(Iq, "1.5685", "1.5689"))
t3 &= check("C3f", "mu_NPT in [0.2051, 0.2054]", inside(mu, "0.2051", "0.2054"))

# ---- C4: Choi PT spectrum -------------------------------------------------
t4 = True
t4 &= check("C4a", "lam1(H0) in [-0.2054, -0.2051]", inside(Hev[0], "-0.2054", "-0.2051"))
t4 &= check("C4b", "lam2(H0) in [0.3123, 0.3127]", inside(Hev[1], "0.3123", "0.3127"))
t4 &= check("C4c", "lam3(H0) in [0.4199, 0.4203]", inside(Hev[2], "0.4199", "0.4203"))
t4 &= check("C4d", "lam4(H0) in [0.4725, 0.4729]", inside(Hev[3], "0.4725", "0.4729"))

# ---- C5: Proposition 3 boundary -------------------------------------------
delta3 = arb("0.01632312") * PI
check("C5", "8 sin(delta/2) < 0.2051 at delta/pi = 0.01632312 (Prop. 3)",
      bool(8 * (delta3 / 2).sin() < arb("0.2051")))

# ---- C6: Proposition 4 margins --------------------------------------------
# computed from the paper's unfavorable Table II endpoints, as in the proof
Aup, Rup, vup = arb("0.71333"), arb("2.3874"), arb("0.5852")
Clo, Ilo, mulo = arb("0.5710"), arb("1.5685"), arb("0.2051")
sq = 2 * ((arb("0.0013") * PI) / 2).sin()
Ds = Rup * (4 * arb(3).sqrt() * sq + 12 * sq * vup) / (1 - 12 * sq * Rup)
tau = 2 * sq + Ds / 2
t6 = True
t6 &= check("C6a", "stability margin 1-(0.71333+12s) > 0.23766",
            bool(1 - (Aup + 12 * sq) > arb("0.23766")))
t6 &= check("C6b", "coherence margin 0.5710 - D(s) > 0.41695",
            bool(Clo - Ds > arb("0.41695")))
t6 &= check("C6c", "tau(s) < 1/2 (validity of Fannes-Audenaert regime)",
            bool(tau < arb(1) / 2))
t6 &= check("C6d", "information margin 1.5685 - [3 h2(tau) + tau log2 3] > 0.17283",
            bool(Ilo - (3 * h2(tau) + tau * log2ball(arb(3))) > arb("0.17283")))
t6 &= check("C6e", "NPT margin 0.2051 - 4s > 0.18876",
            bool(mulo - 4 * sq > arb("0.18876")))

# ---- C7: threshold safety (Eq. 22) ----------------------------------------
t7 = True
t7 &= check("C7a", "0.71333 + 12*0.02388916 < 1",
            bool(Aup + 12 * arb("0.02388916") < 1))
sc = arb("0.01149723")
Dc = Rup * (4 * arb(3).sqrt() * sc + 12 * sc * vup) / (1 - 12 * sc * Rup)
t7 &= check("C7b", "D(0.01149723) < 0.5710", bool(Dc < Clo))
si = arb("0.00471753")
Di = Rup * (4 * arb(3).sqrt() * si + 12 * si * vup) / (1 - 12 * si * Rup)
ti = 2 * si + Di / 2
t7 &= check("C7c", "3 h2(tau) + tau log2 3 < 1.5685 at s = 0.00471753",
            bool(3 * h2(ti) + ti * log2ball(arb(3)) < Ilo))
# 4*0.05127500 = 0.2051 exactly: an equality cannot be certified by ball
# comparison (the balls overlap), so it is checked in exact rational arithmetic
from flint import fmpq
t7 &= check("C7d", "4*0.05127500 <= 0.2051 (exact rational arithmetic)",
            4 * fmpq(5127500, 10 ** 8) <= fmpq(2051, 10 ** 4))

# ---- C8: closed-form family spot-check ------------------------------------
ok8 = True
for kf, bf in [("0.30", "0.20"), ("0.10", "0.40"), ("0.43230980", "0.23903823")]:
    k, b = arb(kf) * PI, arb(bf) * PI
    Uc = U_round(arb(0), k, PI / 2, b)
    Ac, cc = bloch_data(Uc)
    Aan = [[arb(0), arb(0), -(b.cos() * k.sin())],
           [arb(0), arb(0), arb(0)],
           [arb(0), arb(0), b.sin() * k.sin()]]
    can = [b.sin() * k.cos(), arb(0), b.cos() * k.cos()]
    for i in range(3):
        ok8 = ok8 and bool(abs(cc[i] - can[i]) < TINY)
        for j in range(3):
            ok8 = ok8 and bool(abs(Ac[i][j] - Aan[i][j]) < TINY)
check("C8", "closed-form A(kappa,beta), c(kappa,beta) of Prop. 1 (3 samples)", ok8)

# ================================================================ REPORT ==
enclosures = {
    "norm_A0": str(nA), "resolvent_norm": str(Rnorm), "norm_v0": str(vnorm),
    "coherence_C0": str(coh), "info_I0_bits": str(Iq),
    "choi_pt_eigs": [str(l) for l in Hev], "mu_NPT": str(mu),
    "s_square_0.0013pi": str(sq), "D(s)": str(Ds), "tau(s)": str(tau),
}
all_pass = all(ch["passed"] for ch in checks)
with open(__file__, "rb") as f:
    script_sha = hashlib.sha256(f.read()).hexdigest()
cert = {
    "paper": "Stable, informative, coherent, and non-entanglement-breaking "
             "fixed points of multi-time quantum processes",
    "author": "Eran Kopel (Tel Aviv University)",
    "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "environment": {
        "python": sys.version,
        "python-flint": flint.__version__,
        "arb_precision_bits": ctx.prec,
        "platform": platform.platform(),
    },
    "parameters_over_pi": {
        "theta0": "0.16345853", "kappa0": "0.43230980",
        "phi0": "0.20061939", "beta0": "0.23903823",
    },
    "rigorous_enclosures": enclosures,
    "checks": checks,
    "all_passed": all_pass,
    "script_sha256": script_sha,
    "note": "Ball comparisons are certified: True only if the relation holds "
            "for all values in the enclosing balls. Rigor is conditional on "
            "the correctness of Arb/python-flint.",
}
with open("certificate.json", "w") as f:
    json.dump(cert, f, indent=2)
print("=" * 74)
print("Rigorous enclosures (ball notation [mid +/- rad]):")
for k_, v_ in enclosures.items():
    if isinstance(v_, list):
        for i, s_ in enumerate(v_):
            print(f"  {k_}[{i}] = {s_}")
    else:
        print(f"  {k_} = {v_}")
print("=" * 74)
print(f"RESULT: {'ALL %d CHECKS PASSED' % len(checks) if all_pass else '*** FAILURES PRESENT ***'}")
print("certificate.json written (sha256 of this script: %s)" % script_sha[:16])
sys.exit(0 if all_pass else 1)
