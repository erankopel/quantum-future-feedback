"""Independent verification of all quoted numbers in the Quantum Future Feedback draft.

Written from scratch against the paper's definitions (not copied from the project code):
registers ordered M (x) F (x) L, ancillas |00>_FL,
U_round = Ry^M(beta) . U_fb(phi) . U_weak(kappa) . U_W(theta).
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)
P0 = np.array([[1, 0], [0, 0]], complex)
P1 = np.array([[0, 0], [0, 1]], complex)
PAULI = [X, Y, Z]

def Ry(a):
    return np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]], complex)

def kron(*ops):
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out

# SWAP between qubit 0 (M) and qubit 2 (L) in an 8-dim space, basis index 4m+2f+l
SWAP_ML = np.zeros((8, 8), complex)
for m in range(2):
    for f in range(2):
        for l in range(2):
            SWAP_ML[4 * l + 2 * f + m, 4 * m + 2 * f + l] = 1

def U_W(th):
    return kron(P0, Ry(np.pi - 2 * th), I2) + kron(P1, Ry(2 * th), I2)

def U_weak(k):
    return expm(-1j * (k / 2) * kron(I2, Z, Y))

def U_fb(ph):
    return np.cos(ph) * np.eye(8) - 1j * np.sin(ph) * SWAP_ML

def U_round(th, k, ph, b):
    return kron(Ry(b), I2, I2) @ U_fb(ph) @ U_weak(k) @ U_W(th)

ANC = np.zeros((4, 4), complex); ANC[0, 0] = 1  # |00><00|_FL

def ptrace(rho, keep, dims=(2, 2, 2)):
    n = len(dims)
    arr = rho.reshape(dims + dims)
    for j in sorted((i for i in range(n) if i not in keep), reverse=True):
        arr = np.trace(arr, axis1=j, axis2=j + arr.ndim // 2)
    d = int(np.prod([dims[i] for i in keep]))
    return arr.reshape(d, d)

def channel(rho_M, par):
    U = U_round(*par)
    return ptrace(U @ np.kron(rho_M, ANC) @ U.conj().T, (0,))

def bloch(rho):
    return np.array([np.trace(rho @ P).real for P in PAULI])

def affine(par):
    c = bloch(channel(I2 / 2, par))
    A = np.zeros((3, 3))
    for j, P in enumerate(PAULI):
        A[:, j] = bloch(channel((I2 + P) / 2, par)) - c
    return A, c

def rho_of_v(v):
    return (I2 + v[0] * X + v[1] * Y + v[2] * Z) / 2

def entropy(rho):
    ev = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    ev = ev[ev > 1e-14]
    return float(-(ev * np.log2(ev)).sum())

def h2(x):
    return 0.0 if x <= 0 or x >= 1 else float(-x * np.log2(x) - (1 - x) * np.log2(1 - x))

def info_FL(par, rho_M):
    """I(F:L) immediately after U_weak U_W (pre-feedback)."""
    th, k, ph, b = par
    V = U_weak(k) @ U_W(th)
    q = V @ np.kron(rho_M, ANC) @ V.conj().T
    rFL = ptrace(q, (1, 2)); rF = ptrace(q, (1,)); rL = ptrace(q, (2,))
    return entropy(rF) + entropy(rL) - entropy(rFL)

def choi(par):
    J = np.zeros((4, 4), complex)
    for i in range(2):
        for j in range(2):
            E = np.zeros((2, 2), complex); E[i, j] = 1
            J += np.kron(E, channel(E, par)) / 2
    return J

def pt_ref(J):
    return J.reshape(2, 2, 2, 2).transpose(2, 1, 0, 3).reshape(4, 4)

def all_metrics(par):
    A, c = affine(par)
    v = np.linalg.solve(np.eye(3) - A, c)
    nA = np.linalg.norm(A, 2)
    R = np.linalg.norm(np.linalg.inv(np.eye(3) - A), 2)
    C = np.hypot(v[0], v[1])
    Iq = info_FL(par, rho_of_v(v))
    H = pt_ref(choi(par))
    lam = np.linalg.eigvalsh((H + H.conj().T) / 2)
    return dict(A=A, c=c, v=v, nA=nA, R=R, vn=np.linalg.norm(v), C=C, I=Iq, lam=lam,
                specA=np.abs(np.linalg.eigvals(A)).max())

PI = np.pi
par0 = (0.16345853 * PI, 0.43230980 * PI, 0.20061939 * PI, 0.23903823 * PI)

print("=" * 78)
print("[1] REFERENCE POINT vs quoted working enclosures")
m0 = all_metrics(par0)
checks = [
    ("||A0||_2", m0["nA"], 0.71330, 0.71333),
    ("||(I-A0)^-1||_2", m0["R"], 2.3872, 2.3874),
    ("||v0*||_2", m0["vn"], 0.5850, 0.5852),
    ("C0", m0["C"], 0.5710, 0.5712),
    ("I0(F:L)", m0["I"], 1.5685, 1.5689),
    ("lam_min(H0)", m0["lam"][0], -0.2054, -0.2051),
    ("lam_2(H0)", m0["lam"][1], 0.3123, 0.3127),
    ("lam_3(H0)", m0["lam"][2], 0.4199, 0.4203),
    ("lam_4(H0)", m0["lam"][3], 0.4725, 0.4729),
]
for name, val, lo, hi in checks:
    ok = lo <= val <= hi
    print(f"  {name:22s} = {val:+.8f}   quoted [{lo}, {hi}]   {'OK' if ok else '*** OUTSIDE ***'}")
print(f"  fixed point v0* = {m0['v']}")
print(f"  unitarity residual ||U'U-I|| = {np.linalg.norm(U_round(*par0).conj().T @ U_round(*par0) - np.eye(8)):.2e}")

print("=" * 78)
print("[2] PROP 4(old)/1(new): closed form of restricted family theta=0, phi=pi/2")
rng = np.random.default_rng(7)
worst = 0
for _ in range(60):
    k, b = rng.uniform(0.02, 0.48 * PI, 2)
    A, c = affine((0.0, k, PI / 2, b))
    Aan = np.array([[0, 0, -np.cos(b) * np.sin(k)], [0, 0, 0], [0, 0, np.sin(b) * np.sin(k)]])
    can = np.array([np.sin(b) * np.cos(k), 0, np.cos(b) * np.cos(k)])
    v = np.linalg.solve(np.eye(3) - A, c)
    lam = np.sin(b) * np.sin(k)
    zst = np.cos(b) * np.cos(k) / (1 - lam)
    xst = np.cos(k) * (np.sin(b) - np.sin(k)) / (1 - lam)
    worst = max(worst, np.abs(A - Aan).max(), np.abs(c - can).max(),
                abs(v[0] - xst), abs(v[1]), abs(v[2] - zst))
print(f"  max deviation analytic vs numeric over 60 random (kappa,beta): {worst:.2e}")

print("=" * 78)
print("[3] No-go lemma: Choi negativity == 0 at theta=0 (random kappa,phi,beta)")
mx = 0
for _ in range(300):
    par = (0.0, rng.uniform(0, PI / 2), rng.uniform(0, PI / 2), rng.uniform(0, PI / 2))
    lam = np.linalg.eigvalsh(pt_ref(choi(par)))
    mx = max(mx, -lam[0])
print(f"  max(-lam_min) over 300 samples at theta=0: {mx:.2e}  (should be ~0)")

print("=" * 78)
print("[4] PERTURBATION CONSTANTS: empirical check that bounds hold (and how loose)")
# ||dU|| <= 2 sin(|dth|/2) + 2 sin(|dph|/2) = 2s ; ||dA||<=12s ; ||dc||<=4√3 s ; ||dJ||_F<=4s
rng2 = np.random.default_rng(3)
viol = []
for _ in range(200):
    dth, dph = rng2.uniform(-0.02, 0.02, 2) * PI
    par = (par0[0] + dth, par0[1], par0[2] + dph, par0[3])
    s = np.sin(abs(dth) / 2) + np.sin(abs(dph) / 2)
    if s < 1e-12: continue
    A, c = affine(par)
    dU = np.linalg.norm(U_round(*par) - U_round(*par0), 2)
    dA = np.linalg.norm(A - m0["A"], 2)
    dc = np.linalg.norm(c - m0["c"])
    dJF = np.linalg.norm(pt_ref(choi(par)) - pt_ref(choi(par0)), "fro")
    viol.append((dU / (2 * s), dA / (12 * s), dc / (4 * np.sqrt(3) * s), dJF / (4 * s)))
viol = np.array(viol)
print(f"  max ratio dU/(2s)      = {viol[:,0].max():.4f}  (must be <=1)")
print(f"  max ratio dA/(12s)     = {viol[:,1].max():.4f}  (must be <=1)")
print(f"  max ratio dc/(4rt3 s)  = {viol[:,2].max():.4f}  (must be <=1)")
print(f"  max ratio dJ_F/(4s)    = {viol[:,3].max():.4f}  (must be <=1)")

print("=" * 78)
print("[5] THRESHOLDS as quoted (uncorrected, info uses T<=2s ignoring fixed-point drift)")
Aup, Rup, vup, Clo, Ilo, mulo = 0.71333, 2.3874, 0.5852, 0.5710, 1.5685, 0.2051
s_npt = mulo / 4
s_stab = (1 - Aup) / 12
f_coh = lambda s: Rup * (4 * np.sqrt(3) * s + 12 * s * vup) / (1 - Rup * 12 * s) - Clo
s_coh = brentq(f_coh, 1e-12, 0.03)
f_info_old = lambda s: 3 * h2(2 * s) + 2 * s * np.log2(3) - Ilo
s_info_old = brentq(f_info_old, 1e-12, 0.2)
print(f"  s_npt  = {s_npt:.8f}   quoted 0.05127500")
print(f"  s_stab = {s_stab:.8f}   quoted 0.02388916 (truncated)")
print(f"  s_coh  = {s_coh:.8f}   quoted 0.01149723")
print(f"  s_info = {s_info_old:.8f}   quoted 0.05013 [UNCORRECTED, diagnostic-input only]")
for name, s in [("npt", s_npt), ("info_old", s_info_old), ("stab", s_stab), ("coh", s_coh)]:
    print(f"    delta/pi ({name:8s}) = {2*np.arcsin(s/2)/PI:.8f}")

print("=" * 78)
print("[6] *** CORRECTED info certificate: include fixed-point drift in T(rho_FL) ***")
# T <= 2 sin(|dth|/2) + (1/2)||dv*||  <= 2s + dv(s)/2   with dv(s) the resolvent bound
def dv_bound(s):
    return Rup * (4 * np.sqrt(3) * s + 12 * s * vup) / (1 - Rup * 12 * s)
def T_bound(s):
    return 2 * s + dv_bound(s) / 2
f_info_new = lambda s: 3 * h2(T_bound(s)) + T_bound(s) * np.log2(3) - Ilo
s_info_new = brentq(f_info_new, 1e-12, 0.02)
print(f"  corrected s_info = {s_info_new:.8f}  (was {s_info_old:.8f})")
print(f"  corrected delta/pi threshold = {2*np.arcsin(s_info_new/2)/PI:.8f}")
print(f"  ==> binding constraint is now {'INFO' if s_info_new < s_coh else 'COHERENCE'}")

print("=" * 78)
print("[7] MARGINS on candidate squares (using unfavorable endpoints)")
def margins(dpi):
    d = dpi * PI
    s = 2 * np.sin(d / 2)
    da, dc = 12 * s, 4 * np.sqrt(3) * s
    stab = 1 - (Aup + da)
    dv = dv_bound(s)
    coh = Clo - dv
    info_old = Ilo - (3 * h2(2 * s) + 2 * s * np.log2(3))
    T = T_bound(s)
    info_new = Ilo - (3 * h2(T) + T * np.log2(3))
    npt = mulo - 4 * s
    return s, stab, coh, info_old, info_new, npt
for dpi in (0.0034, 0.0015, 0.0014, 0.0013, 0.0012, 0.0010):
    s, stab, coh, io, iN, npt = margins(dpi)
    print(f"  half-width {dpi:.4f}pi: s={s:.6f} stab={stab:+.5f} coh={coh:+.5f} "
          f"info_old={io:+.5f} info_CORR={iN:+.5f} npt={npt:+.5f}")

print("=" * 78)
print("[8] REALITY CHECK: direct numerics on corners of squares (true values, not bounds)")
for dpi in (0.0034, 0.0013):
    print(f"  square half-width {dpi}pi:")
    for sx in (-1, 1):
        for sy in (-1, 1):
            par = (par0[0] + sx * dpi * PI, par0[1], par0[2] + sy * dpi * PI, par0[3])
            m = all_metrics(par)
            print(f"    corner({sx:+d},{sy:+d}): 1-||A||={1-m['nA']:.5f} C={m['C']:.5f} "
                  f"I={m['I']:.5f} -lam_min={-m['lam'][0]:.5f}")

print("=" * 78)
print("[9] Classical anti-predictor closed form (Sec. 3)")
for q in (0.6, 0.8, 0.95):
    for r in (0.7, 1.0):
        a = (1 - q) + (2 * q - 1) * r
        lam = -(2 * q - 1) * (2 * r - 1)
        p = 0.5
        for _ in range(200):
            p = a + lam * p
        assert abs(p - 0.5) < 1e-12 or abs(lam) >= 1
        print(f"  q={q} r={r}: p*={a/(1-lam):.6f} lambda={lam:+.4f} I=1-h2(1-q)={1-h2(1-q):.6f}")
print("ALL DONE")
