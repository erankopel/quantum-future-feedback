# quantum-future-feedback

Code and machine-verified certificate for the paper

> **Stable, informative, coherent, and non-entanglement-breaking fixed points
> of multi-time quantum processes**
> Eran Kopel (Tel Aviv University) — arXiv:quant-ph (submission in preparation)

The paper studies quantum processes in which information extracted from a
forward simulation is returned as an input to an earlier internal time of the
simulated dynamics, classifies the fixed points of the induced message channel
(stability, informativeness, feedability, coherence, non-entanglement
breaking), and certifies an explicit parameter square on which all four
nontrivial properties provably coexist.

## Layout

| Path | Contents |
|---|---|
| `src/` | Original research scripts: model construction, four-parameter search, Choi/NPT analysis, certificate thresholds and margins |
| `crosscheck/` | Independent re-implementation that re-derives every number quoted in the paper (floating point) |
| `figures/` | Figure-generation scripts |
| `verified/` | **Machine-verified certificate** (rigorous ball arithmetic) |

## The verified certificate

`verified/verified_certificate.py` executes the directed-rounding workflow of
Appendix B of the paper using rigorous ball arithmetic
([Arb](https://arblib.org/) via [python-flint](https://pypi.org/project/python-flint/),
256-bit precision). Every quantity is enclosed in a ball provably containing
the exact value, and every assertion uses certified ball comparisons (true
only if the relation holds for *all* values in the balls). It verifies, among
23 checks:

- unitarity of the round unitary and the CPTP structure of the induced channel;
- containment of all six reference quantities in the working enclosures of
  Table II of the paper (typical certified enclosure radius ~1e-75);
- the four Choi partial-transpose eigenvalue intervals and the NPT margin;
- the Proposition 3 NPT-neighborhood boundary;
- the four margins of Proposition 4 on the certified square of half-width
  0.0013 pi, including the drift-corrected information budget;
- safety of the four thresholds quoted in Eq. (22) of the paper;
- the closed-form family of Proposition 1 against the full unitary construction.

Run it with:

```bash
pip install python-flint
cd verified
python verified_certificate.py   # exit code 0 iff all checks pass
```

It regenerates `certificate.json` (machine-readable certificate with rigorous
enclosures, environment metadata, and the script's SHA-256). The committed
`certificate_report.txt` is the console output of the committed run.

## Second-stack audit

`verified/audit_mpmath.py` independently re-verifies every certified claim
with a different verified-arithmetic stack and different algorithms:

- directed-rounded interval arithmetic from `mpmath.iv` (pure Python; no
  Arb/FLINT anywhere in the script);
- self-contained spectral enclosures: approximate diagonalizers are converted
  entry-exactly to rational matrices, inverted exactly over the rationals,
  and Gershgorin discs of the exactly-similar interval matrix bound the
  spectrum, with cluster counting for multiplicities;
- the doubly degenerate zero eigenvalue of the pre-feedback state rho_FL is
  handled by cluster counting (count = 2 in a ~1e-16 interval), independently
  confirming the rank-2 Gram reduction used by the Arb stack;
- interval Gaussian elimination for the fixed point; exact rational
  arithmetic for the one boundary-case equality.

All 24 audit checks pass and agree with the Arb certificate:

```bash
cd verified
python audit_mpmath.py           # exit code 0 iff all checks pass
```

**Rigor caveat.** Both verifications are computer-assisted; their rigor is
conditional on the correctness of the underlying arithmetic libraries (Arb
and mpmath — distinct codebases and algorithms, so implementation bugs are
uncorrelated). A fully external audit on a third toolchain (e.g., INTLAB or
Julia's IntervalArithmetic.jl) by an independent party would strengthen the
result further.

## Reproducing the exploratory results

```bash
pip install -r requirements.txt
python src/partial_swap_extension.py     # 4-parameter search; writes nonEB_result.npz
python crosscheck/independent_crosscheck.py
```

Note: some scripts in `src/` are preserved verbatim from the research
workflow and expect a `/mnt/data` prefix for intermediate files; adjust paths
as needed. `src/safe_interval_certificate.py` implements the *uncorrected*
information budget (T <= 2s) discussed in Remark 1 of the paper and is kept
for the historical record; the paper's Proposition 4 uses the drift-corrected
budget verified in `verified/`.

## Citation

See `CITATION.cff`. License: MIT.
