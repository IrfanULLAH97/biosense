"""
Numerical solution of the conductive matrix enzyme electrode models.

Implements the finite difference schemes of the thesis section
"Numerical approximation of the mathematical models":

    Model A  original          de2/dt = kcat*Cs*(eS-e2)/(Km+Cs) - ka*e2
                               dCs/dt = De*d2Cs/dx2 - R(Cs,e2)
    Model B  auxiliary         quasi steady state only (A1), MM kinetics retained
    Model C  reduced (Wu)      QSS + excess mediator + first order  (A1,A2,A3)

Geometry:  x = 0  conductive base (zero flux),  x = L  membrane/bulk interface.
Units:     SI throughout.  Concentrations in mol/m^3  (1 mM = 1 mol/m^3).

Run:  python biosensor.py          (writes figures and tables to ./results)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace

import numpy as np

# numpy >= 2.0 renamed trapz to trapezoid; support both
_trapz = getattr(np, "trapezoid", None) or np.trapz

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Params:
    """Model parameters.  Defaults are the reference set of Table 1."""

    L: float = 63.4e-6          # membrane thickness, m
    Ae: float = 7.065e-6        # electrode surface area, m^2
    De: float = 2.066e-11       # effective diffusivity, m^2/s  (= alpha * L^2)
    Km: float = 33.0            # Michaelis constant, mol/m^3
    kcat: float = 10.0          # catalytic rate constant, 1/s   (assumed, see text)
    ka: float = 1.0e4           # k * a, mediator reduction rate, 1/s
    q: float = 7.525e4          # charge yield adopted from Wu et al., C/mol
    Cs0: float = 4.0            # bulk substrate concentration, mol/m^3
    s: float = 9.25             # enzyme loading, Unit/mg graphite

    K1_PRIME = 5.24e-4          # Wu fitted k1', mg/(Unit*s)

    @property
    def k1(self) -> float:
        """First-order rate constant, k1 = s * k1' = kcat*eS/Km."""
        return self.K1_PRIME * self.s

    @property
    def eS(self) -> float:
        """Total active enzyme concentration in the membrane, eS = k1*Km/kcat."""
        return self.k1 * self.Km / self.kcat

    @property
    def alpha(self) -> float:
        """Diffusion rate constant De/L^2, 1/s."""
        return self.De / self.L**2

    # dimensionless groups
    @property
    def beta(self) -> float:
        return self.kcat * self.L**2 / self.De

    @property
    def gamma(self) -> float:
        return self.ka * self.L**2 / self.De

    @property
    def delta(self) -> float:
        return self.eS / self.Km

    @property
    def bd(self) -> float:
        """beta*delta = k1*L^2/De = k1/alpha, the diffusion module."""
        return self.k1 / self.alpha

    @property
    def psiS_inf(self) -> float:
        return self.Cs0 / self.Km


# --------------------------------------------------------------------------
# Analytical solutions of the reduced model
# --------------------------------------------------------------------------


def analytic_profile(p: Params, u: np.ndarray) -> np.ndarray:
    """Steady dimensionless substrate profile, psiS*(u), u = x/L, u = 0 at base."""
    r = math.sqrt(p.bd)
    return p.psiS_inf * np.cosh(r * u) / math.cosh(r)


def analytic_steady_current(p: Params) -> float:
    """Steady current of the reduced model, A  (Wu Eq. 18 times Cs0)."""
    return (p.q * p.Ae * math.sqrt(p.k1 * p.De)
            * math.tanh(math.sqrt(p.k1 / p.De) * p.L) * p.Cs0)


def analytic_psiI(p: Params, theta: np.ndarray, rtol: float = 1e-12,
                  n_cap: int = 2_000_000) -> tuple[np.ndarray, int]:
    """
    Transient dimensionless current of the reduced model (Wu Eq. 14).

    The series is truncated when the next term falls below `rtol` times the
    steady term.  Returns (psiI, n_max actually used).
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    r = math.sqrt(p.bd)
    steady = math.tanh(r) / r

    # n_max is set by the smallest strictly positive theta (slowest decay of terms)
    th_min = theta[theta > 0].min() if np.any(theta > 0) else 0.0
    if th_min > 0:
        # need exp(-((2n+1)pi/2)^2 * th) < rtol*steady  =>  solve for n
        need = math.log(1.0 / (rtol * steady))
        n_max = int(math.ceil((math.sqrt(need / th_min) * 2 / math.pi - 1) / 2)) + 5
        n_max = max(10, min(n_max, n_cap))
    else:
        n_max = 10_000

    n = np.arange(n_max + 1)
    lam = p.bd + ((2 * n + 1) * math.pi / 2) ** 2          # (n,)
    den = (2 * n + 1) ** 2 * math.pi**2 + 4 * p.bd         # (n,)

    # chunk over theta to bound memory
    out = np.empty_like(theta)
    chunk = max(1, int(4e6 // (n_max + 1)))
    for a in range(0, theta.size, chunk):
        th = theta[a:a + chunk][:, None]
        out[a:a + chunk] = steady - np.sum(8.0 * np.exp(-lam * th) / den, axis=1)
    return p.psiS_inf * (p.beta / p.gamma) * out, n_max


# --------------------------------------------------------------------------
# Stability
# --------------------------------------------------------------------------


def stability_dt(p: Params, N: int, model: str, e2_update: str) -> dict:
    """Both stability bounds and the admissible time step."""
    dx = p.L / N
    dt_diff = 2.0 / (4.0 * p.De / dx**2 + p.k1)
    dt_reac = 2.0 / (p.kcat + p.ka)
    if model == "A" and e2_update == "euler":
        dt = min(dt_diff, dt_reac)
    else:
        dt = dt_diff
    return {"dx": dx, "dt_diffusion": dt_diff, "dt_reaction": dt_reac, "dt_max": dt}


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------


def simulate(p: Params, model: str = "A", bc: str = "ghost",
             e2_update: str = "exp", N: int = 100, dt: float | None = None,
             eps: float = 1e-4, safety: float = 0.9, t_max: float = 5000.0,
             store_every: int = 1, check_bounds: bool = True) -> dict:
    """
    Solve one model to the stopping time.

    model      'A' original | 'B' auxiliary (QSS only) | 'C' reduced
    bc         'ghost' second-order zero flux | 'copy' first-order Cs0 = Cs1
    e2_update  'exp' exact exponential update | 'euler' forward Euler (model A only)
    eps        stopping tolerance in  (t/I)|dI/dt| < eps
    """
    if model not in ("A", "B", "C"):
        raise ValueError("model must be 'A', 'B' or 'C'")
    if bc not in ("ghost", "copy"):
        raise ValueError("bc must be 'ghost' or 'copy'")

    st = stability_dt(p, N, model, e2_update)
    if dt is None:
        dt = safety * st["dt_max"]
    elif dt > st["dt_max"]:
        raise ValueError(f"dt={dt:.3e} exceeds the stability bound "
                         f"{st['dt_max']:.3e} s")
    dx = st["dx"]

    Cs = np.zeros(N + 1)
    Cs[N] = p.Cs0
    e2 = np.zeros(N + 1)

    diff = p.De / dx**2
    n_max_steps = int(t_max / dt) + 1
    t_hist = np.empty(n_max_steps // store_every + 2)
    I_hist = np.empty_like(t_hist)

    def rate_and_e2(Cs_):
        """Reaction rate R and reduced-enzyme concentration for models B and C."""
        if model == "B":
            e2_ = p.kcat * p.eS * Cs_ / (p.ka * (p.Km + Cs_) + p.kcat * Cs_)
            return p.ka * e2_, e2_
        e2_ = (p.k1 / p.ka) * Cs_               # model C
        return p.k1 * Cs_, e2_

    if model in ("B", "C"):
        _, e2 = rate_and_e2(Cs)

    I_prev = p.q * p.ka * p.Ae * _trapz(e2, dx=dx)
    t_hist[0], I_hist[0] = 0.0, I_prev
    k = 1
    t = 0.0
    stopped = False
    a2_violated = False          # model C only: e2 > eS means (A2) has failed
    lap = np.empty(N + 1)

    for step in range(1, n_max_steps + 1):
        if model == "A":
            mm = p.kcat * Cs / (p.Km + Cs)              # kcat*Cs/(Km+Cs)
            if e2_update == "exp":
                A = mm * p.eS
                B = mm + p.ka
                e2_new = A / B + (e2 - A / B) * np.exp(-B * dt)
            else:
                e2_new = e2 + dt * (mm * (p.eS - e2) - p.ka * e2)
            R = mm * (p.eS - e2)                        # uses old e2, time level j
        else:
            R, _ = rate_and_e2(Cs)
            e2_new = None

        lap[1:N] = Cs[2:] - 2.0 * Cs[1:N] + Cs[:N - 1]
        Cs_new = Cs.copy()
        Cs_new[1:N] = Cs[1:N] + dt * (diff * lap[1:N] - R[1:N])
        if bc == "ghost":
            Cs_new[0] = Cs[0] + dt * (2.0 * diff * (Cs[1] - Cs[0]) - R[0])
        else:
            Cs_new[0] = Cs_new[1]
        Cs_new[N] = p.Cs0

        Cs = Cs_new
        if model == "A":
            e2 = e2_new
        else:
            _, e2 = rate_and_e2(Cs)

        t += dt
        I = p.q * p.ka * p.Ae * _trapz(e2, dx=dx)

        if check_bounds and step % 500 == 0:
            bad = ((not np.all(np.isfinite(Cs))) or Cs.min() < -1e-12 * p.Cs0
                   or e2.min() < -1e-12 * p.eS)
            # e2 <= eS is a physical bound of models A and B; a violation there
            # indicates instability.  In model C, e2 = (k1/ka)Cs is unbounded,
            # so a violation means the approximation (A2) has broken down.
            if model in ("A", "B") and e2.max() > p.eS * (1.0 + 1e-9):
                bad = True
            if bad:
                raise RuntimeError(
                    f"physical bounds violated at t={t:.4g}s: "
                    f"min(Cs)={Cs.min():.3e}, max(e2)/eS={e2.max()/p.eS:.6f}")
            if e2.max() > p.eS:
                a2_violated = True

        if step % store_every == 0:
            t_hist[k], I_hist[k] = t, I
            k += 1

        if I > 0.0 and step > 1:
            if (t / I) * abs(I - I_prev) / dt < eps:
                stopped = True
                break
        I_prev = I

    t_hist, I_hist = t_hist[:k], I_hist[:k]
    if not stopped:
        raise RuntimeError(f"stopping criterion not reached within t_max={t_max}s")

    Istar = I
    # response times, linearly interpolated between stored points so that the
    # resolution is not limited by dt
    Talpha = {}
    for a in (0.5, 0.9, 0.95):
        target = a * Istar
        idx = int(np.searchsorted(I_hist, target))
        if idx <= 0:
            Talpha[a] = t_hist[0]
        elif idx >= len(t_hist):
            Talpha[a] = t_hist[-1]
        else:
            i0, i1 = idx - 1, idx
            dI = I_hist[i1] - I_hist[i0]
            w = 0.0 if dI == 0.0 else (target - I_hist[i0]) / dI
            Talpha[a] = t_hist[i0] + w * (t_hist[i1] - t_hist[i0])

    return {
        "p": p, "model": model, "bc": bc, "e2_update": e2_update,
        "N": N, "dx": dx, "dt": dt, "steps": step, "stability": st,
        "x": np.linspace(0.0, p.L, N + 1), "Cs": Cs, "e2": e2,
        "t": t_hist, "I": I_hist, "Tss": t, "Istar": Istar,
        "T05": Talpha[0.5], "T09": Talpha[0.9], "T095": Talpha[0.95],
        "psiS": Cs / p.Km, "u": np.linspace(0.0, 1.0, N + 1),
        "theta": t_hist * p.alpha,
        "psiI": I_hist / (p.q * p.ka * p.eS * p.Ae * p.L),
        "e2max_over_eS": float(e2.max() / p.eS),
        "A2_violated": a2_violated,
    }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def compare_transient(r1: dict, r2: dict, n: int = 4000) -> dict:
    """
    Compare two transient currents on a common time grid.

    The max-norm difference is dominated by t = 0, where the trapezoidal weight
    of the outer boundary node makes the difference proportional to dx; it is
    therefore reported only as a diagnostic.  The time-averaged (L1) measure is
    mesh convergent and is the one used for the error study.
    """
    T = min(r1["t"][-1], r2["t"][-1])
    tt = np.linspace(0.0, T, n)
    I1 = np.interp(tt, r1["t"], r1["I"])
    I2 = np.interp(tt, r2["t"], r2["I"])
    Istar = r1["Istar"]
    return {
        "L1": float(_trapz(np.abs(I2 - I1), tt) / (Istar * T)),
        "max": float(np.max(np.abs(I2 - I1)) / Istar),
        "max_excl_t0": float(np.max(np.abs(I2 - I1)[tt > T * 1e-3]) / Istar),
    }


def v1_analytical(p: Params, N: int = 100, eps: float = 1e-4) -> dict:
    """V1: reduced model against the analytical solutions."""
    r = simulate(p, model="C", bc="ghost", N=N, eps=eps)
    exact = analytic_profile(p, r["u"])
    prof_err = np.max(np.abs(r["psiS"] - exact)) / p.psiS_inf
    I_an = analytic_steady_current(p)
    cur_err = abs(r["Istar"] - I_an) / I_an
    psiI_an, n_max = analytic_psiI(p, r["theta"])
    m = r["theta"] > 0
    tr_err = np.max(np.abs(r["psiI"][m] - psiI_an[m])) / psiI_an[-1]
    return {"run": r, "exact_profile": exact, "psiI_an": psiI_an, "n_max": n_max,
            "profile_err": prof_err, "current_err": cur_err, "transient_err": tr_err,
            "I_analytic": I_an}


def v2_asymptotic(p: Params, N: int = 100, eps: float = 1e-4) -> dict:
    """V2: original vs reduced deep inside the regime where (A1)-(A3) hold."""
    pa = replace(p, Cs0=1e-3 * p.Km, ka=p.kcat / 1e-3)
    A = simulate(pa, model="A", bc="ghost", N=N, eps=eps)
    C = simulate(pa, model="C", bc="ghost", N=N, eps=eps)
    return {"p": pa, "A": A, "C": C,
            "rel_diff": abs(A["Istar"] - C["Istar"]) / A["Istar"],
            "T095_diff": abs(A["T095"] - C["T095"]) / A["T095"]}


def v3_refinement(p: Params, eps: float = 1e-4) -> dict:
    """V3: time-step and spatial refinement of the reduced model."""
    time_rows = []
    dt0 = 0.9 * stability_dt(p, 200, "C", "exp")["dt_max"]
    prev = None
    for f in (1, 2, 4):
        r = simulate(p, model="C", bc="ghost", N=200, dt=dt0 / f, eps=eps)
        d = None if prev is None else abs(prev["Istar"] - r["Istar"]) / r["Istar"]
        dT = None if prev is None else abs(prev["T095"] - r["T095"]) / r["T095"]
        time_rows.append({"dt": r["dt"], "Istar": r["Istar"], "T095": r["T095"],
                          "E": d, "ET": dT})
        prev = r

    space_rows = []
    dt_fine = 0.9 * stability_dt(p, 400, "C", "exp")["dt_max"]
    prev = None
    for N in (100, 200, 400):
        r = simulate(p, model="C", bc="ghost", N=N, dt=dt_fine, eps=eps)
        if prev is None:
            EI = EC = None
        else:
            EI = abs(prev["Istar"] - r["Istar"]) / r["Istar"]
            step = r["N"] // prev["N"]
            EC = np.max(np.abs(prev["Cs"] - r["Cs"][::step])) / p.Cs0
        space_rows.append({"N": N, "Istar": r["Istar"], "EI": EI, "EC": EC})
        prev = r
    return {"time": time_rows, "space": space_rows}


def boundary_order(p: Params, eps: float = 1e-4) -> list[dict]:
    """Order of accuracy of the two zero-flux treatments, against the exact profile."""
    rows = []
    for N in (25, 50, 100, 200):
        row = {"N": N}
        for bc in ("copy", "ghost"):
            r = simulate(p, model="C", bc=bc, N=N, eps=1e-6)
            exact = analytic_profile(p, r["u"])
            row[bc] = np.max(np.abs(r["psiS"] - exact)) / p.psiS_inf * 100
        rows.append(row)
    return rows


def e2_update_agreement(p: Params, N: int = 100) -> dict:
    """Forward Euler vs exponential update of e2 (model A), same dt."""
    dt = 0.9 * stability_dt(p, N, "A", "euler")["dt_max"]
    a = simulate(p, model="A", e2_update="euler", N=N, dt=dt, eps=1e-3,
                 store_every=200)
    b = simulate(p, model="A", e2_update="exp", N=N, dt=dt, eps=1e-3,
                 store_every=200)
    c = simulate(p, model="A", e2_update="exp", N=N, eps=1e-3)   # own larger dt
    return {"euler": a, "exp_same_dt": b, "exp_own_dt": c,
            "rel_diff_same_dt": abs(a["Istar"] - b["Istar"]) / a["Istar"],
            "rel_diff_own_dt": abs(a["Istar"] - c["Istar"]) / a["Istar"],
            "speedup": a["steps"] / c["steps"]}


def model_differences(p: Params, ratios=(0.2, 0.1, 0.05, 0.01, 0.001),
                      N: int = 100, eps: float = 1e-4) -> list[dict]:
    """Steady and transient errors of the reduction over Cs0/Km."""
    rows = []
    for rt in ratios:
        pr = replace(p, Cs0=rt * p.Km)
        A = simulate(pr, model="A", N=N, eps=eps)
        B = simulate(pr, model="B", N=N, eps=eps)
        C = simulate(pr, model="C", N=N, eps=eps)
        rows.append({
            "Cs0/Km": rt,
            "E_star": 100 * abs(C["Istar"] - A["Istar"]) / A["Istar"],
            "E_AB_star": 100 * abs(B["Istar"] - A["Istar"]) / A["Istar"],
            "E_QSS_T": 100 * abs(B["T095"] - A["T095"]) / A["T095"],
            "E_QSS_L1": 100 * compare_transient(A, B)["L1"],
            "E_tr_L1": 100 * compare_transient(A, C)["L1"],
            "E_T": 100 * abs(C["T095"] - A["T095"]) / A["T095"],
            "A2_ratio": C["e2max_over_eS"],
        })
    return rows


def mediator_sweep(p: Params, ratios=(1e-3, 1e-2, 1e-1, 1.0),
                   N: int = 200, eps: float = 1e-4) -> list[dict]:
    """Error of the reduction as a function of kcat/ka, the mediator ratio."""
    rows = []
    for rt in ratios:
        pr = replace(p, ka=p.kcat / rt)
        A = simulate(pr, model="A", N=N, eps=eps)
        B = simulate(pr, model="B", N=N, eps=eps)
        C = simulate(pr, model="C", N=N, eps=eps)
        rows.append({
            "kcat/ka": rt,
            "E_AB_star": 100 * abs(B["Istar"] - A["Istar"]) / A["Istar"],
            "E_QSS_T": 100 * abs(B["T095"] - A["T095"]) / A["T095"],
            "E_star": 100 * abs(C["Istar"] - A["Istar"]) / A["Istar"],
            "A2_ratio": C["e2max_over_eS"],
            "e2A": A["e2max_over_eS"],
        })
    return rows


# --------------------------------------------------------------------------
# Steady state solved directly (used for the error maps)
# --------------------------------------------------------------------------


def steady_bvp(p: Params, model: str, N: int = 400) -> tuple[np.ndarray, float]:
    """
    Newton solution of the stationary problem  De*Cs'' = R(Cs),
    Cs'(0) = 0 (ghost node), Cs(L) = Cs0.   Returns (Cs, I_steady).

    Models A and B share the same stationary equations, so 'A' and 'B' are
    equivalent here; 'C' uses the first-order rate.
    """
    from scipy.linalg import solve_banded

    dx = p.L / N
    if model == "C":
        rate = lambda c: p.k1 * c
        drate = lambda c: np.full_like(c, p.k1)
    else:
        f = 1.0 + p.kcat / p.ka
        rate = lambda c: p.kcat * p.eS * c / (p.Km + f * c)
        drate = lambda c: p.kcat * p.eS * p.Km / (p.Km + f * c) ** 2

    C = np.full(N + 1, p.Cs0)
    for _ in range(100):
        F = np.zeros(N + 1)
        lo = np.zeros(N + 1); di = np.zeros(N + 1); up = np.zeros(N + 1)
        d2 = p.De / dx**2
        F[1:N] = d2 * (C[2:] - 2 * C[1:N] + C[:N - 1]) - rate(C[1:N])
        lo[0:N - 1] = d2                      # sub-diagonal, shifted
        di[1:N] = -2 * d2 - drate(C[1:N])
        up[2:N + 1] = d2
        F[0] = d2 * (2 * C[1] - 2 * C[0]) - rate(C[0:1])[0]
        di[0] = -2 * d2 - drate(C[0:1])[0]
        up[1] = 2 * d2
        F[N] = p.Cs0 - C[N]
        di[N] = -1.0
        ab = np.zeros((3, N + 1))
        ab[0, 1:] = up[1:]
        ab[1, :] = di
        ab[2, :-1] = lo[:-1]
        d = solve_banded((1, 1), ab, -F)
        C = C + d
        if np.max(np.abs(d)) < 1e-16 * p.Cs0:
            break
    I = p.q * p.Ae * _trapz(rate(C), dx=dx)
    return C, I


def error_map(p: Params, n_c: int = 45, n_b: int = 45,
              c_range=(1e-3, 1.0), bd_range=(0.05, 20.0)) -> dict:
    """Steady state error of the reduced model over (Cs0/Km, beta*delta)."""
    cs = np.logspace(math.log10(c_range[0]), math.log10(c_range[1]), n_c)
    bds = np.logspace(math.log10(bd_range[0]), math.log10(bd_range[1]), n_b)
    E = np.zeros((n_b, n_c))
    for i, bd in enumerate(bds):
        s_val = bd * p.alpha / Params.K1_PRIME          # k1 = bd*alpha
        for j, c in enumerate(cs):
            pr = replace(p, s=s_val, Cs0=c * p.Km)
            _, IA = steady_bvp(pr, "A")
            _, IC = steady_bvp(pr, "C")
            E[i, j] = 100.0 * abs(IC - IA) / IA
    return {"cs": cs, "bd": bds, "E": E}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def make_figures(p: Params, outdir: str = "results") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    v1 = v1_analytical(p)
    r = v1["run"]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(r["u"], v1["exact_profile"], "-", lw=2, label="analytical, Eq. (56)")
    ax[0].plot(r["u"][::5], r["psiS"][::5], "o", ms=4, mfc="none",
               label=f"numerical, N={r['N']}")
    ax[0].set_xlabel("$u = x/L$"); ax[0].set_ylabel(r"$\psi_S$")
    ax[0].set_title("Steady substrate profile"); ax[0].legend()
    ax[1].semilogy(r["u"], np.abs(r["psiS"] - v1["exact_profile"]) / p.psiS_inf)
    ax[1].set_xlabel("$u = x/L$")
    ax[1].set_ylabel(r"$|\psi_S^{num}-\psi_S^{exact}|/\psi_S^\infty$")
    ax[1].set_title("Pointwise error")
    fig.tight_layout(); fig.savefig(f"{outdir}/A_steady_profile.png", dpi=150)
    plt.close(fig)

    m = r["theta"] > 0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r["theta"][m], v1["psiI_an"][m], "-", lw=2, label="analytical, Eq. (59)")
    sl = slice(None, None, max(1, m.sum() // 40))
    ax.plot(r["theta"][m][sl], r["psiI"][m][sl], "o", ms=4, mfc="none",
            label="numerical")
    ax.set_xlabel(r"$\theta = D_e t/L^2$"); ax.set_ylabel(r"$\psi_I$")
    ax.set_title("Transient dimensionless current"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{outdir}/B_transient_current.png", dpi=150)
    plt.close(fig)

    v2 = v2_asymptotic(p)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(v2["A"]["t"], v2["A"]["I"] * 1e9, "-", lw=2, label="original (A)")
    sl = slice(None, None, max(1, len(v2["C"]["t"]) // 40))
    ax.plot(v2["C"]["t"][sl], v2["C"]["I"][sl] * 1e9, "o", ms=4, mfc="none",
            label="reduced (C)")
    ax.set_xlabel("$t$, s"); ax.set_ylabel("$I$, nA")
    ax.set_title(r"Asymptotic limit  $C_{s0}/K_m=k_{cat}/k_a=10^{-3}$")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{outdir}/C_asymptotic.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for bc, mk in (("copy", "s"), ("ghost", "o")):
        rr = simulate(p, model="C", bc=bc, N=25, eps=1e-4)
        ax.plot(rr["u"][:6], rr["psiS"][:6], mk, ms=7, mfc="none",
                label=f"numerical, {bc}")
    uu = np.linspace(0, 5 / 25, 200)
    ax.plot(uu, analytic_profile(p, uu), "-", lw=1.5, label="analytical")
    ax.set_xlabel("$u = x/L$"); ax.set_ylabel(r"$\psi_S$")
    ax.set_title("Zero-flux boundary, first five nodes ($N=25$)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{outdir}/D_boundary_zoom.png", dpi=150)
    plt.close(fig)


def make_result_figures(p: Params, outdir: str = "results") -> dict:
    """Figures and tables for the investigation of the approximation error."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    out = {}

    # E* vs Cs0/Km for several diffusion modules -------------------------
    cs = np.logspace(-3, 0, 40)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for bd, ls in ((0.1, ":"), (0.943, "-"), (5.0, "--"), (20.0, "-.")):
        s_val = bd * p.alpha / Params.K1_PRIME
        E = []
        for c in cs:
            pr = replace(p, s=s_val, Cs0=c * p.Km)
            E.append(100 * abs(steady_bvp(pr, "C")[1] - steady_bvp(pr, "A")[1])
                     / steady_bvp(pr, "A")[1])
        ax.loglog(cs, E, ls, lw=1.8, label=rf"$\beta\delta={bd:g}$")
        out[f"E_bd{bd}"] = np.array(E)
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.text(1.1e-3, 1.15, "1 %", color="0.4", fontsize=9)
    ax.set_xlabel(r"$C_{s0}/K_m$")
    ax.set_ylabel(r"$E^{*}$, %")
    ax.set_title("Steady state error of the reduced model")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(f"{outdir}/E_error_vs_Cs.png", dpi=150)
    plt.close(fig)
    out["cs"] = cs

    # contour map --------------------------------------------------------
    m = error_map(p)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    lv = [0.1, 0.5, 1, 2, 5, 10, 20, 40]
    cf = ax.contourf(m["cs"], m["bd"], m["E"], levels=lv, cmap="viridis",
                     extend="both")
    cl = ax.contour(m["cs"], m["bd"], m["E"], levels=[1.0, 5.0],
                    colors="w", linewidths=1.6)
    ax.clabel(cl, fmt="%g %%", fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$C_{s0}/K_m$")
    ax.set_ylabel(r"$\beta\delta = k_1 L^2/D_e$")
    ax.set_title(r"$E^{*}$ [%], steady state error of the reduced model")
    fig.colorbar(cf, ax=ax, label=r"$E^{*}$, %")
    fig.tight_layout(); fig.savefig(f"{outdir}/F_error_map.png", dpi=150)
    plt.close(fig)
    out["map"] = m

    # mediator ratio -----------------------------------------------------
    rows = mediator_sweep(p, ratios=(1e-3, 1e-2, 1e-1, 0.5, 1.0))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    rt = [r["kcat/ka"] for r in rows]
    ax.semilogx(rt, [r["E_QSS_T"] for r in rows], "o-", lw=1.8,
                label=r"$E_{T}$, models A vs B (quasi steady state)")
    ax.semilogx(rt, [r["E_AB_star"] for r in rows], "s--", lw=1.8,
                label=r"$E^{*}$, models A vs B (steady state)")
    ax.set_xlabel(r"$k_{cat}/k_a$"); ax.set_ylabel("error, %")
    ax.set_yscale("log")
    ax.set_title("Effect of the quasi steady state assumption")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(f"{outdir}/G_qss_vs_mediator.png", dpi=150)
    plt.close(fig)
    out["mediator"] = rows
    return out


# --------------------------------------------------------------------------


def main() -> None:
    p = Params()
    st = stability_dt(p, 100, "A", "euler")
    print("=" * 74)
    print("PARAMETERS")
    print(f"  k1 = {p.k1:.6g} 1/s   eSigma = {p.eS:.6g} mol/m3 "
          f"({p.eS*1e3:.3g} uM)   alpha = {p.alpha:.4g} 1/s")
    print(f"  beta*delta = k1/alpha = {p.bd:.6f}   psiS_inf = {p.psiS_inf:.6f}")
    print(f"  L^2/De = {p.L**2/p.De:.1f} s     1/ka = {1/p.ka:.1e} s")
    print("STABILITY (N=100)")
    print(f"  dt_diffusion <= {st['dt_diffusion']:.4e} s")
    print(f"  dt_reaction  <= {st['dt_reaction']:.4e} s   "
          f"(ratio {st['dt_diffusion']/st['dt_reaction']:.1f})")

    print("=" * 74)
    print("V1  reduced model vs analytical solution")
    v1 = v1_analytical(p)
    print(f"  max profile error / psiS_inf : {v1['profile_err']:.3e}")
    print(f"  steady current, numerical    : {v1['run']['Istar']*1e9:.4f} nA")
    print(f"  steady current, analytical   : {v1['I_analytic']*1e9:.4f} nA")
    print(f"  relative difference          : {v1['current_err']:.3e}")
    print(f"  max transient current error  : {v1['transient_err']:.3e}"
          f"   (series n_max = {v1['n_max']})")
    print(f"  T_ss = {v1['run']['Tss']:.1f} s, T0.95 = {v1['run']['T095']:.1f} s")

    print("=" * 74)
    print("V2  original vs reduced, Cs0/Km = kcat/ka = 1e-3")
    v2 = v2_asymptotic(p)
    print(f"  I*_A = {v2['A']['Istar']*1e12:.5f} pA   "
          f"I*_C = {v2['C']['Istar']*1e12:.5f} pA")
    print(f"  relative difference in I*    : {v2['rel_diff']:.3e}")
    print(f"  relative difference in T0.95 : {v2['T095_diff']:.3e}")

    print("=" * 74)
    print("V3  refinement of the reduced model")
    v3 = v3_refinement(p)
    print("  time step (N=200):   (I* is a fixed point of the spatial scheme and")
    print("                        is nearly independent of dt; T0.95 is not)")
    for r in v3["time"]:
        e = "   --   " if r["E"] is None else f"{r['E']:.3e}"
        eT = "   --   " if r["ET"] is None else f"{r['ET']:.3e}"
        print(f"    dt={r['dt']:.4e} s   I*={r['Istar']*1e9:.6f} nA   "
              f"E_tau^I={e}   T0.95={r['T095']:.3f} s   E_tau^T={eT}")
    print("  spatial grid:")
    for r in v3["space"]:
        eI = "   --   " if r["EI"] is None else f"{r['EI']:.3e}"
        eC = "   --   " if r["EC"] is None else f"{r['EC']:.3e}"
        print(f"    N={r['N']:4d}   I*={r['Istar']*1e9:.6f} nA   "
              f"E_h^I={eI}   E_h^C={eC}")

    print("=" * 74)
    print("Boundary treatment, max |psiS - exact| / psiS_inf  [%]")
    print("     N        Cs0=Cs1      ghost node")
    for r in boundary_order(p):
        print(f"  {r['N']:4d}    {r['copy']:10.5f}    {r['ghost']:12.7f}")

    print("=" * 74)
    print("e2 update: forward Euler vs exponential (model A)")
    ag = e2_update_agreement(p)
    print(f"  same dt   : relative difference in I* = "
          f"{ag['rel_diff_same_dt']:.3e}")
    print(f"  own dt    : relative difference in I* = "
          f"{ag['rel_diff_own_dt']:.3e}")
    print(f"  steps: euler {ag['euler']['steps']} vs exponential "
          f"{ag['exp_own_dt']['steps']}  (speed-up {ag['speedup']:.0f}x)")

    print("=" * 74)
    print("Error of the reduction vs Cs0/Km   [%]      (N=100)")
    print("  Cs0/Km     E*(A-C)   E*(A-B)    E_T(A-C)  E_QSS_T   L1(A-C)")
    for r in model_differences(p):
        print(f"  {r['Cs0/Km']:<8g} {r['E_star']:9.4f} {r['E_AB_star']:10.2e}"
              f" {r['E_T']:9.4f} {r['E_QSS_T']:9.5f} {r['E_tr_L1']:9.4f}")
    print("  E*(A-B) is zero to round-off at every substrate concentration:")
    print("  the quasi steady state assumption has no effect on the steady state.")

    print("=" * 74)
    print("Error of the reduction vs kcat/ka   [%]     (N=200)")
    print("  kcat/ka    E*(A-B)    E_QSS_T   E*(A-C)   max e2/eS: model C   model A")
    for r in mediator_sweep(p):
        flag = "  <- (A2) fails" if r["A2_ratio"] > 1.0 else ""
        print(f"  {r['kcat/ka']:<9g} {r['E_AB_star']:9.2e} {r['E_QSS_T']:9.5f}"
              f" {r['E_star']:9.4f}      {r['A2_ratio']:8.4f}  {r['e2A']:8.4f}{flag}")
    print("  The reduced model gives e2 = (k1/ka)Cs, so max e2/eS ="
          " (kcat/ka)(Cs0/Km);")
    print("  it exceeds unity, and the reduced model becomes unphysical, when"
          " that product > 1.")

    print("=" * 74)
    make_figures(p)
    print("figures written to ./results")


if __name__ == "__main__":
    main()
