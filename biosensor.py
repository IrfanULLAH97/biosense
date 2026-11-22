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
import pickle
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


def steady_bvp(p: Params, model: str, N: int = 400,
               guess: "np.ndarray | None" = None, tol: float = 1e-13,
               max_iter: int = 200) -> tuple[np.ndarray, float]:
    """
    Damped Newton solution of the stationary problem  De*Cs'' = R(Cs),
    Cs'(0) = 0 (ghost node), Cs(L) = Cs0.   Returns (Cs, I_steady).

    Models A and B share the same stationary equations, so 'A' and 'B' are
    equivalent here; 'C' uses the first-order rate.

    At a large diffusion module and a large bulk concentration the kinetics
    saturate and the substrate is driven towards zero in the interior of the
    membrane.  An undamped Newton step then overshoots into negative
    concentrations, where the reaction term is unphysical and the iteration
    fails to converge.  The step is therefore damped so that the iterate stays
    positive, with backtracking on the residual norm, and a continuation guess
    may be supplied.  Non-convergence raises rather than returning silently.
    """
    from scipy.linalg import solve_banded

    dx = p.L / N
    d2 = p.De / dx**2
    if model == "C":
        rate = lambda c: p.k1 * c
        drate = lambda c: np.full_like(c, p.k1)
    else:
        f = 1.0 + p.kcat / p.ka
        rate = lambda c: p.kcat * p.eS * c / (p.Km + f * c)
        drate = lambda c: p.kcat * p.eS * p.Km / (p.Km + f * c) ** 2

    def residual(C):
        F = np.zeros(N + 1)
        F[1:N] = d2 * (C[2:] - 2 * C[1:N] + C[:N - 1]) - rate(C[1:N])
        F[0] = d2 * (2 * C[1] - 2 * C[0]) - rate(C[0:1])[0]
        F[N] = p.Cs0 - C[N]
        return F

    if guess is None:
        C = np.full(N + 1, p.Cs0)
    else:
        C = np.interp(np.linspace(0.0, 1.0, N + 1),
                      np.linspace(0.0, 1.0, len(guess)), guess).copy()
    C = np.maximum(C, 1e-300)

    F = residual(C)
    for _ in range(max_iter):
        lo = np.zeros(N + 1); di = np.zeros(N + 1); up = np.zeros(N + 1)
        lo[0:N - 1] = d2
        di[1:N] = -2 * d2 - drate(C[1:N])
        up[2:N + 1] = d2
        di[0] = -2 * d2 - drate(C[0:1])[0]
        up[1] = 2 * d2
        di[N] = -1.0
        ab = np.zeros((3, N + 1))
        ab[0, 1:] = up[1:]; ab[1, :] = di; ab[2, :-1] = lo[:-1]
        step = solve_banded((1, 1), ab, -F)

        # damp so that the iterate cannot fall below a tenth of its value
        neg = step < 0.0
        lam = 1.0
        if np.any(neg):
            lam = min(1.0, 0.9 * np.min(C[neg] / (-step[neg] + 1e-300)))
            lam = max(lam, 1e-4)
        f0 = np.linalg.norm(F)
        for _ in range(40):                       # backtracking line search
            Cn = np.maximum(C + lam * step, 0.0)
            Fn = residual(Cn)
            if np.linalg.norm(Fn) <= (1.0 - 1e-4 * lam) * f0 or lam < 1e-12:
                break
            lam *= 0.5
        C, F = Cn, Fn
        if np.max(np.abs(lam * step)) < tol * max(p.Cs0, 1e-300):
            break
    else:
        raise RuntimeError(
            f"steady_bvp did not converge (model {model}, N={N}, "
            f"Cs0/Km={p.Cs0/p.Km:.3g}, beta*delta={p.bd:.3g})")

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


def make_figures(p: Params, v1: dict, v2: dict, outdir: str = "results") -> None:
    """Validation figures (Section 5 of the thesis)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    r = v1["run"]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(r["u"], v1["exact_profile"], "-", lw=2, label="analytical")
    ax[0].plot(r["u"][::5], r["psiS"][::5], "o", ms=4, mfc="none",
               label=f"numerical, N={r['N']}")
    ax[0].set_xlabel("$u = x/L$"); ax[0].set_ylabel(r"$\psi_S$")
    ax[0].set_title("Steady state substrate profile"); ax[0].legend()
    ax[1].semilogy(r["u"], np.abs(r["psiS"] - v1["exact_profile"]) / p.psiS_inf)
    ax[1].set_xlabel("$u = x/L$")
    ax[1].set_ylabel(r"$|\psi_S^{num}-\psi_S^{exact}|/\psi_S^\infty$")
    ax[1].set_title("Pointwise error")
    fig.tight_layout(); fig.savefig(f"{outdir}/A_steady_profile.png", dpi=150)
    plt.close(fig)

    m = r["theta"] > 0
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(r["theta"][m], v1["psiI_an"][m], "-", lw=2, label="analytical")
    sl = slice(None, None, max(1, int(m.sum()) // 40))
    ax.plot(r["theta"][m][sl], r["psiI"][m][sl], "o", ms=4, mfc="none",
            label="numerical")
    ax.set_xlabel(r"$\theta = D_e t/L^2$"); ax.set_ylabel(r"$\psi_I$")
    ax.set_title("Transient dimensionless current"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{outdir}/B_transient_current.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(v2["A"]["t"], v2["A"]["I"] * 1e9, "-", lw=2, label="detailed (A)")
    sl = slice(None, None, max(1, len(v2["C"]["t"]) // 40))
    ax.plot(v2["C"]["t"][sl], v2["C"]["I"][sl] * 1e9, "o", ms=4, mfc="none",
            label="reduced (C)")
    ax.set_xlabel("$t$, s"); ax.set_ylabel("$I$, nA")
    ax.set_title(r"$C_{s0}/K_m=k_{cat}/k_a=10^{-3}$")
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


def error_thresholds(p: Params, bds=(0.1, 0.5, 0.943, 2.0, 5.0, 10.0, 20.0),
                     levels=(1.0, 5.0)) -> list[tuple]:
    """Values of Cs0/Km at which E* reaches the given levels, per beta*delta."""
    from scipy.optimize import brentq

    def Estar(c, bd):
        pr = replace(p, s=bd * p.alpha / Params.K1_PRIME, Cs0=c * p.Km)
        IA = steady_bvp(pr, "A")[1]
        IC = steady_bvp(pr, "C")[1]
        return 100.0 * abs(IC - IA) / IA

    out = []
    for bd in bds:
        row = [bd]
        for lv in levels:
            row.append(brentq(lambda c: Estar(c, bd) - lv, 1e-5, 1.0))
        out.append(tuple(row))
    return out


def make_result_figures(p: Params, em: dict, ms: list, outdir: str = "results"):
    """Error study figures (Section 6 of the thesis)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    cs = np.logspace(-3, 0, 40)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for bd, ls in ((0.1, ":"), (0.943, "-"), (5.0, "--"), (20.0, "-.")):
        E = []
        for c in cs:
            pr = replace(p, s=bd * p.alpha / Params.K1_PRIME, Cs0=c * p.Km)
            IA = steady_bvp(pr, "A")[1]
            IC = steady_bvp(pr, "C")[1]
            E.append(100 * abs(IC - IA) / IA)
        ax.loglog(cs, E, ls, lw=1.8, label=rf"$\beta\delta={bd:g}$")
    ax.axhline(1.0, color="0.5", lw=0.8)
    ax.text(1.1e-3, 1.15, "1 %", color="0.4", fontsize=9)
    ax.set_xlabel(r"$C_{s0}/K_m$"); ax.set_ylabel(r"$E^{*}$, %")
    ax.set_title("Steady state error of the reduced model")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(f"{outdir}/E_error_vs_Cs.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    lv = [0.1, 0.5, 1, 2, 5, 10, 20, 40]
    cf = ax.contourf(em["cs"], em["bd"], em["E"], levels=lv, cmap="viridis",
                     extend="both")
    cl = ax.contour(em["cs"], em["bd"], em["E"], levels=[1.0, 5.0],
                    colors="w", linewidths=1.6)
    ax.clabel(cl, fmt="%g %%", fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$C_{s0}/K_m$")
    ax.set_ylabel(r"$\beta\delta = k_1 L^2/D_e$")
    ax.set_title(r"$E^{*}$ [%], steady state error of the reduced model")
    fig.colorbar(cf, ax=ax, label=r"$E^{*}$, %")
    fig.tight_layout(); fig.savefig(f"{outdir}/F_error_map.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    rt = [r["kcat/ka"] for r in ms]
    ax.semilogx(rt, [r["E_QSS_T"] for r in ms], "o-", lw=1.8,
                label=r"$E_{T,\mathrm{AB}}$ (response time)")
    ax.semilogx(rt, [max(r["E_AB_star"], 1e-13) for r in ms], "s--", lw=1.8,
                label=r"$E^{*}_{\mathrm{AB}}$ (steady state)")
    ax.set_xlabel(r"$k_{cat}/k_a$"); ax.set_ylabel("error, %")
    ax.set_yscale("log")
    ax.set_title("Effect of the quasi steady state assumption")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(f"{outdir}/G_qss_vs_mediator.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Dimensionless structure and wide-range parameter studies
# --------------------------------------------------------------------------


def group_invariance_check(p: Params, f: float = 2.5) -> dict:
    """
    Confirm that the dimensionless response depends only on
    (beta, gamma, beta*delta, psiS_inf).

    A second parameter set is built with different dimensional values but the
    same four groups; the dimensionless responses must coincide.
    """
    alt = replace(p, L=p.L * f, De=p.De * f**2)          # beta, gamma, delta fixed
    r1 = simulate(p, model="A", N=100, eps=1e-4)
    r2 = simulate(alt, model="A", N=100, eps=1e-4)
    th = np.linspace(0.0, min(r1["theta"][-1], r2["theta"][-1]), 400)[1:]
    a = np.interp(th, r1["theta"], r1["psiI"])
    b = np.interp(th, r2["theta"], r2["psiI"])
    return {
        "groups": (p.beta, p.gamma, p.bd, p.psiS_inf),
        "groups_alt": (alt.beta, alt.gamma, alt.bd, alt.psiS_inf),
        "max_rel_dev": float(np.max(np.abs(a - b)) / r1["psiI"][-1]),
        "I_dimensional": (r1["Istar"], r2["Istar"]),
        "psiI_star": (float(r1["psiI"][-1]), float(r2["psiI"][-1])),
    }


def _bvp_N(bd: float) -> int:
    """Grid fine enough to resolve the reaction boundary layer of width 1/sqrt(bd)."""
    return int(min(3200, max(400, 200 * math.sqrt(max(bd, 1.0)))))


def _Estar(p: Params, psi: float, bd: float, ratio: float) -> float:
    """Steady state error of the reduced model, per cent, from the direct solver."""
    pr = replace(p, s=bd * p.alpha / Params.K1_PRIME, Cs0=psi * p.Km,
                 ka=p.kcat / ratio)
    N = _bvp_N(bd)
    IA = steady_bvp(pr, "A", N=N)[1]
    IC = steady_bvp(pr, "C", N=N)[1]
    return 100.0 * abs(IC - IA) / IA


def error_map_wide(p: Params, n_psi: int = 40, n_bd: int = 40,
                   psi_range=(1e-3, 10.0), bd_range=(0.01, 100.0),
                   ratio: float = 1e-3) -> dict:
    """Steady state error over four decades of psiS_inf and of beta*delta."""
    psis = np.logspace(math.log10(psi_range[0]), math.log10(psi_range[1]), n_psi)
    bds = np.logspace(math.log10(bd_range[0]), math.log10(bd_range[1]), n_bd)
    E = np.zeros((n_bd, n_psi))
    for i, bd in enumerate(bds):
        for j, psi in enumerate(psis):
            E[i, j] = _Estar(p, psi, bd, ratio)
    return {"psi": psis, "bd": bds, "E": E, "ratio": ratio}


def steady_error_vs_mediator(p: Params,
                             ratios=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
                             psis=(0.01, 0.1, 1.0), bd: float = 0.943) -> list:
    """Dependence of the steady state error on kcat/ka."""
    rows = []
    for r in ratios:
        row = {"kcat/ka": r}
        for psi in psis:
            row[psi] = _Estar(p, psi, bd, r)
            row[f"A2_{psi}"] = r * psi          # Eq. max e2/eSigma for model C
        rows.append(row)
    return rows


def steady_error_gamma_independence(p: Params,
                                    gammas=(1e2, 1e4, 1e6, 1e8)) -> list:
    """
    The steady state error must not depend on gamma.

    gamma is varied at fixed beta*delta, psiS_inf and kcat/ka by scaling kcat
    and ka together; eSigma = k1*Km/kcat then adjusts so that k1 is unchanged.
    """
    ratio = p.kcat / p.ka
    rows = []
    for g in gammas:
        kcat = g * ratio * p.De / p.L**2         # gamma = ka L^2/De, ka = kcat/ratio
        pr = replace(p, kcat=kcat, ka=kcat / ratio)
        N = _bvp_N(pr.bd)
        IA = steady_bvp(pr, "A", N=N)[1]
        IC = steady_bvp(pr, "C", N=N)[1]
        rows.append({"gamma": pr.gamma, "bd": pr.bd, "kcat/ka": pr.kcat / pr.ka,
                     "E_star": 100.0 * abs(IC - IA) / IA})
    return rows


def qss_vs_gamma(p: Params, gammas=(1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7),
                 N: int = 200, eps: float = 1e-4) -> list:
    """
    Error of the quasi steady state assumption against the dimensionless
    mediator reduction rate gamma = ka L^2/De, at fixed beta*delta, psiS_inf
    and kcat/ka.
    """
    ratio = p.kcat / p.ka
    rows = []
    for g in gammas:
        kcat = g * ratio * p.De / p.L**2
        pr = replace(p, kcat=kcat, ka=kcat / ratio)
        A = simulate(pr, model="A", N=N, eps=eps)
        B = simulate(pr, model="B", N=N, eps=eps)
        rows.append({
            "gamma": pr.gamma,
            "E_star_AB": 100.0 * abs(B["Istar"] - A["Istar"]) / A["Istar"],
            "E_T_AB": 100.0 * abs(B["T095"] - A["T095"]) / A["T095"],
            "L1_AB": 100.0 * compare_transient(A, B)["L1"],
            "e2max_A": A["e2max_over_eS"],
        })
    return rows


def make_wide_figures(p: Params, mw: dict, qg: list, outdir: str = "results"):
    """Figures for the wide-range error study (Chapter 6)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    r0 = p.kcat / p.ka

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    lv = [0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250]
    cf = ax.contourf(mw["psi"], mw["bd"], mw["E"], levels=lv, cmap="viridis",
                     extend="both")
    cl = ax.contour(mw["psi"], mw["bd"], mw["E"], levels=[1.0, 5.0, 10.0],
                    colors="w", linewidths=1.6)
    ax.clabel(cl, fmt="%g %%", fontsize=9)
    ax.axvline(0.2, color="r", ls="--", lw=1.3)
    ax.text(0.215, 0.013, r"$\Psi=0.2$", color="r", fontsize=9,
            rotation=90, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\Psi=(1+k_{cat}/k_a)\,C_{s0}/K_m$")
    ax.set_ylabel(r"$\beta\delta=k_1L^2/D_e$")
    ax.set_title(r"$E^{*}$ [%] over the complete governing plane")
    fig.colorbar(cf, ax=ax, label=r"$E^{*}$, %")
    fig.tight_layout(); fig.savefig(f"{outdir}/H_error_map_wide.png", dpi=150)
    plt.close(fig)

    Psi = np.logspace(-3, 1, 50)
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    for bd, ls in ((0.01, ":"), (0.943, "-"), (10.0, "--"), (100.0, "-.")):
        E = [_Estar(p, P / (1 + r0), bd, r0) for P in Psi]
        ax.loglog(Psi, E, ls, lw=1.8, label=rf"$\beta\delta={bd:g}$")
    ax.axhline(1.0, color="0.55", lw=0.8)
    ax.text(1.2e-3, 1.15, "1 %", color="0.4", fontsize=9)
    ax.axvline(0.2, color="r", ls="--", lw=1.1)
    ax.text(0.22, 2e-2, r"$\Psi=0.2$", color="r", fontsize=9)
    ax.set_xlabel(r"$\Psi=(1+k_{cat}/k_a)\,C_{s0}/K_m$")
    ax.set_ylabel(r"$E^{*}$, %")
    ax.set_title("Steady state error of the reduced model")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(f"{outdir}/I_error_vs_Psi.png", dpi=150)
    plt.close(fig)

    g = np.array([r["gamma"] for r in qg])
    eT = np.array([r["E_T_AB"] for r in qg])
    eS = np.array([max(r["E_star_AB"], 1e-13) for r in qg])
    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.loglog(g, eT, "o-", lw=1.8, label=r"$E_{T,\mathrm{AB}}$ (response time)")
    ax.loglog(g, eS, "s--", lw=1.8, label=r"$E^{*}_{\mathrm{AB}}$ (steady state)")
    gg = np.logspace(1, 7, 50)
    ax.loglog(gg, 124.6 / gg, "k:", lw=1.2, label=r"$124.6/\gamma$")
    ax.axhline(1.0, color="0.55", lw=0.8)
    ax.set_xlabel(r"$\gamma=k_aL^2/D_e$"); ax.set_ylabel("error, %")
    ax.set_title("Error of the quasi steady state assumption")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(f"{outdir}/J_qss_vs_gamma.png", dpi=150)
    plt.close(fig)


def _wide_thresholds(p: Params,
                     bds=(0.01, 0.1, 0.5, 0.943, 2.0, 5.0, 10.0, 30.0, 100.0),
                     levels=(1.0, 5.0, 10.0)) -> list:
    """Psi at which the steady state error reaches each level, per beta*delta."""
    from scipy.optimize import brentq
    r0 = p.kcat / p.ka
    out = []
    for bd in bds:
        row = [bd]
        for lv in levels:
            row.append(brentq(
                lambda P: _Estar(p, P / (1.0 + r0), bd, r0) - lv, 1e-6, 60.0))
        out.append(tuple(row))
    return out


def qss_constant_scan(p: Params,
                      cases=((0.943, 0.121212), (10.0, 0.121212), (0.943, 1.0)),
                      gammas=(100.0, 1000.0), N: int = 200,
                      eps: float = 1e-4) -> list:
    """
    The proportionality constant in E_{T,AB} ~ C/gamma, at several values of
    the diffusion module and the dimensionless bulk concentration.

    gamma is varied at fixed beta*delta, psiS_inf and kcat/ka by scaling kcat
    and ka together, so that k1 and hence beta*delta are unchanged.
    """
    rows = []
    for bd, psi in cases:
        p0 = replace(p, s=bd * p.alpha / Params.K1_PRIME, Cs0=psi * p.Km)
        ratio = p0.kcat / p0.ka
        row = {"bd": bd, "psi": psi, "C": []}
        for g in gammas:
            kcat = g * ratio * p0.De / p0.L**2
            pr = replace(p0, kcat=kcat, ka=kcat / ratio)
            A = simulate(pr, model="A", N=N, eps=eps)
            B = simulate(pr, model="B", N=N, eps=eps)
            eT = 100.0 * abs(B["T095"] - A["T095"]) / A["T095"]
            row["C"].append(eT * pr.gamma)
        rows.append(row)
    return rows


def timing_comparison(p: Params, N: int = 100, eps: float = 1e-3) -> dict:
    """
    Separate the reduction in the NUMBER of time steps from the reduction in
    wall-clock time: the exponential update costs an exponential per node per
    step, so the two factors are not the same.
    """
    import time
    dt = 0.9 * stability_dt(p, N, "A", "euler")["dt_max"]
    t0 = time.perf_counter()
    a = simulate(p, model="A", e2_update="euler", N=N, dt=dt, eps=eps,
                 store_every=200)
    t_euler = time.perf_counter() - t0
    t0 = time.perf_counter()
    c = simulate(p, model="A", e2_update="exp", N=N, eps=eps)
    t_exp = time.perf_counter() - t0
    return {"steps_euler": a["steps"], "steps_exp": c["steps"],
            "step_factor": a["steps"] / c["steps"],
            "wall_euler": t_euler, "wall_exp": t_exp,
            "wall_factor": t_euler / t_exp,
            "rel_diff_Istar": abs(a["Istar"] - c["Istar"]) / a["Istar"]}


# --------------------------------------------------------------------------
# Stage cache
# --------------------------------------------------------------------------

CACHE_DIR = os.path.join("results", "cache")


def cached(name: str, fn):
    """
    Return fn(), caching the value on disk under results/cache/.

    A complete run takes several minutes.  Caching each stage separately means
    an interrupted run resumes where it stopped.  Delete results/cache to force
    everything to be recomputed.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name + ".pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    value = fn()
    with open(path, "wb") as fh:
        pickle.dump(value, fh)
    return value


# --------------------------------------------------------------------------


def main() -> None:
    """Reproduce every table and figure of Sections 5 and 6 of the thesis."""
    import time
    t0 = time.time()

    def stage(msg):
        print(f"[{time.time()-t0:7.1f} s] {msg}", flush=True)

    p = Params()

    # ---- all computations, each performed exactly once ------------------
    stage("validation against the analytical solution ...")
    v1 = cached("v1", lambda: v1_analytical(p))
    stage("asymptotic regime ...")
    v2 = cached("v2", lambda: v2_asymptotic(p))
    stage("grid refinement ...")
    v3 = cached("v3", lambda: v3_refinement(p))
    stage("boundary treatments ...")
    bo = cached("bo", lambda: boundary_order(p))
    stage("forward Euler vs exponential update ...")
    ag = cached("ag", lambda: e2_update_agreement(p))
    stage("error vs substrate concentration ...")
    md = cached("md", lambda: model_differences(p))
    stage("group invariance ...")
    gi = cached("gi", lambda: group_invariance_check(p))
    stage("timing comparison ...")
    tm = cached("timing", lambda: timing_comparison(p))
    stage("QSS versus gamma ...")
    qg = cached("qss_gamma", lambda: qss_vs_gamma(p))
    stage("proportionality constant in E_T ~ C/gamma ...")
    qc = cached("qss_const", lambda: qss_constant_scan(p))
    stage("gamma independence of the steady error ...")
    gind = cached("gind", lambda: steady_error_gamma_independence(p))
    stage("wide error map ...")
    mw = cached("map_wide", lambda: error_map_wide(p))
    stage("wide thresholds ...")
    thw = cached("thresh_wide", lambda: _wide_thresholds(p))
    stage("computations complete")

    # ---- report ---------------------------------------------------------
    st = stability_dt(p, 100, "A", "euler")
    W = "=" * 74
    print(W); print("PARAMETERS  (Table 5.1, Table 5.2)")
    print(f"  k1 = {p.k1:.6g} 1/s   eSigma = {p.eS:.6g} mol/m3   "
          f"alpha = {p.alpha:.6g} 1/s")
    print(f"  beta*delta = {p.bd:.6f}   psiS_inf = {p.psiS_inf:.6f}   "
          f"L^2/De = {p.L**2/p.De:.1f} s")
    for s_val in (4.63, 9.25, 18.5, 27.8):
        pr = replace(p, s=s_val)
        print(f"    s={s_val:6.2f} Unit/mg   k1={pr.k1:.4e} 1/s   "
              f"eSigma={pr.eS:.4e} mol/m3   beta*delta={pr.bd:.4f}")
    print("STABILITY (N=100), Section 4.7")
    print(f"  diffusion  dt <= {st['dt_diffusion']:.4e} s")
    print(f"  reaction   dt <= {st['dt_reaction']:.4e} s   "
          f"ratio {st['dt_diffusion']/st['dt_reaction']:.1f}")

    print(W); print("VALIDATION (Section 5.2, 5.3, 5.4)")
    print(f"  max profile error / psiS_inf   : {v1['profile_err']:.4e}")
    print(f"  steady current, numerical      : {v1['run']['Istar']*1e9:.6f} nA")
    print(f"  steady current, analytical     : {v1['I_analytic']*1e9:.6f} nA")
    print(f"  relative difference            : {v1['current_err']:.4e}")
    print(f"  max transient current error    : {v1['transient_err']:.4e}")
    print(f"  series terms used (n_max)      : {v1['n_max']}")
    print(f"  T_ss = {v1['run']['Tss']:.2f} s,  T0.95 = {v1['run']['T095']:.2f} s")
    _, I_bvp_C = steady_bvp(p, "C")
    _, I_bvp_A = steady_bvp(p, "A")
    rA = cached("rA", lambda: simulate(p, model="A", N=400, eps=1e-5))
    print(f"  Newton BVP, reduced            : {I_bvp_C*1e9:.6f} nA")
    print(f"  Newton BVP, detailed           : {I_bvp_A*1e9:.6f} nA")
    print(f"  time marching, detailed        : {rA['Istar']*1e9:.6f} nA  "
          f"(rel. diff. {abs(I_bvp_A-rA['Istar'])/rA['Istar']:.2e})")
    print(f"  asymptotic test: I* differ by  : {v2['rel_diff']:.4e}")
    print(f"  asymptotic test: T0.95 differ  : {v2['T095_diff']:.4e}")

    print(W); print("REFINEMENT (Table 5.3)")
    for r in v3["time"]:
        e = "     --    " if r["E"] is None else f"{r['E']:.4e}"
        eT = "     --    " if r["ET"] is None else f"{r['ET']:.4e}"
        print(f"  dt={r['dt']:.4e} s  I*={r['Istar']*1e9:.6f} nA  E_tau^I={e}"
              f"  T0.95={r['T095']:.3f} s  E_tau^T={eT}")
    for r in v3["space"]:
        eI = "     --    " if r["EI"] is None else f"{r['EI']:.4e}"
        eC = "     --    " if r["EC"] is None else f"{r['EC']:.4e}"
        print(f"  N={r['N']:4d}          I*={r['Istar']*1e9:.6f} nA  "
              f"E_h^I={eI}  E_h^C={eC}")

    print(W); print("BOUNDARY TREATMENT (Table 5.4)   max|psiS-exact|/psiS_inf [%]")
    print("      N        Cs,0=Cs,1       ghost node")
    for r in bo:
        print(f"   {r['N']:4d}     {r['copy']:10.5f}     {r['ghost']:12.7f}")

    print(W); print("ENZYME UPDATE (Section 5.6)")
    print(f"  same dt : rel. difference in I* = {ag['rel_diff_same_dt']:.4e}")
    print(f"  own dt  : rel. difference in I* = {ag['rel_diff_own_dt']:.4e}")
    print(f"  steps   : Euler {ag['euler']['steps']}  vs  exponential "
          f"{ag['exp_own_dt']['steps']}   (factor {ag['speedup']:.0f})")

    print(W); print("ERROR VS SUBSTRATE CONCENTRATION (Table 6.2)")
    print("  Cs0/Km     E* [%]     E_T [%]    E_L1 [%]   E*/(Cs0/Km)")
    for r in md:
        print(f"  {r['Cs0/Km']:<9g} {r['E_star']:9.4f} {r['E_T']:9.4f}"
              f" {r['E_tr_L1']:10.4f}   {r['E_star']/r['Cs0/Km']:8.1f}")

    print(W); print("DIMENSIONLESS GROUP INVARIANCE (Section 5.2)")
    print(f"  groups set 1 (beta,gamma,beta*delta,psiS_inf): "
          f"{tuple(round(float(v),6) for v in gi['groups'])}")
    print(f"  groups set 2                                 : "
          f"{tuple(round(float(v),6) for v in gi['groups_alt'])}")
    print(f"  dimensional I*  : {gi['I_dimensional'][0]*1e9:.4f} nA  vs  "
          f"{gi['I_dimensional'][1]*1e9:.4f} nA")
    print(f"  dimensionless psiI* : {gi['psiI_star'][0]:.10e}  vs  "
          f"{gi['psiI_star'][1]:.10e}")
    print(f"  max relative deviation of psiI(theta) : {gi['max_rel_dev']:.3e}")

    print(W); print("TIMING: STEPS VERSUS WALL CLOCK (Section 5.9)")
    print(f"  steps      : Euler {tm['steps_euler']}  exponential "
          f"{tm['steps_exp']}   factor {tm['step_factor']:.1f}")
    print(f"  wall clock : Euler {tm['wall_euler']:.2f} s  exponential "
          f"{tm['wall_exp']:.2f} s   factor {tm['wall_factor']:.1f}")
    print(f"  cost per step ratio (exp/Euler) : "
          f"{(tm['wall_exp']/tm['steps_exp'])/(tm['wall_euler']/tm['steps_euler']):.2f}")
    print(f"  relative difference in I*       : {tm['rel_diff_Istar']:.3e}")

    print(W); print("QSS ASSUMPTION VERSUS gamma = ka L^2/De (Table 6.1)")
    print("     gamma     E*_AB [%]    E_T,AB [%]   L1_AB [%]   E_T,AB * gamma")
    for r in qg:
        print(f"  {r['gamma']:9.3g}   {r['E_star_AB']:10.3e}  {r['E_T_AB']:11.5f}"
              f" {r['L1_AB']:10.5f}   {r['E_T_AB']*r['gamma']:12.1f}")

    print(W); print("CONSTANT IN E_T,AB ~ C/gamma (Section 6.2)")
    print("   beta*delta   psiS_inf    C at gamma=100   C at gamma=1000")
    for r in qc:
        print(f"   {r['bd']:9.3g} {r['psi']:10.4g}   {r['C'][0]:14.1f}  {r['C'][1]:16.1f}")

    print(W); print("STEADY ERROR IS INDEPENDENT OF gamma (Section 6.2)")
    for r in gind:
        print(f"  gamma={r['gamma']:9.3g}  beta*delta={r['bd']:.4f}  "
              f"kcat/ka={r['kcat/ka']:.3g}   E* = {r['E_star']:.6f} %")

    print(W); print("COLLAPSE ONTO Psi = (1+kcat/ka)Cs0/Km (Section 6.3)")
    print("   beta*delta    Psi     E* for kcat/ka = 1e-3 ... 10        spread")
    for bd in (0.1, 0.943, 10.0):
        for Psi in (0.011, 0.11, 1.1):
            v = [_Estar(p, Psi/(1+r), bd, r) for r in (1e-3, 1e-2, 1e-1, 1.0, 10.0)]
            print(f"   {bd:9.3g} {Psi:7.4g}   " + " ".join(f"{x:8.4f}" for x in v)
                  + f"   {(max(v)-min(v))/max(v):.1e}")

    print(W); print("VALIDITY THRESHOLDS IN Psi (Table 6.3)")
    print("   beta*delta     E*=1%      E*=5%     E*=10%")
    for bd, a, b, c in thw:
        print(f"   {bd:9.3g}   {a:8.4f}  {b:8.4f}  {c:8.4f}")
    print(f"  wide map range: E* from {mw['E'].min():.4f} % to {mw['E'].max():.1f} %")

    print(W)
    stage("writing figures ...")
    make_figures(p, v1, v2)
    make_wide_figures(p, mw, qg)
    stage("figures written to ./results")


if __name__ == "__main__":
    main()
