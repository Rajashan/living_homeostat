#!/usr/bin/env python3
"""
living_homeostat.py - simulator, emulator fitting and preregistered analyses for
"The Living Homeostat" (free energy principle vs Nave's bioenactivism, via Ashby).

The same file serves three jobs:

  1. SIMULATOR. A reference model of the rack: a microbial-fuel-cell gut (acetate,
     sugar, acidity, exoelectrogen biomass, sugar-fermenter acclimation), a BQ25504-style
     harvester, a supercap store, three dosing pumps, and Ashby's ultrastable controller
     with a 25-position uniselector that steps when the electronics die and reboot.
     Use it in the pilot to choose store size and dose volume, and as the reference
     implementation for the firmware (Controller class).

  2. EMULATOR FITTING. Fits the frozen twin's metabolism model to a living rack's
     days 1-14, and writes the parameter file the emulator firmware loads.

  3. ANALYSES. The preregistered tests of Acts 1-2:
       Q1 flow stationarity      (path-integral FEP premise vs Nave's intrinsic instability)
       Q2 state stationarity     (2019 NESS premise, secondary)
       Q3 what predicts death     (surprisal under own past density vs metabolic trend)
       Q4 predictive validation  (Isomura-style: fit on 14 days, predict the next 14,
                                  living vs frozen)
       A2 allostasis             (deaths per week by controller x week; anticipation index)

Usage
  python living_homeostat.py selftest
  python living_homeostat.py simulate --days 70 --store-f 1.0 --seed 1 --out sim/
  python living_homeostat.py sweep-store                 # pilot helper: deaths/week vs store size
  python living_homeostat.py fit-emulator runs/L1/telemetry.csv --days 14 --out emulator_L1.json
  python living_homeostat.py act1 runs/L1/telemetry.csv runs/F1/telemetry.csv
  python living_homeostat.py act2 runs/act2/            # folder of rack telemetry CSVs

Telemetry CSV columns (one row per controller tick, default every 30 min):
  t_days, p_in_mw, v_store, alive, dose_acetate, dose_sugar, dose_water, unisel_pos, temp_c

Dependencies: numpy, scipy, pandas, scikit-learn, statsmodels.
Every constant marked PREREGISTER is fixed before real data are seen.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

# ---------------------------------------------------------------------------
# PREREGISTER: analysis constants
# ---------------------------------------------------------------------------
TICK_MIN = 30                 # controller decision interval
WINDOW_DAYS = 7               # Q1/Q2 window
PRED_HORIZON_TICKS = 2        # Q1: predict 1 h ahead
DEATH_LOOKBACK_H = 6          # Q3 features use the 6 h before an event
DEATH_SURPRISAL_FIT_DAYS = 7  # Q3 density fitted on the 7 days before the lookback
Q4_FIT_DAYS = 14
Q4_TEST_DAYS = 14
N_BINS = 6                    # Q4 discretisation of p_in and v_store
SEED_TABLE = 1948             # uniselector table seed (Ashby's year); same for every rack


# ---------------------------------------------------------------------------
# 1. Model of the gut (one 2 L gut, four electrode pairs in parallel, 1.2 L anolyte)
# ---------------------------------------------------------------------------
@dataclass
class GutParams:
    vol_l: float = 1.2          # anolyte volume of a 2 L gut
    i_max_ma: float = 1.0        # current at B = 1, saturating acetate, no acid stress (~0.3 mW)
    v_op: float = 0.35           # operating voltage at the harvester's MPP (V)
    k_a_mm: float = 2.0          # acetate half-saturation (mM)
    ce: float = 0.3              # coulombic efficiency
    mu: float = 0.5              # biomass growth (1/day)
    d: float = 0.05              # biomass decay (1/day)
    d_acid: float = 1.0          # extra decay under acid stress (1/day)
    b_max: float = 1.5
    load_benefit: float = 0.3    # growth factor when no current is drawn (open circuit)
    k_g: float = 5.0             # max sugar fermentation at F = 1 (mM sucrose/day)
    k_g_half: float = 0.5
    y_acetate: float = 3.0       # mM acetate per mM sucrose fermented
    y_acid: float = 4.0          # acid units per mM sucrose fermented
    k_buf: float = 0.5           # acid neutralisation (1/day)
    h50: float = 6.0             # acid level that halves activity
    a_f: float = 0.2             # fermenter acclimation rate (1/day)
    b_f: float = 0.02            # fermenter loss without sugar (1/day)
    q10: float = 2.0
    dose_ml: float = 0.5
    acetate_stock_m: float = 1.0
    sugar_stock_m: float = 0.5


@dataclass
class GutState:
    A: float = 10.0   # acetate (mM)
    G: float = 0.0    # sucrose (mM)
    H: float = 0.0    # excess acidity (a.u.)
    B: float = 0.5    # active exoelectrogen biomass (a.u.)
    F: float = 0.1    # sugar-fermenter capacity (0-1)


def acid_factor(H, h50):
    return 1.0 / (1.0 + (max(H, 0.0) / h50) ** 4)


def gut_step(s: GutState, p: GutParams, dt_days: float, load_on: bool, temp_c: float,
             frozen: bool) -> float:
    """Advance the gut; returns electrical power at the harvester input (mW).
    frozen=True holds the slow, self-produced states (B, F) fixed: metabolism-shaped
    response without self-production."""
    q = p.q10 ** ((temp_c - 25.0) / 10.0)
    phi = acid_factor(s.H, p.h50)
    sat = s.A / (p.k_a_mm + s.A)
    i_ma = p.i_max_ma * s.B * sat * phi * q if load_on else 0.0
    # acetate consumed by current (and by non-electrogenic routes, via CE)
    dA = -(i_ma * 1e-3 * 86400 / (96485 * 8) * 1e3 / p.ce) / p.vol_l
    r_g = p.k_g * s.F * s.G / (p.k_g_half + s.G) * q
    dA += p.y_acetate * r_g
    dG = -r_g
    dH = p.y_acid * r_g - p.k_buf * s.H
    s.A = max(0.0, s.A + dA * dt_days)
    s.G = max(0.0, s.G + dG * dt_days)
    s.H = max(0.0, s.H + dH * dt_days)
    if not frozen:
        growth = p.mu * s.B * sat * phi * q * (1.0 if load_on else p.load_benefit) * (1 - s.B / p.b_max)
        decay = p.d * s.B + p.d_acid * s.B * (1 - phi)
        s.B = max(1e-3, s.B + (growth - decay) * dt_days)
        acc = p.a_f * s.G / (p.k_g_half + s.G) * (1 - s.F) - p.b_f * s.F
        s.F = float(np.clip(s.F + acc * dt_days, 0.0, 1.0))
    return i_ma * p.v_op  # mW


def gut_dose(s: GutState, p: GutParams, kind: str):
    if kind == "acetate":
        s.A += p.dose_ml * 1e-3 * p.acetate_stock_m * 1e3 / p.vol_l
    elif kind == "sugar":
        s.G += p.dose_ml * 1e-3 * p.sugar_stock_m * 1e3 / p.vol_l


# ---------------------------------------------------------------------------
# 2. Electronics and Ashby's controller
# ---------------------------------------------------------------------------
@dataclass
class Electronics:
    store_f: float = 1.0          # PREREGISTER after the pilot (sweep-store)
    v_max: float = 4.5            # harvester overvoltage setting
    v_die: float = 3.1            # LDO dropout + MCU brownout: where the controller stops
    v_boot: float = 3.3           # where it restarts
    eta: float = 0.7              # harvester efficiency
    p_base_mw: float = 0.02       # MCU sleep + radio + ADC average
    e_pump_j: float = 3.0         # energy per 0.5 mL dose (pilot measures it)


def uniselector_table(seed=SEED_TABLE, positions=25, n_out=3, n_in=4):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=(positions, n_out, n_in))


class Controller:
    """Reference implementation of the firmware's Ashby unit.
    Inputs: normalised power, its 6 h trend, normalised store, bias.
    Outputs: three non-negative dose drives (acetate, sugar, water) that charge
    accumulators; a dose fires when its accumulator reaches 1. No setpoint, no guard."""
    KINDS = ("acetate", "sugar", "water")

    def __init__(self, table, el: Electronics, pos=0, doses_per_day_per_unit=2.0):
        self.table, self.el, self.pos = table, el, pos
        self.acc = np.zeros(3)
        self.rate = doses_per_day_per_unit
        self.hist = []

    def step_uniselector(self):
        self.pos = (self.pos + 1) % len(self.table)
        self.acc[:] = 0
        self.hist = []

    def decide(self, p_mw, v_store, dt_days):
        self.hist.append(p_mw)
        lag = int(6 * 60 / TICK_MIN)
        trend = p_mw - self.hist[-lag - 1] if len(self.hist) > lag else 0.0
        x = np.array([p_mw / 0.5, trend / 0.1,
                      (v_store - self.el.v_die) / (self.el.v_max - self.el.v_die), 1.0])
        drive = np.maximum(0.0, self.table[self.pos] @ x)
        self.acc += drive * self.rate * dt_days
        fire = [k for k in range(3) if self.acc[k] >= 1.0]
        for k in fire:
            self.acc[k] -= 1.0
        return [self.KINDS[k] for k in fire]


# ---------------------------------------------------------------------------
# 3. Whole-rack simulation
# ---------------------------------------------------------------------------
def simulate(days=70.0, seed=0, gut: GutState | None = None, gp: GutParams | None = None,
             el: Electronics | None = None, frozen_from_day: float | None = None,
             temp_fn=None, start_pos=0, controller="ashby", aif=None, substeps=6,
             v0=None, probe_days=(), t0=0.0, acetate_probe_days=()):
    """probe_days: times (absolute days) at which the experimenter adds one standard
    sugar probe dose at no energy cost to the robot (Act 1 history-specificity test)."""
    rng = np.random.default_rng(seed)
    gp, el = gp or GutParams(), el or Electronics()
    s = gut or GutState()
    table = uniselector_table()
    ctl = Controller(table, el, pos=start_pos)
    dt = TICK_MIN / (60 * 24)
    v = el.v_boot if v0 is None else v0
    alive = v >= el.v_die
    rows = []
    n = int(days / dt)
    probes = sorted(probe_days)
    a_probes = sorted(acetate_probe_days)
    for i in range(n):
        t = t0 + i * dt
        temp = temp_fn(t) if temp_fn else 25.0
        frozen = frozen_from_day is not None and t >= frozen_from_day
        probe = 0
        while probes and probes[0] <= t:
            probes.pop(0)
            gut_dose(s, gp, "sugar")
            probe = 1
        a_probe = 0
        while a_probes and a_probes[0] <= t:
            a_probes.pop(0)
            gut_dose(s, gp, "acetate")
            a_probe += 1
        p_acc = 0.0
        for _ in range(substeps):
            p_mw = gut_step(s, gp, dt / substeps, True, temp, frozen)
            p_mw *= np.exp(0.03 * rng.standard_normal())       # cell-level noise
            p_acc += p_mw / substeps
        e = 0.5 * el.store_f * v ** 2
        e += el.eta * p_acc * 1e-3 * dt * 86400
        if alive:
            e -= el.p_base_mw * 1e-3 * dt * 86400
        e = min(e, 0.5 * el.store_f * el.v_max ** 2)
        v = np.sqrt(max(2 * e / el.store_f, 0.0))
        doses = {k: 0 for k in Controller.KINDS}
        if alive:
            if controller == "ashby":
                fire = ctl.decide(p_acc, v, dt)
            else:
                fire = aif.decide(t, p_acc, v, temp)
            for kind in fire:
                e = 0.5 * el.store_f * v ** 2 - el.e_pump_j
                v = np.sqrt(max(2 * e / el.store_f, 0.0))
                gut_dose(s, gp, kind)
                doses[kind] += 1
                if v < el.v_die:
                    break
        if alive and v < el.v_die:
            alive = False
        elif not alive and v > el.v_boot:
            alive = True
            if controller == "ashby":
                ctl.step_uniselector()
            else:
                aif.reboot()
        rows.append(dict(t_days=t, p_in_mw=p_acc, v_store=v, alive=int(alive),
                         dose_acetate=doses["acetate"], dose_sugar=doses["sugar"],
                         dose_water=doses["water"], unisel_pos=ctl.pos, temp_c=temp,
                         probe=probe, a_probe=a_probe, _A=s.A, _G=s.G, _H=s.H, _B=s.B, _F=s.F))
    return pd.DataFrame(rows), s


def deaths(df):
    a = df["alive"].values
    return df["t_days"].values[1:][(a[:-1] == 1) & (a[1:] == 0)]


def fork(days_total=70.0, fork_day=14.0, seed=0, start_pos=0, probe_every=7.0, gut=None, **kw):
    """Run the living rack; at fork_day copy its exact state (gut, store, uniselector
    position) into a frozen twin. Both then receive the same weekly sugar probes."""
    import copy
    probes_all = list(np.arange(probe_every / 2, days_total, probe_every)) if probe_every else []
    # system identification in days 0-14: extra experimenter pulses (no energy cost) so the
    # day-14 model sees a wide range of feeding: double acetate pulses and extra sugar
    sysid_sugar = [5.0, 5.02, 9.0, 12.0, 12.02]
    sysid_acetate = [2.0, 2.02, 6.0, 6.02, 6.04, 11.0]
    pre, s_pre = simulate(days=fork_day, seed=seed, start_pos=start_pos, gut=gut,
                          probe_days=sorted([p for p in probes_all if p < fork_day] + sysid_sugar),
                          acetate_probe_days=sysid_acetate, **kw)
    last = pre.iloc[-1]
    probes = [p for p in probes_all if p >= fork_day]
    common = dict(start_pos=int(last.unisel_pos), v0=float(last.v_store), t0=fork_day,
                  probe_days=probes, **kw)
    rest = days_total - fork_day
    live, _ = simulate(days=rest, seed=seed + 1000, gut=copy.deepcopy(s_pre), **common)
    froz, _ = simulate(days=rest, seed=seed + 2000, gut=copy.deepcopy(s_pre),
                       frozen_from_day=fork_day, **common)
    return pd.concat([pre, live], ignore_index=True), pd.concat([pre, froz], ignore_index=True)


def probe_responses(df, window_h=12.0):
    """Excess energy harvested in the 12 h after each standard probe, relative to the
    12 h before (J). A history-specific (self-produced) change shows as a trend across
    probes in the living rack and none in the frozen twin."""
    per_h = 60 / TICK_MIN
    w = int(window_h * per_h)
    dt_s = TICK_MIN * 60
    idx = np.where(df.get("probe", pd.Series(0, index=df.index)).values == 1)[0]
    out = []
    for i in idx:
        if i - w < 0 or i + w > len(df):
            continue
        after = df.p_in_mw.values[i:i + w].sum() * 1e-3 * dt_s
        before = df.p_in_mw.values[i - w:i].sum() * 1e-3 * dt_s
        out.append((df.t_days.values[i], after - before))
    out = np.array(out)
    if len(out) < 3:
        return {"n": len(out), "slope_j_per_week": np.nan, "responses": out}
    slope = np.polyfit(out[:, 0] / 7, out[:, 1], 1)[0]
    return {"n": len(out), "slope_j_per_week": float(slope), "responses": out}


# ---------------------------------------------------------------------------
# 4. Emulator fitting (frozen twin)
# ---------------------------------------------------------------------------
FIT_NAMES = ["B", "F", "i_max_ma", "k_a_mm", "h50", "k_buf"]


def _replay(theta, df, gp0: GutParams):
    B, F, i_max, k_a, h50, k_buf = theta
    gp = GutParams(**{**asdict(gp0), "i_max_ma": i_max, "k_a_mm": k_a, "h50": h50, "k_buf": k_buf})
    s = GutState(A=10.0, B=B, F=F)
    dt = np.median(np.diff(df.t_days.values))
    out = np.empty(len(df))
    for i, r in enumerate(df.itertuples()):
        out[i] = gut_step(s, gp, dt, True, getattr(r, "temp_c", 25.0), frozen=True)
        for _ in range(int(r.dose_acetate) + int(getattr(r, "a_probe", 0))):
            gut_dose(s, gp, "acetate")
        for _ in range(int(r.dose_sugar) + int(getattr(r, "probe", 0))):
            gut_dose(s, gp, "sugar")
    return out


def fit_emulator(df, days=14.0, gp0: GutParams | None = None):
    gp0 = gp0 or GutParams()
    d = df[df.t_days < days].reset_index(drop=True)
    y = d.p_in_mw.values
    x0 = [0.5, 0.2, 2.0, 2.0, 6.0, 0.5]
    lo = [0.01, 0.0, 0.1, 0.1, 0.5, 0.05]
    hi = [1.5, 1.0, 10.0, 20.0, 30.0, 3.0]
    res = optimize.least_squares(lambda th: _replay(th, d, gp0) - y, x0, bounds=(lo, hi),
                                 diff_step=1e-2, max_nfev=200)
    rmse = float(np.sqrt(np.mean(res.fun ** 2)))
    return dict(zip(FIT_NAMES, map(float, res.x))), rmse


# ---------------------------------------------------------------------------
# 5. Analyses
# ---------------------------------------------------------------------------
def _features(df):
    """Per-tick features for the flow model of the gut: power history and feeding history
    (every source of input to the metabolism, including probes). The controller's
    configuration is deliberately NOT a feature: a frozen gut maps feeding history to
    power the same way whatever the controller does, so only a gut that changes itself
    should make a model trained in one window fail in another."""
    per_h = int(60 / TICK_MIN)
    d = df.copy()
    feed = {"acetate": d.dose_acetate, "sugar": d.dose_sugar + d.get("probe", 0)}
    cols = []
    for k, s in feed.items():
        for hrs in (6, 24, 72):
            c = f"n_{k}_{hrs}"
            d[c] = s.rolling(hrs * per_h, min_periods=1).sum()
            cols.append(c)
    for lag_h in (0, 1, 3, 12):
        c = f"p_lag{lag_h}"
        d[c] = d.p_in_mw.shift(lag_h * per_h).bfill()
        cols.append(c)
    X = d[cols].values
    y = d.p_in_mw.shift(-PRED_HORIZON_TICKS).values
    ok = ~np.isnan(y)
    return X[ok], y[ok], d.t_days.values[ok]


def q1_flow_stationarity(df, start_day=14.0):
    """Train a flow model in each window, test it on every other window.
    Drift index = slope of normalised test error against window separation (per week)."""
    X, y, t = _features(df[df.t_days >= start_day])
    edges = np.arange(t.min(), t.max() + 1e-9, WINDOW_DAYS)
    wins = [(t >= a) & (t < a + WINDOW_DAYS) for a in edges[:-1]]
    wins = [w for w in wins if w.sum() > 50]
    k = len(wins)
    L = np.full((k, k), np.nan)
    for i in range(k):
        m = HistGradientBoostingRegressor(max_iter=150, max_depth=3, learning_rate=0.1,
                                          random_state=0).fit(X[wins[i]], y[wins[i]])
        for j in range(k):
            r = y[wins[j]] - m.predict(X[wins[j]])
            L[i, j] = np.mean(r ** 2) / (np.var(y[wins[j]]) + 1e-12)
    sep, err = [], []
    for i in range(k):
        for j in range(k):
            if i != j:
                sep.append(abs(i - j)); err.append(L[i, j] - L[j, j])
    slope = np.polyfit(sep, err, 1)[0] if len(sep) > 2 else np.nan
    return {"drift_index": float(slope), "transfer_matrix": L}


def q1_replay_drift(df, fit_days=14.0, start_day=14.0, params=None):
    """PRIMARY Q1 measure. Fit the frozen-gut model on days 0-14, then replay the rack's
    own feeding history (doses and probes) through it for the rest of the run. Because the
    replay uses exactly what this gut was fed, controller rewiring cannot create a residual;
    a growing residual means the gut now maps the same inputs to different outputs.
    Extrapolation guard: a model can also fail because the controller later feeds the gut
    in ways never seen in days 0-14. Only ticks whose 24 h and 72 h feeding counts lie inside
    the range seen while fitting are scored; the in-range fraction is reported.
    Returns the weekly mean normalised residual and its slope per week."""
    if params is None:
        params, _ = fit_emulator(df, days=fit_days)
    theta = [params[k] for k in FIT_NAMES]
    d = df.reset_index(drop=True)
    pred = _replay(theta, d, GutParams())
    res = (d.p_in_mw.values - pred) / (d[d.t_days < fit_days].p_in_mw.mean() + 1e-9)
    t = d.t_days.values
    per_h = int(60 / TICK_MIN)
    sugar = d.dose_sugar + d.get("probe", 0)
    acet = d.dose_acetate + d.get("a_probe", 0)
    feats = np.column_stack([s.rolling(h * per_h, min_periods=1).sum().values
                             for s in (acet, sugar) for h in (24, 72)])
    fit_mask = t < fit_days
    lo, hi = feats[fit_mask].min(0), feats[fit_mask].max(0)
    in_range = np.all((feats >= lo) & (feats <= hi), axis=1)
    weeks = np.arange(start_day, t.max(), 7.0)
    wk = []
    for w in weeks:
        m = (t >= w) & (t < w + 7) & in_range
        if m.sum() > 24:
            wk.append((w, float(np.mean(np.abs(res[m])))))
    wk = np.array(wk)
    slope = float(np.polyfit(wk[:, 0] / 7, wk[:, 1], 1)[0]) if len(wk) > 2 else np.nan
    after = t >= start_day
    return {"replay_residual_by_week": wk, "replay_drift_per_week": slope,
            "in_range_fraction": float(in_range[after].mean())}


def q2_state_stationarity(df, start_day=14.0, bins=10):
    d = df[df.t_days >= start_day]
    pe = np.linspace(0, max(d.p_in_mw.max(), 1e-6), bins + 1)
    ve = np.linspace(d.v_store.min(), d.v_store.max() + 1e-9, bins + 1)
    out = []
    for a in np.arange(d.t_days.min(), d.t_days.max() - WINDOW_DAYS + 1e-9, WINDOW_DAYS):
        w = d[(d.t_days >= a) & (d.t_days < a + WINDOW_DAYS)]
        h, _, _ = np.histogram2d(w.p_in_mw, w.v_store, [pe, ve])
        out.append((h.ravel() + 0.5) / (h.sum() + 0.5 * h.size))
    kl = [stats.entropy(out[i + 1], out[i]) for i in range(len(out) - 1)]
    return {"kl_successive": kl, "kl_mean": float(np.mean(kl)) if kl else np.nan}


def _metabolic_state(df):
    """Metabolic state only (power and its 1 h change). The store voltage is excluded on
    purpose: death is defined by the store, so any feature built from it predicts death
    trivially."""
    per_h = int(60 / TICK_MIN)
    p = df.p_in_mw.values
    dp = np.r_[np.zeros(per_h), p[per_h:] - p[:-per_h]]
    return np.column_stack([p, dp])


def _q3_windows(df, rng):
    per_h = 60 / TICK_MIN
    look = int(DEATH_LOOKBACK_H * per_h)
    fit = int(DEATH_SURPRISAL_FIT_DAYS * 24 * per_h)
    alive = df.alive.values
    di = np.where((alive[:-1] == 1) & (alive[1:] == 0))[0] + 1
    ev = [i for i in di if i > fit + look]
    safe = [i for i in np.where(alive == 1)[0] if i > fit + look and
            not np.any((di > i) & (di < i + int(24 * per_h)))]
    ctl = list(rng.choice(safe, size=min(len(safe), 3 * len(ev)), replace=False)) if ev and safe else []
    Z = _metabolic_state(df)
    p = df.p_in_mw.values
    tw = int(12 * per_h)
    rows = []
    for i, lab in [(i, 1) for i in ev] + [(i, 0) for i in ctl]:
        gmm = GaussianMixture(2, random_state=0).fit(Z[i - look - fit:i - look])
        surpr = -gmm.score_samples(Z[i - look:i]).mean()
        seg = p[max(0, i - tw):i]
        trend = -np.polyfit(np.arange(len(seg)), seg, 1)[0]     # falling income = positive
        rows.append((surpr, trend, lab))
    return rows


def excursion_survival(df, q=0.95):
    """FEP reading: excursions into improbable metabolic states precede failure.
    Returns the fraction of excursions (surprisal above the q-quantile of the previous
    7 days) followed by a death within 24 h, against the base rate for any tick."""
    per_h = int(60 / TICK_MIN)
    fit = int(DEATH_SURPRISAL_FIT_DAYS * 24 * per_h)
    Z = _metabolic_state(df)
    alive = df.alive.values
    di = np.where((alive[:-1] == 1) & (alive[1:] == 0))[0] + 1
    death_soon = np.zeros(len(df), bool)
    for i in di:
        death_soon[max(0, i - 24 * per_h):i] = True
    exc, step = [], 24 * per_h
    for a in range(fit, len(df) - step, step):
        gmm = GaussianMixture(2, random_state=0).fit(Z[a - fit:a])
        thr = np.quantile(-gmm.score_samples(Z[a - fit:a]), q)
        s = -gmm.score_samples(Z[a:a + step])
        exc.extend([(death_soon[a + k], s[k] > thr) for k in range(step) if alive[a + k]])
    exc = np.array(exc, dtype=bool)
    if len(exc) == 0 or exc[:, 1].sum() == 0:
        return {"excursions": 0, "p_death_given_excursion": np.nan, "p_death_base": np.nan}
    return {"excursions": int(exc[:, 1].sum()),
            "p_death_given_excursion": float(exc[exc[:, 1], 0].mean()),
            "p_death_base": float(exc[:, 0].mean())}


def _auc(rows):
    if not rows or sum(r[2] for r in rows) < 3:
        return {"n_events": int(sum(r[2] for r in rows)) if rows else 0,
                "auc_surprisal": np.nan, "auc_trend": np.nan}
    F = np.array(rows)
    return {"n_events": int(F[:, 2].sum()), "auc_surprisal": roc_auc_score(F[:, 2], F[:, 0]),
            "auc_trend": roc_auc_score(F[:, 2], F[:, 1])}


def q3_death_predictors(df, rng=None):
    """AUC of (a) surprisal of the metabolic state under its own past density (FEP) and
    (b) the trend of metabolic income toward the boundary (Di Paolo's 'tendency'), for
    windows ending in a death vs matched windows that do not."""
    return _auc(_q3_windows(df, rng or np.random.default_rng(0)))


def q3_pooled(dfs, rng=None):
    """Deaths are rare per rack, so the preregistered Q3 test pools windows over racks."""
    rng = rng or np.random.default_rng(0)
    rows = []
    for df in dfs:
        rows += _q3_windows(df, rng)
    return _auc(rows)


def _discretise(df, edges=None):
    if edges is None:
        edges = (np.quantile(df.p_in_mw, np.linspace(0, 1, N_BINS + 1)[1:-1]),
                 np.quantile(df.v_store, np.linspace(0, 1, N_BINS + 1)[1:-1]))
    o = np.digitize(df.p_in_mw, edges[0]) * N_BINS + np.digitize(df.v_store, edges[1])
    a = (df.dose_acetate.values > 0) * 1 + (df.dose_sugar.values > 0) * 2  # 0..3
    return o.astype(int), a.astype(int), edges


def q4_predictive_validation(df, fit_start=14.0):
    """Isomura-style test with a count-based generative model p(o' | o, a) learned
    from 14 days (Dirichlet prior 1); score on the next 14 days in nats per tick.
    The FEP side may substitute its own POMDP (e.g. pymdp) through the same interface."""
    fit = df[(df.t_days >= fit_start) & (df.t_days < fit_start + Q4_FIT_DAYS)]
    test = df[(df.t_days >= fit_start + Q4_FIT_DAYS) &
              (df.t_days < fit_start + Q4_FIT_DAYS + Q4_TEST_DAYS)]
    o, a, edges = _discretise(fit)
    K = N_BINS * N_BINS
    C = np.ones((4, K, K))
    for t in range(len(o) - 1):
        C[a[t], o[t], o[t + 1]] += 1
    P = C / C.sum(2, keepdims=True)
    o2, a2, _ = _discretise(test, edges)
    ll = np.mean([np.log(P[a2[t], o2[t], o2[t + 1]]) for t in range(len(o2) - 1)])
    return {"loglik_per_tick": float(ll)}


def act1_report(live, froz, fork_day=14.0):
    out = {}
    for name, d in (("living", live), ("frozen", froz)):
        q1 = q1_flow_stationarity(d, start_day=fork_day)
        q1r = q1_replay_drift(d, fit_days=fork_day, start_day=fork_day)
        pr = probe_responses(d)
        out[name] = {"replay_drift_per_week": q1r["replay_drift_per_week"],
                     "drift_index": q1["drift_index"],
                     "kl_mean": q2_state_stationarity(d, start_day=fork_day)["kl_mean"],
                     "deaths_after_fork": int(np.sum(deaths(d) >= fork_day)),
                     "rewirings_after_fork": int(d[d.t_days >= fork_day].unisel_pos.diff().fillna(0).ne(0).sum()),
                     "probe_slope_j_per_week": pr["slope_j_per_week"],
                     **excursion_survival(d[d.t_days >= fork_day - DEATH_SURPRISAL_FIT_DAYS].reset_index(drop=True)),
                     **q4_predictive_validation(d, fit_start=fork_day)}
    return out


# --- Act 2: allostasis ---------------------------------------------------------
def cold_cycle(t_days, period_h=24.0, warm=25.0, cold=15.0):
    return warm if (t_days * 24 % period_h) < period_h / 2 else cold


class AllostaticAgent:
    """Minimal model-based anticipatory controller (stand-in for the FEP side's own
    active-inference agent, which they should supply). It learns mean power by phase of
    the day and doses acetate ahead of predicted shortfalls so that the store is high at
    cold onset; preferences = the empirical store distribution of living Ashby racks."""
    def __init__(self, el: Electronics, preferred_v: float, phases=48, lr=0.1):
        self.el, self.v_pref = el, preferred_v
        self.mean_p = np.full(phases, np.nan)     # learned income by phase of day (mW)
        self.phases, self.lr = phases, lr
        self.need = 0.0                           # learned feeding need (doses/day)
        self.last_dose_t = -1.0

    def reboot(self):
        pass

    def _store_after(self, v, e_delta_j):
        e = 0.5 * self.el.store_f * v ** 2 + e_delta_j
        e = min(e, 0.5 * self.el.store_f * self.el.v_max ** 2)
        return np.sqrt(max(2 * e / self.el.store_f, 0.0))

    def decide(self, t, p_mw, v_store, temp):
        ph = int((t % 1.0) * self.phases)
        m = self.mean_p[ph]
        self.mean_p[ph] = p_mw if np.isnan(m) else m + self.lr * (p_mw - m)
        known = ~np.isnan(self.mean_p)
        if known.sum() < self.phases:          # first day: feed only when it is affordable
            ok = self._store_after(v_store, -self.el.e_pump_j) > self.el.v_die + 0.1
            if ok and t - self.last_dose_t > 0.5:
                self.last_dose_t = t
                return ["acetate"]
            return []
        # acetate limitation: income below what this phase usually yields
        limited = p_mw < 0.85 * self.mean_p[ph]
        # predicted store at the next income minimum, with and without a dose now
        steps = [(ph + k) % self.phases for k in range(1, self.phases)]
        income = np.cumsum(self.mean_p[steps]) * 1e-3 * (86400 / self.phases) * self.el.eta
        base = np.cumsum(np.full(len(steps), self.el.p_base_mw)) * 1e-3 * (86400 / self.phases)
        trough = int(np.argmin(self.mean_p[steps]))
        v_no = self._store_after(v_store, income[trough] - base[trough])
        v_yes = self._store_after(v_store, income[trough] - base[trough] - self.el.e_pump_j)
        # expected free energy, crudely: risk = falling below the preferred store at the trough;
        # ambiguity ignored. Dose only if it keeps the trough above preference.
        v_now = self._store_after(v_store, -self.el.e_pump_j)       # store right after pumping
        if ((limited or t - self.last_dose_t > 1.0) and v_yes >= self.v_pref
                and v_now > self.el.v_die + 0.1 and t - self.last_dose_t > 0.3):
            self.last_dose_t = t
            return ["acetate"]
        return []


def anticipation_index(df, period_h=24.0):
    """Store level at cold onset minus store level at mid-warm, averaged over cycles."""
    ph = (df.t_days * 24) % period_h
    onset = df[(ph >= period_h / 2 - 0.5) & (ph < period_h / 2)].v_store.mean()
    mid = df[(ph >= period_h / 4 - 0.5) & (ph < period_h / 4)].v_store.mean()
    return float(onset - mid)


def act2_tests(racks: dict):
    """racks: name -> (controller label, telemetry df). Poisson GLM deaths ~ ctl * week."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    rows = []
    for name, (ctl, d) in racks.items():
        dd = deaths(d)
        for w in range(int(d.t_days.max() // 7)):
            rows.append(dict(rack=name, ctl=ctl, week=w,
                             deaths=int(np.sum((dd >= 7 * w) & (dd < 7 * (w + 1))))))
    R = pd.DataFrame(rows)
    # primary: exact conditional test of death rates (robust when one arm has zero deaths)
    tot = R.groupby("ctl").agg(deaths=("deaths", "sum"), weeks=("week", "count"))
    ctls = list(tot.index)
    primary = None
    if len(ctls) == 2 and tot.deaths.sum() > 0:
        a, b = ctls
        p_exp = tot.loc[a, "weeks"] / tot.weeks.sum()
        primary = {"deaths": tot.deaths.to_dict(), "weeks": tot.weeks.to_dict(),
                   "p_two_sided": stats.binomtest(int(tot.loc[a, "deaths"]), int(tot.deaths.sum()), p_exp).pvalue}
    # secondary: erosion of any advantage over time (controller x week), only if estimable
    glm = None
    if (R.groupby("ctl").deaths.sum() > 0).all():
        m = smf.glm("deaths ~ C(ctl) * week", R, family=sm.families.Poisson()).fit()
        glm = pd.DataFrame({"coef": m.params, "p": m.pvalues})
    ai = {name: anticipation_index(d) for name, (ctl, d) in racks.items()}
    return {"primary": primary, "glm": glm, "anticipation": ai}


# ---------------------------------------------------------------------------
# Self-test and pilot helpers
# ---------------------------------------------------------------------------
def selftest():
    def show_fork(title, gp, seeds, positions, b0=0.5):
        print(title)
        lives = []
        for seed, pos in zip(seeds, positions):
            live, froz = fork(seed=seed, start_pos=pos, gp=gp, gut=GutState(B=b0))
            lives.append(live)
            r = act1_report(live, froz)
            for nm in ("living", "frozen"):
                x = r[nm]
                print(f"   rack {seed} {nm:6s}: replay-drift={x['replay_drift_per_week']:+.3f}/wk "
                      f"transfer-drift={x['drift_index']:+.2f} KL={x['kl_mean']:.3f} "
                      f"deaths={x['deaths_after_fork']:2d} probe-slope={x['probe_slope_j_per_week']:+.2f} J/wk "
                      f"Q4={x['loglik_per_tick']:.2f} nats/tick "
                      f"P(death|excursion)={x['p_death_given_excursion']:.3f} vs base {x['p_death_base']:.3f}")
            print(f"   rack {seed} sugar fermenter F: day14={live[live.t_days < 14]._F.iloc[-1]:.2f} "
                  f"day70={live._F.iloc[-1]:.2f}")
        q3 = q3_pooled(lives)
        print(f"   Q3 pooled over living racks: n={q3['n_events']} deaths, "
              f"AUC surprisal={q3['auc_surprisal']:.2f}, AUC trend={q3['auc_trend']:.2f}")

    seeds, positions = (1, 2, 3), (3, 11, 19)     # preregistered random start positions
    show_fork("1a. Fork with default biology (weak acclimation): is there anything to detect?",
              GutParams(), seeds, positions)
    show_fork("1b. Positive control: a biofilm still maturing (B starts at 0.2, growth 0.08/day) and "
              "strong sugar acclimation. Replay drift and the probe slope should separate living from frozen.",
              GutParams(a_f=1.0, mu=0.08), seeds, positions, b0=0.2)

    print("2. Emulator fit on a living rack's first 14 days")
    live, _ = fork(seed=1, start_pos=3)
    params, rmse = fit_emulator(live, days=14)
    print("  ", {k: round(v, 3) for k, v in params.items()}, f"RMSE={rmse:.4f} mW "
          f"(mean power {live[live.t_days < 14].p_in_mw.mean():.3f} mW)")

    print("3. Act 2: Ashby vs allostatic stand-in under a 24 h 25/15 C cycle, 42 days")
    racks = {}
    for seed, pos in zip((11, 12, 13), (5, 14, 22)):
        d, _ = simulate(days=42, seed=seed, temp_fn=cold_cycle, start_pos=pos)
        racks[f"ashby{seed}"] = ("ashby", d)
        agent = AllostaticAgent(Electronics(), preferred_v=3.6)
        d2, _ = simulate(days=42, seed=seed, temp_fn=cold_cycle, controller="aif", aif=agent)
        racks[f"aif{seed}"] = ("aif", d2)
    res = act2_tests(racks)
    for k, (c, d) in racks.items():
        print(f"   {k:8s} deaths={len(deaths(d)):3d} anticipation={res['anticipation'][k]:+.3f} V")
    print("   primary:", res["primary"])
    if res["glm"] is not None:
        print(res["glm"].round(3).to_string())
    print("\nSelf-test finished.")


def sweep_store(sizes=(0.5, 1.0, 2.0, 5.0), seeds=(1, 2, 3), days=21):
    """Pilot helper: choose the store so a random homeostat dies 1-3 times a week."""
    for c in sizes:
        rates = []
        for s in seeds:
            d, _ = simulate(days=days, seed=s, el=Electronics(store_f=c))
            rates.append(len(deaths(d)) / (days / 7))
        print(f"store {c:4.1f} F: deaths/week = {np.mean(rates):.2f} (seeds {np.round(rates, 2)})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    a = sub.add_parser("simulate"); a.add_argument("--days", type=float, default=70)
    a.add_argument("--store-f", type=float, default=1.0); a.add_argument("--seed", type=int, default=1)
    a.add_argument("--out", default="sim")
    sub.add_parser("sweep-store")
    a = sub.add_parser("fit-emulator"); a.add_argument("csv"); a.add_argument("--days", type=float, default=14)
    a.add_argument("--out", default="emulator.json")
    a = sub.add_parser("act1"); a.add_argument("living_csv"); a.add_argument("frozen_csv")
    a = sub.add_parser("act2"); a.add_argument("folder")
    args = ap.parse_args(argv)

    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "simulate":
        live, froz = fork(days_total=args.days, seed=args.seed, el=Electronics(store_f=args.store_f))
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        live.to_csv(out / "living.csv", index=False); froz.to_csv(out / "frozen.csv", index=False)
        print(f"wrote {out}/living.csv and {out}/frozen.csv")
    elif args.cmd == "sweep-store":
        sweep_store()
    elif args.cmd == "fit-emulator":
        params, rmse = fit_emulator(pd.read_csv(args.csv), days=args.days)
        Path(args.out).write_text(json.dumps({"params": params, "rmse_mw": rmse}, indent=2))
        print(f"wrote {args.out} (RMSE {rmse:.4f} mW)")
    elif args.cmd == "act1":
        r = act1_report(pd.read_csv(args.living_csv), pd.read_csv(args.frozen_csv))
        for k in r:
            r[k].pop("transfer_matrix", None)
        print(json.dumps(r, indent=2, default=float))
    elif args.cmd == "act2":
        racks = {}
        for f in sorted(Path(args.folder).glob("*.csv")):
            d = pd.read_csv(f)
            ctl = "aif" if f.stem.startswith("aif") else "ashby"
            racks[f.stem] = (ctl, d)
        res = act2_tests(racks)
        print(res["glm"].to_string()); print(json.dumps(res["anticipation"], indent=2))


if __name__ == "__main__":
    sys.exit(main())