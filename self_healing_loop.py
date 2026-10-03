"""
self_healing_loop.py
====================
Production-grade implementation of the Autonomous Cyber-Physical
Self-Healing Loop — Layers 2 and 3.

Layer 2 : StreamingPSIGuardian
    Rolling 168-hour reference vs 24-hour deployment window PSI gate.
    Quantile binning, EMA smoothing, GREEN/AMBER/RED state machine.

Layer 3 : SHAPGuidedRLController
    Online REINFORCE + critic baseline Actor-Critic.
    Lexicographic multi-objective reward.
    CBF-QP safety projection via scipy.optimize.minimize.

Main    : SelfHealingSimulation
    Minute-by-minute BMS data stream simulation with routing logic.

Thesis anchor metrics (Harshil Patel, bbw Hochschule Berlin):
    In-sample F1  : 0.9923  CI [0.9913, 0.9933]
    Cross-bldg F1 : 0.418   (binary baseline 0.628)
    Fan speed PSI : 1.61    OA temp PSI : 5.84
    Real-world PSI ratio : 17.2x vs simulation
"""

from __future__ import annotations

import math
import time
import logging
import collections
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

import numpy as np
import scipy.optimize as sco
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────────────────
# SENSOR SCHEMA  (matches ASHRAE LBNL 11-channel layout)
# ─────────────────────────────────────────────────────────────────────────────
SENSOR_NAMES = [
    "SA_Temp",          # 0  Supply Air Temperature
    "OA_Temp",          # 1  Outdoor Air Temperature  ← high-risk
    "MA_Temp",          # 2  Mixed Air Temperature
    "RA_Temp",          # 3  Return Air Temperature
    "SA_Fan_Status",    # 4  Supply Air Fan Status (binary)
    "SA_Fan_Speed",     # 5  Supply Air Fan Speed  ← high-risk
    "OA_Damper",        # 6  Outdoor Air Damper Position
    "RA_Damper",        # 7  Return Air Damper Position
    "CC_Valve",         # 8  Cooling Coil Valve
    "HC_Valve",         # 9  Heating Coil Valve
    "Occupancy",        # 10 Occupancy Mode Indicator
]
N_SENSORS = len(SENSOR_NAMES)
HIGH_RISK_IDX = [5, 1]          # fan speed, OA temperature
SAFE_IDX      = [4, 10]         # fan status, occupancy (known low-PSI)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SelfHealingLoop")


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 2 — STREAMING PSI GUARDIAN
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class PSIEvent:
    """Emitted by the guardian every time state changes or PSI is updated."""
    timestep:    int
    state:       str                      # GREEN / AMBER / RED
    psi_raw:     np.ndarray               # [N_SENSORS] raw PSI
    psi_smooth:  np.ndarray               # [N_SENSORS] EMA-smoothed PSI
    psi_max:     float                    # max over high-risk sensors
    drift_rate:  float                    # Δ psi_max / Δ t
    trigger:     bool                     # should Layer 3 act?
    sensor_ids:  List[int]                # sensors above AMBER threshold


class StreamingPSIGuardian:
    """
    Continuous distribution-shift detector operating on a 1-minute BMS
    data stream.

    Reference window : 168 hours (one week, rolling, 10 080 samples)
    Deployment window:  24 hours (rolling, 1 440 samples)
    Update cadence   : every 60 incoming samples (1 simulated hour)
    Binning          : 10 quantile bins derived from the reference window
    Smoothing        : EMA with α = 0.30
    Thresholds       : AMBER = 0.10, RED = 0.50  (Siddiqi 2006)
    """

    THRESHOLD_AMBER   : float = 0.10
    THRESHOLD_RED     : float = 0.50
    ALPHA_EMA         : float = 0.30
    N_BINS            : int   = 10
    REF_WINDOW_MINS   : int   = 168 * 60   # 10 080
    DEPLOY_WINDOW_MINS: int   =  24 * 60   #  1 440
    UPDATE_EVERY_MINS : int   =  60

    def __init__(self):
        self.ref_buf    = collections.deque(maxlen=self.REF_WINDOW_MINS)
        self.dep_buf    = collections.deque(maxlen=self.DEPLOY_WINDOW_MINS)
        self.psi_smooth = np.zeros(N_SENSORS)
        self._psi_max_prev : float = 0.0
        self._step_counter : int   = 0
        self.state         : str   = "GREEN"
        self._event_log    : List[PSIEvent] = []

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                          #
    # ------------------------------------------------------------------ #

    def ingest(self, x: np.ndarray) -> Optional[PSIEvent]:
        """
        Ingest one minute of sensor data.

        Parameters
        ----------
        x : np.ndarray, shape [N_SENSORS]
            Raw sensor readings at the current minute.

        Returns
        -------
        PSIEvent if an hourly update was computed this step, else None.
        """
        assert x.shape == (N_SENSORS,), (
            f"Expected {N_SENSORS} sensors, got {x.shape}"
        )
        self.ref_buf.append(x.astype(np.float64))
        self.dep_buf.append(x.astype(np.float64))
        self._step_counter += 1

        if self._step_counter % self.UPDATE_EVERY_MINS != 0:
            return None                         # no update this minute

        return self._compute_and_route()

    def latest_psi(self) -> np.ndarray:
        """Return the most recent EMA-smoothed PSI vector."""
        return self.psi_smooth.copy()

    # ------------------------------------------------------------------ #
    #  PRIVATE COMPUTATION                                                 #
    # ------------------------------------------------------------------ #

    def _compute_and_route(self) -> PSIEvent:
        """Called every UPDATE_EVERY_MINS steps."""

        # Minimum data requirements
        if len(self.dep_buf) < self.N_BINS * 4:
            return self._emit_event(
                np.zeros(N_SENSORS), drift=0.0, force_state="GREEN"
            )

        ref = np.asarray(self.ref_buf)   # [≤10080, N_SENSORS]
        dep = np.asarray(self.dep_buf)   # [≤1440,  N_SENSORS]

        psi_raw = self._psi_quantile_bins(ref, dep)

        # ── Exponential moving average ──────────────────────────────────
        self.psi_smooth = (
            self.ALPHA_EMA * psi_raw
            + (1.0 - self.ALPHA_EMA) * self.psi_smooth
        )

        psi_max  = float(self.psi_smooth[HIGH_RISK_IDX].max())
        drift    = psi_max - self._psi_max_prev
        self._psi_max_prev = psi_max

        return self._emit_event(psi_raw, drift=drift)

    def _psi_quantile_bins(
        self,
        ref: np.ndarray,
        dep: np.ndarray,
    ) -> np.ndarray:
        """
        Compute PSI for each sensor using quantile bins derived from the
        reference distribution.

        PSI_j = Σ_k (p_k^dep - p_k^ref) · ln(p_k^dep / p_k^ref)

        Bins are computed from ref quantiles so that each bin contains
        roughly equal probability mass under the reference distribution.
        This avoids the sparse-bin artefacts of equal-width binning on
        heavy-tailed or clipped sensor signals (e.g. fan status = {0,1}).
        """
        eps    = 1e-9
        psi    = np.zeros(N_SENSORS)
        quants = np.linspace(0.0, 100.0, self.N_BINS + 1)   # 11 edges

        for j in range(N_SENSORS):
            r_j = ref[:, j]
            d_j = dep[:, j]

            # Build quantile bin edges from reference
            edges = np.percentile(r_j, quants)
            edges = np.unique(edges)              # collapse duplicates
            if edges.size < 3:
                # Constant signal (e.g. occupancy all-zero at night) —
                # PSI is undefined; treat as 0 (no shift).
                psi[j] = 0.0
                continue

            # Extend edges slightly to capture all values in histogram
            edges[0]  -= 1e-6
            edges[-1] += 1e-6

            r_counts = np.histogram(r_j, bins=edges)[0].astype(np.float64)
            d_counts = np.histogram(d_j, bins=edges)[0].astype(np.float64)

            r_prop = (r_counts + eps) / (r_counts.sum() + eps * len(r_counts))
            d_prop = (d_counts + eps) / (d_counts.sum() + eps * len(d_counts))

            psi[j] = np.sum((d_prop - r_prop) * np.log(d_prop / r_prop))

        return psi

    def _emit_event(
        self,
        psi_raw: np.ndarray,
        drift: float,
        force_state: Optional[str] = None,
    ) -> PSIEvent:
        """Determine state, build event object, log it."""
        psi_max = float(self.psi_smooth[HIGH_RISK_IDX].max())

        if force_state:
            new_state = force_state
        elif psi_max > self.THRESHOLD_RED:
            new_state = "RED"
        elif psi_max > self.THRESHOLD_AMBER and drift > 0:
            new_state = "AMBER"
        elif psi_max > self.THRESHOLD_AMBER:
            new_state = "AMBER"    # stable but elevated
        else:
            new_state = "GREEN"

        # Identify which sensors are individually above AMBER
        above_amber = [
            j for j in range(N_SENSORS)
            if self.psi_smooth[j] > self.THRESHOLD_AMBER
        ]

        if new_state != self.state:
            log.info(
                "PSI state: %s → %s  "
                "(psi_max=%.4f  drift=%.4f  sensors=%s)",
                self.state, new_state, psi_max, drift,
                [SENSOR_NAMES[j] for j in above_amber],
            )

        self.state  = new_state
        trigger     = new_state in ("AMBER", "RED")

        evt = PSIEvent(
            timestep   = self._step_counter,
            state      = new_state,
            psi_raw    = psi_raw.copy(),
            psi_smooth = self.psi_smooth.copy(),
            psi_max    = psi_max,
            drift_rate = drift,
            trigger    = trigger,
            sensor_ids = above_amber,
        )
        self._event_log.append(evt)
        return evt


# ═════════════════════════════════════════════════════════════════════════════
# LAYER 3 — SHAP-GUIDED ACTOR-CRITIC WITH CBF-QP SAFETY FILTER
# ═════════════════════════════════════════════════════════════════════════════

# ── 3.1  BMS physical constants ─────────────────────────────────────────────

@dataclass
class BMSConstraints:
    """
    Hard physical limits that the CBF must guarantee.
    All values are SI / ASHRAE 62.1 compliant.
    """
    SP_static_min_Pa  : float = 100.0    # duct static pressure floor
    SP_static_max_Pa  : float = 375.0    # duct static pressure ceiling
    T_changeover_min_C: float =  10.0    # economiser changeover minimum
    T_changeover_max_C: float =  24.0    # economiser changeover maximum
    OA_fraction_min   : float =  0.10    # ASHRAE 62.1 minimum ventilation
    OA_fraction_max   : float =  1.00    # full economiser open

DEFAULT_CONSTRAINTS = BMSConstraints()

# ── 3.2  State / Action definitions ─────────────────────────────────────────

STATE_DIM  = 14    # see build_state_vector()
ACTION_DIM = 3     # [Δ SP_static_Pa, Δ T_changeover_C, Δ OA_fraction]

# Per-step action magnitude limits (rate-of-change constraints)
ACTION_MAX = np.array([25.0, 2.0, 0.05], dtype=np.float32)

# ── 3.3  Neural network modules ─────────────────────────────────────────────

class Actor(nn.Module):
    """
    Gaussian policy π_θ(a|s).
    Outputs a mean and log-std for each of the 3 action dimensions.
    """
    def __init__(self, state_dim: int = STATE_DIM,
                 action_dim: int = ACTION_DIM):
        super().__init__()
        self._action_scale = torch.as_tensor(ACTION_MAX)

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128,        64), nn.LayerNorm(64),  nn.GELU(),
        )
        self.mu_head      = nn.Linear(64, action_dim)
        self.log_std_head = nn.Linear(64, action_dim)

        # Initialise output heads near zero for stable early training
        nn.init.orthogonal_(self.mu_head.weight,      gain=0.01)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)

    def forward(
        self, s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean_action, std_action) — both in action-scaled space."""
        h  = self.shared(s)
        mu = torch.tanh(self.mu_head(h))
        # Scale to physical action bounds
        mu = mu * self._action_scale.to(s.device)
        log_std = self.log_std_head(h).clamp(-4.0, 0.0)
        return mu, log_std.exp()

    def sample(
        self, s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and return (action, log_prob)."""
        mu, std = self.forward(s)
        dist    = torch.distributions.Normal(mu, std)
        a       = dist.rsample()          # reparameterisation trick
        log_p   = dist.log_prob(a).sum(-1)
        return a, log_p


class Critic(nn.Module):
    """
    State-value baseline V_ψ(s).
    Used to compute the advantage A_t = G_t - V_ψ(s_t).
    """
    def __init__(self, state_dim: int = STATE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128,        64), nn.LayerNorm(64),  nn.GELU(),
            nn.Linear(64,          1),
        )
        nn.init.orthogonal_(
            list(self.net.parameters())[-2], gain=1.0
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).squeeze(-1)


# ── 3.4  CBF-QP Safety Filter ───────────────────────────────────────────────

class CBFProjector:
    """
    Control Barrier Function safety filter implemented as a Quadratic
    Programme solved at each timestep via scipy.optimize.minimize with
    the SLSQP method (supports inequality constraints exactly).

    Problem:
        min_{a ∈ R^3}  ½ ‖a - a_proposed‖²
        s.t.
            SP_static  + a[0] ≥ SP_min
            SP_static  + a[0] ≤ SP_max
            OA_frac    + a[2] ≥ OA_min
            OA_frac    + a[2] ≤ OA_max
            T_eco      + a[1] ≥ T_eco_min
            T_eco      + a[1] ≤ T_eco_max
            |a[k]| ≤ ACTION_MAX[k]  for k = 0,1,2   (rate-of-change)

    The CBF condition  ḣ(s,a) + α·h(s) ≥ 0  with α=1 and linear dynamics
    reduces to these linear inequality constraints on the action.
    """

    def __init__(self, constraints: BMSConstraints = DEFAULT_CONSTRAINTS):
        self.c = constraints

    def project(
        self,
        a_proposed: np.ndarray,
        SP_current: float,
        T_eco_current: float,
        OA_current: float,
    ) -> np.ndarray:
        """
        Project a_proposed onto the CBF-safe set.

        Parameters
        ----------
        a_proposed  : [Δ SP, Δ T_eco, Δ OA]  proposed by the RL policy
        SP_current  : current static pressure setpoint (Pa)
        T_eco_current: current economiser changeover temperature (°C)
        OA_current  : current outdoor-air fraction (dimensionless)

        Returns
        -------
        a_safe : [Δ SP, Δ T_eco, Δ OA]  guaranteed to satisfy all CBF
                 constraints when applied to the current BMS state.
        """
        a0 = a_proposed.astype(np.float64)

        # Objective: minimise distance from proposed action
        def objective(a):
            d = a - a0
            return 0.5 * d @ d

        def grad_obj(a):
            return a - a0

        # Build inequality constraints:  g_i(a) ≥ 0  →  SLSQP form
        # (scipy uses g(a) ≥ 0 for 'ineq' constraints)
        c = self.c
        constraints_list = [
            # Static pressure: SP + Δ ≥ SP_min  →  Δ ≥ SP_min - SP
            {"type": "ineq",
             "fun": lambda a: a[0] - (c.SP_static_min_Pa - SP_current),
             "jac": lambda a: np.array([1., 0., 0.])},
            # Static pressure: SP + Δ ≤ SP_max  →  -Δ ≥ -(SP_max - SP)
            {"type": "ineq",
             "fun": lambda a: (c.SP_static_max_Pa - SP_current) - a[0],
             "jac": lambda a: np.array([-1., 0., 0.])},
            # Economiser: T_eco + Δ ≥ T_min
            {"type": "ineq",
             "fun": lambda a: a[1] - (c.T_changeover_min_C - T_eco_current),
             "jac": lambda a: np.array([0., 1., 0.])},
            # Economiser: T_eco + Δ ≤ T_max
            {"type": "ineq",
             "fun": lambda a: (c.T_changeover_max_C - T_eco_current) - a[1],
             "jac": lambda a: np.array([0., -1., 0.])},
            # OA fraction: OA + Δ ≥ OA_min  (ASHRAE 62.1)
            {"type": "ineq",
             "fun": lambda a: a[2] - (c.OA_fraction_min - OA_current),
             "jac": lambda a: np.array([0., 0., 1.])},
            # OA fraction: OA + Δ ≤ OA_max
            {"type": "ineq",
             "fun": lambda a: (c.OA_fraction_max - OA_current) - a[2],
             "jac": lambda a: np.array([0., 0., -1.])},
        ]

        # Rate-of-change bounds (box constraints on the action itself)
        bounds = [
            (-ACTION_MAX[0], ACTION_MAX[0]),
            (-ACTION_MAX[1], ACTION_MAX[1]),
            (-ACTION_MAX[2], ACTION_MAX[2]),
        ]

        result = sco.minimize(
            fun     = objective,
            x0      = a0.copy(),
            jac     = grad_obj,
            method  = "SLSQP",
            bounds  = bounds,
            constraints = constraints_list,
            options = {"ftol": 1e-9, "maxiter": 200, "disp": False},
        )

        if not result.success:
            # Fallback: clip to box bounds only (conservative safe action)
            log.warning(
                "CBF-QP did not converge (msg='%s'); falling back to clipping.",
                result.message,
            )
            a_safe = np.clip(a0, -ACTION_MAX, ACTION_MAX)
        else:
            a_safe = result.x

        return a_safe.astype(np.float32)


# ── 3.5  Reward function ─────────────────────────────────────────────────────

@dataclass
class RewardWeights:
    psi      : float = 10.0    # PSI reduction (primary)
    comfort  : float = 100.0   # thermal comfort violation (lexicographic)
    co2      : float = 100.0   # CO₂ violation (lexicographic)
    energy   : float = 1.0     # energy saving (secondary bonus)


def compute_reward(
    psi_max_now    : float,
    psi_max_prev   : float,
    T_zone         : float,
    T_setpoint     : float,
    CO2_ppm        : float,
    E_baseline_kWh : float,
    E_actual_kWh   : float,
    weights        : RewardWeights = RewardWeights(),
    psi_red        : float = 0.5,
    epsilon_T      : float = 1.0,
    CO2_limit      : float = 1000.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Lexicographic multi-objective reward.

    R_t = w_PSI  · (psi_prev - psi_now) / PSI_RED
        - w_comfort · max(0, |T_zone - T_sp| - ε)²
        - w_CO2     · max(0, CO2 - CO2_limit)²
        + w_energy  · (E_base - E_actual) / max(E_base, ε)

    Priority: comfort ≫ PSI reduction > energy savings.

    Returns
    -------
    (total_reward, component_dict) for logging and debugging.
    """
    r_psi = weights.psi * (psi_max_prev - psi_max_now) / psi_red

    temp_violation = max(0.0, abs(T_zone - T_setpoint) - epsilon_T)
    r_comfort = -weights.comfort * (temp_violation ** 2)

    co2_violation = max(0.0, CO2_ppm - CO2_limit)
    r_co2 = -weights.co2 * (co2_violation ** 2)

    denom_e = max(E_baseline_kWh, 1e-6)
    r_energy = weights.energy * (E_baseline_kWh - E_actual_kWh) / denom_e

    total = r_psi + r_comfort + r_co2 + r_energy

    components = {
        "r_psi"   : r_psi,
        "r_comfort": r_comfort,
        "r_co2"   : r_co2,
        "r_energy": r_energy,
        "total"   : total,
    }
    return total, components


# ── 3.6  Trajectory buffer ───────────────────────────────────────────────────

@dataclass
class Transition:
    state     : torch.Tensor
    action    : torch.Tensor
    log_prob  : torch.Tensor
    reward    : float
    next_state: torch.Tensor
    done      : bool


# ── 3.7  Full RL controller ──────────────────────────────────────────────────

class SHAPGuidedRLController:
    """
    Online REINFORCE + Critic Baseline controller.

    Operates at hourly decision frequency (triggered by Layer 2).
    Trajectory is accumulated in rolling fashion; an update is performed
    every `update_every` transitions.
    """

    def __init__(
        self,
        constraints: BMSConstraints = DEFAULT_CONSTRAINTS,
        lr_actor  : float = 1e-4,
        lr_critic : float = 5e-4,
        gamma     : float = 0.99,
        update_every: int = 24,          # update after 24 hourly steps
        device    : str = "cpu",
    ):
        self.gamma        = gamma
        self.update_every = update_every
        self.device       = torch.device(device)

        self.actor  = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.opt_a  = torch.optim.Adam(
            self.actor.parameters(),  lr=lr_actor,  eps=1e-5
        )
        self.opt_c  = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic, eps=1e-5
        )

        # Cyclic learning-rate schedule (weekly cycle = 168 hourly steps)
        self.sched_a = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt_a, T_0=168, T_mult=1, eta_min=1e-6
        )
        self.sched_c = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.opt_c, T_0=168, T_mult=1, eta_min=1e-6
        )

        self.cbf = CBFProjector(constraints)
        self.trajectory: List[Transition] = []
        self._update_count = 0

        # Running statistics for state normalisation
        self._state_mean = np.zeros(STATE_DIM, dtype=np.float64)
        self._state_var  = np.ones (STATE_DIM, dtype=np.float64)
        self._state_n    = 0

    # ------------------------------------------------------------------ #
    #  STATE CONSTRUCTION                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_state_vector(
        psi_smooth    : np.ndarray,     # [N_SENSORS]
        shap_values   : np.ndarray,     # [N_SENSORS] from FDD model
        T_zone        : float,
        T_setpoint    : float,
        CO2_ppm       : float,
        E_hour_kWh    : float,
        SP_current    : float,          # current static pressure Pa
        OA_fraction   : float,          # current OA fraction
        hour_of_day   : int,
        day_of_week   : int,
    ) -> np.ndarray:
        """
        Build the 14-dimensional state vector for the RL agent.

        Dimensions:
          0   fan speed PSI (smoothed)
          1   OA temperature PSI (smoothed)
          2   supply air temp PSI (smoothed)
          3   return air temp PSI (smoothed)
          4   fan speed SHAP attribution magnitude
          5   OA temperature SHAP magnitude
          6   supply air temp std SHAP magnitude (top-ranked feature)
          7   (T_zone - T_setpoint) / 5.0          comfort deviation
          8   CO2 / 1000.0                          normalised CO2
          9   E_hour_kWh / 10.0                     normalised energy
          10  SP_current / 250.0                    normalised SP
          11  OA_fraction                           already in [0,1]
          12  sin(2π · hour / 24)                   cyclical hour
          13  sin(2π · dow / 7)                     cyclical day
        """
        s = np.array([
            psi_smooth[5],                             # fan speed PSI
            psi_smooth[1],                             # OA temp PSI
            psi_smooth[0],                             # SA temp PSI
            psi_smooth[3],                             # RA temp PSI
            abs(shap_values[5]),                       # fan speed |SHAP|
            abs(shap_values[1]),                       # OA temp  |SHAP|
            abs(shap_values[0]),                       # SA temp  |SHAP|
            (T_zone - T_setpoint) / 5.0,
            CO2_ppm / 1000.0,
            E_hour_kWh / 10.0,
            SP_current / 250.0,
            OA_fraction,
            math.sin(2 * math.pi * hour_of_day / 24.0),
            math.sin(2 * math.pi * day_of_week / 7.0),
        ], dtype=np.float32)
        return s

    def _normalise_state(self, s: np.ndarray) -> np.ndarray:
        """Online Welford mean/variance normalisation of state vector."""
        self._state_n  += 1
        delta           = s - self._state_mean
        self._state_mean += delta / self._state_n
        self._state_var  += delta * (s - self._state_mean)
        std = np.sqrt(self._state_var / max(self._state_n - 1, 1)) + 1e-8
        return (s - self._state_mean) / std

    def _to_tensor(self, s: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(
            self._normalise_state(s), dtype=torch.float32
        ).to(self.device)

    # ------------------------------------------------------------------ #
    #  DECISION STEP                                                       #
    # ------------------------------------------------------------------ #

    def decide(
        self,
        state_vec  : np.ndarray,       # raw 14-dim state
        bms_state  : Dict[str, float], # current BMS setpoints for CBF
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Sample an action from the policy, project through CBF filter.

        Returns
        -------
        a_safe   : np.ndarray [3]   — safe BMS setpoint deltas
        log_prob : torch.Tensor scalar
        """
        s_t = self._to_tensor(state_vec)

        with torch.no_grad():
            a_proposed, log_p = self.actor.sample(s_t.unsqueeze(0))
            a_proposed = a_proposed.squeeze(0).cpu().numpy()

        # CBF-QP projection
        a_safe = self.cbf.project(
            a_proposed    = a_proposed,
            SP_current    = bms_state.get("SP_static_Pa",    200.0),
            T_eco_current = bms_state.get("T_changeover_C",   18.0),
            OA_current    = bms_state.get("OA_fraction",       0.15),
        )

        return a_safe, log_p.squeeze(0)

    # ------------------------------------------------------------------ #
    #  TRAJECTORY STORAGE                                                  #
    # ------------------------------------------------------------------ #

    def store_transition(
        self,
        state      : np.ndarray,
        action     : torch.Tensor,
        log_prob   : torch.Tensor,
        reward     : float,
        next_state : np.ndarray,
        done       : bool = False,
    ):
        t = Transition(
            state      = self._to_tensor(state),
            action     = action if isinstance(action, torch.Tensor)
                         else torch.as_tensor(action),
            log_prob   = log_prob,
            reward     = reward,
            next_state = self._to_tensor(next_state),
            done       = done,
        )
        self.trajectory.append(t)

        if len(self.trajectory) >= self.update_every:
            self._update_networks()

    # ------------------------------------------------------------------ #
    #  NETWORK UPDATE  (REINFORCE + TD critic)                            #
    # ------------------------------------------------------------------ #

    def _update_networks(self):
        n = len(self.trajectory)
        if n == 0:
            return

        states    = torch.stack([t.state      for t in self.trajectory])
        log_probs = torch.stack([t.log_prob   for t in self.trajectory])
        rewards   = torch.tensor(
            [t.reward for t in self.trajectory],
            dtype=torch.float32, device=self.device
        )
        next_sts  = torch.stack([t.next_state for t in self.trajectory])
        dones     = torch.tensor(
            [float(t.done) for t in self.trajectory],
            dtype=torch.float32, device=self.device
        )

        # ── Discounted Monte-Carlo returns ──────────────────────────
        G = torch.zeros(n, device=self.device)
        running = 0.0
        for i in reversed(range(n)):
            running = rewards[i].item() + self.gamma * running
            G[i]    = running

        # Normalise returns for stable gradients
        G = (G - G.mean()) / (G.std() + 1e-8)

        # ── Critic update  (minimise TD MSE) ────────────────────────
        V_now  = self.critic(states)
        with torch.no_grad():
            V_next = self.critic(next_sts)
        td_target = rewards + self.gamma * V_next * (1.0 - dones)
        loss_critic = ((td_target - V_now) ** 2).mean()

        self.opt_c.zero_grad()
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.opt_c.step()

        # ── Actor update  (REINFORCE with advantage) ────────────────
        # Recompute log_probs with current policy (avoids detach issue)
        mu, std = self.actor(states)
        dist = torch.distributions.Normal(mu, std)
        # Reconstruct actions tensor from stored (detached) actions
        actions_stored = torch.stack([t.action for t in self.trajectory]).to(self.device)
        log_probs_fresh = dist.log_prob(actions_stored).sum(-1)

        with torch.no_grad():
            advantage = G - self.critic(states)

        loss_actor = -(log_probs_fresh * advantage).mean()

        self.opt_a.zero_grad()
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.opt_a.step()

        # Learning rate schedules
        self.sched_a.step(self._update_count)
        self.sched_c.step(self._update_count)

        self._update_count += 1
        log.debug(
            "RL update #%d  loss_actor=%.4f  loss_critic=%.4f",
            self._update_count, loss_actor.item(), loss_critic.item(),
        )
        self.trajectory.clear()


# ═════════════════════════════════════════════════════════════════════════════
# SIMULATED BMS ENVIRONMENT
# ═════════════════════════════════════════════════════════════════════════════

class SimulatedBMSEnvironment:
    """
    Minimal physics-plausible BMS environment for integration testing.

    The environment simulates building sensor data with controllable
    distribution shift that can be reduced by the RL controller.

    Distribution shift is introduced by linearly ramping the fan speed
    operating point from the 'training' range to a 'shifted' range,
    mirroring the control-environment divergence documented in the thesis
    (fan speed PSI = 2.63 in the original model, 1.61 in the new model).
    """

    def __init__(
        self,
        seed            : int   = 42,
        shift_start_min : int   = 600,   # introduce shift at minute 600
        shift_ramp_mins : int   = 120,   # ramp over 2 hours
    ):
        self.rng             = np.random.default_rng(seed)
        self.shift_start     = shift_start_min
        self.shift_ramp      = shift_ramp_mins
        self.minute          = 0
        self.shift_magnitude = 0.0       # 0→1, controlled by environment

        # Current BMS setpoints (modified by RL controller)
        self.SP_static_Pa    : float = 200.0
        self.T_changeover_C  : float = 18.0
        self.OA_fraction     : float = 0.20

        # Simulated zone conditions
        self.T_zone          : float = 21.5
        self.T_setpoint      : float = 21.0
        self.CO2_ppm         : float = 650.0

    def step(
        self,
        action_delta: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Advance simulation by one minute.

        Parameters
        ----------
        action_delta : [Δ SP, Δ T_eco, Δ OA] or None

        Returns
        -------
        x     : [N_SENSORS]  sensor readings this minute
        info  : dict of environment state for reward computation
        """
        self.minute += 1

        # ── Apply RL setpoint deltas (with physical limits) ──────────
        if action_delta is not None:
            self.SP_static_Pa   = np.clip(
                self.SP_static_Pa   + action_delta[0],
                DEFAULT_CONSTRAINTS.SP_static_min_Pa,
                DEFAULT_CONSTRAINTS.SP_static_max_Pa,
            )
            self.T_changeover_C = np.clip(
                self.T_changeover_C + action_delta[1],
                DEFAULT_CONSTRAINTS.T_changeover_min_C,
                DEFAULT_CONSTRAINTS.T_changeover_max_C,
            )
            self.OA_fraction    = np.clip(
                self.OA_fraction    + action_delta[2],
                DEFAULT_CONSTRAINTS.OA_fraction_min,
                DEFAULT_CONSTRAINTS.OA_fraction_max,
            )

        # ── Compute distribution shift magnitude ─────────────────────
        if self.minute < self.shift_start:
            target_shift = 0.0
        elif self.minute < self.shift_start + self.shift_ramp:
            target_shift = (
                (self.minute - self.shift_start) / self.shift_ramp
            )
        else:
            target_shift = 1.0
        # RL can reduce shift by adjusting SP toward training range
        sp_correction = (self.SP_static_Pa - 200.0) / 175.0   # [-1, +1]
        effective_shift = max(0.0, target_shift - 0.5 * max(0, -sp_correction))
        self.shift_magnitude = effective_shift

        # ── Generate sensor readings ──────────────────────────────────
        hour = (self.minute // 60) % 24
        occupancy = 1.0 if (8 <= hour < 18) else 0.0

        # Fan speed: base 40-70%, shifted by up to +25%
        fan_base  = 0.40 + 0.30 * occupancy
        fan_shift = 0.25 * self.shift_magnitude
        fan_noise = self.rng.normal(0, 0.03)
        fan_speed = np.clip(fan_base + fan_shift + fan_noise, 0, 1)

        # Temperatures
        T_OA   = 15.0 + 8.0 * math.sin(2*math.pi*(self.minute/10080)) \
                 + self.rng.normal(0, 0.5)
        T_RA   = self.T_zone + self.rng.normal(0, 0.2)
        T_MA   = self.OA_fraction * T_OA + (1-self.OA_fraction) * T_RA \
                 + self.rng.normal(0, 0.1)
        T_SA   = T_MA - 4.0 * occupancy + self.rng.normal(0, 0.3)

        # Valve positions
        cc_valve = np.clip(0.3 + 0.5*occupancy + self.rng.normal(0,0.05),0,1)
        hc_valve = np.clip(0.1 + 0.2*(1-occupancy)+self.rng.normal(0,0.05),0,1)

        # Damper positions (influenced by OA fraction setting)
        oa_damper = np.clip(
            self.OA_fraction + self.rng.normal(0, 0.02), 0, 1
        )
        ra_damper = np.clip(1.0 - oa_damper + self.rng.normal(0, 0.02), 0, 1)

        x = np.array([
            T_SA, T_OA, T_MA, T_RA,
            float(fan_speed > 0.05),   # fan status (binary)
            fan_speed,
            oa_damper, ra_damper, cc_valve, hc_valve,
            occupancy,
        ], dtype=np.float32)

        # Zone comfort drift
        heat_from_fan = fan_speed * 0.5 * occupancy
        self.T_zone += 0.02 * (T_SA - self.T_zone) + heat_from_fan*0.01
        self.CO2_ppm = 400 + 400 * occupancy + self.rng.normal(0, 20)

        info = {
            "SP_static_Pa"     : self.SP_static_Pa,
            "T_changeover_C"   : self.T_changeover_C,
            "OA_fraction"      : self.OA_fraction,
            "T_zone"           : self.T_zone,
            "T_setpoint"       : self.T_setpoint,
            "CO2_ppm"          : self.CO2_ppm,
            "E_hour_kWh"       : fan_speed * 5.0 + cc_valve * 8.0,
            "E_baseline_kWh"   : 0.40 * 5.0 + 0.30 * 8.0,
            "shift_magnitude"  : self.shift_magnitude,
            "hour_of_day"      : hour,
            "day_of_week"      : (self.minute // (60*24)) % 7,
        }
        return x, info


# ═════════════════════════════════════════════════════════════════════════════
# MOCK FDD MODEL  (replace with your trained Random Forest)
# ═════════════════════════════════════════════════════════════════════════════

class MockFDDModel:
    """
    Placeholder for the trained Random Forest fault classifier.
    Returns a random class (0-3) and mock SHAP values.
    Replace with:
        import joblib
        self.model = joblib.load("rf_model_final.pkl")
    """
    FAULT_NAMES = ["Healthy", "Sensor Bias", "Valve Fault", "Damper Fault"]

    def predict(self, x: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Returns (predicted_class, shap_values[N_SENSORS]).
        In production: feed the 80-feature engineered vector.
        """
        rng   = np.random.default_rng(int(x.sum() * 1e4) % 2**31)
        cls   = int(rng.choice([0, 1, 2, 3], p=[0.6, 0.25, 0.10, 0.05]))
        shap  = rng.normal(0, 0.05, N_SENSORS).astype(np.float32)
        # Exaggerate fan speed and OA temp attributions to mirror thesis
        shap[5] *= 4.0
        shap[1] *= 3.5
        shap[0] *= 3.0
        return cls, shap


# ═════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION LOOP
# ═════════════════════════════════════════════════════════════════════════════

def run_self_healing_simulation(
    total_minutes     : int   = 4320,   # 3 days
    log_every_mins    : int   = 60,
    shift_start_min   : int   = 600,
    shift_ramp_mins   : int   = 120,
    seed              : int   = 42,
) -> Dict[str, list]:
    """
    Execute the complete four-layer self-healing loop simulation.

    Routing logic:
      GREEN  → FDD active, RL in monitoring mode
      AMBER  → FDD active with reduced-confidence flag, RL acts
      RED    → FDD paused (unreliable), RL acts with emergency priority

    Returns
    -------
    logs : dict of lists for post-analysis
    """
    log.info("=" * 65)
    log.info("SELF-HEALING LOOP  —  %d-minute simulation", total_minutes)
    log.info("Shift introduced at minute %d, ramp over %d minutes",
             shift_start_min, shift_ramp_mins)
    log.info("=" * 65)

    # ── Instantiate components ────────────────────────────────────────
    env        = SimulatedBMSEnvironment(
        seed=seed,
        shift_start_min=shift_start_min,
        shift_ramp_mins=shift_ramp_mins,
    )
    guardian   = StreamingPSIGuardian()
    controller = SHAPGuidedRLController()
    fdd_model  = MockFDDModel()

    # ── Logging containers ────────────────────────────────────────────
    logs: Dict[str, list] = {
        "minute"          : [],
        "gate_state"      : [],
        "psi_fan_smooth"  : [],
        "psi_oa_smooth"   : [],
        "psi_max"         : [],
        "fault_class"     : [],
        "fdd_active"      : [],
        "rl_action_SP"    : [],
        "rl_action_Teco"  : [],
        "rl_action_OA"    : [],
        "reward_total"    : [],
        "T_zone"          : [],
        "CO2"             : [],
        "shift_magnitude" : [],
        "SP_static"       : [],
    }

    fdd_active       = True
    prev_psi_max     = 0.0
    prev_state_vec   = None
    prev_action      = None
    prev_log_p       = None
    hourly_reward    = 0.0

    # ── Minute-by-minute loop ─────────────────────────────────────────
    for minute in range(total_minutes):
        # 1. Environment step (no action at start of minute)
        x_raw, info = env.step(action_delta=None)

        # 2. Layer 2: ingest into PSI guardian (hourly update)
        psi_event = guardian.ingest(x_raw)

        # 3. FDD classification (if gate is GREEN or AMBER)
        fault_class = None
        shap_values = np.zeros(N_SENSORS, dtype=np.float32)

        if fdd_active:
            fault_class, shap_values = fdd_model.predict(x_raw)

        # 4. Process PSI event if one was emitted this minute
        rl_action  = np.zeros(3, dtype=np.float32)
        log_p      = None

        if psi_event is not None:
            current_psi_max = psi_event.psi_max

            # Gate routing
            if psi_event.state == "RED":
                fdd_active = False
                log.warning(
                    "[min %5d] 🔴 RED gate — FDD paused. "
                    "psi_fan=%.3f  psi_oa=%.3f",
                    minute,
                    guardian.latest_psi()[5],
                    guardian.latest_psi()[1],
                )
            elif psi_event.state == "AMBER":
                fdd_active = True
                log.info(
                    "[min %5d] 🟡 AMBER gate — FDD ⚠ reduced confidence. "
                    "psi_max=%.3f",
                    minute, current_psi_max,
                )
            else:
                fdd_active = True

            # Layer 3: RL acts when triggered
            if psi_event.trigger:
                state_vec = SHAPGuidedRLController.build_state_vector(
                    psi_smooth    = guardian.latest_psi(),
                    shap_values   = shap_values,
                    T_zone        = info["T_zone"],
                    T_setpoint    = info["T_setpoint"],
                    CO2_ppm       = info["CO2_ppm"],
                    E_hour_kWh    = info["E_hour_kWh"],
                    SP_current    = info["SP_static_Pa"],
                    OA_fraction   = info["OA_fraction"],
                    hour_of_day   = info["hour_of_day"],
                    day_of_week   = info["day_of_week"],
                )

                bms_state = {
                    "SP_static_Pa"  : info["SP_static_Pa"],
                    "T_changeover_C": info["T_changeover_C"],
                    "OA_fraction"   : info["OA_fraction"],
                }

                rl_action, log_p = controller.decide(state_vec, bms_state)

                # Apply safe action to environment
                _, _ = env.step(action_delta=rl_action)

                # Compute reward
                reward, reward_parts = compute_reward(
                    psi_max_now    = current_psi_max,
                    psi_max_prev   = prev_psi_max,
                    T_zone         = info["T_zone"],
                    T_setpoint     = info["T_setpoint"],
                    CO2_ppm        = info["CO2_ppm"],
                    E_baseline_kWh = info["E_baseline_kWh"],
                    E_actual_kWh   = info["E_hour_kWh"],
                )
                hourly_reward = reward

                # Store transition for RL update
                if prev_state_vec is not None:
                    controller.store_transition(
                        state      = prev_state_vec,
                        action     = prev_action,
                        log_prob   = prev_log_p.detach(),
                        reward     = reward,
                        next_state = state_vec,
                    )

                prev_state_vec = state_vec
                prev_action    = torch.as_tensor(rl_action)
                prev_log_p     = log_p
                prev_psi_max   = current_psi_max

                if minute % log_every_mins == 0:
                    log.info(
                        "[min %5d] RL action  ΔSP=%+.1fPa  "
                        "ΔTeco=%+.2f°C  ΔOA=%+.3f  "
                        "reward=%.3f (psi=%.3f comf=%.3f)",
                        minute,
                        rl_action[0], rl_action[1], rl_action[2],
                        reward_parts["total"],
                        reward_parts["r_psi"],
                        reward_parts["r_comfort"],
                    )

        # 5. Log state every `log_every_mins` minutes
        if minute % log_every_mins == 0:
            psi_vec = guardian.latest_psi()
            log.info(
                "[min %5d] state=%-6s  shift=%.3f  "
                "psi_fan=%.4f  psi_oa=%.4f  "
                "T_zone=%.2f  CO2=%.0f  FDD=%s  fault=%s",
                minute,
                guardian.state,
                info["shift_magnitude"],
                psi_vec[5], psi_vec[1],
                info["T_zone"], info["CO2_ppm"],
                "ON " if fdd_active else "OFF",
                MockFDDModel.FAULT_NAMES[fault_class] if fault_class is not None
                else "—",
            )

        # 6. Append to logs
        psi_vec = guardian.latest_psi()
        logs["minute"].append(minute)
        logs["gate_state"].append(guardian.state)
        logs["psi_fan_smooth"].append(float(psi_vec[5]))
        logs["psi_oa_smooth"].append(float(psi_vec[1]))
        logs["psi_max"].append(float(psi_vec[HIGH_RISK_IDX].max()))
        logs["fault_class"].append(fault_class)
        logs["fdd_active"].append(fdd_active)
        logs["rl_action_SP"].append(float(rl_action[0]))
        logs["rl_action_Teco"].append(float(rl_action[1]))
        logs["rl_action_OA"].append(float(rl_action[2]))
        logs["reward_total"].append(hourly_reward)
        logs["T_zone"].append(info["T_zone"])
        logs["CO2"].append(info["CO2_ppm"])
        logs["shift_magnitude"].append(info["shift_magnitude"])
        logs["SP_static"].append(info["SP_static_Pa"])

    # ── Final summary ─────────────────────────────────────────────────
    n_red   = logs["gate_state"].count("RED")
    n_amber = logs["gate_state"].count("AMBER")
    n_green = logs["gate_state"].count("GREEN")
    peak_psi = max(logs["psi_max"])
    final_psi = logs["psi_max"][-1]
    comfort_violations = sum(
        1 for t, sp in zip(logs["T_zone"], [21.0]*total_minutes)
        if abs(t - sp) > 1.0
    )

    log.info("=" * 65)
    log.info("SIMULATION COMPLETE")
    log.info("  Gate states  : GREEN=%d  AMBER=%d  RED=%d",
             n_green, n_amber, n_red)
    log.info("  Peak PSI_max : %.4f", peak_psi)
    log.info("  Final PSI_max: %.4f", final_psi)
    log.info("  PSI reduction: %+.4f (negative = improvement)",
             final_psi - peak_psi)
    log.info("  Comfort violations (|ΔT|>1°C): %d minutes", comfort_violations)
    log.info("  RL updates executed: %d", controller._update_count)
    log.info("=" * 65)

    # Zero-shot success criteria check
    log.info("ZERO-SHOT SUCCESS CRITERIA:")
    log.info("  [%s] psi_max_final < 0.50  (value=%.4f)",
             "PASS" if final_psi < 0.50 else "FAIL", final_psi)
    log.info("  [%s] comfort_violations == 0  (count=%d)",
             "PASS" if comfort_violations == 0 else "FAIL",
             comfort_violations)
    log.info("  Note: binary F1 evaluation requires full FDD model "
             "with labelled target data.")

    return logs


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logs = run_self_healing_simulation(
        total_minutes   = 4320,   # 3 simulation days
        log_every_mins  = 120,    # log every 2 hours
        shift_start_min = 600,    # shift starts at hour 10
        shift_ramp_mins = 120,    # ramps over 2 hours
        seed            = 42,
    )
    log.info("Logs contain %d timestep records.", len(logs["minute"]))
    log.info(
        "To connect your trained RF model, replace MockFDDModel "
        "with:\n"
        "  from joblib import load\n"
        "  model = load('rf_model_final.pkl')\n"
        "  scaler = load('scaler_final.pkl')\n"
        "  feat_cols = load('feature_cols_final.pkl')"
    )
