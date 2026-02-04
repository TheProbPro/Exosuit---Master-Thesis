from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional
import math


# TODO: Look through this code to ensure that it is correct. This is a initial implementation from ChatGPT.


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


@dataclass
class HillMuscleParams:
    # Strength / geometry
    f_max: float                  # [N] maximum isometric force
    l_opt: float                  # [m] optimal fiber length
    l_slack: float                # [m] tendon slack length
    moment_arm: float             # [m] moment arm about the joint (sign sets flex/ext direction)

    # Pennation (optional; 0 is fine for many uses)
    pennation_at_opt: float = 0.0 # [rad] fiber pennation angle at l_opt

    # Activation dynamics
    tau_act: float = 0.015        # [s]
    tau_deact: float = 0.050      # [s]

    # Force-length curve shape
    fl_width: float = 0.45        # dimensionless width; larger = broader active FL curve

    # Force-velocity properties
    v_max: float = 10.0           # [l_opt/s] max shortening velocity in optimal-lengths per second
    ecc_factor: float = 1.4       # eccentric force enhancement cap (>=1)

    # Passive fiber (parallel elastic)
    passive_strain_start: float = 1.0  # normalized length where passive starts (1.0 = at l_opt)
    passive_k: float = 5.0             # steepness of passive curve

    # Tendon compliance
    tendon_compliant: bool = False
    tendon_strain_at_fmax: float = 0.04  # tendon strain at f_max (typ ~ 0.03-0.06)


class HillTypeMuscle:
    """
    Hill-type muscle model producing force from activation and musculotendon kinematics.

    Inputs each step:
      - u: neural excitation/command [0..1]
      - l_mt: musculotendon length [m]
      - v_mt: musculotendon velocity [m/s] (positive lengthening)

    Output:
      - tendon force (or muscle force if rigid tendon) [N]
    """

    def __init__(self, p: HillMuscleParams):
        self.p = p
        self.a = 0.0  # activation state [0..1]

        # Internal fiber length state only needed for compliant tendon;
        # for rigid tendon we can take l_f from geometry directly.
        self.l_f = p.l_opt

    # ---------- Curve models (normalized) ----------
    def _force_length_active(self, l_fn: float) -> float:
        # Gaussian-ish around 1.0
        # width is in normalized lengths; typical ~0.4-0.6
        w = self.p.fl_width
        return math.exp(-((l_fn - 1.0) / w) ** 2)

    def _force_length_passive(self, l_fn: float) -> float:
        # Zero below passive_strain_start, rises nonlinearly above it.
        s0 = self.p.passive_strain_start
        if l_fn <= s0:
            return 0.0
        # Exponential-like growth; scaled so it’s moderate around +10-20% strain
        k = self.p.passive_k
        # normalize strain above s0
        x = (l_fn - s0) / (1.6 - s0)  # map toward ~1 at l_fn=1.6
        x = clamp(x, 0.0, 2.0)
        return (math.exp(k * x) - 1.0) / (math.exp(k) - 1.0)

    def _force_velocity(self, v_fn: float) -> float:
        """
        v_fn: normalized fiber velocity in l_opt/s. Positive = lengthening.
        Returns FV multiplier.
        """
        # Hill-type-ish: reduced force in concentric (v<0), enhanced up to ecc_factor in eccentric (v>0)
        if v_fn < 0.0:
            # Concentric: force drops hyperbolically with speed
            # Simple stable form:
            # fv = (1 - v/vmax) / (1 + (v/(a*vmax)))  with a ~ 0.25
            a = 0.25
            vmax = self.p.v_max
            v = clamp(v_fn, -vmax, 0.0)
            return (1.0 - v / (-vmax)) / (1.0 + (-v) / (a * vmax))
        else:
            # Eccentric: rises and saturates
            # Simple saturating curve to ecc_factor
            ef = self.p.ecc_factor
            return 1.0 + (ef - 1.0) * (v_fn / (v_fn + 1.0))

    # ---------- Pennation (optional) ----------
    def _cos_pennation(self, l_f: float) -> float:
        # Constant thickness assumption: l_f * sin(alpha) = l_opt * sin(alpha0)
        a0 = self.p.pennation_at_opt
        if a0 <= 1e-8:
            return 1.0
        h = self.p.l_opt * math.sin(a0)
        # avoid invalid if l_f < h
        l_f = max(l_f, h + 1e-9)
        sin_a = h / l_f
        sin_a = clamp(sin_a, 0.0, 0.999999)
        cos_a = math.sqrt(1.0 - sin_a * sin_a)
        return cos_a

    # ---------- Tendon ----------
    def _tendon_force_norm(self, l_t: float) -> float:
        """
        Normalized tendon force (relative to f_max) from tendon length.
        Nonlinear toe then linear-ish. Zero below slack.
        """
        p = self.p
        if l_t <= p.l_slack:
            return 0.0
        # strain
        e = (l_t - p.l_slack) / p.l_slack
        # scale so that e = tendon_strain_at_fmax => f = 1
        e_ref = max(p.tendon_strain_at_fmax, 1e-6)
        x = e / e_ref

        # toe region up to x=1, then linear-ish beyond
        if x <= 1.0:
            # smooth toe (quadratic)
            return x * x
        else:
            # continue with slope 2 at x=1: f = 1 + 2*(x-1)
            return 1.0 + 2.0 * (x - 1.0)

    def _tendon_length_from_force_norm(self, f_tn: float) -> float:
        """
        Invert the tendon curve approximately:
        given normalized force f_tn, return tendon length l_t.
        """
        p = self.p
        f_tn = max(f_tn, 0.0)
        e_ref = max(p.tendon_strain_at_fmax, 1e-6)

        if f_tn <= 1.0:
            # f = x^2 => x = sqrt(f)
            x = math.sqrt(f_tn)
        else:
            # f = 1 + 2*(x-1) => x = 1 + (f-1)/2
            x = 1.0 + (f_tn - 1.0) / 2.0

        e = x * e_ref
        return p.l_slack * (1.0 + e)

    # ---------- Dynamics update ----------
    def step(self, u: float, l_mt: float, v_mt: float, dt: float) -> float:
        """
        Advance state by dt and return force [N] transmitted to the joint (along tendon line).
        """
        if dt <= 0:
            raise ValueError("dt must be > 0")
        u = clamp(u, 0.0, 1.0)

        # Activation dynamics (first-order, separate time constants)
        tau = self.p.tau_act if u > self.a else self.p.tau_deact
        # stable Euler update
        self.a += (u - self.a) * (dt / max(tau, 1e-6))
        self.a = clamp(self.a, 0.0, 1.0)

        p = self.p

        if not p.tendon_compliant:
            # Rigid tendon assumption: l_t = l_slack, fiber length is remaining
            l_t = p.l_slack
            l_f = max(l_mt - l_t, 1e-9)
            cos_a = self._cos_pennation(l_f)

            # Fiber velocity ~ musculotendon velocity projected (simplified)
            v_f = v_mt  # if pennation small; for more accuracy divide by cos_a
            l_fn = l_f / p.l_opt
            v_fn = v_f / (p.l_opt)  # [l_opt/s]

            f_a = self._force_length_active(l_fn) * self._force_velocity(v_fn)
            f_p = self._force_length_passive(l_fn)

            f_total = p.f_max * (self.a * f_a + f_p) * cos_a
            return max(0.0, f_total)

        # Compliant tendon: solve force equilibrium roughly.
        # We do a simple fixed-point iteration on tendon force.
        # Start with previous-ish estimate from current fiber length state.
        l_f = self.l_f
        for _ in range(6):
            cos_a = self._cos_pennation(l_f)
            l_t = max(l_mt - l_f * cos_a, 0.0)
            f_tn = self._tendon_force_norm(l_t)

            l_fn = l_f / p.l_opt
            # approximate fiber velocity from mt velocity (ignoring tendon dynamics):
            v_f = v_mt
            v_fn = v_f / p.l_opt

            f_a = self._force_length_active(l_fn) * self._force_velocity(v_fn)
            f_p = self._force_length_passive(l_fn)

            # muscle force along tendon line, normalized
            f_mn = (self.a * f_a + f_p) * cos_a

            # equilibrium: f_mn ~= f_tn
            # adjust fiber length using a small correction based on force mismatch
            err = f_mn - f_tn
            l_f += -0.15 * err * p.l_opt  # heuristic gain
            l_f = clamp(l_f, 0.2 * p.l_opt, 1.8 * p.l_opt)

        # Integrate fiber length state using mt velocity (simple)
        self.l_f = l_f

        # Final tendon force
        l_t = max(l_mt - l_f * self._cos_pennation(l_f), 0.0)
        f_tn = self._tendon_force_norm(l_t)
        return p.f_max * f_tn


class HillMuscleTorqueController:
    """
    Converts muscle activations to joint torque using a set of Hill-type muscles.

    Provide each muscle with:
      - musculotendon length function l_mt(q)
      - musculotendon velocity function v_mt(q, qd)
    """

    def __init__(
        self,
        muscles: List[HillTypeMuscle],
        lmt_funcs: List[Callable[[float], float]],
        vmt_funcs: List[Callable[[float, float], float]],
    ):
        if not (len(muscles) == len(lmt_funcs) == len(vmt_funcs)):
            raise ValueError("muscles, lmt_funcs, vmt_funcs must have same length")
        self.muscles = muscles
        self.lmt_funcs = lmt_funcs
        self.vmt_funcs = vmt_funcs

    def step(self, activations: List[float], q: float, qd: float, dt: float) -> float:
        if len(activations) != len(self.muscles):
            raise ValueError("activations length must match number of muscles")

        torque = 0.0
        for u, m, lmt_f, vmt_f in zip(activations, self.muscles, self.lmt_funcs, self.vmt_funcs):
            l_mt = lmt_f(q)
            v_mt = vmt_f(q, qd)
            F = m.step(u, l_mt, v_mt, dt)
            torque += F * m.p.moment_arm
        return torque
