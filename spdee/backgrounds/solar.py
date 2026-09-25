"""spdee.backgrounds.solar — adapter for tabulated solar interior data.

Provides `Background.Scalar` and `Background.Vector` instances backed by the
Model S (JCD) interior table and HMI helioseismic differential-rotation
inversions. Constructed once for a given geometry; interpolates to whatever
spdee mesh is passed at solve time.

Data files (in `data_dir`)
--------------------------
* `model_s_corrected.txt` — Model S radial structure table (2482 points,
  r from 0 to 1.00071 R_sun): density, pressure, temperature, Γ₁ and the
  buoyancy column pN2 = r N²/g.
* `HMI/rot2d.hmiv72d.ave` + `HMI/rmesh.orig` — raw HMI Ω(r, θ) inversion
  (151 radii × 49 northern latitudes, nHz), the preferred rotation source.
* `omega_fit.npy` — pre-fitted Ω(r, θ) samples, shape (Nt, Nr) = (512, 256),
  on Chebyshev nodes in r ∈ [0.5, 1.0] × Legendre roots in θ, with a tanh
  blend to uniform rotation below r ≈ 0.55 baked in. Used only when the raw
  tables are absent.

Units
-----
By default all backgrounds are CGS with frequencies in rad/s. With
`nondimensional=True` each background is divided by its anchor from the
`normalization` dict (rho/rho_max, Ω/Ω_max, ...).

Geometry
--------
  geometry = ('shell', r0, r1)
      Annulus r ∈ [r0, r1] in R_sun units. Both must lie within the Model S
      extent [0, 1.00071]; radial profiles raise ValueError when evaluated
      outside the data range.
  geometry = ('ball', r1)
      Full ball of outer radius r1 ≤ 1.00071.

Ω(r, θ) is clamped to its data extent in θ and above the outer data radius,
returns the uniform core rate below the inner data radius (only the
pre-fitted samples have one, at r = 0.5), and with `uniform_core_blend=True`
is blended smoothly to that core rate below `blend_radius`.

Public API
----------
SolarBackgrounds(data_dir, geometry, nondimensional=False, ...)

Properties (spdee Background fields):
    rho          — density ρ₀(r)              [g/cm³] or dimensionless
    pressure     — pressure p₀(r)             [dyn/cm²] or dimensionless
    sound_speed  — adiabatic c_s(r)           [cm/s] or dimensionless
    gravity      — |g(r)| scalar              [cm/s²] or dimensionless
    gamma1       — adiabatic exponent Γ₁(r)   [dimensionless either way]
    temperature  — T₀(r)                      [K] or dimensionless
    cp           — specific heat c_p(r)       [erg/g/K] or dimensionless
    N2           — buoyancy frequency² N²(r)  [rad²/s²] or dimensionless
    grad_s0      — entropy-gradient proxy (N²/g) r̂ [1/cm] or dimensionless
    grad_rho     — density gradient (dρ₀/dr) r̂ [g/cm³ per R_sun] or dimensionless
    g_vec        — gravity vector −|g(r)| r̂   [cm/s²] or dimensionless
    Omega        — rotation rate Ω(r,θ)       [rad/s] or dimensionless
    U0           — azimuthal flow Ω r sinθ φ̂  [R_sun·rad/s] or dimensionless

`.normalization` — dict of dimensional anchors used to nondimensionalize.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.interpolate import (
    PchipInterpolator, RectBivariateSpline, RegularGridInterpolator,
)
from scipy.special import roots_legendre

from spdee.ir.fields import Background
from spdee.ir.backgrounds import NumericComponent


G_NEWTON = 6.67e-8  # gravitational constant, CGS


# ---------------------------------------------------------------------------
# Model S parser
# ---------------------------------------------------------------------------

def _parse_model_s(path: Path) -> dict:
    """Parse the Model S structure file into 1D radial arrays.

    Header line 4 holds the point count, line 5 the total mass, radius and
    luminosity (16-character fields); from line 8 each point occupies five
    lines of five 16-character floats. Column 0 is r, 1 is ln(M(r)/M), 2 is T,
    3 is p, 4 is ρ, 9 is Γ₁ and 14 is pN2 = r N²/g. Model S lists the surface
    first; the returned arrays are sorted so that r increases. Returns a dict
    of arrays plus the anchors `Mass`, `Radius_sun`, `Ls`.
    """
    with open(path) as f:
        data = f.readlines()
    npts = int(data[4][6:10])
    Mass = float(data[5][0:16])
    Radius_sun = float(data[5][16:32])
    Ls = float(data[5][32:48])
    radius = np.zeros(npts)
    lnq = np.zeros(npts)
    temperature = np.zeros(npts)
    pressure = np.zeros(npts)
    density = np.zeros(npts)
    gamma1 = np.zeros(npts)
    pN2 = np.zeros(npts)
    darray = np.zeros(25)
    tmp = np.zeros(5)
    ll = 8
    for ii in range(npts):
        for jj in range(5):
            for kk in range(5):
                tmp[kk] = float(data[ll][16 * kk:16 * (kk + 1)])
            darray[5 * jj:5 * (jj + 1)] = tmp
            ll += 1
        radius[ii] = darray[0]
        lnq[ii] = darray[1]
        temperature[ii] = darray[2]
        pressure[ii] = darray[3]
        density[ii] = darray[4]
        gamma1[ii] = darray[9]
        pN2[ii] = darray[14]
    r_norm = radius / Radius_sun

    # Derived quantities (CGS): c_s² = Γ₁ p/ρ, g = G M(r)/r² with
    # M(r) = Mass exp(lnq), N² = g pN2/r. Non-finite values (0/0 at r = 0)
    # are set to 0.
    sound_speed = np.sqrt(gamma1 * pressure / density)
    with np.errstate(divide='ignore', invalid='ignore'):
        gravity = np.exp(lnq) * Mass * G_NEWTON / radius**2
    gravity[~np.isfinite(gravity)] = 0.0
    with np.errstate(divide='ignore', invalid='ignore'):
        N2 = gravity * pN2 / radius
    N2[~np.isfinite(N2)] = 0.0

    # Sort to increasing r.
    order = np.argsort(r_norm)
    r_norm = r_norm[order]
    density = density[order]
    pressure = pressure[order]
    gamma1 = gamma1[order]
    sound_speed = sound_speed[order]
    gravity = gravity[order]
    N2 = N2[order]
    temperature = temperature[order]

    # The innermost Model S point has r ≈ 1e-49. Snap it to exactly 0 so ball
    # geometries can query r = 0, and set gravity and N² there to their
    # physical limits (M ∝ r³ near the origin, so g → 0).
    if r_norm[0] < 1e-10:
        r_norm[0] = 0.0
        gravity[0] = 0.0
        N2[0] = 0.0

    # Drop consecutive duplicate radii (PCHIP needs strictly increasing r).
    keep = np.concatenate([[True], np.diff(r_norm) > 1e-15])
    r_norm = r_norm[keep]
    density = density[keep]
    pressure = pressure[keep]
    gamma1 = gamma1[keep]
    sound_speed = sound_speed[keep]
    gravity = gravity[keep]
    N2 = N2[keep]
    temperature = temperature[keep]

    return {
        "r": r_norm,
        "density": density,
        "pressure": pressure,
        "gamma1": gamma1,
        "sound_speed": sound_speed,
        "gravity": gravity,
        "N2": N2,
        "temperature": temperature,
        "Mass": Mass,
        "Radius_sun": Radius_sun,
        "Ls": Ls,
    }


# ---------------------------------------------------------------------------
# Ω(r, θ) loaders
# ---------------------------------------------------------------------------

def _load_omega(
    data_dir: Path,
    uniform_core_blend: bool = False,
    blend_radius: Optional[float] = None,
    blend_steepness: float = 200.0,
) -> dict:
    """Load the HMI Ω(r, θ) inversion and return an evaluator dict.

    Sources, in priority order:

      1. Raw HMI tables ``HMI/rot2d.hmiv72d.ave`` (151 × 49) and
         ``HMI/rmesh.orig`` (601 radii, every 4th used). A bicubic
         `RectBivariateSpline` is fitted directly to these values.
      2. Pre-fitted samples ``omega_fit.npy`` (Chebyshev × Legendre,
         256 × 512), interpolated bilinearly. Coarser; used only when the raw
         tables are absent.

    The raw table covers the northern hemisphere only (colatitude 0 to π/2);
    the inversion assumes north-south symmetry, so the data are mirrored
    across the equator to cover θ ∈ [0, π].

    With ``uniform_core_blend=True`` the profile is blended toward a uniform
    core rate below the transition radius:

        Ω = Ω_data (1 + tanh(s (r − r_c)))/2 + Ω_core (1 − tanh(s (r − r_c)))/2

    with r_c = ``blend_radius`` and s = ``blend_steepness`` (transition width
    ≈ 1/s). When ``blend_radius`` is None, r_c is detected from the data as the
    largest r where the sin θ-weighted latitudinal mean Ω̄(r) crosses Ω_core
    (the tachocline, r ≈ 0.71 for HMI); a profile with no crossing raises.
    Both parameters are ignored when the blend is off, and the detection is
    then skipped.

    Every latitudinal mean here (Ω_core and Ω̄) is the solid-angle average
    over θ ∈ [0, π], i.e. weighted by sin θ.

    Below the radial data extent both sources return ``Omega_core_uniform``
    (for the raw tables the extent starts at r = 0, so this only acts on the
    pre-fitted samples, which start at r = 0.5); above it and outside
    [0, π] in θ the boundary value is returned.

    Returns a dict with:
      - ``evaluator``: ``callable(R, T) -> np.ndarray`` (any shape; Ω in rad/s).
      - ``r_min``, ``r_max``: data extents in R_sun.
      - ``Omega_core_uniform``: uniform core rate (rad/s) used by the blend.
      - ``Omega_max``: peak |Ω| over the data (the dimensional anchor).
      - ``blend_radius_used``: the transition radius, None when the blend is
        off.
      - ``source``: 'raw' or 'prefit'.
    """
    hmi_data_path = data_dir / "HMI" / "rot2d.hmiv72d.ave"
    hmi_rmesh_path = data_dir / "HMI" / "rmesh.orig"
    npy_path = data_dir / "omega_fit.npy"
    if hmi_data_path.exists() and hmi_rmesh_path.exists():
        return _build_omega_from_raw(
            hmi_data_path, hmi_rmesh_path,
            uniform_core_blend=uniform_core_blend,
            blend_radius=blend_radius,
            blend_steepness=blend_steepness,
        )
    elif npy_path.exists():
        return _build_omega_from_prefit(
            npy_path,
            uniform_core_blend=uniform_core_blend,
            blend_radius=blend_radius,
            blend_steepness=blend_steepness,
        )
    else:
        raise FileNotFoundError(
            f"No Ω data found under {data_dir}. Expected either "
            f"HMI/rot2d.hmiv72d.ave + HMI/rmesh.orig (raw, preferred) "
            f"or omega_fit.npy (pre-fitted, fallback)."
        )


def _sin_weighted_mean(values, theta, axis=-1):
    """Solid-angle mean over colatitude: Σ f sin θ / Σ sin θ along ``axis``."""
    w = np.sin(theta)
    shape = [1] * values.ndim
    shape[axis] = w.size
    w = w.reshape(shape)
    return (values * w).sum(axis=axis) / w.sum(axis=axis)


def _auto_blend_radius(
    omega_evaluator,
    omega_core: float,
    r_min: float = 0.05,
    r_max: float = 0.99,
    n_r: int = 2000,
    n_theta: int = 41,
) -> float:
    """Largest radius in [r_min, r_max] where the sin θ-weighted latitudinal
    mean Ω̄(r) = ⟨Ω(r, θ)⟩_θ crosses ``omega_core`` (the tachocline).

    The mean is over a uniform θ grid; the crossing is refined by linear
    interpolation. Raises ValueError when Ω̄ never crosses ``omega_core`` on
    the scanned extent (pass ``blend_radius`` explicitly then).
    """
    r_grid = np.linspace(r_min, r_max, n_r)
    theta_grid = np.linspace(0.01, np.pi - 0.01, n_theta)
    R, T = np.meshgrid(r_grid, theta_grid, indexing='ij')
    Omega = omega_evaluator(R, T)
    Omega_bar = _sin_weighted_mean(Omega, theta_grid, axis=1)
    diff = Omega_bar - omega_core
    sign_change = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_change) == 0:
        nhz = 1.0 / (2.0 * np.pi * 1e-9)
        raise ValueError(
            f"uniform-core blend: the latitudinal mean rotation rate never "
            f"crosses the core rate {omega_core * nhz:.2f} nHz on "
            f"r in [{r_min}, {r_max}] (mean rate spans "
            f"{Omega_bar.min() * nhz:.2f}..{Omega_bar.max() * nhz:.2f} nHz); "
            f"pass blend_radius explicitly.")
    i = sign_change[-1]
    r_lo, r_hi = r_grid[i], r_grid[i + 1]
    d_lo, d_hi = diff[i], diff[i + 1]
    return float(r_lo + (r_hi - r_lo) * (-d_lo) / (d_hi - d_lo))


def _build_omega_from_raw(
    data_path: Path,
    rmesh_path: Path,
    uniform_core_blend: bool = False,
    blend_radius: Optional[float] = None,
    blend_steepness: float = 200.0,
) -> dict:
    """Ω evaluator dict from the raw HMI tables (see `_load_omega`).

    ``rot2d.hmiv72d.ave`` is an (nr, nth) = (151, 49) array in nHz; rows are
    radii, columns are latitudes from the pole (90°) to the equator (0°) in
    steps of 15/8 degrees. ``rmesh.orig`` holds 601 radii of which every 4th
    matches the data rows. The northern data are mirrored across the equator
    (equator row not duplicated) and a bicubic spline is fitted on the full
    (r, θ) grid. The uniform core rate is the sin θ-weighted latitudinal mean
    at the radius nearest r = 0.5.
    """
    data_nHz = np.genfromtxt(data_path)          # (nr, nth)
    radius_full = np.genfromtxt(rmesh_path)
    radius_sub = radius_full[::4]                # (nr,) in R_sun units
    nr, nth = data_nHz.shape
    if radius_sub.shape[0] != nr:
        raise ValueError(
            f"HMI rmesh and rot2d row counts disagree: "
            f"{radius_sub.shape[0]} vs {nr}"
        )

    # Northern colatitudes θ_n = π/2 − lat, lat = 90° → 0°; mirror to (π/2, π].
    lat_deg = 90.0 - np.arange(nth) * 15.0 / 8.0
    theta_north = np.pi / 2.0 - lat_deg * np.pi / 180.0
    theta_south = np.pi - theta_north[:-1][::-1]
    theta_full = np.concatenate([theta_north, theta_south])
    data_full = np.concatenate(
        [data_nHz, data_nHz[:, :-1][:, ::-1]], axis=1,
    )
    order = np.argsort(radius_sub)
    radius_sub = radius_sub[order]
    data_full = data_full[order, :]
    spline = RectBivariateSpline(
        radius_sub, theta_full, data_full,
        kx=3, ky=3, s=0,
    )
    nHz_to_rad_s = 2.0 * np.pi * 1e-9

    # Uniform-core asymptote: solid-angle mean at the grid radius nearest 0.5.
    i_05 = int(np.argmin(np.abs(radius_sub - 0.5)))
    Omega_core_uniform_nHz = float(_sin_weighted_mean(data_full[i_05, :], theta_full))
    Omega_core_uniform = Omega_core_uniform_nHz * nHz_to_rad_s
    Omega_max = float(np.abs(data_full).max()) * nHz_to_rad_s

    def _raw_eval(R, T):
        R_c = np.clip(R, radius_sub[0], radius_sub[-1])
        T_c = np.clip(T, 0.0, np.pi)
        vals = spline.ev(R_c.ravel(), T_c.ravel()).reshape(R.shape) * nHz_to_rad_s
        return np.where(R < radius_sub[0], Omega_core_uniform, vals)

    if not uniform_core_blend:
        blend_radius_used = None
    elif blend_radius is None:
        blend_radius_used = _auto_blend_radius(_raw_eval, Omega_core_uniform)
    else:
        blend_radius_used = float(blend_radius)

    def evaluator(R, T):
        R_arr = np.asarray(R, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        vals = _raw_eval(R_arr, T_arr)
        if uniform_core_blend:
            s = blend_steepness * (R_arr - blend_radius_used)
            weight_data = (1.0 + np.tanh(s)) / 2.0
            weight_core = 1.0 - weight_data
            vals = weight_data * vals + weight_core * Omega_core_uniform
        return vals

    return {
        "evaluator": evaluator,
        "r_min": float(radius_sub[0]),
        "r_max": float(radius_sub[-1]),
        "Omega_core_uniform": Omega_core_uniform,
        "Omega_max": Omega_max,
        "blend_radius_used": blend_radius_used,
        "source": "raw",
    }


def _build_omega_from_prefit(
    path: Path,
    uniform_core_blend: bool = False,
    blend_radius: Optional[float] = None,
    blend_steepness: float = 200.0,
) -> dict:
    """Ω evaluator dict from the pre-fitted samples ``omega_fit.npy``.

    The file already contains a tanh blend to uniform rotation (r_c = 0.55,
    steepness 200), which cannot be undone; ``Omega_core_uniform`` is the
    Gauss-Legendre (solid-angle) mean of the innermost column (r = 0.5).
    Below r = 0.5 the evaluator returns ``Omega_core_uniform``. With
    ``uniform_core_blend=True`` a second blend with the requested radius and
    steepness is applied on top.
    """
    Om_nHz = np.load(path)                       # shape (Nt, Nr)
    Nt, Nr = Om_nHz.shape
    r_min, r_max = 0.5, 1.0
    r_cheb = np.flip(
        0.5 * (r_max + r_min) + 0.5 * (r_max - r_min)
        * np.cos(np.pi * (2 * np.arange(Nr) + 1) / (2 * Nr))
    )
    mu_leg, w_leg = roots_legendre(Nt)
    theta_leg = np.arccos(mu_leg)
    order = np.argsort(theta_leg)
    theta_leg = theta_leg[order]
    w_leg = w_leg[order]
    Om_nHz = Om_nHz[order, :]
    nHz_to_rad_s = 2.0 * np.pi * 1e-9
    Om_rad_s = Om_nHz * nHz_to_rad_s
    Omega_core_uniform = float((w_leg * Om_rad_s[:, 0]).sum() / w_leg.sum())
    Omega_max = float(np.abs(Om_rad_s).max())
    interp = RegularGridInterpolator(
        (theta_leg, r_cheb), Om_rad_s,
        method='linear', bounds_error=False, fill_value=None,
    )

    def _raw_eval(R, T):
        R_c = np.clip(R, r_min, r_max)
        T_c = np.clip(T, theta_leg.min(), theta_leg.max())
        pts = np.stack([T_c.ravel(), R_c.ravel()], axis=-1)
        vals = interp(pts).reshape(R.shape)
        return np.where(R < r_min, Omega_core_uniform, vals)

    # Auto-detection scans [0.505, 0.99] since the samples start at r = 0.5.
    if not uniform_core_blend:
        blend_radius_used = None
    elif blend_radius is None:
        blend_radius_used = _auto_blend_radius(
            _raw_eval, Omega_core_uniform,
            r_min=r_min + 0.005, r_max=r_max - 0.01,
        )
    else:
        blend_radius_used = float(blend_radius)

    def evaluator(R, T):
        R_arr = np.asarray(R, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        vals = _raw_eval(R_arr, T_arr)
        if uniform_core_blend:
            s = blend_steepness * (R_arr - blend_radius_used)
            w_data = (1.0 + np.tanh(s)) / 2.0
            vals = w_data * vals + (1.0 - w_data) * Omega_core_uniform
        return vals

    return {
        "evaluator": evaluator,
        "r_min": r_min,
        "r_max": r_max,
        "Omega_core_uniform": Omega_core_uniform,
        "Omega_max": Omega_max,
        "blend_radius_used": blend_radius_used,
        "source": "prefit",
    }


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Geometry:
    """Validated geometry: `kind` 'shell' or 'ball'; `r_min` (0 for a ball)
    and `r_max` in R_sun units."""
    kind: str
    r_min: float
    r_max: float

    @classmethod
    def parse(cls, geom) -> "_Geometry":
        """Parse ('shell', r0, r1) / {'shell': (r0, r1)} or
        ('ball', r1) / {'ball': r1}."""
        if isinstance(geom, dict):
            if 'shell' in geom:
                r0, r1 = geom['shell']
                return cls('shell', float(r0), float(r1))
            if 'ball' in geom:
                return cls('ball', 0.0, float(geom['ball']))
            raise ValueError(f"Unknown geometry dict keys: {list(geom)}")
        if isinstance(geom, (tuple, list)):
            if len(geom) == 3 and geom[0] == 'shell':
                return cls('shell', float(geom[1]), float(geom[2]))
            if len(geom) == 2 and geom[0] == 'ball':
                return cls('ball', 0.0, float(geom[1]))
        raise ValueError(
            f"Unrecognized geometry spec: {geom!r}. Use ('shell', r0, r1) "
            f"or ('ball', r1)."
        )


# ---------------------------------------------------------------------------
# Interpolator builders
# ---------------------------------------------------------------------------

def _build_radial_interpolator(r_data, values, geom: _Geometry, label: str):
    """1D PCHIP interpolator of a radial Model S profile as `f(R, T)`.

    PCHIP is monotone and non-oscillatory, which suits sharp features such as
    the sound-speed transition at the base of the convection zone. The
    geometry's [r_min, r_max] must lie within the data range; a query outside
    the data range raises ValueError. On a ball the profiles are even in r,
    so a query at r < 0 (a finite-difference probe across the origin)
    evaluates the reflection |r|.
    """
    r_data_min, r_data_max = float(r_data.min()), float(r_data.max())
    if geom.r_max > r_data_max + 1e-9:
        raise ValueError(
            f"Solar background {label!r}: requested r_max = {geom.r_max} "
            f"exceeds Model S data extent ({r_data_max:.5f} R_sun)."
        )
    if geom.r_min < r_data_min - 1e-9:
        raise ValueError(
            f"Solar background {label!r}: requested r_min = {geom.r_min} "
            f"below Model S data extent ({r_data_min:.5f} R_sun)."
        )
    interp = PchipInterpolator(r_data, values, extrapolate=False)
    even = geom.kind == "ball"

    def f(R, T):
        R = np.abs(R) if even else np.asarray(R)
        out = interp(R)
        if np.any(np.isnan(out)):
            mask = np.isnan(out)
            r_bad = R[mask]
            raise ValueError(
                f"Background {label!r}: query at r = {r_bad.min():.5f}..."
                f"{r_bad.max():.5f} is outside data range "
                f"[{r_data_min:.5f}, {r_data_max:.5f}]"
            )
        return out
    return f


def _build_omega_interpolator(omega_data: dict, geom: _Geometry):
    """Return the Ω evaluator of `_load_omega` after checking that the
    geometry's r_max lies within the data extent. There is no r_min check:
    below the data the evaluator clamps (or blends to the uniform core)."""
    r_max_data = omega_data["r_max"]
    if geom.r_max > r_max_data + 1e-9:
        raise ValueError(
            f"Solar background 'Omega': requested r_max = {geom.r_max} "
            f"exceeds HMI data extent ({r_max_data:.5f} R_sun)."
        )
    return omega_data["evaluator"]


# ---------------------------------------------------------------------------
# Adapter class
# ---------------------------------------------------------------------------

class SolarBackgrounds:
    """Solar interior backgrounds for spdee.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``model_s_corrected.txt`` and the rotation data
        (raw ``HMI/rot2d.hmiv72d.ave`` + ``HMI/rmesh.orig`` preferred,
        ``omega_fit.npy`` as fallback).
    geometry : tuple or dict
        ``('shell', r0, r1)`` or ``('ball', r1)``, in R_sun units.
    nondimensional : bool, default False
        Divide each background by its anchor from ``.normalization``.
    uniform_core_blend : bool, default False
        Blend Ω toward the uniform core rate (the sin θ-weighted latitudinal
        mean at r = 0.5) below ``blend_radius`` with a tanh transition.
    blend_radius : float or None, default None
        Center of the tanh transition in R_sun. None selects the tachocline
        found from the data (r ≈ 0.71 for HMI; a profile whose mean never
        crosses the core rate raises). Ignored when the blend is off, and
        ``blend_radius_resolved`` is then None.
    blend_steepness : float, default 200.0
        s in tanh(s (r − r_c)); 200 gives a transition ≈ 0.005 R_sun wide.
        Ignored when the blend is off.

    Examples
    --------
    >>> bg = SolarBackgrounds("data/solar", ('shell', 0.715, 0.99))
    >>> rho_field = bg.rho           # Background.Scalar
    >>> g_field = bg.g_vec           # Background.Vector (= -|g| r̂)
    >>> bg = SolarBackgrounds("data/solar", ('ball', 0.5))
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        geometry,
        nondimensional: bool = False,
        uniform_core_blend: bool = False,
        blend_radius: Optional[float] = None,
        blend_steepness: float = 200.0,
    ):
        self.data_dir = Path(data_dir)
        self.geom = _Geometry.parse(geometry)
        self.nondimensional = bool(nondimensional)
        self.uniform_core_blend = bool(uniform_core_blend)
        self.blend_radius = (
            None if blend_radius is None else float(blend_radius)
        )
        self.blend_steepness = float(blend_steepness)
        model_s_path = self.data_dir / "model_s_corrected.txt"
        if not model_s_path.exists():
            raise FileNotFoundError(
                f"Model S file not found: {model_s_path}"
            )
        self._modelS = _parse_model_s(model_s_path)
        self._omega = _load_omega(
            self.data_dir,
            uniform_core_blend=self.uniform_core_blend,
            blend_radius=self.blend_radius,
            blend_steepness=self.blend_steepness,
        )
        self.blend_radius_resolved = self._omega["blend_radius_used"]

        # Dimensional anchors: the maximum of each PCHIP-interpolated profile
        # over the Model S points inside the geometry plus both endpoints, so
        # nondimensional values are ≤ 1 wherever a mesh can query.
        r_data = self._modelS["r"]
        r_fine = np.concatenate([
            np.array([self.geom.r_min, self.geom.r_max]),
            r_data[(r_data >= self.geom.r_min) & (r_data <= self.geom.r_max)],
        ])
        r_fine = np.unique(r_fine)
        if r_fine.size < 2:
            raise ValueError(
                f"Geometry {self.geom} contains no Model S grid points; "
                f"r data ∈ [{r_data.min():.5f}, {r_data.max():.5f}]"
            )
        rho_max = float(PchipInterpolator(r_data, self._modelS["density"])(r_fine).max())
        p_max = float(PchipInterpolator(r_data, self._modelS["pressure"])(r_fine).max())
        cs_max = float(PchipInterpolator(r_data, self._modelS["sound_speed"])(r_fine).max())
        g_max = float(PchipInterpolator(r_data, self._modelS["gravity"])(r_fine).max())
        T_max = float(PchipInterpolator(r_data, self._modelS["temperature"])(r_fine).max())
        N2_vals_fine = PchipInterpolator(r_data, self._modelS["N2"])(r_fine)
        N2_scale = float(np.max(np.abs(N2_vals_fine)))
        if N2_scale == 0.0:
            N2_scale = 1.0
        # c_p = c_s² / ((Γ₁ − 1) T) evaluated on the same fine grid.
        gamma_fine = PchipInterpolator(r_data, self._modelS["gamma1"])(r_fine)
        T_fine = PchipInterpolator(r_data, self._modelS["temperature"])(r_fine)
        cs_fine = PchipInterpolator(r_data, self._modelS["sound_speed"])(r_fine)
        cp_vals = cs_fine**2 / ((gamma_fine - 1.0) * T_fine + 1e-30)
        cp_max = float(cp_vals.max())
        Omega_max = self._omega["Omega_max"]
        self.normalization = {
            "rho0": rho_max,             # g/cm³
            "p0": p_max,                  # dyn/cm²
            "cs0": cs_max,                # cm/s
            "g0": g_max,                  # cm/s²
            "T0": T_max,                  # K
            "cp0": cp_max,                # erg/g/K
            "N20": N2_scale,              # rad²/s²
            "Omega0": Omega_max,          # rad/s
            "R_sun": self._modelS["Radius_sun"],  # cm
            "Mass_sun": self._modelS["Mass"],     # g
        }

    # ----- internal helpers -----

    def _scalar_radial(self, key: str, anchor_key: Optional[str]) -> Background:
        """Background.Scalar from a 1D radial Model S column."""
        r_data = self._modelS["r"]
        values = self._modelS[key]
        if self.nondimensional and anchor_key is not None:
            anchor = self.normalization[anchor_key]
            if anchor != 0.0:
                values = values / anchor
        f = _build_radial_interpolator(r_data, values, self.geom, key)
        nc = NumericComponent(f=f, description=f"solar_{key}")
        return Background.Scalar(f"solar_{key}", nc)

    # ----- scalar properties -----

    @property
    def rho(self) -> Background:
        """Density ρ₀(r). [g/cm³] or dimensionless."""
        return self._scalar_radial("density", "rho0")

    @property
    def pressure(self) -> Background:
        """Pressure p₀(r). [dyn/cm²] or dimensionless."""
        return self._scalar_radial("pressure", "p0")

    @property
    def sound_speed(self) -> Background:
        """Adiabatic sound speed c_s(r). [cm/s] or dimensionless."""
        return self._scalar_radial("sound_speed", "cs0")

    @property
    def gravity(self) -> Background:
        """Gravity magnitude |g(r)|. [cm/s²] or dimensionless."""
        return self._scalar_radial("gravity", "g0")

    @property
    def gamma1(self) -> Background:
        """Adiabatic exponent Γ₁(r), dimensionless (no anchor)."""
        return self._scalar_radial("gamma1", None)

    @property
    def temperature(self) -> Background:
        """Temperature T₀(r). [K] or dimensionless."""
        return self._scalar_radial("temperature", "T0")

    @property
    def cp(self) -> Background:
        """Specific heat c_p(r) = c_s² / ((Γ₁ − 1) T). [erg/g/K] or
        dimensionless (anchor cp0 = max c_p).

        Computed on the Model S grid from c_s, Γ₁ and T, then PCHIP
        interpolated; points with Γ₁ ≤ 1 or T ≤ 0 are set to 0.
        """
        r_data = self._modelS["r"]
        cs = self._modelS["sound_speed"]
        gamma = self._modelS["gamma1"]
        T = self._modelS["temperature"]
        with np.errstate(divide='ignore', invalid='ignore'):
            cp_vals = np.where(
                (gamma > 1.0) & (T > 0),
                cs**2 / ((gamma - 1.0) * T + 1e-30),
                0.0,
            )
        if self.nondimensional:
            anchor = self.normalization["cp0"]
            if anchor != 0.0:
                cp_vals = cp_vals / anchor
        f = _build_radial_interpolator(r_data, cp_vals, self.geom, "cp")
        nc = NumericComponent(f=f, description="solar_cp")
        return Background.Scalar("solar_cp", nc)

    @property
    def N2(self) -> Background:
        """Buoyancy frequency squared N²(r) = g pN2 / r. [rad²/s²] or
        dimensionless (anchor N20 = max |N²|)."""
        return self._scalar_radial("N2", "N20")

    # ----- vector properties -----

    @property
    def grad_s0(self) -> Background:
        """Entropy-gradient proxy (N²/g) r̂ for the linearized energy equation
        ∂_t s + Dot(grad_s0, v) = κ ∇²s. Positive where stably stratified,
        negative in the convection zone; 0 where g = 0.

        In nondimensional mode divided by N20/g0.
        """
        r_data = self._modelS["r"]
        N2 = self._modelS["N2"]
        g = self._modelS["gravity"]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(g > 0, N2 / g, 0.0)
        if self.nondimensional:
            anchor = self.normalization["N20"] / max(
                self.normalization["g0"], 1e-30,
            )
            ratio = ratio / max(anchor, 1e-30)
        f_r = _build_radial_interpolator(r_data, ratio, self.geom, "grad_s0_r")
        nc_r = NumericComponent(f=f_r, description="solar_grad_s0_r")
        return Background.Vector(
            "solar_grad_s0", v_r=nc_r,
            v_theta=None, v_phi=None,
        )

    @property
    def grad_rho(self) -> Background:
        """Density gradient (dρ₀/dr) r̂ for the linearized continuity equation
        ∂_t ρ₁ + ρ₀ Div(v) + Dot(grad_rho, v) = 0.

        dρ₀/dr is a central finite difference of the Model S table in r
        (R_sun units), so the values are g/cm³ per R_sun; in nondimensional
        mode divided by rho0.
        """
        r_data = self._modelS["r"]
        rho = self._modelS["density"]
        drho_dr = np.gradient(rho, r_data)
        if self.nondimensional:
            anchor = self.normalization["rho0"]
            if anchor != 0.0:
                drho_dr = drho_dr / anchor
        f_r = _build_radial_interpolator(
            r_data, drho_dr, self.geom, "grad_rho_r",
        )
        nc_r = NumericComponent(f=f_r, description="solar_grad_rho_r")
        return Background.Vector(
            "solar_grad_rho", v_r=nc_r,
            v_theta=None, v_phi=None,
        )

    @property
    def g_vec(self) -> Background:
        """Gravity vector −|g(r)| r̂ (pointing inward). [cm/s²] or
        dimensionless."""
        r_data = self._modelS["r"]
        g = self._modelS["gravity"]
        values = -g
        if self.nondimensional:
            anchor = self.normalization["g0"]
            if anchor != 0.0:
                values = values / anchor
        f_r = _build_radial_interpolator(r_data, values, self.geom, "g_vec_r")
        nc_r = NumericComponent(f=f_r, description="solar_g_vec_r")
        return Background.Vector(
            "solar_g_vec", v_r=nc_r,
            v_theta=None, v_phi=None,
        )

    # ----- rotation -----

    @property
    def Omega(self) -> Background:
        """Differential rotation rate Ω(r, θ) from the HMI inversion. [rad/s]
        or dimensionless (anchor Omega0 = max |Ω|).

        Outside the data extent the value is clamped to the boundary; with
        `uniform_core_blend` it is blended to the uniform core rate below
        `blend_radius_resolved`.
        """
        f = _build_omega_interpolator(self._omega, self.geom)
        if self.nondimensional:
            anchor = self.normalization["Omega0"]
            if anchor != 0.0:
                inner = f
                def f_nd(R, T):
                    return inner(R, T) / anchor
                f = f_nd
        nc = NumericComponent(f=f, description="solar_Omega")
        return Background.Scalar("solar_Omega", nc)

    @property
    def U0(self) -> Background:
        """Azimuthal background flow U₀ = Ω(r, θ) r sin θ φ̂, with r in R_sun
        units, so U₀ is in R_sun·rad/s (or Ω/Omega0 · r sin θ when
        nondimensional).

        Linearizing about this flow instead of solid-body rotation brings
        (U₀·∇)v and (v·∇)U₀ into the momentum equation.
        """
        f_Omega = _build_omega_interpolator(self._omega, self.geom)
        if self.nondimensional:
            anchor = self.normalization["Omega0"]
            if anchor != 0.0:
                inner = f_Omega
                def f_Omega_nd(R, T):
                    return inner(R, T) / anchor
                f_Omega = f_Omega_nd

        def f_phi(R, T):
            return f_Omega(R, T) * R * np.sin(T)

        nc_phi = NumericComponent(f=f_phi, description="solar_U0_phi")
        return Background.Vector(
            "solar_U0",
            v_r=None, v_theta=None, v_phi=nc_phi,
        )
