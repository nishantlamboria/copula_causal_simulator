"""Synthetic bivariate copula models with known ground truth.

This module creates explicitly specified pair-copula models and samples
pseudo-observations from them. It is used to test whether the project's
calibration pipeline can recover known copula families and parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sin
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray
import pyvinecopulib as pv


FloatArray = NDArray[np.float64]


# Canonical names used in configurations and result tables.
_FAMILY_ALIASES = {
    "indep": "indep",
    "independence": "indep",
    "gaussian": "gaussian",
    "normal": "gaussian",
    "student": "student",
    "student_t": "student",
    "student-t": "student",
    "t": "student",
    "clayton": "clayton",
    "gumbel": "gumbel",
    "frank": "frank",
}


@dataclass(frozen=True)
class PairCopulaSimulation:
    """Result of one synthetic pair-copula simulation.

    Attributes
    ----------
    family:
        Canonical name of the true generating family.
    tau:
        True population Kendall's tau.
    rotation:
        Rotation of the true copula. The first validation stage uses
        unrotated copulas only.
    seed:
        Seed passed to pyvinecopulib.
    model:
        True pyvinecopulib bivariate copula model.
    data:
        Simulated pseudo-observations with shape ``(n, 2)``.
    """

    family: str
    tau: float
    rotation: int
    seed: int
    model: pv.Bicop
    data: FloatArray

    @property
    def parameters(self) -> FloatArray:
        """Return a defensive copy of the true parameter matrix."""
        return np.asarray(
            self.model.parameters,
            dtype=float,
        ).copy()


def normalize_family_name(family: str) -> str:
    """Return the canonical name for a supported copula family."""
    if not isinstance(family, str):
        raise TypeError(
            "family must be a string, "
            f"received {type(family).__name__}."
        )

    key = family.strip().lower()

    try:
        return _FAMILY_ALIASES[key]
    except KeyError as error:
        supported = ", ".join(
            [
                "indep",
                "gaussian",
                "student",
                "clayton",
                "gumbel",
                "frank",
            ]
        )
        raise ValueError(
            f"Unsupported copula family {family!r}. "
            f"Supported families are: {supported}."
        ) from error


def _family_enum(family: str):
    """Resolve a canonical family name to the installed API enum.

    The fallback supports older pyvinecopulib versions that exposed
    family constants directly on the package namespace.
    """
    canonical = normalize_family_name(family)

    if hasattr(pv, "BicopFamily"):
        return getattr(
            pv.BicopFamily,
            canonical,
        )

    try:
        return getattr(pv, canonical)
    except AttributeError as error:
        raise RuntimeError(
            f"The installed pyvinecopulib version does not expose "
            f"the family {canonical!r}."
        ) from error


def _construct_bicop(
    family: str,
    rotation: int = 0,
    parameters: FloatArray | None = None,
) -> pv.Bicop:
    """Construct a Bicop while supporting multiple library versions."""
    family_value = _family_enum(family)

    if parameters is None:
        parameters = np.empty(
            (0, 0),
            dtype=float,
            order="F",
        )
    else:
        parameters = np.asfortranarray(
            parameters,
            dtype=float,
        )

    # Newer pyvinecopulib API.
    if hasattr(pv.Bicop, "from_family"):
        return pv.Bicop.from_family(
            family=family_value,
            rotation=rotation,
            parameters=parameters,
        )

    # Compatibility with versions using the direct constructor.
    return pv.Bicop(
        family_value,
        rotation,
        parameters,
    )


def _validate_tau(
    family: str,
    tau: float,
) -> float:
    """Validate the requested population Kendall's tau."""
    if not isinstance(tau, Real):
        raise TypeError(
            "tau must be a real number, "
            f"received {type(tau).__name__}."
        )

    tau = float(tau)

    if not np.isfinite(tau):
        raise ValueError("tau must be finite.")

    if not -1.0 < tau < 1.0:
        raise ValueError(
            f"tau must lie strictly between -1 and 1; received {tau}."
        )

    if family == "indep":
        if not np.isclose(tau, 0.0, atol=1e-12):
            raise ValueError(
                "The independence copula requires tau = 0."
            )
        return 0.0

    # The first experiment uses unrotated Clayton and Gumbel copulas,
    # which represent positive dependence.
    if family in {"clayton", "gumbel"} and tau <= 0.0:
        raise ValueError(
            f"An unrotated {family} copula requires positive tau. "
            "Rotated negative-dependence cases will be added in a "
            "later experiment."
        )

    return tau


def make_pair_copula_from_tau(
    family: str,
    tau: float,
    *,
    rotation: int = 0,
    student_df: float = 4.0,
) -> pv.Bicop:
    """Construct a true pair copula from a target Kendall's tau.

    Parameters
    ----------
    family:
        One of ``indep``, ``gaussian``, ``student``, ``clayton``,
        ``gumbel`` or ``frank``.
    tau:
        Requested population Kendall's tau.
    rotation:
        Copula rotation. The initial validation experiment deliberately
        supports rotation zero only.
    student_df:
        Degrees of freedom used for the Student t copula.

    Returns
    -------
    pv.Bicop
        Explicitly parameterized ground-truth copula.
    """
    canonical = normalize_family_name(family)
    tau = _validate_tau(
        canonical,
        tau,
    )

    if rotation != 0:
        raise NotImplementedError(
            "The first synthetic-validation milestone supports only "
            "unrotated copulas. Rotations will be added after the "
            "basic recovery experiment is working."
        )

    if canonical == "indep":
        return _construct_bicop(
            family=canonical,
            rotation=0,
        )

    if canonical == "student":
        if not isinstance(student_df, Real):
            raise TypeError(
                "student_df must be a real number."
            )

        student_df = float(student_df)

        if not np.isfinite(student_df) or student_df <= 2.0:
            raise ValueError(
                "student_df must be finite and greater than 2."
            )

        # Gaussian and Student t copulas have
        #
        #     tau = (2 / pi) * arcsin(rho),
        #
        # hence rho = sin(pi * tau / 2).
        rho = sin(pi * tau / 2.0)

        parameters = np.array(
            [
                [rho],
                [student_df],
            ],
            dtype=float,
        )

        model = _construct_bicop(
            family=canonical,
            rotation=0,
            parameters=parameters,
        )
    else:
        # tau_to_parameters is available for the one-parameter
        # families used here.
        template = _construct_bicop(
            family=canonical,
            rotation=0,
        )

        parameters = np.asarray(
            template.tau_to_parameters(tau),
            dtype=float,
        )

        model = _construct_bicop(
            family=canonical,
            rotation=0,
            parameters=parameters,
        )

    recovered_tau = float(model.tau)

    if not np.isclose(
        recovered_tau,
        tau,
        atol=1e-8,
        rtol=1e-8,
    ):
        raise RuntimeError(
            "Constructed copula does not have the requested Kendall's "
            f"tau: requested={tau}, constructed={recovered_tau}."
        )

    return model


def simulate_pair_copula(
    family: str,
    tau: float,
    n: int,
    seed: int,
    *,
    rotation: int = 0,
    student_df: float = 4.0,
) -> PairCopulaSimulation:
    """Generate reproducible pseudo-observations from a known copula.

    The returned data contain two continuous uniform coordinates in
    the open unit square. No marginal distributions are applied at
    this stage, so this experiment isolates copula calibration.

    Parameters
    ----------
    family:
        True generating copula family.
    tau:
        True population Kendall's tau.
    n:
        Number of observations.
    seed:
        Non-negative random seed.
    rotation:
        Copula rotation; currently only zero is supported.
    student_df:
        Degrees of freedom for the Student t family.

    Returns
    -------
    PairCopulaSimulation
        Ground-truth model, metadata, and simulated observations.
    """
    if not isinstance(n, Integral):
        raise TypeError(
            f"n must be an integer, received {type(n).__name__}."
        )

    n = int(n)

    if n < 2:
        raise ValueError(
            f"n must be at least 2; received {n}."
        )

    if not isinstance(seed, Integral):
        raise TypeError(
            "seed must be an integer, "
            f"received {type(seed).__name__}."
        )

    seed = int(seed)

    if seed < 0:
        raise ValueError(
            f"seed must be non-negative; received {seed}."
        )

    canonical = normalize_family_name(family)

    model = make_pair_copula_from_tau(
        family=canonical,
        tau=tau,
        rotation=rotation,
        student_df=student_df,
    )

    data = np.asarray(
        model.simulate(
            n=n,
            qrng=False,
            seeds=[seed],
        ),
        dtype=float,
    )

    if data.shape != (n, 2):
        raise RuntimeError(
            "pyvinecopulib returned an unexpected sample shape: "
            f"expected {(n, 2)}, received {data.shape}."
        )

    if not np.all(np.isfinite(data)):
        raise RuntimeError(
            "Simulated pseudo-observations contain non-finite values."
        )

    if np.any(data <= 0.0) or np.any(data >= 1.0):
        raise RuntimeError(
            "Simulated pseudo-observations must lie in the open "
            "interval (0, 1)."
        )

    return PairCopulaSimulation(
        family=canonical,
        tau=float(model.tau),
        rotation=rotation,
        seed=seed,
        model=model,
        data=data,
    )
