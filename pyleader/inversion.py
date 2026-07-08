"""LEADER linear inversion: solve for the (p, beta) occupation numbers.

Ported from the ``leader_invert`` cell.  The notebook read ``CDFA`` from a
global and returned a 10-tuple; here ``Asort``/``CDFA`` are arguments and the
results are bundled in an :class:`InversionResult` dataclass so that plotting
and post-processing receive them explicitly instead of via globals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear


@dataclass
class InversionResult:
    """Everything downstream plotting / post-processing needs from one inversion."""

    pmax: float
    betamax: float
    relerr: float
    M: np.ndarray
    W_back: np.ndarray      # solution as a flat vector
    W: np.ndarray           # solution reshaped to (len(P), len(BETA))
    P: np.ndarray
    BETA: np.ndarray
    P_Gr: np.ndarray
    BETA_Gr: np.ndarray
    Asort: np.ndarray
    CDFA: np.ndarray


def leader_invert(
    Asort: np.ndarray,
    CDFA: np.ndarray,
    *,
    gridtype: str | None = None,
    deltaP: float = 0.1,
    deltaB: float = 1.0,
    grid_jitter: bool = True,
    verbose: bool = True,
) -> InversionResult:
    """Solve ``M w = CDF(A)`` for the occupation numbers ``w`` over the (p, beta) grid.

    ``grid_jitter=False`` keeps the canonical un-perturbed (P, BETA) bin grids
    (used by the correction basis runs so all runs share one grid); the default
    reproduces the original LEADER behaviour of randomly perturbing the bins.
    """
    NP = 20
    NBETA = 29

    P = np.linspace(0.025, 0.975, NP)
    BETA = np.linspace(0.025, 1.55, NBETA)

    if gridtype == "dynamic":
        temp = np.zeros(2)
        temp[0] = P[np.random.randint(0, 5)]   # p < 0.25
        temp[1] = P[np.random.randint(5, 8)]   # 0.25 < p < 0.4
        temp = np.concatenate([temp, P[8:]])
        P = temp

    coeff = 0.015

    def truncated_gaussian_noise(arr, max_val=0.025):
        noise = coeff * np.random.randn(len(arr))
        for i in range(len(noise)):
            while abs(noise[i]) > max_val:
                noise[i] = coeff * np.random.randn()
        return noise

    if grid_jitter:
        P = P + truncated_gaussian_noise(P)

    if gridtype == "dynamic":
        temp = np.zeros(5)
        temp[0] = BETA[np.random.randint(0, 6)]
        temp[1] = BETA[np.random.randint(6, 8)]
        temp[2] = BETA[np.random.randint(8, 10)]
        temp[3] = BETA[np.random.randint(10, 12)]
        temp[4] = BETA[np.random.randint(12, 14)]
        temp = np.concatenate([temp, BETA[14:]])
        BETA = temp

    if grid_jitter:
        BETA = np.minimum(BETA + truncated_gaussian_noise(BETA), np.pi / 2 - np.finfo(float).eps)

    # Build matrix M for the linear system M w = C
    M = np.zeros((len(Asort), len(P) * len(BETA)))
    ind = 0
    for j in range(len(P)):
        for k in range(len(BETA)):
            for i in range(len(Asort)):
                A_val = Asort[i]
                beta_val = BETA[k]
                p_val = P[j]
                if A_val <= p_val:
                    M[i, ind] = 0
                elif A_val < np.sqrt(np.sin(beta_val) ** 2 + p_val ** 2 * np.cos(beta_val) ** 2):
                    val = np.sqrt((A_val ** 2 - p_val ** 2) / (1 - p_val ** 2))
                    M[i, ind] = np.pi / 2 - np.arccos(val / np.sin(beta_val))
                else:
                    M[i, ind] = np.pi / 2
            ind += 1

    # Regularization matrices RP and RB
    NN = len(P)
    MM = len(BETA)
    RP = np.zeros(((NN - 1) * MM, NN * MM))
    RB = np.zeros((NN * (MM - 1), NN * MM))

    for k in range(RP.shape[0]):
        pindex = math.ceil((k + 1) / MM) - 1
        RP[k, k] = -1 / (P[pindex + 1] - P[pindex])
        RP[k, k + MM] = 1 / (P[pindex + 1] - P[pindex])

    for k in range(RB.shape[0]):
        ll = k + math.ceil((k + 1) / (MM - 1)) - 1
        bindex = k % (MM - 1)
        if bindex == 0 and (k + 1) % (MM - 1) == 0:
            bindex = MM - 1
        RB[k, ll] = -1 / (BETA[bindex + 1] - BETA[bindex])
        RB[k, ll + 1] = 1 / (BETA[bindex + 1] - BETA[bindex])

    # Extended (regularized) system
    Mtilde = np.vstack([M, np.sqrt(deltaP) * RP, np.sqrt(deltaB) * RB])
    Mtilde = np.nan_to_num(Mtilde, nan=0.0)
    Ctilde = np.concatenate([CDFA, np.zeros(RP.shape[0]), np.zeros(RB.shape[0])])
    Ctilde = np.nan_to_num(Ctilde, nan=0.0)

    # Solve with positivity constraint
    if verbose:
        print("Solving the weights w_ij for the bins (p_i, beta_j)...")
    result = lsq_linear(Mtilde, Ctilde, bounds=(0, np.inf), method="trf", lsmr_tol="auto", verbose=0)
    W = result.x
    if verbose:
        print("Solution obtained!")

    W_back = W.copy()
    W = W.reshape((len(P), len(BETA)))

    # Peak of the distribution
    max_idx = np.unravel_index(np.argmax(W), W.shape)
    pmax = P[max_idx[0]]
    betamax = BETA[max_idx[1]]

    P_Gr, BETA_Gr = np.meshgrid(P, BETA, indexing="ij")

    relerr = np.linalg.norm(M @ W_back - CDFA) / np.linalg.norm(CDFA)

    if verbose:
        print(f"The highest peak: P = {pmax}, BETA = {betamax}")
        print(f"Relative error: {relerr}")

    return InversionResult(
        pmax=pmax,
        betamax=betamax,
        relerr=relerr,
        M=M,
        W_back=W_back,
        W=W,
        P=P,
        BETA=BETA,
        P_Gr=P_Gr,
        BETA_Gr=BETA_Gr,
        Asort=np.asarray(Asort),
        CDFA=np.asarray(CDFA),
    )
