"""Post-processing / smoothing of the LEADER solution.

Ported from the ``leader_postprocess_WISE`` cell.  State previously read from
globals (``W``, ``P``, ``BETA``, ``outdir``, ``trial``) is now passed in.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .inversion import InversionResult


def leader_postprocess_WISE(
    result: InversionResult,
    outdir: str,
    trial: int,
    *,
    allow_p_spread: bool = False,
    show: bool = False,
    verbose: bool = True,
) -> None:
    """Damp the solution away from its peak and write the smoothed contour plot."""
    if verbose:
        print("Smoothing the solution...")

    W, P, BETA = result.W, result.P, result.BETA

    dampen = 0.1 if allow_p_spread else 1.0

    # Peak index
    pind, bind = np.unravel_index(np.argmax(W), W.shape)

    # Damp values by distance from the peak
    W_after = W.copy()
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_after[i, j] = W[i, j] / ((dampen * abs(pind - i) + abs(bind - j) + 1) ** 1)

    # Shift P values to the right by a constant step
    Pshift = 0.1
    PP = P.copy()
    PP[1:] = np.minimum(P[1:] + Pshift, 1.0)

    # Keep PP strictly increasing where it saturates at 1
    ind = np.where(PP == 1.0)[0]
    if len(ind) > 1:
        temp = PP[ind[0] - 1]
        for i in range(len(ind) - 1):
            PP[ind[i]] = temp + (i + 1) / len(ind) * (1 - temp)

    BB = BETA.copy()

    plt.figure()
    cp = plt.contourf(PP, BB, W_after.T, levels=100, cmap="viridis")
    plt.colorbar(cp)
    plt.xlabel("p")
    plt.ylabel(r"$\beta$")
    plt.title("Smoothed joint distribution f(p, β)")
    plt.tight_layout()
    plt.savefig(f"{outdir}/Trial{trial + 1}/Solutions_smoothed_{trial + 1}.png", dpi=300)
    if show:
        plt.show()
    plt.close()
