# Figure 9 resolution — balanced full-shell convergence

The preregistered diagonal was `L_j=max(1,floor(sqrt(r_inj,j)))`. All available certified levels have injectivity radius only 1–3, so every selected shell radius is `L_j=1`; neither `L_j→∞` nor `L_j/r_inj→0` is demonstrated.

All six error components were saved independently and recombined exactly. Operator surrogates, spectral-island errors, bandwidth, gap, projector, C1 velocity, and C2 Hodge-Hessian inheritance were computed in the same declared two-dimensional moment-Jacobi space; no unequal-dimensional zero padding was used. These implementation and inheritance checks pass (`R9-02`–`R9-04`).

The theorem-level balanced diagonal remains unavailable, so `R9-01=INCONCLUSIVE` and `R9-05=INCONCLUSIVE`. The revised figure shows the true discrete diagonal without connecting unrelated towers into one artificial line.
