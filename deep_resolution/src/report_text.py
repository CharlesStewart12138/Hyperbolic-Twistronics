from __future__ import annotations


CURVATURE_THEORY_TEX = r"""\documentclass[11pt]{article}
\usepackage{amsmath,amssymb,booktabs,geometry}
\geometry{margin=1in}
\title{Curvature-Relevant Universality: A Conditional Operator Theory and Its Certified Scope}
\author{Computer-assisted deep-resolution extension}
\date{}
\begin{document}
\maketitle

\section{Assumptions and comparison contract}
We fix the target fiber, energy origin, energy scale, momentum path, orbital
basis, shell convention, and normalized tunnelling observable.  The manuscript
identity
\[
  \frac{a}{R}=2\operatorname{arcosh}\!\left[
  \frac{\cos(\pi/p)}{\sin(\pi/q)}\right]
\]
implies that changing $R$ at fixed $(p,q)$ does not vary dimensionless
curvature.  A pure curvature diagnostic must instead keep $a$ fixed while
varying $g=(a/R)^2$, or change tessellation data with the coordination effect
tracked separately.  The global fixed-octagon family and the local fixed-$a$
fiber are therefore distinct comparison classes.

\section{Microscopic normalized operator}
For the registered fixed-$a$, coordination-eight local geodesic-star fiber,
the normalized operator has the explicit structure
\[
 \widehat H(q;X,g,\vartheta,S_\parallel,S_\perp)
 =\frac{H_\parallel(q;g,S_\parallel)-E_0 I}{E_M(g,\vartheta)}
 +X\,\tau(q;S_\perp) V,
\]
where $\vartheta=\theta^2$, $S_\parallel=a/\lambda_\parallel$,
$S_\perp=\lambda_\perp/a$, and $E_M=(a/\xi_M)^2$.  The exact local moir\'e
length is
\[
 \xi_M=R\operatorname{arsinh}\!\left[
 \frac{\sinh(a/2R)}{\sin(\theta/2)}\right].
\]
The Slater--Koster blocks are obtained by parallel transport on the hyperboloid,
and $\tau$ is a finite positive geodesic-shell kernel normalized by $\tau(0)=1$.

Around a fixed comparison point the preregistered expansion is
\[
 \widehat H=\widehat H_\star(X)+\delta g\,\Phi_g
 +\delta\vartheta\,\Phi_\vartheta
 +\delta g\,\delta\vartheta\,\Phi_{g\vartheta}
 +\delta S_\parallel\Phi_{S_\parallel}
 +\delta S_\perp\Phi_{S_\perp}+\mathcal R.
\]
Each $\Phi$ is the centered five-point derivative of the displayed
microscopic operator; no correction direction was invented from holdout
residuals.  The derivatives were checked at a half step.

\section{Operator rank lemma}
Use the discrete $C^2$ Hodge--Hilbert--Schmidt metric
\[
 \langle A,B\rangle_{C^2}=\frac1{3|Q|}\sum_{q\in Q}
 \operatorname{ReTr}\bigl(A_0^\dagger B_0+A_1^\dagger B_1+A_2^\dagger B_2\bigr).
\]
After unit normalization, the registered local fiber gives a stable
$\{\Phi_X,\Phi_g\}$ rank-two direction on every one of nine momentum blocks.
The three-field Gram calculation has $s_2/s_1=0.27144$.  This proves local
operator sensitivity to fixed-$a$ curvature in that fiber.  It does not prove
global bulk curvature relevance: the registered scalar-observable Jacobian has
no stable second singular-value gap.

\section{Conditional two-parameter theorem}
Let $\lambda_n$ remain in one fixed global comparison package and suppose
\[
 \|\mathcal R(\lambda_n)\|_{C^k}\to0,\qquad k\in\{0,1,2\}.
\]
Then
\[
 \widehat H(\lambda_n)=\widehat H_\star(X_n,g_n)
 +Y_{\theta,n}\Psi_\theta(X_n,g_n)+o_{C^k}(1)
\]
implies convergence of every licensed homogeneous spectral observable.  Weyl
gives eigenvalue and bandwidth errors bounded by the $C^0$ remainder; a fixed
gap gives Riesz-projector error $O(\|\mathcal R\|/\mathrm{gap})$; $C^1$ and
$C^2$ remainders control velocity and Hodge-Hessian observables separately.
This is a conditional theorem.  The present holdout does not satisfy its
remainder premise uniformly and therefore does not certify the global
two-parameter conclusion.

\section{Angular and shape corrections}
Even twist symmetry selects $\vartheta=\theta^2$ as the leading angular field.
The angular Taylor direction reduces the one-parameter residual locally, but
the joint holdout remains outside the preregistered absolute $C^0/C^1/C^2$
gates.  In the fixed regular-octagon radius path, $g$ is constant and the
derived profile coordinate is $S_\perp=\lambda_\perp/a$.  A quadratic Taylor
expansion in this coordinate reduces the median old-radius $C^0$ residual by
$99.9936\%$ and passes its isolated holdout.  This identifies the old radius
effect as a microscopic shape correction, not as a pure curvature theorem.

\section{Certified conclusion and limitations}
The final classification is \texttt{PASS\_RESTRICTED\_CLASS}.  There is local
operator-level evidence for an independent fixed-$a$ curvature tangent, but no
stable scalar identifiability rank and no successful global H3/H4 holdout.
Neither \texttt{PASS\_TWO\_PARAMETER\_CURVATURE} nor
\texttt{PASS\_THREE\_FIELD} is claimed.  No registered manuscript theorem is
contradicted, because the manuscript already fixes $a/R$ in a regular
tessellation and states one-parameter collapse conditionally with explicit
correction fields.
\end{document}
"""


MANUSCRIPT_REVISION_TEX = r"""\documentclass[11pt]{article}
\usepackage{amsmath,amssymb,geometry}
\geometry{margin=1in}
\title{Proposed Manuscript Revision After Deep Resolution}
\date{}
\begin{document}
\maketitle
\section*{Abstract replacement}
Within a fixed transported microscopic class, the effective coupling remains
the leading one-parameter coordinate.  New exact congruence-height bounds
construct three non-Abelian cover towers with certified growing injectivity
radius and a genuine shell law $L=1,2,3,4,5$.  Retained projective-sector
spectral measures pass independent KPM/SLQ weak-CDF checks with a frozen
vanishing-broadening schedule, while strong full-spectrum no-pollution and a
tower-uniform density regularity theorem remain open.  A fixed-spacing local
operator has an independent curvature tangent, but observable identifiability
and global two-parameter holdout tests do not close.  The supported
universality statement therefore remains restricted to the fixed comparison
class.

\section*{Introduction addition}
For a regular $\{p,q\}$ tessellation, $a/R$ is fixed.  Radius scans at fixed
$(p,q)$ must not be described as pure dimensionless curvature scans when any
physical hopping length is held fixed.  They activate profile fields such as
$\lambda_\perp/a$.  We distinguish these paths from fixed-$a$ local curvature
diagnostics and from true cross-tessellation sequences.

\section*{Universality-section replacement}
Replace any unrestricted equal-$X$ assertion by the conditional statement
\[
 \widehat H=\widehat H_\star(X)+\sum_a y_a\mathcal O_a+\mathcal R,
 \qquad \|\mathcal R\|_{C^k}\to0,
\]
with the comparison package and all active fields declared.  The deep
holdout supports the fixed-class statement only.  A local fixed-$a$ tangent
test finds $\operatorname{rank}\{\Phi_X,\Phi_g\}=2$ in the operator metric,
but the registered scalar-observable Jacobian and global H3 holdout do not
support an independently identifiable bulk curvature coordinate.

\section*{Discussion replacement}
The deeper covers remove the earlier $L=1$ design defect: all three
non-Abelian congruence towers now have exact arithmetic levels realizing
$L=1$ through $5$, and the height proof extends indefinitely.  The current
packing bound, however, is too conservative to prove that the physical and
Hodge tails vanish with word-shell radius.  KPM and SLQ agree strongly in the
retained projective sector, and the frozen $\eta_N=\sqrt{\kappa_N}$ schedule
passes its finite holdout, but no uniform limiting coarea/Morse regularity
certificate is available.  These are genuine remaining mathematical gaps.

\section*{Conclusion replacement}
The strongest supported conclusions are: weak spectral-measure consistency in
the explicitly retained projective sector; existence of a true growing-shell
cover diagonal without certified physical-tail closure; and restricted-class
operator universality with the old radius effect identified as a microscopic
shape field.  Strong bulk no-pollution, local unsmoothed DOS convergence, and
global curvature-relevant two-parameter universality remain inconclusive.  No
genuine \texttt{FAIL\_THEORY} was found.

\section*{Proposed Figure 16 caption}
One-parameter, angular, curvature, and shape-field operator tests on independent
holdouts.  The local fixed-spacing fiber has a rank-two $\{X,g\}$ operator
tangent, whereas the scalar-observable Jacobian and global H3/H4 holdouts fail
their preregistered gates.  The fixed-octagon radius residual closes under the
derived $\lambda_\perp/a$ profile coordinate.  The resulting classification is
\texttt{PASS\_RESTRICTED\_CLASS}; the panel is not a bulk two-parameter
curvature certificate.
\end{document}
"""


DEEP_REPORT_MD = """# Deep asymptotic closure and curvature-relevant universality

The second extension completed all registered computations while preserving the original project and first extension. One implementation-only R10 run is frozen separately; its recovery run passed all numerical method checks.

## Figure 8

Three inequivalent non-Abelian congruence towers now have five exact, certified levels each. The height/norm argument gives a quantitative injectivity-radius lower bound growing linearly with congruence level. Local finite-propagation no-loss tests pass. Independent projective-sector KPM/SLQ measures pass a weak cross-tower CDF test. Strong full-regular-spectrum edge/gap no-pollution remains open. Classification: `PASS_WEAK_BULK` only in the explicitly retained projective-sector/local-measure scope; the strong physical bulk claim is `INCONCLUSIVE`.

## Figure 9

The preregistered law `L=floor(sqrt(r_inj_lower))` realizes `L=1,2,3,4,5` on every tower and has the required analytic limits. Common-Hilbert-space transport is exact on the rooted labelled balls. The frozen physical packing bound does not decrease with word-shell radius, so C0/C1/C2 error closure fails all four estimators. Classification: `INCONCLUSIVE`, now because of a precise tail theorem gap rather than a defective `L=1` sequence.

## Figure 10

Matrix-free projective actions up to dimension 953,312 were evaluated by Jackson KPM and fully reorthogonalized SLQ. All method disagreements pass; the three deepest holdouts have `kappa≈4.6e-4–8.1e-4` and the frozen `eta=sqrt(kappa)` schedule passes. A tower-uniform limiting regularity theorem is absent, so local/unsmoothed/coherence DOS convergence is not certified. Classification: weak CDF and finite broadening schedule pass; local DOS remains `INCONCLUSIVE`.

## Figure 16 and Figure 18

The manuscript contract fixes `a/R` for a regular tessellation. A fixed-`a` local q=8 fiber has a stable rank-two operator tangent, but the scalar-observable Jacobian lacks a stable second singular value and H3/H4 holdouts fail. The old fixed-octagon radius residual is instead closed by the derived profile coordinate `lambda_perp/a` (median reduction 99.9936%). Classification: `PASS_RESTRICTED_CLASS`. Curvature is not globally certified as an independent relevant coordinate.

No genuine `FAIL_THEORY` was found. The reports preserve every inconclusive result and the separate `FAIL_IMPLEMENTATION` recovery history.
"""

