"""Optimal-separation initial values for the OBCA dual variables (no ROS).

This is NOT in Zhang, Liniger, Borrelli. It is the warm-start idea from TDR-OBCA
(He et al., arXiv 2009.11345, Sec. III-B), applied to eq. (15) of the OBCA paper.

Conventions (identical to the collision constraints in controller.py):
  * the car body is {z : G z <= g} in a frame whose origin is the footprint
    CENTRE, so the world position of that origin is t = model.translation(state)
    (NOT the rear axle that the state (X, Y) refers to) and R = rotation(phi);
  * an obstacle is {y : A y <= b} in the world frame.
"""

from __future__ import annotations

import logging
from time import perf_counter

import casadi as ca
import numpy as np

logger = logging.getLogger(__name__)

# I call a pair (k, m) "colliding" when the optimal gap is below this, in metres.
COLLISION_TOLERANCE = 1e-6


def _build_separation_problem(n_walls: int, G: np.ndarray, g: np.ndarray):
    """Build the small problem for one (k, m) ONCE; only its parameters change.

    Unknowns: lambda (one weight per obstacle wall), mu (one weight per car
    wall) and the scalar d. Parameters: the obstacle (A, b) and the car pose
    (t, R). The dynamics, bounds and cost of the full problem only involve the
    states and inputs; I fix the states, so none of them appear here.
    """
    opti = ca.Opti()
    lam = opti.variable(n_walls)
    mu = opti.variable(G.shape[0])
    d = opti.variable()
    A = opti.parameter(n_walls, 2)
    b = opti.parameter(n_walls)
    t = opti.parameter(2)
    R = opti.parameter(2, 2)
    G_dm, g_dm = ca.DM(G), ca.DM(g)

    world_normal = A.T @ lam  # the shared normal  A^T lambda

    # (1) d = -s, where s = -g^T mu + (A t - b)^T lambda is the gap between the
    #     two half-planes. Maximizing the gap s is the same as minimizing d.
    opti.subject_to(-ca.dot(g_dm, mu) + ca.dot(A @ t - b, lam) + d == 0)
    # (2) Both shapes must share ONE separating line. Weighting the obstacle
    #     walls by lambda >= 0 gives the half-plane  (A^T lambda).y <= lambda^T b
    #     that contains the obstacle. Weighting the car walls by mu >= 0 gives,
    #     in the world frame, a half-plane with normal -R G^T mu that contains
    #     the car. Equation (2) says G^T mu = -R^T A^T lambda, i.e. these two
    #     normals are the same vector, so the two half-planes are parallel and
    #     the car lies on the far side of the obstacle's line.
    opti.subject_to(G_dm.T @ mu + R.T @ world_normal == 0)
    # (3) Without a scale constraint (c*lambda, c*mu) is feasible for every
    #     c > 0 and the gap grows with c. Bounding the normal to length <= 1
    #     fixes the scale, so that at the optimum -d is the true distance in
    #     metres between car and obstacle.
    opti.subject_to(ca.sumsqr(world_normal) <= 1.0)
    # (4) Weights must be non-negative so that the weighted sums really are
    #     valid (implied) half-planes.
    opti.subject_to(lam >= 0)
    opti.subject_to(mu >= 0)
    # (5) d <= 0 means s >= 0 (collision free). It also makes lambda = 0,
    #     mu = 0, d = 0 always feasible, so the solver never reports infeasible.
    opti.subject_to(d <= 0)
    opti.minimize(d)
    opti.solver(
        "ipopt",
        {"print_time": False},
        {"print_level": 0, "sb": "yes", "tol": 1e-9, "max_iter": 200},
    )
    return opti, lam, mu, d, A, b, t, R


def compute_dual_warmstart(X, Y, PHI, obstacles, G, g):
    """Optimal-separation lambda/mu for a FIXED trajectory.

    X, Y, PHI: (N+1,) footprint-centre position and yaw of the guess. Because
    the states are fixed, t_k = [X_k, Y_k] and R_k are plain numbers, and every
    collision constraint of eq. (15) only involves the lambda_{m,k}, mu_{m,k}
    of its own (m, k). The problem therefore splits into independent small
    problems, one per (k, m), each of which I solve separately.

    obstacles: objects with attributes ``A`` (l_m, 2) and ``b`` (l_m,).
    Returns, per obstacle m: lam[m] (N+1, l_m), mu[m] (N+1, rows of G),
    dist[m] (N+1,) = -d (metres, 0 when touching), collides[m] (N+1,) bool.
    """
    X, Y, PHI = (np.asarray(v, dtype=float).reshape(-1) for v in (X, Y, PHI))
    G, g = np.asarray(G, dtype=float), np.asarray(g, dtype=float)
    n_steps = len(X)
    started = perf_counter()

    problems = {}  # one compiled problem per distinct number of obstacle walls
    lam_out, mu_out, dist_out, collides_out = [], [], [], []
    for m, obstacle in enumerate(obstacles):
        A, b = np.asarray(obstacle.A, dtype=float), np.asarray(obstacle.b, dtype=float)
        n_walls = len(A)
        if n_walls not in problems:
            problems[n_walls] = _build_separation_problem(n_walls, G, g)
        opti, lam, mu, d, pA, pb, pt, pR = problems[n_walls]

        lam_m = np.zeros((n_steps, n_walls))
        mu_m = np.zeros((n_steps, G.shape[0]))
        dist_m = np.zeros(n_steps)
        collides_m = np.zeros(n_steps, dtype=bool)
        for k in range(n_steps):
            c, s = np.cos(PHI[k]), np.sin(PHI[k])
            opti.set_value(pA, ca.DM(A))
            opti.set_value(pb, ca.DM(b))
            opti.set_value(pt, ca.DM([X[k], Y[k]]))
            opti.set_value(pR, ca.DM([[c, -s], [s, c]]))
            opti.set_initial(lam, 0.0)
            opti.set_initial(mu, 0.0)
            opti.set_initial(d, 0.0)
            try:
                solution = opti.solve()
                lam_k = np.clip(np.asarray(solution.value(lam)).reshape(-1), 0.0, None)
                mu_k = np.clip(np.asarray(solution.value(mu)).reshape(-1), 0.0, None)
                distance = max(0.0, -float(solution.value(d)))
            except RuntimeError as error:
                # Should not happen (lambda = mu = d = 0 is feasible), but never
                # let the warm start crash the real solve.
                logger.warning("dual warm start failed at k=%d, m=%d: %s", k, m, error)
                lam_k, mu_k, distance = np.zeros(n_walls), np.zeros(G.shape[0]), 0.0
            lam_m[k], mu_m[k], dist_m[k] = lam_k, mu_k, distance
            # d > -1e-6: the best separating line has no room, so the guess
            # touches or overlaps the obstacle at this step and lambda, mu are
            # meaningless there (they would be ~0).
            collides_m[k] = distance < COLLISION_TOLERANCE
        lam_out.append(lam_m)
        mu_out.append(mu_m)
        dist_out.append(dist_m)
        collides_out.append(collides_m)

    logger.info(
        "optimal-separation dual warm start: %d steps x %d obstacles in %.3f s",
        n_steps, len(lam_out), perf_counter() - started,
    )
    return lam_out, mu_out, dist_out, collides_out
