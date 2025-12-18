import numpy as np

mu = 3.986e5  # km^3/s^2

def rk4(rhs, x0, dt):
    k1 = rhs(0, x0)
    k2 = rhs(0, x0 + 0.5 * dt * k1)
    k3 = rhs(0, x0 + 0.5 * dt * k2)
    k4 = rhs(0, x0 + dt * k3)

    return x0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
def two_body_rhs(_, x):
    r = x[:3]
    v = x[3:]
    rnorm = np.linalg.norm(r)

    a = -mu * r / rnorm**3
    return np.hstack((v, a))

def propagate_func(x0, t0, t, dt_step=1.0):
    """
    Propagate from t0 to t using RK4 with fixed substeps of size dt_step (seconds).
    """
    dt_total = float(t - t0)
    if dt_total == 0.0:
        return x0.copy()

    # handle backwards propagation just in case
    sgn = 1.0 if dt_total > 0 else -1.0
    dt_step = abs(dt_step) * sgn

    n_steps = int(abs(dt_total) // abs(dt_step))
    dt_rem  = dt_total - n_steps * dt_step

    x = x0.copy()

    for _ in range(n_steps):
        x = rk4(two_body_rhs, x, dt_step)

    if abs(dt_rem) > 1e-12:
        x = rk4(two_body_rhs, x, dt_rem)

    return x

