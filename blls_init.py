import numpy as np

def blls_x0(raw_data, x0_nom, P0_nom, R_meas, stations_eci_func, propagate_func, meas_func, H_func, t_window=50.0, t0=None):

    if t0 is None:
        t0 = raw_data[0, 0]

    # Select measurements in first t_window seconds
    mask = (raw_data[:, 0] >= t0) & (raw_data[:, 0] <= t0 + t_window)
    meas = raw_data[mask]

    if meas.shape[0] < 3:
        raise ValueError("Not enough measurements for BLLS initialization.")

    Rinv = np.linalg.inv(R_meas)

    GTWG = np.zeros((6, 6))
    GTWdy = np.zeros((6, 1))

    for row in meas:
        tk = float(row[0])
        site_id = int(row[1])
        yk = np.array([row[2], row[3]])

        # Ground station state
        r_site, v_site = stations_eci_func(tk, site_id)

        # Propagate nominal state
        xk_nom = propagate_func(x0_nom, t0, tk)

        # Predicted measurement and residual
        yk_hat = meas_func(xk_nom, r_site, v_site)
        dy = (yk - yk_hat).reshape(2, 1)

        # Measurement Jacobian
        Hk = H_func(xk_nom, r_site, v_site)   # 2x6

        # Accumulate normal equations
        GTWG += Hk.T @ Rinv @ Hk
        GTWdy += Hk.T @ Rinv @ dy

    # Optional prior (regularization)
    if P0_nom is not None:
        P0inv = np.linalg.inv(P0_nom)
        GTWG += P0inv

    # Solve for correction
    dx0 = np.linalg.solve(GTWG, GTWdy).flatten()

    x0_blls = x0_nom + dx0
    P0_blls = np.linalg.inv(GTWG)

    return x0_blls, P0_blls, dx0