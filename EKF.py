import matplotlib.pyplot as plt
import numpy as np
import time
from stations import stations_eci_func

def integrator(X_dot_fcn, X_0, t_vec):
    from scipy.integrate import solve_ivp
    t_span = [t_vec[0], t_vec[-1]]
    t_eval = t_vec
    sol = solve_ivp(X_dot_fcn, t_span, y0=np.asarray(X_0, dtype = float), t_eval=t_eval, method = "RK45", rtol=1e-9, atol=1e-9,)
    return np.asarray(sol.y)    

def propagate_LTV_system_numerically(X_0, x_dot_fcn, A_fcn, t_vec):
    # Return trajectory and STM over time where
    # np.shape(X_t_vec) = (len(t_vec), len(X_0))
    # np.shape(phi_t_vec) = (len(t_vec), len(X_0), len(X_0))
    N = 6
    t0 = float(t_vec[0])
    tf = float(t_vec[-1])

    if np.isclose(tf, t0):
        X_t_vec = np.vstack([X_0, X_0])                # (2,6)
        phi_t_vec = np.stack([np.eye(N), np.eye(N)])   # (2,6,6)
        return X_t_vec, phi_t_vec
    phi_0 = np.eye(N).flatten()
    Z_0 = np.concatenate((X_0, phi_0))
    
    def Z_fcn(t, Z):
        x=Z[0:N]
        phi=Z[N:].reshape(N,N)

        x_dot = x_dot_fcn(t,x)
        A = A_fcn(x)

        phi_dot = (A @ phi).flatten()
        return np.concatenate((x_dot, phi_dot))

    Z_t = integrator(Z_fcn, Z_0, t_vec)
    X_t_vec = Z_t[:N, :].T
    phi_t_vec = Z_t[N:, :].T.reshape(len(t_vec), N, N)
    
    return X_t_vec, phi_t_vec

def xdot_2bp(t, x, mu):
    r = x[0:3]
    v = x[3:6]
    rnorm = np.linalg.norm(r)
    a = -mu * r / (rnorm**3)
    return np.hstack((v, a))

def A_2bp(x, mu):
    r = x[0:3]
    rnorm = np.linalg.norm(r)
    I3 = np.eye(3)
    rrT = np.outer(r, r)
    dadr = -mu * (I3/(rnorm**3) - 3.0*rrT/(rnorm**5))

    Z3 = np.zeros((3,3))
    A = np.block([
        [Z3, I3],
        [dadr, Z3]
    ])
    return A

def h_rho_rhodot(x, t, site_id):
    r = x[0:3]
    v = x[3:6]

    Rsite, Vsite = stations_eci_func(t, site_id)

    dr = r - Rsite
    dv = v - Vsite

    rho = np.linalg.norm(dr)
    rhodot = (dr @ dv) / rho
    return np.array([rho, rhodot])

def H_rho_rhodot(x, t, site_id):
    r = x[0:3]
    v = x[3:6]
    Rsite, Vsite = stations_eci_func(t, site_id)

    dr = r - Rsite
    dv = v - Vsite
    rho = np.linalg.norm(dr)
    rhodot = (dr @ dv) / rho
    d_rhodot_dr = (dv / rho - (rhodot / rho) * (dr / rho)).reshape(1, 3)
    d_rhodot_dv = (dr / rho).reshape(1, 3)
    H = np.block([[dr/rho, np.zeros((1,3))],
                  [d_rhodot_dr, d_rhodot_dv]])
    return H

def run_EKF_prediction_only(meas, length, mu0, P0, mu, a):
    mu_minus = np.zeros((length, 6))
    P_minus  = np.zeros((length, 6, 6))
    sigma_minus = np.zeros((length, 6))

    mu_prev = mu0
    P_prev  = P0

    t_start = time.time()

    for k in range(1, length):
        t_prev = float(meas[k-1, 0])
        t_k = float(meas[k, 0])
        dt = t_k - t_prev

        # def nonlinear dynamics and STM functions, propagate
        x_dot_fcn = lambda t, x: xdot_2bp(t, x, mu)
        A_fcn = lambda x: A_2bp(x, mu)

        t_vec = [t_prev, t_k]
        X_t_vec, phi_t_vec = propagate_LTV_system_numerically(
            mu_prev, x_dot_fcn, A_fcn, t_vec)

        mu_pred = X_t_vec[-1]
        Fk = phi_t_vec[-1]

        # update process noise
        
        Qk = a**2 * np.block([
            [(dt**3)/3 * np.eye(3), np.zeros((3,3))],
            [np.zeros((3,3)), dt        * np.eye(3)]])

        P_pred = Fk @ P_prev @ Fk.T + Qk

        mu_minus[k-1] = mu_pred
        P_minus[k-1]  = P_pred
        sigma_minus[k-1] = np.sqrt(np.diag(P_pred))

        mu_prev = mu_pred
        P_prev  = P_pred

    t_end = time.time()

    return {
        'mu_minus': mu_minus,
        'P_minus': P_minus,
        'sigma_minus': sigma_minus,
        'sigma3_minus': 3 *sigma_minus,
        't_execution': t_end - t_start,
        't': meas[1:length, 0]

    }

def plot_pure_prediction(results):
    
    mu_minus = results['mu_minus']
    P_minus = results['P_minus']
    t = results['t']

    #clip prediction to fit in time span of data
    N = len(t)
    mu_minus = mu_minus[:N]
    P_minus  = P_minus[:N]

    state_labels = ["x", "y", "z", "vx", "vy", "vz"]

    fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()
    plt.suptitle("EKF Pure Prediction (-)")

    for i in range(6):
        mu_i = mu_minus[:, i]
        sigma_i = np.sqrt(P_minus[:, i, i])
        
        axes[i].plot(t, mu_i, label="mu- (prediction)")
        axes[i].plot(t, mu_i + 3*sigma_i, 'r--', label="+3sigma")
        axes[i].plot(t, mu_i - 3*sigma_i, 'r--', label="-3sigma")
        
        axes[i].set_ylabel(state_labels[i])
        axes[i].grid(True)
        axes[i].set_xlabel("Time (t)")

    axes[0].legend()
        
    plt.tight_layout()

    return fig

def plot_orbit_xy_samples(results, x_dot_fcn=None, A_fcn=None, dense=False):

    mu = results['mu_minus']
    t  = results['t']

    N = len(t)
    mu = mu[:N]

    r = mu[:, 0:3]

    plt.figure(figsize=(6,6))

    plt.scatter(r[:,0], r[:,1], s=12, label='Predicted Samples')

    if dense:
        t_dense = np.linspace(t[0], t[-1], 5000)
        X_dense, _ = propagate_LTV_system_numerically(
            mu[0], x_dot_fcn, A_fcn, t_dense
        )
        plt.plot(X_dense[:,0], X_dense[:,1],
                 linewidth=1.5, label='Dense Propagation')

    plt.axis('equal')
    plt.xlabel('x [km]')
    plt.ylabel('y [km]')
    plt.title('Predicted Orbit in ECI')
    plt.grid(True)
    plt.legend()

    return plt.gcf()

def plot_prediction_covariance_envelope(results):

    P = results['P_minus']
    t = results['t']

    N = len(t)
    P = P[:N]

    state_labels = ['x (km)', 'y (km)', 'z (km)',
                    'vx (km/s)', 'vy (km/s)', 'vz (km/s)']

    fig1, axes = plt.subplots(3, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()

    for i in range(6):
        sigma = np.sqrt(P[:, i, i])

        axes[i].plot(t,  3*sigma, 'r')
        axes[i].plot(t, -3*sigma, 'r')
        axes[i].axhline(0, color='k', linewidth=0.5)

        axes[i].set_ylabel(state_labels[i])
        axes[i].grid(True)

        if i >= 4:
            axes[i].set_xlabel('Time [s]')

    fig1.suptitle('Predicted State Covariance', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig2, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    sigma_x  = np.sqrt(P[:, 0, 0])
    sigma_y  = np.sqrt(P[:, 1, 1])
    sigma_z  = np.sqrt(P[:, 2, 2])

    sigma_vx = np.sqrt(P[:, 3, 3])
    sigma_vy = np.sqrt(P[:, 4, 4])
    sigma_vz = np.sqrt(P[:, 5, 5])

    ax[0].plot(t,  3*sigma_x, color='C2', label='±3σ x')
    ax[0].plot(t, -3*sigma_x, color='C2')

    ax[0].plot(t,  3*sigma_y, color='C0', label='±3σ y')
    ax[0].plot(t, -3*sigma_y, color='C0')

    ax[0].plot(t,  3*sigma_z, color='C3', label='±3σ z',linestyle='--')
    ax[0].plot(t, -3*sigma_z, color='C3',linestyle='--')

    ax[0].set_ylabel('Position Uncertainty [km]')
    ax[0].set_title('Position Uncertainty Components')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(t,  3*sigma_vx, color='C2', label='±3σ vx')
    ax[1].plot(t, -3*sigma_vx, color='C2')

    ax[1].plot(t,  3*sigma_vy, color='C0', label='±3σ vy')
    ax[1].plot(t, -3*sigma_vy, color='C0')

    ax[1].plot(t,  3*sigma_vz, color='C3', label='±3σ vz',linestyle='--')
    ax[1].plot(t, -3*sigma_vz, color='C3',linestyle='--')

    ax[1].set_ylabel('Velocity Uncertainty [km/s]')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_title('Velocity Uncertainty Components')
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()

    return fig1, fig2

def run_EKF(meas, length, mu0, P0, mu, a, Rk):
    mu_minus = np.zeros((length, 6))
    P_minus  = np.zeros((length, 6, 6))
    sigma_minus = np.zeros((length, 6))

    mu_plus  = np.zeros((length, 6))
    P_plus   = np.zeros((length, 6, 6))
    sigma_plus = np.zeros((length, 6))

    residual_prefit  = np.zeros((length, 2))
    residual_postfit = np.zeros((length, 2))

    mu_plus[0] = mu0
    P_plus[0]  = P0
    sigma_plus[0] = np.sqrt(np.diag(P0))

    I6 = np.eye(6)

    t_start = time.time()

    for k in range(1, length):
        t_prev = float(meas[k-1, 0])
        t_k    = float(meas[k, 0])
        i_k    = int(meas[k, 1])
        y_k    = meas[k, 2:4]

        dt = t_k - t_prev

        x_dot_fcn = lambda t, x: xdot_2bp(t, x, mu)
        A_fcn     = lambda x: A_2bp(x, mu)

        t_vec = [t_prev, t_k]
        X_t_vec, phi_t_vec = propagate_LTV_system_numerically(
            mu_plus[k-1], x_dot_fcn, A_fcn, t_vec
        )

        mu_pred = X_t_vec[-1]
        Fk = phi_t_vec[-1]

        Qk = a**2 * np.block([
            [(dt**3)/3 * np.eye(3), np.zeros((3,3))],
            [np.zeros((3,3)),       dt * np.eye(3)]
        ])

        P_pred = Fk @ P_plus[k-1] @ Fk.T + Qk

        mu_minus[k] = mu_pred
        P_minus[k]  = P_pred
        sigma_minus[k] = np.sqrt(np.diag(P_pred))

        y_hat = h_rho_rhodot(mu_pred, t_k, i_k)
        Hk = H_rho_rhodot(mu_pred, t_k, i_k)

        residual_prefit[k] = y_k - y_hat

        K = P_pred @ Hk.T @ np.linalg.inv(Hk @ P_pred @ Hk.T + Rk)

        mu_upd = mu_pred + K @ (y_k - y_hat)
        P_upd = (I6 - K @ Hk) @ P_pred @ (I6 - K @ Hk).T + K @ Rk @ K.T

        mu_plus[k] = mu_upd
        P_plus[k]  = P_upd
        sigma_plus[k] = np.sqrt(np.diag(P_upd))

        y_post = h_rho_rhodot(mu_upd, t_k, i_k)
        residual_postfit[k] = y_k - y_post

    t_end = time.time()

    return {
        't': meas[0:length, 0],

        'mu_minus': mu_minus,
        'P_minus': P_minus,
        'sigma_minus': sigma_minus,
        'sigma3_minus': 3 * sigma_minus,

        'mu_plus': mu_plus,
        'P_plus': P_plus,
        'sigma_plus': sigma_plus,
        'sigma3_plus': 3 * sigma_plus,

        'residual_prefit': residual_prefit,
        'residual_postfit': residual_postfit,

        'x_final': mu_plus[length-1],
        'P_final': P_plus[length-1],
        't_execution': t_end - t_start
    }

def plot_EKF_covariance_envelope(results):
    P_minus = results['P_minus']
    P_plus  = results['P_plus']
    t = results['t']

    N = len(t)
    P_minus = P_minus[:N]
    P_plus  = P_plus[:N]

    state_labels = [
        'x (km)', 'y (km)', 'z (km)',
        'vx (km/s)', 'vy (km/s)', 'vz (km/s)'
    ]

    colors = ['C2', 'C0', 'C3', 'C2', 'C0', 'C3']

    fig1, axes = plt.subplots(3, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()

    for i in range(6):
        sig_m = np.sqrt(P_minus[:, i, i])
        sig_p = np.sqrt(P_plus[:, i, i])
        c = colors[i]

        if i == 0:
            axes[i].plot(t,  3*sig_m, color=c, linestyle='--', label='Pre ±3σ')
            axes[i].plot(t, -3*sig_m, color=c, linestyle='--')

            axes[i].plot(t,  3*sig_p, color=c, linestyle='-', label='Post ±3σ')
            axes[i].plot(t, -3*sig_p, color=c, linestyle='-')
        else:
            axes[i].plot(t,  3*sig_m, color=c, linestyle='--')
            axes[i].plot(t, -3*sig_m, color=c, linestyle='--')

            axes[i].plot(t,  3*sig_p, color=c, linestyle='-')
            axes[i].plot(t, -3*sig_p, color=c, linestyle='-')

        axes[i].axhline(0, color='k', linewidth=0.5)
        axes[i].set_ylabel(state_labels[i])
        axes[i].grid(True)

        if i >= 4:
            axes[i].set_xlabel('Time [s]')

    axes[0].legend(fontsize=9)
    fig1.suptitle('EKF State Covariance: Pre vs Post ±3σ')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig2, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for idx, lbl, ax_i in [
        ([0, 1, 2], 'Position Uncertainty [km]', ax[0]),
        ([3, 4, 5], 'Velocity Uncertainty [km/s]', ax[1])
    ]:
        for j in idx:
            sig_m = np.sqrt(P_minus[:, j, j])
            sig_p = np.sqrt(P_plus[:, j, j])
            c = colors[j]

            ax_i.plot(t,  3*sig_m, color=c, linestyle='--')
            ax_i.plot(t, -3*sig_m, color=c, linestyle='--')

            ax_i.plot(t,  3*sig_p, color=c, linestyle='-')
            ax_i.plot(t, -3*sig_p, color=c, linestyle='-')

        ax_i.set_ylabel(lbl)
        ax_i.grid(True)

    ax[1].set_xlabel('Time [s]')
    ax[0].set_title('Position Uncertainty Components')
    ax[1].set_title('Velocity Uncertainty Components')

    from matplotlib.lines import Line2D

    pos_legend = [
    Line2D([0], [0], color='C2', lw=2, label='x'),
    Line2D([0], [0], color='C0', lw=2, label='y'),
    Line2D([0], [0], color='C3', lw=2, label='z')
    ]

    vel_legend = [
        Line2D([0], [0], color='C2', lw=2, label='vx'),
        Line2D([0], [0], color='C0', lw=2, label='vy'),
        Line2D([0], [0], color='C3', lw=2, label='vz')
    ]

    ax[0].legend(handles=pos_legend, fontsize=9)
    ax[1].legend(handles=vel_legend, fontsize=9)

    plt.tight_layout()

    return fig1, fig2

def plot_EKF_state_update_difference(results):
    t = results['t']

    mu_minus = results['mu_minus']
    mu_plus  = results['mu_plus']
    P_minus  = results['P_minus']

    delta_mu = mu_plus - mu_minus

    state_labels = [
        'x (km)', 'y (km)', 'z (km)',
        'vx (km/s)', 'vy (km/s)', 'vz (km/s)'
    ]

    colors = ['C2', 'C0', 'C3', 'C2', 'C0', 'C3']

    fig, axes = plt.subplots(3, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()

    for i in range(6):
        sigma_minus = np.sqrt(P_minus[:, i, i])
        c = colors[i]

        axes[i].plot(t,  delta_mu[:, i], color=c, label='Post − Pre')
        axes[i].plot(t,  3*sigma_minus, 'k--')
        axes[i].plot(t, -3*sigma_minus, 'k--')

        axes[i].axhline(0, color='k', linewidth=0.5)
        axes[i].set_ylabel(state_labels[i])
        axes[i].grid(True)

        if i >= 4:
            axes[i].set_xlabel('Time [s]')

    axes[0].legend(fontsize=9)
    fig.suptitle(
        'EKF Measurement Update Δμ within Pre-Update ±3σ Bounds',
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig
