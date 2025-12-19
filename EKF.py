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
    dadr = -mu * (I3/(rnorm**3) - 3.0*rrT/(rnorm**5))  # ∂a/∂r

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

def load_numpy_data(file_path):
    import os
    cur_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    data = np.load(cur_dir + file_path, allow_pickle=True)
    print(f"Loaded data from {file_path}")
    return data

def run_EKF(length, mu0, P0, a, Rk):

    #Set up blank matricies
    mu_plus_vec = np.zeros((length + 1, 6))
    P_plus_vec = np.zeros((length + 1, 6, 6))
    mu_minus_vec = np.zeros((length, 6))
    P_minus_vec  = np.zeros((length, 6, 6))

    meas = np.load("Project-Measurements-Easy.npy")
    mu = 3.986e5

    # Set initialize mu, P
    mu_plus_vec[0] = mu0
    P_plus_vec[0]  = P0

    I6 = np.eye(6)
    t_start = time.time()

    for k in range(length):
        mu_prev = mu_plus_vec[k]
        P_prev  = P_plus_vec[k]

        # measurement time for this step
        t_k = float(meas[k, 0])
        i_k = int(meas[k, 1])
        y_k = meas[k, 2:4]  # [rho, rhodot]
        
        if k == 0:
            t_prev = t_k
        else:
            t_prev = float(meas[k-1, 0])

        t_vec = [t_prev, t_k]

        x_dot_fcn = lambda t, x: xdot_2bp(t, x, mu)
        A_fcn     = lambda x: A_2bp(x, mu)

        X_t_vec, phi_t_vec = propagate_LTV_system_numerically(mu_prev, x_dot_fcn, A_fcn, t_vec)

        mu_minus = X_t_vec[-1]
        Fk = phi_t_vec[-1] # STM from t_prev -> t_k

        dt = t_k - t_prev

        sigma_dd = a
        sigma_d = sigma_dd*dt
        sigma = sigma_dd*(dt**3)/2

        Q11 = sigma**2 * np.eye(3)
        Q22 = sigma_d**2 * np.eye(3)
        Qk = np.block([[Q11, np.zeros((3,3))],
                       [np.zeros((3,3)), Q22]
                       ])

        P_minus = Fk @ P_prev @ Fk.T + Qk

        mu_minus_vec[k] = mu_minus
        P_minus_vec[k]  = P_minus

        # Correct: nonlinear measurement 
        y_hat = h_rho_rhodot(mu_minus, t_k, i_k)
        Hk = H_rho_rhodot(mu_minus, t_k, i_k)

        K = P_minus @ Hk.T @ np.linalg.inv(Hk @ P_minus @ Hk.T + Rk)

        mu_plus = mu_minus + K @ (y_k - y_hat)

       
        P_plus = (I6 - K @ Hk) @ P_minus @ (I6 - K @ Hk).T + K @ Rk @ K.T

        mu_plus_vec[k+1] = mu_plus
        P_plus_vec[k+1]  = P_plus

        
    t_end = time.time()

    results_dict = {
        'mu_minus': mu_minus_vec,
        'P_minus':  P_minus_vec,
        'mu_plus':  mu_plus_vec,
        'P_plus':   P_plus_vec,
        'x_final':  mu_plus_vec[-1],
        'P_final':  P_plus_vec[-1],
        't_execution': t_end - t_start
    }
    return results_dict

def plot_pure_prediction(results_EKF):
    
    mu_minus = results_EKF['mu_minus']
    P_minus = results_EKF['P_minus']

    K = mu_minus.shape[0]    # number of prediction steps
    t = np.arange(K)

    state_labels = ["x", "y", "z", "vx", "vy", "vz"]

    fig, axes = plt.subplots(6, 1, figsize=(10, 18), sharex=True)

    for i in range(6):
        mu_i = mu_minus[:, i]
        sigma_i = np.sqrt(P_minus[:, i, i])
        
        axes[i].plot(t, mu_i, label="μ⁻ (prediction)")
        axes[i].plot(t, mu_i + 3*sigma_i, 'r--', label="+3σ")
        axes[i].plot(t, mu_i - 3*sigma_i, 'r--', label="-3σ")
        
        axes[i].set_ylabel(state_labels[i])
        axes[i].grid(True)

    axes[-1].set_xlabel("Time step k")
    axes[0].legend()
    plt.tight_layout()

    return fig

def plot_with_updates(results_EKF):
    mu_plus = results_EKF['mu_plus']
    P_plus = results_EKF['P_plus']

    state_names = ["x", "y", "z", "vx", "vy", "vz"]
    k_vec = np.arange(mu_plus.shape[0])
    
    fig, axes = plt.subplots(6, 1, figsize=(10, 16), sharex=True)
    plt.suptitle("Measurement Corrected Estimates (-)")

    for i in range(6):
        mu_i = mu_plus[:, i]
        sigma_i = np.sqrt(P_plus[:, i, i])

        axes[i].plot(k_vec, mu_i, label=f'{state_names[i]} estimate [km]')
        axes[i].fill_between(k_vec,
                            mu_i - 3*sigma_i,
                            mu_i + 3*sigma_i,
                            alpha=0.2,
                            label='+/-3 sigma bounds')
        axes[i].set_ylabel(state_names[i])
        axes[i].grid(True)
        axes[i].legend(loc='upper right')

    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()

    return fig