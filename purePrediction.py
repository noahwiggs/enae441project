import matplotlib.pyplot as plt
import numpy as np
import time
from stations import stations_eci_func

def h_rho_rhodot(x, t, site_id):
    r = x[0:3]
    v = x[3:6]

    rs, vs = stations_eci_func(t, site_id)

    dr = r - rs
    dv = v - vs

    rho = np.linalg.norm(dr)
    rhodot = (dr @ dv) / rho

    return np.array([rho, rhodot])
def load_numpy_data(file_path):
    import os
    cur_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    data = np.load(cur_dir + file_path, allow_pickle=True)
    print(f"Loaded data from {file_path}")
    return data

def run_EKF(length, y, mu0, P0, F, H, Q, R):

    mu_plus_vec = np.zeros((length + 1, 6))
    P_plus_vec = np.zeros((length + 1, 6, 6))
    mu_minus_vec = np.zeros((length, 6))
    P_minus_vec  = np.zeros((length, 6, 6))

    mu_plus_vec[0] = mu0
    P_plus_vec[0]  = P0

    t_start = time.time()

    for k in range(length):
        mu_prev = mu_plus_vec[k]
        P_prev = P_plus_vec[k]
        
        # Predict
        mu_minus = F @ mu_prev
        P_minus = F @ P_prev @ F.T + Q

        mu_minus_vec[k] = mu_minus
        P_minus_vec[k]  = P_minus

        # Correct
        K = P_minus @ H.T @ np.linalg.inv(H @ P_minus @ H.T + R)
        mu_plus = mu_minus + K @ (y[k] - H @ mu_minus)
        P_plus = (np.eye(6) - K@H) @ P_minus 

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

def plot_pure_prediction(file_name):
    
    raw=load_numpy_data(file_name)
    data = raw.item()
    y = data['Y']                  
    t = data['t']
    dt = t[-1]/(len(t)-1) #should be 1 second
    length = y.shape[0]
    
    #Givens
    x0_hat = [0, 0, 500, 0, 0, 0] #km
    
    P0 = np.block([[50 * np.eye(3),     np.zeros((3,3))],
                   [np.zeros((3,3)),    0.1 * np.eye(3)]])


    W = 1e-5*np.eye(3) 
    V = 1e3*np.eye(3)

    #DT state space setup
    
    F = np.array([
        [1, 0, 0, dt, 0,  0],
        [0, 1, 0, 0,  dt, 0],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0],
        [0, 0, 0, 0,  1,  0],
        [0, 0, 0, 0,  0,  1]])
    
    H = np.array([
        [1, 0, 0, 0, 0,  0],
        [0, 1, 0, 0,  0, 0],
        [0, 0, 1, 0,  0,  0]])
    
    Q = np.block([[W*(dt**3)/3, W*(dt**2)/2],
                  [W*(dt**2)/2, W*dt]]) #6x6
    R = V #3x3

    results_KF = run_EKF(length, y, x0_hat, P0, F, H, Q, R)

    mu_minus = results_KF['mu_minus']
    P_minus = results_KF['P_minus']

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
    plt.show()

    return fig
