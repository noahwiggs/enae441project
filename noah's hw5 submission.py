import matplotlib.pyplot as plt
import numpy as np
import time


def load_numpy_data(file_path):
    import os
    cur_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    data = np.load(cur_dir + file_path, allow_pickle=True)
    print(f"Loaded data from {file_path}")
    return data

###############################################
# REQUIRED FUNCTIONS FOR AUTOGRADER
# Keep the function signatures the same!!
###############################################

def run_BLLS(length,R,H,F,y_vec):
    
    R_inv_vec = np.eye(length) * (1/R)

    H_vec = np.zeros((length, 6))
    for k in range(length):
        H_vec[k, :] = H

    t_start = time.time()
    x_vec = np.linalg.pinv(H_vec.T @ R_inv_vec @ H_vec) @ H_vec.T @ R_inv_vec @ y_vec
    t_end = time.time()
    t_execution = t_end-t_start

    Pe = np.linalg.pinv(H_vec.T @ R_inv_vec @ H_vec)
    sigma2= np.diag(Pe)
    sigma = np.sqrt(sigma2)

    results_dict = {
        'x_vec': x_vec,
        't_execution': t_execution,
        'Pe' : Pe,
        'sigma' : sigma
    }
    return results_dict

def run_RLS(length, R, H, F, y_vec, x0_vec, P0):
    
    x_k = np.zeros((length + 1, 6))
    P_k = np.zeros((length + 1, 6, 6))
    K_k = np.zeros((length, 6))
    
    x_k[0, :] = x0_vec
    P_k[0, :, :] = P0
    I = np.eye(6)

    # run loop and save at each step
    t_start = time.time()
    for k in range(length):
        x_prev = x_k[k, :]
        P_prev = P_k[k, :, :]
        
        K = P_prev @ H.T / (H @ P_prev @ H.T + R)
        
        x_upd = x_prev + K * (y_vec[k] - H @ x_prev)

        KH = np.outer(K, H)
        KRK = np.outer(K,K) * R
        P_upd = (I - KH) @ P_prev @ (I - KH).T + KRK

        x_k[k + 1, :] = x_upd
        P_k[k + 1, :, :] = P_upd
        K_k[k, :] = K.flatten()

    
    t_end = time.time()
    t_execution = t_end-t_start

    results_dict = {
        'x_k_vec': x_k,
        't_execution': t_execution,
        'P_k_vec': P_k,
        'K_k_vec': K_k,
        'x_final': x_k[-1, :],
        'P_final' : P_k[-1, :, :],
        'sigma_final' : np.sqrt(np.diag(P_k[-1, :, :]))
    }

    return results_dict

def run_KF(length, y, mu0, P0, F, H, Q, R):

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

#######################
# Problem 1
#######################


# REQUIRED --- 1b
def plot_batch_least_squares_single_trial():
    raw = load_numpy_data('HW5Measurements-P1.npy')
    data = np.array(raw) 

    y_vec = data[0, :] #extract first trial
    length = y_vec.size
    
    #Givens from problem statement
    R = 100
    dt = 10 #seconds
    
    F = np.array([
        [1, 0, 0, dt, 0,  0],
        [0, 1, 0, 0,  dt, 0],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0],
        [0, 0, 0, 0,  1,  0],
        [0, 0, 0, 0,  0,  1]
        ])
    
    H = np.array([0, 0, 1, 0, 0, 0])

    results_BLLS = run_BLLS(length,R,H,F,y_vec)
    x0_vec = results_BLLS['x_vec']
    
    X_traj = np.zeros((length, 6))
    X_traj[0, :] = x0_vec
    for k in range(1, length):
        X_traj[k, :] = F @ X_traj[k-1, :]

    fig, ax = plt.subplots()
    k_vec = np.arange(length)
    ax.plot(k_vec, X_traj[:, 0], label='x')
    ax.plot(k_vec, X_traj[:, 1], label='y')
    ax.plot(k_vec, X_traj[:, 2], label='z')
    ax.set_xlabel('k')
    ax.set_ylabel('Estimated Position [m]')
    ax.set_title('Trial 1: k = {1,...,200}')
    ax.grid(True)
    ax.legend()

    return fig

# REQUIRED --- 1c
def plot_batch_least_squares_all_trials():

    raw = load_numpy_data('HW5Measurements-P1.npy')
    data = np.array(raw)

    n_trials, length = data.shape

    R = 100
    dt = 10  # seconds

    F = np.array([
        [1, 0, 0, dt, 0,  0],
        [0, 1, 0, 0,  dt, 0],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0],
        [0, 0, 0, 0,  1,  0],
        [0, 0, 0, 0,  0,  1]
    ])

    H = np.array([0, 0, 1, 0, 0, 0])

    X_traj_all = np.zeros((n_trials, length, 6))

    for i in range(n_trials):
        y_vec = data[i, :]
        results_BLLS = run_BLLS(length, R, H, F, y_vec)
        x0_vec = results_BLLS['x_vec']
        sigma = results_BLLS['sigma'] 

        X_traj = np.zeros((length, 6))
        X_traj[0, :] = x0_vec
        for k in range(1, length):
            X_traj[k, :] = F @ X_traj[k-1, :]

        X_traj_all[i, :, :] = X_traj

    X_mean = np.mean(X_traj_all, axis=0)

    k_vec = np.arange(length)
    fig, ax = plt.subplots()

    # plot one trial line with label
    ax.plot(k_vec,
            X_traj_all[0, :, 2],
            color='orange',
            linewidth=0.6,
            label='individual trials')

    # plot remaining trials to not mess up legend
    for i in range(1, n_trials):
        ax.plot(k_vec,
                X_traj_all[i, :, 2],
                color='orange',
                linewidth=0.6)

    
    ax.plot(k_vec, X_mean[:, 2], label='mean z', color='C0')
    ax.fill_between(k_vec,
                    X_mean[:, 2] - 3*sigma[2],
                    X_mean[:, 2] + 3*sigma[2],
                    alpha=0.2,
                    label='+/-3 sigma z')

    ax.set_xlabel('k')
    ax.set_ylabel('Estimated Position [m]')
    ax.set_title('All Trials: k = {1,...,200}')

    ax.grid(True)
    ax.legend()

    return fig


# REQUIRED --- 1d
def plot_state_estimate_histograms():
    
    raw=load_numpy_data('HW5Measurements-P1.npy')  
    data = np.array(raw)

    n_trials, length = data.shape

    R = 100
    dt = 10  # seconds

    F = np.array([
        [1, 0, 0, dt, 0,  0],
        [0, 1, 0, 0,  dt, 0],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0],
        [0, 0, 0, 0,  1,  0],
        [0, 0, 0, 0,  0,  1]
    ])

    H = np.array([0, 0, 1, 0, 0, 0])

    X_traj_all = np.zeros((n_trials, length, 6))

    for i in range(n_trials):
        y_vec = data[i, :]
        results_BLLS = run_BLLS(length, R, H, F, y_vec)
        x0_vec = results_BLLS['x_vec']

        X_traj = np.zeros((length, 6))
        X_traj[0, :] = x0_vec
        for k in range(1, length):
            X_traj[k, :] = F @ X_traj[k-1, :]

        X_traj_all[i, :, :] = X_traj
    
    
    k_indices = [9, 49, 199]   # k= 10, 50, 200 
    k_labels = [10, 50, 200]
    state_labels = ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']

    figs = []

    for k_idx, k_label in zip(k_indices, k_labels):
        X_k = X_traj_all[:, k_idx, :]   # shape (n_trials, 6)

        mu_k = np.nanmean(X_k, axis=0)
        sigma_k = np.nanstd(X_k,axis=0)

        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        axes = axes.flatten()

        for j in range(6):
            ax = axes[j]
            ax.hist(X_k[:, j], bins=7)
            ax.axvline(mu_k[j], linestyle='--')
            ax.set_title(state_labels[j])

            ax.text(
                0.05,
                0.95,
                f"mu = {mu_k[j]:.3e}\nsigma^2 = {sigma_k[j]:.3e}",
                transform=ax.transAxes,
                va='top'
            )

        fig.suptitle(f'Histograms of state estimates at k = {k_label}')
        fig.tight_layout()
        figs.append(fig)
    
    description = """
        The histograms show that only the z state has spread across trials. 
        This is expected because H = [0 0 1 0 0 0] only appears in the z position, 
        and the system is otherwise static, so x, y, xdot, ydot, and zdot remain zero 
        for all 50 trials. Their means are zero and their variances are also effectively zero, 
        so all samples fall in a single histogram bin in the center. In contrast, the z position 
        shows a roughly Gaussian distribution with a variance reflecting the measurement noise 
        which was propagated through the BLLS estimator.
    """
    return figs, description


# REQUIRED --- 1e
def plot_execution_time_vs_measurements():
    raw=load_numpy_data('HW5Measurements-P1.npy')
    data = np.array(raw)
    n_trials, length = data.shape

    R = 100
    dt = 10

    F = np.array([
        [1, 0, 0, dt, 0,  0],
        [0, 1, 0, 0,  dt, 0],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0],
        [0, 0, 0, 0,  1,  0],
        [0, 0, 0, 0,  0,  1]
    ])

    H = np.array([0, 0, 1, 0, 0, 0])

    k_vec = np.arange(1, length + 1)
    t_mean = np.zeros(length)

    for idx, k in enumerate(k_vec):
        t_sum = 0.0
        for i in range(n_trials):
            y_vec = data[i, :k]
            results = run_BLLS(k, R, H, F, y_vec)
            t_sum += results['t_execution']
        t_mean[idx] = t_sum / n_trials

    fig, ax = plt.subplots()
    ax.plot(k_vec, t_mean)
    ax.set_xlabel('k')
    ax.set_ylabel('Average Computation Time [s]')
    ax.set_title('Average BLLS Computation Time vs. k')
    ax.grid(True)

    return fig


#######################
# Problem 2
#######################


# REQUIRED --- Problem 2a
def plot_recursive_lease_squares():
    
    raw=load_numpy_data('HW5Measurements-P1.npy')
    data = np.array(raw)
    n_trials, length = data.shape
    
    x0_vec = [0, 0, 42000*1e3, 0, 0, 0]
    
    km2_to_m2 = 1000.0**2
    
    P0 = np.zeros((6, 6))
    P0[0:3, 0:3] = 50.0 * km2_to_m2 * np.eye(3)
    P0[3:6, 3:6] = 1.0 * km2_to_m2 * np.eye(3)
    
    R = 100
    dt = 10  # seconds

    F = np.array([
        [1, 0, 0, dt, 0,  0],
        [0, 1, 0, 0,  dt, 0],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0],
        [0, 0, 0, 0,  1,  0],
        [0, 0, 0, 0,  0,  1]
    ])

    H = np.array([0, 0, 1, 0, 0, 0])

    X_traj_all = np.zeros((n_trials, length, 6))
    #print("Running RLS")
    for i in range(n_trials):
        #print("Trial " + str(i+1))
        y_vec = data[i, :]
        results_RLS = run_RLS(length, R, H, F, y_vec, x0_vec, P0)
        x_vec_final = results_RLS['x_final']

        X_traj = np.zeros((length, 6))
        X_traj[0, :] = x_vec_final
        for k in range(1, length):
            X_traj[k, :] = F @ X_traj[k-1, :]

        X_traj_all[i, :, :] = X_traj

    k_vec = np.arange(length)
    fig, ax = plt.subplots()

    # plot one trial line with label
    ax.plot(k_vec,
            X_traj_all[0, :, 2],
            color='orange',
            linewidth=0.6,
            label='individual trials')

    # plot remaining trials to not mess up legend
    for i in range(1, n_trials):
        ax.plot(k_vec,
                X_traj_all[i, :, 2],
                color='orange',
                linewidth=0.6)

    ax.set_xlabel('k')
    ax.set_ylabel('Estimated Position [m]')
    ax.set_title('All Trials: k = {1,...,200}')

    ax.grid(True)
    ax.legend()
    return fig


# REQUIRED --- Problem 2b
def plot_and_describe_sample_mean():
    
    raw=load_numpy_data('HW5Measurements-P1.npy')
    data = np.array(raw)
    n_trials, length = data.shape
    
    x0_vec = [0, 0, 42000*1e3, 0, 0, 0]
    
    km2_to_m2 = 1000.0**2
    
    P0 = np.zeros((6, 6))
    P0[0:3, 0:3] = 50.0 * km2_to_m2 * np.eye(3)
    P0[3:6, 3:6] = 1.0 * km2_to_m2 * np.eye(3)
    
    R = 100
    dt = 10  # seconds

    F = np.array([
        [1, 0, 0, dt, 0,  0],
        [0, 1, 0, 0,  dt, 0],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0],
        [0, 0, 0, 0,  1,  0],
        [0, 0, 0, 0,  0,  1]
    ])

    H = np.array([0, 0, 1, 0, 0, 0])

    X_traj_all = np.zeros((n_trials, length, 6))
    #print("Running RLS")
    for i in range(n_trials):
        #print("Trial " + str(i+1))
        y_vec = data[i, :]
        results_RLS = run_RLS(length, R, H, F, y_vec, x0_vec, P0)
        x_vec_final = results_RLS['x_final']
        sigma_final = results_RLS['sigma_final']

        X_traj = np.zeros((length, 6))
        X_traj[0, :] = x_vec_final
        for k in range(1, length):
            X_traj[k, :] = F @ X_traj[k-1, :]

        X_traj_all[i, :, :] = X_traj

        X_mean = np.mean(X_traj_all, axis=0)

    k_vec = np.arange(length)
    fig, ax = plt.subplots()

    # plot one trial line with label
    ax.plot(k_vec,
            X_traj_all[0, :, 2],
            color='orange',
            linewidth=0.6,
            label='individual trials')

    # plot remaining trials to not mess up legend
    for i in range(1, n_trials):
        ax.plot(k_vec,
                X_traj_all[i, :, 2],
                color='orange',
                linewidth=0.6)

    
    ax.plot(k_vec, X_mean[:, 2], label='mean z', color='C0')
    ax.fill_between(k_vec,
                    X_mean[:, 2] - 3*sigma_final[2],
                    X_mean[:, 2] + 3*sigma_final[2],
                    alpha=0.2,
                    label='+/-3 sigma z')

    ax.set_xlabel('k')
    ax.set_ylabel('Estimated Position [m]')
    ax.set_title('All Trials: k = {1,...,200}')

    ax.grid(True)
    ax.legend()
    
    
    description = """
        The RLS results match the behavior predicted by BLLS. 
        Across all trials, the mean z trajectory produced by RLS is essentially 
        identical to the BLLS result, showing that RLS is also an unbiased estimator. 
        The +/- 3 sigma bands also match the BLLS envelopes at k = 200, which 
        demonstrates that the recursive solution converges to the BLLS covariance
        once all measurements have been processed. Differences at small k are expected, 
        because RLS uncertainty is generally large early on but at k= 200, it decreased
        as more measurements were incorporated. Overall, the RLS implementation reproduces 
        the BLLS mean and final covariance, as expected for a linear Gaussian system.
    """
    return fig, description


# REQUIRED --- Problem 2c
def plot_and_describe_time():
    
    raw=load_numpy_data('HW5Measurements-P1.npy')
    data = np.array(raw)
    n_trials, length = data.shape

    x0_vec = [0, 0, 42000*1e3, 0, 0, 0]
    
    km2_to_m2 = 1000.0**2
    
    P0 = np.zeros((6, 6))
    P0[0:3, 0:3] = 50.0 * km2_to_m2 * np.eye(3)
    P0[3:6, 3:6] = 1.0 * km2_to_m2 * np.eye(3)

    R = 100
    dt = 10

    F = np.array([
        [1, 0, 0, dt, 0,  0],
        [0, 1, 0, 0,  dt, 0],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0],
        [0, 0, 0, 0,  1,  0],
        [0, 0, 0, 0,  0,  1]
    ])

    H = np.array([0, 0, 1, 0, 0, 0])

    k_vec = np.arange(1, length + 1)
    t_mean = np.zeros(length)

    for idx, k in enumerate(k_vec):
        t_sum = 0.0
        for i in range(n_trials):
            y_vec = data[i, :k]
            results = run_RLS(k, R, H, F, y_vec, x0_vec, P0)
            t_sum += results['t_execution']
        t_mean[idx] = t_sum / n_trials

    fig, ax = plt.subplots()
    ax.plot(k_vec, t_mean)
    ax.set_xlabel('k')
    ax.set_ylabel('Average Computation Time [s]')
    ax.set_title('Average RLS Computation Time vs. k')
    ax.grid(True)

    description = """
        The RLS timing grows almost perfectly linearly, but the BLLS 
        estimator, while noisier, remains much faster. RLS is linear with k because 
        the algorithm performs one "fixed cost" update per measurement. 
        
        BLLS, with a fixed 6DOF system, only inverts a small 6x6 matrix and its cost comes 
        mainly from building 200x6 arrays, which NumPy seems to handle very efficiently. 
        
        RLS runs many small 6x6 operations inside a loop, giving it a much larger 
        constant cost per measurement. As a result, for k ~= 200 the BLLS implementation is faster. 
        
        In application, RLS would be preferred for large sets of measurements because as k increases,
        the R and H matricies become impossible to manage and the program would crash. Also, RLS
        allows updating as measurements are streamed in because it updates the estimate incrementally 
        and avoids recomputing the entire batch solution whenever a new measurement arrives.
    """
    return fig, description


#######################
# Problem 3
#######################


# REQUIRED --- Problem 3b
def compute_final_x_and_P():
    
    raw=load_numpy_data('HW5Measurements-P3.npy')
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

    results_KF = run_KF(length, y, x0_hat, P0, F, H, Q, R)

    x_final = results_KF['x_final']
    P_final = results_KF['P_final']

    return x_final, P_final


# REQUIRED --- Problem 3c
def plot_pure_prediction():
    
    raw=load_numpy_data('HW5Measurements-P3.npy')
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

    results_KF = run_KF(length, y, x0_hat, P0, F, H, Q, R)

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


# REQUIRED --- Problem 3d
def plot_with_measurement_updates():
    return fig


# REQUIRED --- Problem 3e
def describe_differences():
    description = """
        Write your answer here.
    """
    return description


# REQUIRED --- Problem 3f
def plot_and_describe_residuals():
    fig = None
    description = """
        Write your answer here.
        """
    return fig, description


###############################################
# Main Script to test / debug your code
# This will not be run by the autograder
# the individual functions above will be called and tested
###############################################


def main():
    # Problem 1
    # plot_batch_least_squares_single_trial()
    # plot_batch_least_squares_all_trials()
    # plot_state_estimate_histograms()
    # plot_execution_time_vs_measurements()

    # # # Problem 2
    # plot_recursive_lease_squares()
    # plot_and_describe_sample_mean()
    # plot_and_describe_time()

    # # Problem 3
    print(compute_final_x_and_P())
    plot_pure_prediction()
    # plot_with_measurement_updates()
    # describe_differences()
    # plot_and_describe_residuals()

    plt.show()


if __name__ == "__main__":
    main()
