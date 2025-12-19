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
    update_times = np.zeros(length)

    x_k[0, :] = x0_vec
    P_k[0, :, :] = P0

    I = np.eye(6)

    t_start = time.time()
    for k in range(length):

        t0 = time.time()   

        x_prev = x_k[k, :]
        P_prev = P_k[k, :, :]

        K = P_prev @ H.T / (H @ P_prev @ H.T + R)
        x_upd = x_prev + K * (y_vec[k] - H @ x_prev)

        KH = np.outer(K, H)
        KRK = np.outer(K, K) * R
        P_upd = (I - KH) @ P_prev @ (I - KH).T + KRK

        x_k[k + 1, :] = x_upd
        P_k[k + 1, :, :] = P_upd
        K_k[k, :] = K.flatten()
        update_times[k] = time.time() - t0  

    t_execution = time.time() - t_start

    sigma_over_k = np.zeros((length, 6))
    for k in range(1, length + 1):
        sigma_over_k[k - 1, :] = np.sqrt(np.diag(P_k[k]))

    results = {
        'x_k_vec': x_k,
        'P_k_vec': P_k,
        'K_k_vec': K_k,
        't_execution': t_execution,
        'update_times': update_times,
        'x_final': x_k[-1, :],
        'P_final': P_k[-1, :, :],
        'sigma_final': np.sqrt(np.diag(P_k[-1])),
        'estimates_over_measurements': x_k[1:, :],
        'sigma_over_measurements': sigma_over_k
    }

    return results

def run_KF_prediction_only(length, mu0, P0, F, Q):
    mu_minus = np.zeros((length, 6))
    P_minus  = np.zeros((length, 6, 6))
    sigma_minus = np.zeros((length, 6))

    mu_prev = mu0
    P_prev  = P0

    t_start = time.time()

    for k in range(length):
        mu_pred = F @ mu_prev
        P_pred  = F @ P_prev @ F.T + Q

        mu_minus[k] = mu_pred
        P_minus[k]  = P_pred
        sigma_minus[k] = np.sqrt(np.diag(P_pred))  

        mu_prev = mu_pred
        P_prev  = P_pred

    t_end = time.time()

    return {
        'mu_minus': mu_minus,             
        'P_minus': P_minus,                
        'sigma_minus': sigma_minus,        
        'sigma3_minus': 3 * sigma_minus,   
        't_execution': t_end - t_start
    }

def run_KF(length, y, mu0, P0, F, H, Q, R):

    mu_plus_vec = np.zeros((length + 1, 6))
    P_plus_vec = np.zeros((length + 1, 6, 6))
    mu_minus_vec = np.zeros((length, 6))
    P_minus_vec = np.zeros((length, 6, 6))
    del_y_vec = np.zeros((length, 3))

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
        del_y = y[k] - H @ mu_plus

        del_y_vec[k] = del_y
        mu_plus_vec[k+1] = mu_plus
        P_plus_vec[k+1]  = P_plus
        
    t_end = time.time()

    results_dict = {
        'mu_minus': mu_minus_vec,
        'P_minus': P_minus_vec,
        'mu_plus': mu_plus_vec,
        'P_plus': P_plus_vec,
        'x_final': mu_plus_vec[-1],
        'P_final': P_plus_vec[-1],
        'del_y': del_y_vec,
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

    final_results = np.zeros((200, 6))

    for L in range(1, 201):
        y_sub = y_vec[:L]
        results_BLLS = run_BLLS(L, R, H, F, y_sub)
        x_hat = results_BLLS['x_vec']
        final_results[L-1, :] = x_hat

    fig, ax = plt.subplots()
    k_vec = np.arange(length)
    ax.plot(k_vec, final_results[:,2], label='z', color="orange")
    ax.set_xlabel('k')
    ax.set_ylabel('Estimated z [m]')
    ax.set_title('BLLS z Estimate for Trial 1')
    ax.grid(True)

    return fig

# REQUIRED --- 1c
def plot_batch_least_squares_all_trials():

    raw = load_numpy_data('HW5Measurements-P1.npy')
    data = np.array(raw)

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

    n_trials = data.shape[0]
    n_meas = data.shape[1]

    Z_all = np.zeros((n_trials, n_meas))
    sigma_z = np.zeros(n_meas)

    for L in range(1, n_meas + 1):
        y_sub_first_trial = data[0, :L]
        results = run_BLLS(L, R, H, F, y_sub_first_trial)
        sigma_z[L-1] = results["sigma"][2]

    for trial in range(n_trials):
        y_vec = data[trial, :]
        for L in range(1, n_meas + 1):
            y_sub = y_vec[:L]
            results = run_BLLS(L, R, H, F, y_sub)
            Z_all[trial, L-1] = results['x_vec'][2]

    Z_mean = np.mean(Z_all, axis=0)

    k_vec = np.arange(1, n_meas + 1)
    fig, ax = plt.subplots()

    ax.plot(k_vec,
            Z_all[0, :],
            color='orange',
            linewidth=0.6,
            label='individual trials')

    for i in range(1, n_trials):
        ax.plot(k_vec,
                Z_all[i, :],
                color='orange',
                linewidth=0.6)

    ax.plot(k_vec,
            Z_mean,
            color='C0',
            linewidth=2,
            label='mean')

    ax.fill_between(k_vec,
                    Z_mean - 3 * sigma_z,
                    Z_mean + 3 * sigma_z,
                    color='C0',
                    alpha=0.2,
                    label='+/-3sigma bounds')

    ax.set_xlabel('k')
    ax.set_ylabel('Estimated z [m]')
    ax.set_title('BLLS z Estimate Across All Trials')
    ax.grid(True)
    ax.legend()

    return fig


# REQUIRED --- 1d
def plot_state_estimate_histograms():
    raw = load_numpy_data('HW5Measurements-P1.npy')
    data = np.array(raw)

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

    n_trials = data.shape[0]
    n_meas = data.shape[1]

    # store x_hat for all trials, all L
    X_all = np.zeros((n_trials, n_meas, 6))

    for trial in range(n_trials):
        y_vec = data[trial, :]
        for L in range(1, n_meas + 1):
            y_sub = y_vec[:L]
            results = run_BLLS(L, R, H, F, y_sub)
            X_all[trial, L-1, :] = results["x_vec"]

    k_indices = [9, 49, 199]   # 10, 50, 200
    k_labels = [10, 50, 200]

    state_labels = ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot']

    figs = []

    for k_idx, k_label in zip(k_indices, k_labels):

        X_k = X_all[:, k_idx, :]   # shape (n_trials, 6)
        mu_k = np.mean(X_k, axis=0)
        sigma_k = np.std(X_k, axis=0)

        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        axes = axes.flatten()

        for j in range(6):
            ax = axes[j]
            ax.hist(X_k[:, j], bins=7, color="C0", edgecolor="black")
            ax.axvline(mu_k[j], linestyle='--', color='red')
            ax.set_title(state_labels[j])

            ax.text(
                0.05,
                0.95,
                f"μ = {mu_k[j]:.3e}\nsigma^2 = {sigma_k[j]**2:.3e}",
                transform=ax.transAxes,
                va='top'
            )

        fig.suptitle(f"Histograms of State Estimates at k = {k_label}")
        fig.tight_layout()
        figs.append(fig)
    
    description = """
        The histograms show that only the z state has spread across trials. 
        This is expected because H = [0 0 1 0 0 0] only appears in the z position, 
        and the system is otherwise static, so x, y, xdot, ydot, and zdot remain zero 
        for all 50 trials. Their means are zero and their variances are also effectively zero, 
        so all samples fall in a single histogram bin in the center. In contrast, the z position 
        shows a roughly Gaussian distribution with a variance reflecting the measurement noise 
        which was propagated through the BLLS estimator. As more measurements are used, the histogram
        approaches a more gaussian shape, appearing the most normal at 200 measurements.
    """
    return figs, description


# REQUIRED --- 1e
def plot_execution_time_vs_measurements():
    raw = load_numpy_data('HW5Measurements-P1.npy')
    data = np.array(raw)

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

    n_trials, length = data.shape

    k_vec = np.arange(1, length + 1)
    t_mean = np.zeros(length)

    for k in k_vec:
        t_sum = 0.0
        for trial in range(n_trials):
            y_vec = data[trial, :k]
            results = run_BLLS(k, R, H, F, y_vec)
            t_sum += results["t_execution"]

        t_mean[k - 1] = t_sum / n_trials

    fig, ax = plt.subplots()
    ax.plot(k_vec, t_mean)
    ax.set_xlabel("k")
    ax.set_ylabel("Average Computation Time [s]")
    ax.set_title("Average BLLS Computation Time vs. k")
    ax.grid(True)

    return fig


#######################
# Problem 2
#######################


# REQUIRED --- Problem 2a
def plot_recursive_lease_squares():
    
    raw = load_numpy_data('HW5Measurements-P1.npy')
    data = np.array(raw)
    n_trials, length = data.shape

    x0_vec = [0, 0, 42000 * 1e3, 0, 0, 0]

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

    X_all = np.zeros((n_trials, length, 6))

    for trial in range(n_trials):
        y_vec = data[trial, :]
        results_RLS = run_RLS(length, R, H, F, y_vec, x0_vec, P0)

        X_all[trial, :, :] = results_RLS['estimates_over_measurements']

    k_vec = np.arange(1, length + 1)

    fig, ax = plt.subplots()

    ax.plot(k_vec,
            X_all[0, :, 2],
            color='orange',
            linewidth=0.6,
            label='individual trials')

    for trial in range(1, n_trials):
        ax.plot(k_vec,
                X_all[trial, :, 2],
                color='orange',
                linewidth=0.6)

    ax.set_xlabel('k')
    ax.set_ylabel('Estimated z [m]')
    ax.set_title('RLS z Estimate Across All Trials')
    ax.grid(True)
    ax.legend()


# REQUIRED --- Problem 2b
def plot_and_describe_sample_mean():
    raw = load_numpy_data('HW5Measurements-P1.npy')
    data = np.array(raw)
    n_trials, length = data.shape

    x0_vec = [0, 0, 42000 * 1e3, 0, 0, 0]

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

    # Store trajectories and sigma from RLS
    X_all = np.zeros((n_trials, length, 6))
    sigma_all = np.zeros((n_trials, length, 6))

    for trial in range(n_trials):
        y_vec = data[trial, :]
        results = run_RLS(length, R, H, F, y_vec, x0_vec, P0)
        X_all[trial, :, :] = results["estimates_over_measurements"]
        sigma_all[trial, :, :] = results["sigma_over_measurements"]

    X_mean = np.mean(X_all, axis=0)

    sigma_mean = np.mean(sigma_all, axis=0)  

    k_vec = np.arange(1, length + 1)

    fig, ax = plt.subplots()

    ax.plot(k_vec,
            X_all[0, :, 2],
            color='orange',
            linewidth=0.6,
            label='individual trials')

    for trial in range(1, n_trials):
        ax.plot(k_vec,
                X_all[trial, :, 2],
                color='orange',
                linewidth=0.6)

    ax.plot(k_vec,
            X_mean[:, 2],
            label='mean z',
            color='C0',
            linewidth=2)

    ax.fill_between(k_vec,
                    X_mean[:, 2] - 3 * sigma_mean[:, 2],
                    X_mean[:, 2] + 3 * sigma_mean[:, 2],
                    color='C0',
                    alpha=0.2,
                    label='+/-3sigma')
    #ax.scatter(0, x0_vec[2], color='red', label='initial guess')
    ax.set_xlabel('k')
    ax.set_ylabel('Estimated z [m]')
    ax.set_title('RLS z Estimate Across All Trials')
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
    
    raw = load_numpy_data('HW5Measurements-P1.npy')
    data = np.array(raw)
    n_trials, length = data.shape

    x0_vec = [0, 0, 42000 * 1e3, 0, 0, 0]

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

    t_mean = np.zeros(length)
    k_vec = np.arange(1, length + 1)

    for k in k_vec:
        t_sum = 0.0

        for trial in range(n_trials):
            y_vec = data[trial, :k]
            results = run_RLS(k, R, H, F, y_vec, x0_vec, P0)
            t_sum += results["update_times"][k - 1]

        t_mean[k - 1] = t_sum / n_trials

    fig, ax = plt.subplots()
    ax.plot(k_vec, t_mean)
    ax.set_xlabel("k")
    ax.set_ylabel("Average Update Time [s]")
    ax.set_title("Average Per-Update RLS Computation Time vs k")
    ax.grid(True)

    description = """
        The BLLS time to compute at each step steadily increases as the number of measurements increases.
        BLLS, with a fixed 6DOF system, only inverts a small 6x6 matrix and its cost comes 
        mainly from building 200x6 arrays, which NumPy seems to handle very efficiently. The time to compute
        per added measurememt is increasing but remains small.
        
        The RLS timing is constant per measurement. There is some noise, but generally the time to compute each
        step does not increase as the number of measurments increases. RLS runs many small 6x6 operations 
        inside a loop, giving it a larger constant cost per measurement. At low measurment counts, computing
        overhead in Python may cause RLS to be slower than BLLS.
        
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
    x0_hat = [0, 0, 500, 0.01, 0, 0.01] #km
    
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
    R = V/dt #3x3

    results_KF = run_KF(length, y, x0_hat, P0, F, H, Q, R)

    x_final = results_KF['x_final']
    P_final = results_KF['P_final']

    return x_final, P_final


# REQUIRED --- Problem 3c
def plot_pure_prediction():
    
    raw = load_numpy_data('HW5Measurements-P3.npy')
    data = raw.item()
    y = data['Y']                  
    t = data['t']
    dt = t[-1]/(len(t)-1)  # should be 1 second
    length = y.shape[0]
    
    # Givens
    x0_hat = np.array([0, 0, 500, 0.01, 0, 0.01])  # km
    
    P0 = np.block([
        [50 * np.eye(3),     np.zeros((3,3))],
        [np.zeros((3,3)),    0.1 * np.eye(3)]
    ])

    W = 1e-5*np.eye(3)
    V = 1e3*np.eye(3)

    # DT state transition matrix
    F = np.array([
        [1, 0, 0, dt, 0,  0],
        [0, 1, 0, 0,  dt, 0],
        [0, 0, 1, 0,  0,  dt],
        [0, 0, 0, 1,  0,  0],
        [0, 0, 0, 0,  1,  0],
        [0, 0, 0, 0,  0,  1]
    ])
    
    H = np.array([
        [1, 0, 0, 0, 0,  0],
        [0, 1, 0, 0,  0, 0],
        [0, 0, 1, 0,  0,  0]
    ])
    
    Q = np.block([
        [W*(dt**3)/3, W*(dt**2)/2],
        [W*(dt**2)/2, W*dt]
    ])
    
    R = V/dt

    results_KF = run_KF_prediction_only(length, x0_hat, P0, F, Q)

    mu_minus = results_KF['mu_minus'] 
    P_minus  = results_KF['P_minus']

    state_names = ["x", "y", "z", "vx", "vy", "vz"]
    k_vec = np.arange(mu_minus.shape[0])

    fig, axes = plt.subplots(6, 1, figsize=(10, 16), sharex=True)
    plt.suptitle("Pure Prediction Estimates (-)")

    for i in range(6):
        mu_i = mu_minus[:, i]
        sigma_i = np.sqrt(P_minus[:, i, i]) 

        axes[i].plot(k_vec, mu_i, label=f'{state_names[i]} estimate')
        axes[i].fill_between(k_vec,
                             mu_i - 3*sigma_i,
                             mu_i + 3*sigma_i,
                             alpha=0.2,
                             label='±3σ bounds')

        axes[i].set_ylabel(state_names[i])
        axes[i].grid(True)
        axes[i].legend(loc='upper right')

    axes[-1].set_xlabel('time step')
    plt.tight_layout()

    return fig

# REQUIRED --- Problem 3d
def plot_with_measurement_updates():
    
    raw=load_numpy_data('HW5Measurements-P3.npy')
    data = raw.item()
    y = data['Y']                  
    t = data['t']
    dt = t[-1]/(len(t)-1) #should be 1 second
    length = y.shape[0]
    
    #Givens
    x0_hat = [0, 0, 500, 0.01, 0, 0.01] #km
    
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
    R = V/dt #3x3

    results_KF = run_KF(length, y, x0_hat, P0, F, H, Q, R)

    mu_plus = results_KF['mu_plus']
    P_plus = results_KF['P_plus']

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


# REQUIRED --- Problem 3e
def describe_differences():
    description = """
        The pure prediction results (muk-, Pk-) show the expected behavior: the small
        non zero initial velocities produce a drift in x, z positions, and the covariance grows
        since no measurements are used to constrain the estimate. This leads to the
        widening +/-3sigma bands seen in the prediction only plots.

        In contrast, the measurement corrected estimates (muk+, Pk+) do not drift. Each update
        pulls the state back toward the measurements, rapidly shrinking the covariance. The +/-3sigma
        bands shrink and remain that way, demonstrating that measurement updates suppress the growth 
        seen in the prediction case.

    """
    return description


# REQUIRED --- Problem 3f
def plot_and_describe_residuals():
    
    raw=load_numpy_data('HW5Measurements-P3.npy')
    data = raw.item()
    y = data['Y']                  
    t = data['t']
    dt = t[-1]/(len(t)-1) #should be 1 second
    length = y.shape[0]
    
    #Givens
    x0_hat = [0, 0, 500, 0.01, 0, 0.01] #km
    
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
    R = V/dt #3x3

    results_KF = run_KF(length, y, x0_hat, P0, F, H, Q, R)

    del_y = results_KF['del_y']

    k_vec = np.arange(1, length+1)
    fig, ax = plt.subplots()
    ax.plot(k_vec, del_y[:,0], label='x')
    ax.plot(k_vec, del_y[:,1], label='y')
    ax.plot(k_vec, del_y[:,2], label='z')
    ax.set_xlabel('k')
    ax.set_ylabel('Residual [km]')
    ax.set_title('Corrected Measurement Residual (+)')
    ax.legend()
    ax.grid(True)
    
    description = """
        The residual starts large in in z because of the initial state uncertainty,
        but rapidly shrinks and the filter converges. After the first 100 steps all
        x, y and z all fluctuate around zero with no bias and with variance consistent 
        with the measurement noise. This behavior matches the expected behavior of a correct 
        Kalman filter.
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

    # # # # Problem 2
    # plot_recursive_lease_squares()
    # plot_and_describe_sample_mean()
    # plot_and_describe_time()

    # # Problem 3
    print(compute_final_x_and_P())
    plot_pure_prediction()
    plot_with_measurement_updates()
    describe_differences()
    plot_and_describe_residuals()

    plt.show()


if __name__ == "__main__":
    main()
