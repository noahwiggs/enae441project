import numpy as np
import matplotlib.pyplot as plt

def load_numpy_data(file_path):
    import os
    cur_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    data = np.load(cur_dir + file_path, allow_pickle=True)
    print(f"Loaded data from {file_path}")
    return data

def oe_conversion(X_oe):
    from orbits import orbital_elements_to_state
    X_oe_rad = X_oe.copy()
    X_oe_rad[2:] = np.deg2rad(X_oe_rad[2:])
    x_hat = orbital_elements_to_state(X_oe_rad)
    print("OE conversion x_hat:", x_hat)
    return x_hat

def write_to_csv(arr, filename):
    if arr.ndim != 2:
        arr = arr.reshape(-1, arr.shape[-1])
    arr = arr[~np.all(arr == 0, axis=1)]
    header = "time,siteID,range,range_rate"
    np.savetxt(filename, arr, delimiter=",", fmt="%.10f", header=header, comments="")

def extract_present_data(raw_data):
    N = raw_data.shape[0]
    DSN0_data = []
    DSN1_data = []
    DSN2_data = []

    for i in range(N):
        GS_id = raw_data[i, 1]
        if GS_id == 0:
            DSN0_data.append(raw_data[i]) 
        elif GS_id == 1:
            DSN1_data.append(raw_data[i])
        elif GS_id == 2:
            DSN2_data.append(raw_data[i])
        else:
            print('Error sorting measurement index ' + str(i))

    DSN0_data = np.array(DSN0_data)
    DSN1_data = np.array(DSN1_data)
    DSN2_data = np.array(DSN2_data)

    write_to_csv(raw_data,"raw_data.csv")
    write_to_csv(DSN0_data,"DSN0_data.csv")
    write_to_csv(DSN1_data,"DSN1_data.csv")
    write_to_csv(DSN2_data,"DSN2_data.csv")

    DSN_list = [DSN0_data, DSN1_data, DSN2_data]
    colors = ["tab:green", "tab:orange", "tab:blue"]
    labels = ["DSN0", "DSN1", "DSN2"]

    marker_size = 5

    #Fig 1 (range on top, range rate on bottom)
    fig1, axs = plt.subplots(2, 3, figsize=(10, 7), sharex=True)

    for i, data in enumerate(DSN_list):
        t = data[:,0]
        rho = data[:,2]
        rho_dot = data[:,3]

        axs[0, i].plot(t, rho, '.', color=colors[i], markersize=marker_size)
        axs[0, i].set_title(f"{labels[i]} range")
        axs[0, i].set_ylabel("range [km]")
        axs[0, i].grid(True)

        axs[1, i].plot(t, rho_dot, '.', color=colors[i], markersize=marker_size)
        axs[1, i].set_title(f"{labels[i]} range rate")
        axs[1, i].set_xlabel("time [s]")
        axs[1, i].set_ylabel("range rate [km/s]")
        axs[1, i].grid(True)

    # Figure 2 (combined plots)
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 7), sharex=True)

    for i, data in enumerate(DSN_list):
        t = data[:,0]
        axs2[0].plot(t, data[:,2], '.', color=colors[i], markersize=marker_size, label=labels[i])
        axs2[1].plot(t, data[:,3], '.', color=colors[i], markersize=marker_size, label=labels[i])

    axs2[0].set_title("Measured range vs time")
    axs2[0].set_xlabel("time [s]")
    axs2[0].set_ylabel("range [km]")
    axs2[0].grid(True)
    axs2[0].legend()

    axs2[1].set_title("Measured range rate vs time")
    axs2[1].set_xlabel("time [s]")
    axs2[1].set_ylabel("range rate [km/s]")
    axs2[1].grid(True)
    axs2[1].legend()

    fig1.tight_layout()
    fig2.tight_layout()

def initial_blls_guess(x_hat, raw_data):
    from stations import stations_eci_func
    from dynamics import propagate_func
    from blls_fcns import meas_func, H_func
    from blls_init import blls_x0

    sigma_r0 = 10.0 # km
    sigma_v0 = 0.5 # km/s
    P0_guess = np.diag([sigma_r0**2]*3 + [sigma_v0**2]*3)

    sigma_rho = 1e-3  # km
    sigma_rhodot = 1e-5  # km/s
    R_guess = np.diag([sigma_rho**2, sigma_rhodot**2])

    t_window = 50.0 #seconds
    x0_blls, P0_blls, dx0 = blls_x0(raw_data,x_hat,P0_guess,R_guess,stations_eci_func,propagate_func,meas_func,H_func, t_window)
    
    print("BLLS x0:", x0_blls)
    print("BLLS P0: ", P0_blls)
    return x0_blls, P0_blls

def main():
    # Givens/load data
    X_oe = np.array([7000, 0.2, 45, 0, 270, 78.75], dtype = float)
    mu = 3.986e5
    #a(km), e(dimless), i(deg), omega(deg), Omega(deg), theta(deg)

    raw_data = load_numpy_data('Project-Measurements-Easy.npy')
    length = raw_data.shape[0]

    #preliminary functions
    x_hat = oe_conversion(X_oe)
    #extract_present_data(raw_data)

    #Run BLLS to get better filter initialization
    x0_blls, P0_blls = initial_blls_guess(x_hat, raw_data)

    #Filter initialization
    a = 1e-15 #micrometers/s^2

    R = np.array([[(51*1e-3)**2,0],
                  [0, (1e-5)**2]])

    mu0 =x0_blls

    P0 = P0_blls

    #Run filters and output Q3, 4, 5 results
    #from EKF import run_EKF_prediction_only, plot_pure_prediction, plot_orbit_xy_samples, plot_prediction_covariance_envelope
    #results_prediction = run_EKF_prediction_only(raw_data, length, mu0, P0, mu, a)
    #plot_pure_prediction(results_prediction)
    #plot_orbit_xy_samples(results_prediction)
    #plot_prediction_covariance_envelope(results_prediction)

    from EKF import run_EKF, plot_EKF_covariance_envelope, plot_EKF_state_update_difference
    results_EKF = run_EKF(raw_data, length, mu0, P0, mu, a, R)
    #plot_EKF_covariance_envelope(results_EKF)
    #plot_EKF_state_update_difference(results_EKF)

    from EKF import plot_postfit_residuals, plot_state_estimate_with_3sigma
    plot_postfit_residuals(results_EKF)
    plot_state_estimate_with_3sigma(results_EKF)
    x_final = results_EKF['x_final']
    P_final = results_EKF['P_final']
    sigma_final = np.sqrt(np.diag(P_final))
    print('Final state estimate:')
    print(x_final)
    print('Final 1-sigma uncertainty:')
    print(sigma_final)

    plt.show()

if __name__ == "__main__":
    main()