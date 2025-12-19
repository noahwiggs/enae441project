import numpy as np
import matplotlib.pyplot as plt
from orbits import orbital_elements_to_state
from stations import stations_eci_func
from dynamics import propagate_func
from measurements import meas_func, H_func
from blls_init import blls_x0


## Givens/problem statement

X_oe = np.array([7000, 0.2, 45, 0, 270, 78.75], dtype = float)
#a(km), e(dimless), i(deg), omega(deg), Omega(deg), theta(deg)
X_oe_rad = X_oe.copy()
X_oe_rad[2:] = np.deg2rad(X_oe_rad[2:])
x_hat = orbital_elements_to_state(X_oe_rad)
print("OE conversion x_hat:", x_hat)

##################################################################
## =================== Extract/present data =================== ##
##################################################################

def load_numpy_data(file_path):
    import os
    cur_dir = os.path.dirname(os.path.abspath(__file__)) + "/"
    data = np.load(cur_dir + file_path, allow_pickle=True)
    print(f"Loaded data from {file_path}")
    return data

def write_to_csv(arr, filename):
    if arr.ndim != 2:
        arr = arr.reshape(-1, arr.shape[-1])
    arr = arr[~np.all(arr == 0, axis=1)]
    header = "time,siteID,range,range_rate"
    np.savetxt(filename, arr, delimiter=",", fmt="%.10f", header=header, comments="")

raw_data = load_numpy_data('Project-Measurements-Easy.npy')

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

###########################################################
## =================== 1(e) Plotting =================== ##
###########################################################

DSN_list = [DSN0_data, DSN1_data, DSN2_data]
colors = ["tab:green", "tab:orange", "tab:blue"]
labels = ["DSN0", "DSN1", "DSN2"]

marker_size = 5

#Fig 1 (range on top, range rate on bottom)
fig1, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)

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
fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

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

###########################################################
## =========== 3(a) Initial Guess using BLLS =========== ##
###########################################################
x0_blls, P0_blls, dx0 = blls_x0(
    raw_data=raw_data,
    x0_nom=x_hat,
    P0_nom=P0,
    R_meas=R_blls,
    stations_eci_func=stations_eci_func,
    propagate_func=propagate_func,
    meas_func=meas_func,
    H_func=H_func,
    t_window=50.0
)
print("BLLS x0:", x0_blls)
print("BLLS P0: ", P0_blls)

###########################################################
## ============ Extended Kalman Filter ================= ##
###########################################################
from EKF import run_EKF
meas = np.load("Project-Measurements-Easy.npy")
length = meas.shape[0]

a = 1e-15 #mm/s^2

R = np.array([1e-3**2,0],
             [0, 1e-5**2])

mu0 = np.array([
    4.48528055e+03, -1.26238277e+03,  4.48527073e+03,
    2.15123949e+00,  7.55370288e+00,  2.15134590e+00
], dtype=float)

#run EKF
results = run_EKF(length, mu0, P0_blls, a, R)

#plot EKF results
from EKF import plot_pure_prediction
plot_pure_prediction(results)


from EKF import plot_with_updates
plot_with_updates(results)

plt.show()
