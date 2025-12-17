import numpy as np
import matplotlib.pyplot as plt

## Givens/problem statement

X_oe = [7000, 0.2, 45, 0, 270, 78.75] 
#a(km), e(dimless), i(deg), omega(deg), Omega(deg), theta(deg)

DSN0_loc = [35.297, -116.914] #lat(deg), long(deg)
DSN1_loc = [40.4311, -4.248]
DSN2_loc = [-35.4023, 148.9813]

R_Earth = 6378.137 #km
omega_EN = 7.292115e-5 #rad/sec
gamma_0 = 0 #deg

## Extract/present data

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

## 1(e) plotting

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
plt.show()

## 3(a) Initializing x0 p0 R and q
## Functions OE >> State
def orbital_elements_to_state(oe):
    a, e, i, omega, Omega, f = oe
    # Semilatus rectum
    p = a * (1.0 - e**2)
    # Radius
    r_pf = p / (1.0 + e * np.cos(f))
    # Perifocal position and velocity
    r_PF = np.array([r_pf * np.cos(f), r_pf * np.sin(f), 0.0])
    v_PF = np.sqrt(mu / p) * np.array([-np.sin(f), e + np.cos(f), 0.0])

    # Rotation from PF to inertial: R3(Omega)*R1(i)*R3(omega)
    cO = np.cos(Omega); sO = np.sin(Omega)
    co = np.cos(omega); so = np.sin(omega)
    ci = np.cos(i);     si = np.sin(i)

    R3_O = np.array([[ cO, -sO, 0.0],
                     [ sO,  cO, 0.0],
                     [0.0, 0.0, 1.0]])
    R1_i = np.array([[1.0, 0.0, 0.0],
                     [0.0,  ci, -si],
                     [0.0,  si,  ci]])
    R3_o = np.array([[ co, -so, 0.0],
                     [ so,  co, 0.0],
                     [0.0, 0.0, 1.0]])

    Q = R3_O @ R1_i @ R3_o

    r_N = Q @ r_PF
    v_N = Q @ v_PF
    X_N = np.hstack((r_N, v_N))
    return X_N
    
# --- x0 from the given OE (deg -> rad) ---
oe_deg = np.array([7000, 0.2, 45, 0, 270, 78.75], dtype=float)
oe = oe_deg.copy()
oe[2:] = np.deg2rad(oe[2:])          
x_hat = orbital_elements_to_state(oe) # 6x1 state (km, km/s)

# --- P0 ---
sigma_r0 = 10.0 # km
sigma_v0 = 0.01 # km/s  
P = np.diag([sigma_r0**2]*3 + [sigma_v0**2]*3)

# --- R ---
sigma_rho    = 1e-3 # km 
sigma_rhodot = 1e-5 # km/s 
R = np.diag([sigma_rho**2, sigma_rhodot**2])

# --- Process noise strength ---
sigma_a = 1e-7 # km/s^2 
q_a = sigma_a**2
