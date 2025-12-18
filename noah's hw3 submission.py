import os

import matplotlib.pyplot as plt
import numpy as np

# CONSTANTS
# Gravitational parameter for Earth in km^3/s^2
mu = 398600  # km^3/s^2
omega_EN = 7.2921150e-5  # rad /s

# Problem 2
X_0_spring = np.array([1, 0])  # m, m/s

X_N_0 = np.array([7000, 0, 0, 0, 7.5, 3.5])  # km, km/s
dX_N_0 = np.array([30, 0, 0, 0, 0, 0.1])  # km, km/s

cur_dir = os.path.dirname(__file__)

###############################################
# OPTIONAL FUNCTIONS TO AID IN DEBUGGING
# These are not graded functions, but may help you debug your code.
# Keep the function signatures the same if you want autograder feedback!
###############################################


def load_numpy_data(file_path):
    data = np.load(file_path, allow_pickle=True).item()
    print(f"Data keys loaded from {file_path}: {list(data.keys())}")
    print(
        "Query the data dictionary using `data['key_name']` to access specific data arrays."
    )
    return data

def integrator(X_dot_fcn, X_0, t_vec):
    from scipy.integrate import solve_ivp
    t_span = [t_vec[0], t_vec[-1]]
    t_eval = t_vec
    sol = solve_ivp(X_dot_fcn, t_span, X_0, t_eval=t_eval, rtol=1e-9, atol=1e-9,)
    return sol.y    

def R3(th):
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]])

def r_station_eci(t):
    phi = np.deg2rad(30.0)
    lam = np.deg2rad(60.0)
    RE = 6378.0
    omegaE = 7.2921150e-5
    r_site_ecef = np.array([RE*np.cos(phi)*np.cos(lam),
                            RE*np.cos(phi)*np.sin(lam),
                            RE*np.sin(phi)])
    return R3(omegaE*t) @ r_site_ecef


###############################################
# REQUIRED FUNCTIONS FOR AUTOGRADER
# Keep the function signatures the same!!
###############################################


#######################
# Problem 1
#######################


# REQUIRED --- 1a
def propogate_CT_LTI_numerically(X_0, X_dot_fcn, t_vec):
    # Return trajectory over time where np.shape(X_t) = (len(t_vec), len(X_0))
    X_t = integrator(X_dot_fcn, X_0, t_vec).T

    plt.figure()
    plt.plot(t_vec, X_t[:,0], label="x(t)")
    plt.plot(t_vec, X_t[:,1], label="xdot(t)")
    plt.xlabel("t")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return X_t

def propogate_CT_LTI_numerically_no_plot(X_0, X_dot_fcn, t_vec):
    # Return trajectory over time where np.shape(X_t) = (len(t_vec), len(X_0))
    X_t = integrator(X_dot_fcn, X_0, t_vec).T

    return X_t

# REQUIRED --- 1b
def propogate_CT_LTI_analytically(X_0, A, t_vec):
    # Return trajectory over time where np.shape(X_t) = (len(t_vec), len(X_0))
    from scipy.linalg import expm
    t_0 = t_vec[0]
    soln = []

    for i in range(len(t_vec)):
        element = expm(A * (t_vec[i] - t_0)) @ X_0
        soln.append(element)
    
    X_t = np.vstack(soln)
    return X_t


# REQUIRED --- 1c
def propogate_DT_LTI_analytically(X_0, A, dt, k_max):
    # Return trajectory over time where np.shape(X_t) = (len(t_vec), len(X_0))
    from scipy.linalg import expm
    F = expm(A*dt) 
    
    soln = []
    for k in range(k_max+1):
        element = np.linalg.matrix_power(F,k) @ X_0
        soln.append(element)    
    
    X_t = np.vstack(soln)
    return X_t


# REQUIRED --- 1d
def propagate_LTV_system_numerically(X_0, x_dot_fcn, A_fcn, t_vec):
    # Return trajectory and STM over time where
    # np.shape(X_t_vec) = (len(t_vec), len(X_0))
    # np.shape(phi_t_vec) = (len(t_vec), len(X_0), len(X_0))
    N = 6
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


# REQUIRED --- 1e
def max_allowable_sampling_time(A):
    # compute the maximum allowable sampling time
    eigenvalues, dummy1 = np.linalg.eig(A)
    lambd = np.max(np.abs(eigenvalues))
    dt_max = np.pi/(2*lambd)
    return dt_max


#######################
# Problem 2
#######################


# REQUIRED --- Problem 2b
def plot_trajectories():
    t_end = 10 #sec
    X_0 = X_0_spring
    t_vec = np.linspace(0, t_end, 101)  

    A = np.array([[0, 1],
                  [-4, -0.5]])  
    
    def X_dot_fcn(t,X):
        return A @ X

    #Method 1: LTI CT numerically
    X_t_CT_n = propogate_CT_LTI_numerically_no_plot(X_0, X_dot_fcn, t_vec)

    #Method 2: LTI CT analytically
    X_t_CT_a = propogate_CT_LTI_analytically(X_0,A,t_vec)

    #Method 3: LTI DT analytically
    dt = 0.01
    k_max = int(t_end / dt)
    X_t_DT = propogate_DT_LTI_analytically(X_0, A, dt, k_max)

    #scale time vector for DT plot
    t_vec_DT = np.arange(0, (k_max + 1) * dt, dt)

    #plot
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t_vec, X_t_CT_n[:,0], 'b', label='CT Num x(t)')
    ax1.plot(t_vec, X_t_CT_a[:,0], 'r--', label='CT Ana x(t)')
    ax1.plot(t_vec_DT, X_t_DT[:,0], 'g:', label='DT Ana x(t)')
    ax1.set_ylabel('Displacement')
    ax1.legend()

    ax2.plot(t_vec, X_t_CT_n[:,1], 'b', label='CT Num xdot(t)')
    ax2.plot(t_vec, X_t_CT_a[:,1], 'r--', label='CT Ana xdot(t)')
    ax2.plot(t_vec_DT, X_t_DT[:,1], 'g:', label='DT Ana xdot(t)')
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('Velocity')
    ax2.legend()

    plt.tight_layout()
    plt.show()


    return fig


# REQUIRED --- Problem 2c
def describe_propagation_methods():
    return """
        All three propagation methods produce nearly identical results for this 
        damped spring mass system. The CT numerical and analytical solutions 
        overlap nearly exactly, confirming that the numerical integrator and 
        analytical STM yield the same trajectory. The DT analytical solution with 
        a small sampling period (dt = 0.01 s) the converges smoothly to the continuous 
        ones, showing that finer sampling accurately reproduces the continuous solution. 
        However, setting dt closer to the max sampling frequency (dt = 0.79 s) 
        causes the result to appears jagged and it does not reach the peaks that 
        are expected, oscillating wildly. This is the aliasing behavior discussed in class. 
    """


# REQUIRED --- Problem 2d
def determine_x0():
    data = load_numpy_data(cur_dir + "/HW3-spring-data.npy")
    t_meas = data["t"]
    y_meas = data["y"]

    # Method: X(0) = (QT * Q)-1 * QT * Y
    from scipy.linalg import expm

    A = np.array([[0, 1],
                  [-4, -0.5]])
    C = np.array([[1, 0]])

    Q = np.vstack([C @ expm(A * t) for t in t_meas])

    X0 = np.linalg.inv(Q.T @ Q) @ Q.T @ y_meas

    return X0  # dimension (2,)


# REQUIRED --- Problem 2e
def describe_observability():
    # compute how many measurements you need and return value and discription
    num_measurements = 2

    return f"""
        {num_measurements} are needed to observe the state, because with a 2D state, 
        one sample gives one equation and two samples provide two independent 
        equations - F = exp(A dt) with dt = 1 s makes C and CF linearly independent. 
        Fewer measurements is underdetermined.
    """


#######################
# Problem 3
#######################


# REQUIRED --- Problem 3b
def get_Ak(X_nom):
    r = X_nom[:3]
    v = X_nom[3:]

    rnorm = np.linalg.norm(r)
    G = mu*(3.0*np.outer(r, r)/rnorm**5 - np.eye(3)/rnorm**3)
    
    Ak = np.block([[np.zeros((3,3)), np.eye(3)],
                   [G, np.zeros((3,3))]])
    
    return Ak


# REQUIRED --- Problem 3c
def get_Ck(X_nom, R_obs):
    # X_nom : nominal state at time k
    # R_obs : observer position at time k

    r = X_nom[:3]
    d = r - R_obs
    Cpos = (d/np.linalg.norm(d)).reshape(1,3)
    Ck = np.hstack([Cpos, np.zeros((1,3))])

    return Ck


# REQUIRED --- Problem 3d
def plot_numerical_integration_dX():
    period = 90*60
    t_vec = np.linspace(0, period, 1000)

    def X_dot_fcn(t, X):
        r = X[:3]
        v = X[3:]
        r_norm = np.linalg.norm(r)
        a = -mu * r / (r_norm**3)
        return np.hstack((v, a)) 

    #integrate nominal
    X_nom = integrator(X_dot_fcn, X_N_0, t_vec).T
    
    #integrate nominal + perturbation
    X = integrator(X_dot_fcn, X_N_0+dX_N_0, t_vec).T

    del_r = X[:, :3] - X_nom[:, :3]
    del_r_norm = np.linalg.norm(del_r, axis=1)

    #plot
    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.title("3d.")
    ax[0].plot(t_vec/60.0, del_r[:,0], label='dx')
    ax[0].plot(t_vec/60.0, del_r[:,1], label='dy')
    ax[0].plot(t_vec/60.0, del_r[:,2], label='dz')
    ax[0].set_ylabel('delta r [km]')
    ax[0].legend()

    ax[1].plot(t_vec/60.0, del_r_norm)
    ax[1].set_xlabel('t [min]')
    ax[1].set_ylabel('|delta r| [km]')
    
    plt.tight_layout()
    plt.show()

    return fig


# REQUIRED --- Problem 3e
def plot_analytic_integration_dX():
    period = 90*60
    t_vec = np.linspace(0, period, 1000)

    def X_dot_fcn(t, X):
        r = X[:3]
        v = X[3:]
        r_norm = np.linalg.norm(r)
        a = -mu * r / (r_norm**3)
        return np.hstack((v, a)) 
    
    X_nom = integrator(X_dot_fcn, X_N_0, t_vec).T

    # Propagate STM and nominal
    X_vec, phi_vec = propagate_LTV_system_numerically(X_N_0, X_dot_fcn, get_Ak, t_vec)
    
    del_X = np.array([phi_vec[i] @ dX_N_0 for i in range(len(t_vec))])
    del_r = del_X[:, :3]
    del_r_norm = np.linalg.norm(del_r, axis=1)

    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.title("3e.")
    ax[0].plot(t_vec/60, del_r[:,0], label='dx')
    ax[0].plot(t_vec/60, del_r[:,1], label='dy')
    ax[0].plot(t_vec/60, del_r[:,2], label='dz')

    ax[0].set_ylabel('delta r [km]')
    ax[0].legend()
    ax[1].plot(t_vec/60, del_r_norm)
    ax[1].set_xlabel('t [min]')
    ax[1].set_ylabel('|delta r| [km]')

    plt.tight_layout()
    plt.show()

    return fig


# REQUIRED --- Problem 3f
def plot_critical_dX_neighborhood():
    dX_N_02 = np.array([1000, 0, 0, 0, 0, 0.1])
    period = 90*60
    t_vec = np.linspace(0, period, 1000)

    def X_dot_fcn(t, X):
        r = X[:3]
        v = X[3:]
        r_norm = np.linalg.norm(r)
        a = -mu * r / (r_norm**3)
        return np.hstack((v, a)) 
    
    ##numerical

    #integrate nominal
    X_nom = integrator(X_dot_fcn, X_N_0, t_vec).T
    
    #integrate nominal + perturbation
    X = integrator(X_dot_fcn, X_N_0+dX_N_02, t_vec).T

    del_r = X[:, :3] - X_nom[:, :3]
    del_r_norm = np.linalg.norm(del_r, axis=1)

    #plot
    fig1, ax1 = plt.subplots(2, 1, sharex=True)
    plt.title("3f. Numerical")
    ax1[0].plot(t_vec/60.0, del_r[:,0], label='dx')
    ax1[0].plot(t_vec/60.0, del_r[:,1], label='dy')
    ax1[0].plot(t_vec/60.0, del_r[:,2], label='dz')
    ax1[0].set_ylabel('delta r [km]')
    ax1[0].legend()

    ax1[1].plot(t_vec/60.0, del_r_norm)
    ax1[1].set_xlabel('t [min]')
    ax1[1].set_ylabel('|delta r| [km]')
    
    plt.tight_layout()
    plt.show()

    ##analytical    
    X_nom = integrator(X_dot_fcn, X_N_0, t_vec).T

    # Propagate STM and nominal
    X_vec, phi_vec = propagate_LTV_system_numerically(X_N_0, X_dot_fcn, get_Ak, t_vec)
    
    del_X = np.array([phi_vec[i] @ dX_N_02 for i in range(len(t_vec))])
    del_r = del_X[:, :3]
    del_r_norm = np.linalg.norm(del_r, axis=1)

    fig2, ax2 = plt.subplots(2, 1, sharex=True)
    plt.title("3f. Analytical")
    ax2[0].plot(t_vec/60, del_r[:,0], label='dx')
    ax2[0].plot(t_vec/60, del_r[:,1], label='dy')
    ax2[0].plot(t_vec/60, del_r[:,2], label='dz')

    ax2[0].set_ylabel('delta r [km]')
    ax2[0].legend()
    ax2[1].plot(t_vec/60, del_r_norm)
    ax2[1].set_xlabel('t [min]')
    ax2[1].set_ylabel('|delta r| [km]')

    plt.tight_layout()
    plt.show()
        
    fig = [fig1, fig2]

    return fig


# REQUIRED --- Problem 3g
def describe_neighborhood():
    return """I repeated both methods with a larger perturbation
            dX0 = [1000, 0, 0, 0, 0, 0.1]^T. Both trajectories grow much faster.
            the nonlinear deviation exceeds 1.5e4 km by 90 min, and the STM prediction
            diverges even more. The increasing mismatch shows the expected breakdown
            of the linear approximation as the state leaves the local neighborhood where
            first order terms dominate. Conclusion is that the STM is accurate for small
            deviations and short horizons, but for larger deviations and longer times
            the full nonlinear propagation should be used"""


# REQUIRED --- Problem 3h
def estimtae_dX0():
    data = load_numpy_data(cur_dir + "/HW3-kepler-data.npy")
    y_meas = data["rho"]
    t_meas = data["t"]

    def X_dot_fcn(t, X):
        r = X[:3]; v = X[3:]
        r_norm = np.linalg.norm(r)
        a = -mu * r / (r_norm**3)
        return np.hstack((v, a))

    # Nominal and STM at measurement times
    X_nom, phi_vec = propagate_LTV_system_numerically(X_N_0, X_dot_fcn, get_Ak, t_meas)

    # Build H and ranges
    H_rows = []
    y_nom = []
    for i in range(len(t_meas)):
        t = t_meas[i]
        R_obs_t = r_station_eci(t)          
        C = get_Ck(X_nom[i], R_obs_t)       
        H_rows.append(C @ phi_vec[i])       
        r_nom = X_nom[i, :3]
        y_nom.append(np.linalg.norm(r_nom - R_obs_t))

    H = np.vstack(H_rows)                   
    y_nom = np.array(y_nom)                 
    delta_y = y_meas - y_nom                

    # Least squares without explicit inverse
    dX0_estimated = np.linalg.solve(H.T @ H, H.T @ delta_y)
    return dX0_estimated


# REQUIRED --- Problem 3hi
def explain_approach():
    return """
    Part (h) estimates the initial state deviation del_x(t0) from range measurements. 
    Using the same nominal trajectory as before, the function propagates both the 
    state and STM, then computes the predicted range to the ground station for each 
    measurement time. The partial derivative of range with respect to the state is 
    used to build the matrix H = C(t)*phi(t,t0). The difference between 
    measured and nominal ranges (delta_rho) is then related to the initial deviation by 
    delata_rho = H*del_x(t0). The solution is inverted/transposed and solved for del_x(t0)
    """


###############################################
# Main Script to test / debug your code
# This will not be run by the autograder
# the individual functions above will be called and tested
###############################################


def main():
    # Problem 1
    # propogate_CT_LTI_numerically
    # propogate_CT_LTI_analytically
    # propogate_DT_LTI_analytically
    # propagate_LTV_system_numerically
    # max_allowable_sampling_time

    # Problem 2
    plot_trajectories()
    describe_propagation_methods()
    print(determine_x0())
    describe_observability()

    # Problem 3
    # get_Ak(X_N_0)
    # get_Ck(X_N_0)
    plot_numerical_integration_dX()
    plot_analytic_integration_dX()
    plot_critical_dX_neighborhood()
    describe_neighborhood()
    print(estimtae_dX0())
    explain_approach()
    plt.show()


if __name__ == "__main__":
    main()
