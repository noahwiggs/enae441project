import numpy as np

def meas_func(x, r_site, v_site):
    r = x[:3]
    v = x[3:]

    rho_vec = r - r_site
    rho = np.linalg.norm(rho_vec)
    rho_dot = np.dot(rho_vec, v - v_site) / rho

    return np.array([rho, rho_dot])


def H_func(x, r_site, v_site):
    r = x[:3]
    v = x[3:]

    rho_vec = r - r_site
    rho = np.linalg.norm(rho_vec)
    rho_hat = rho_vec / rho
    v_rel = v - v_site

    H = np.zeros((2, 6))
    H[0, 0:3] = rho_hat
    H[1, 0:3] = (v_rel / rho) - (np.dot(rho_hat, v_rel) / rho) * rho_hat
    H[1, 3:6] = rho_hat

    return H