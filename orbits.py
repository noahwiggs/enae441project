import numpy as np

mu = 3.986e5  # km^3/s^2
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