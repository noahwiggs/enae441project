import numpy as np

R_Earth = 6378.137
omega_EN = 7.292115e-5
gamma_0 = 0.0

DSN_LOCS = {
    0: (35.297, -116.914),
    1: (40.4311, -4.248),
    2: (-35.4023, 148.9813),
}

def stations_eci_func(t, site_id):
    lat_deg, lon_deg = DSN_LOCS[site_id]
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    theta = omega_EN * t + gamma_0

    r_ecef = R_Earth * np.array([
        np.cos(lat)*np.cos(lon),
        np.cos(lat)*np.sin(lon),
        np.sin(lat)
    ])

    R3 = np.array([
        [ np.cos(theta), -np.sin(theta), 0],
        [ np.sin(theta),  np.cos(theta), 0],
        [ 0,              0,             1]
    ])

    r_eci = R3 @ r_ecef
    v_eci = np.cross([0, 0, omega_EN], r_eci)

    return r_eci, v_eci
